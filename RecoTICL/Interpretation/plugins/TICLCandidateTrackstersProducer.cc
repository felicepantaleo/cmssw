// Author: Felice Pantaleo (CERN) - felice.pantaleo@cern.ch
// Date: 11/2026
//
// One trackster per TICL candidate with calorimeter energy, made of layer clusters: those of all the tracksters of
// the candidate, and of the jet trackster of a jet-member track. A layer cluster of a trackster that several
// candidates hold is shared in proportion to the candidate energy (assignment "energyShare"), or goes whole to the
// candidate whose track passes nearest, within radius0 + radiusSlope x depth, else to the largest neutral candidate
// that holds it, else to the nearest track (assignment "nearest"). The second product is the candidate index of each
// trackster.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"
#include "RecoTICL/Common/interface/TICLUtils.h"
#include "RecoTICL/Common/interface/TrackstersPCA.h"
#include "RecoTICL/Interpretation/interface/AssignmentMaps.h"
#include "RecoTICL/Interpretation/interface/TrackImpact.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

using namespace ticl;

class TICLCandidateTrackstersProducer : public edm::stream::EDProducer<edm::stream::WatchRuns> {
public:
  explicit TICLCandidateTrackstersProducer(const edm::ParameterSet &ps);
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginRun(edm::Run const &, edm::EventSetup const &es) override;
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  void produce(edm::Event &evt, const edm::EventSetup &es) override;

  const edm::EDGetTokenT<std::vector<TICLCandidate>> candidates_token_;
  const edm::EDGetTokenT<std::vector<Trackster>> tracksters_token_;
  const edm::EDGetTokenT<std::vector<int>> trackToTrackster_token_;
  const edm::EDGetTokenT<std::vector<int>> trackMode_token_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClusters_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> layerClustersTime_token_;
  const edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
  const edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
  const edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hdc_token_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bfield_token_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagator_token_;
  const bool nearest_;
  const float radius0_;
  const float radiusSlope_;
  ticlgeom::Tools rhtools_;
  edm::ESHandle<MagneticField> bfield_;
  edm::ESHandle<Propagator> propagator_;
  std::array<std::unique_ptr<GeomDet>, 2> frontDisks_;
};

TICLCandidateTrackstersProducer::TICLCandidateTrackstersProducer(const edm::ParameterSet &ps)
    : candidates_token_(consumes<std::vector<TICLCandidate>>(ps.getParameter<edm::InputTag>("candidates"))),
      tracksters_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("interpretations"))),
      trackToTrackster_token_(consumes<std::vector<int>>(
          edm::InputTag(ps.getParameter<edm::InputTag>("interpretations").label(), "trackToTrackster"))),
      trackMode_token_(consumes<std::vector<int>>(
          edm::InputTag(ps.getParameter<edm::InputTag>("interpretations").label(), "trackMode"))),
      layerClusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layerClusters"))),
      layerClustersTime_token_(
          consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("layerClustersTime"))),
      ticlGeomToken_(esConsumes<TICLGeomHost, CaloGeometryRecord, edm::Transition::BeginRun>()),
      ticlGeomLookupToken_(esConsumes<TICLGeomLookupHost, CaloGeometryRecord, edm::Transition::BeginRun>()),
      ticlGeomLayersToken_(esConsumes<TICLGeomLayersHost, CaloGeometryRecord, edm::Transition::BeginRun>()),
      hdc_token_(esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag("", "HGCalEESensitive"))),
      bfield_token_(esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()),
      propagator_token_(esConsumes<Propagator, TrackingComponentsRecord, edm::Transition::BeginRun>(
          edm::ESInputTag("", ps.getParameter<std::string>("propagator")))),
      nearest_(ps.getParameter<std::string>("assignment") == "nearest"),
      radius0_(ps.getParameter<double>("radius0")),
      radiusSlope_(ps.getParameter<double>("radiusSlope")) {
  const auto &assignment = ps.getParameter<std::string>("assignment");
  if (assignment != "nearest" && assignment != "energyShare")
    throw cms::Exception("Configuration") << "assignment must be nearest or energyShare, not " << assignment;
  produces<std::vector<Trackster>>();
  produces<std::vector<int>>("candidateIndex");
}

void TICLCandidateTrackstersProducer::beginRun(edm::Run const &, edm::EventSetup const &es) {
  rhtools_.setGeometry(es.getData(ticlGeomToken_), es.getData(ticlGeomLookupToken_), es.getData(ticlGeomLayersToken_));
  bfield_ = es.getHandle(bfield_token_);
  propagator_ = es.getHandle(propagator_token_);
  frontDisks_ = utils::buildHGCalFirstDisks(es.getData(hdc_token_));
}

void TICLCandidateTrackstersProducer::produce(edm::Event &evt, const edm::EventSetup &) {
  auto const &candidates = evt.get(candidates_token_);
  const auto tracksters_h = evt.getHandle(tracksters_token_);
  auto const &tracksters = *tracksters_h;
  auto const &trackToTrackster = evt.get(trackToTrackster_token_);
  auto const &trackMode = evt.get(trackMode_token_);
  auto const &layerClusters = evt.get(layerClusters_token_);

  // The candidates that hold each trackster, and their summed energy.
  std::vector<std::vector<unsigned int>> holders(tracksters.size());
  std::vector<double> energySum(tracksters.size(), 0.);
  for (unsigned int c = 0; c < candidates.size(); ++c) {
    std::vector<unsigned int> mine;
    for (auto const &ptr : candidates[c].tracksters())
      if (ptr.isNonnull() && ptr.id() == tracksters_h.id())
        mine.push_back(ptr.key());
    auto const track = candidates[c].trackPtr();
    if (track.isNonnull() && track.key() < trackMode.size() &&
        trackMode[track.key()] == static_cast<int>(TrackMode::kJetMember) && trackToTrackster[track.key()] >= 0)
      mine.push_back(trackToTrackster[track.key()]);
    std::sort(mine.begin(), mine.end());
    mine.erase(std::unique(mine.begin(), mine.end()), mine.end());
    for (auto t : mine) {
      holders[t].push_back(c);
      energySum[t] += candidates[c].energy();
    }
  }

  // The HGCAL impact of the track of each candidate, computed when a shared layer cluster needs it.
  std::vector<std::optional<TrackImpact>> impacts(candidates.size());
  auto impactOf = [&](unsigned int c) -> const TrackImpact & {
    if (!impacts[c])
      impacts[c] = propagateToHGCalFront(*candidates[c].trackPtr(), bfield_.product(), *propagator_, frontDisks_);
    return *impacts[c];
  };

  // The fraction of each layer cluster that each candidate takes.
  std::vector<std::map<unsigned int, float>> fractions(candidates.size());
  for (unsigned int t = 0; t < tracksters.size(); ++t) {
    auto const &owners = holders[t];
    if (owners.empty() || energySum[t] <= 0.)
      continue;
    auto const &ts = tracksters[t];
    for (unsigned int k = 0; k < ts.vertices().size(); ++k) {
      const unsigned int lc = ts.vertices()[k];
      const float f = 1.f / ts.vertex_multiplicity()[k];
      if (owners.size() == 1) {
        fractions[owners[0]][lc] += f;
        continue;
      }
      if (!nearest_) {
        for (auto c : owners)
          fractions[c][lc] += f * candidates[c].energy() / energySum[t];
        continue;
      }
      auto const &cluster = layerClusters[lc];
      int closest = -1, neutral = -1;
      float best = std::numeric_limits<float>::infinity();
      float bestRadius = 0.f;
      for (auto c : owners) {
        if (candidates[c].trackPtr().isNull()) {
          if (neutral < 0 || candidates[c].energy() > candidates[neutral].energy())
            neutral = c;
          continue;
        }
        auto const &impact = impactOf(c);
        const float d = impactTransverseDistance(impact, cluster.position());
        if (d < best) {
          best = d;
          closest = c;
          bestRadius =
              radius0_ +
              radiusSlope_ * std::max(0.f, std::abs(static_cast<float>(cluster.z())) - std::abs(impact.position.z()));
        }
      }
      const int winner = closest >= 0 && best < bestRadius ? closest : neutral >= 0 ? neutral : closest;
      if (winner < 0) {
        for (auto c : owners)
          fractions[c][lc] += f * candidates[c].energy() / energySum[t];
        continue;
      }
      fractions[winner][lc] += f;
    }
  }

  auto result = std::make_unique<std::vector<Trackster>>();
  auto candidateIndex = std::make_unique<std::vector<int>>();
  for (unsigned int c = 0; c < candidates.size(); ++c) {
    Trackster merged;
    for (auto const &[lc, f] : fractions[c])
      if (f > 0.f) {
        merged.vertices().push_back(lc);
        merged.vertex_multiplicity().push_back(1.f / f);
      }
    if (merged.vertices().empty())
      continue;
    result->push_back(std::move(merged));
    candidateIndex->push_back(c);
  }
  assignPCAtoTracksters(*result,
                        layerClusters,
                        evt.get(layerClustersTime_token_),
                        rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z(),
                        rhtools_,
                        true);
  evt.put(std::move(result));
  evt.put(std::move(candidateIndex), "candidateIndex");
}

void TICLCandidateTrackstersProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("candidates", edm::InputTag("ticlCandidate"));
  desc.add<edm::InputTag>("interpretations", edm::InputTag("ticlTracksterInterpretations"))
      ->setComment("The final tracksters and the assignment maps of the interpretation stage.");
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("layerClustersTime", edm::InputTag("hgcalMergeLayerClusters", "timeLayerCluster"));
  desc.add<std::string>("assignment", "nearest")
      ->setComment("A layer cluster that several candidates hold: nearest (to one candidate) or energyShare.");
  desc.add<double>("radius0", 3.)->setComment("Track radius [cm] at the HGCAL front for the nearest assignment.");
  desc.add<double>("radiusSlope", 0.05)->setComment("Growth of the track radius per cm of depth.");
  desc.add<std::string>("propagator", "PropagatorWithMaterial");
  descriptions.add("ticlCandidateTrackstersProducer", desc);
}

DEFINE_FWK_MODULE(TICLCandidateTrackstersProducer);
