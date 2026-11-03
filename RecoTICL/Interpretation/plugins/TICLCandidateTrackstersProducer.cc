// Author: Felice Pantaleo (CERN) - felice.pantaleo@cern.ch
// Date: 11/2026
//
// One trackster per TICL candidate with calorimeter energy, made of layer clusters by
// ticl::candidateLayerClusterFractions (CandidateTracksters.h). The second product is the candidate index of each
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
#include "RecoTICL/Interpretation/interface/CandidateTracksters.h"
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
  const CandidateTrackstersParameters parameters_;
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
      parameters_{.nearest = ps.getParameter<std::string>("assignment") == "nearest",
                  .radius0 = static_cast<float>(ps.getParameter<double>("radius0")),
                  .radiusSlope = static_cast<float>(ps.getParameter<double>("radiusSlope")),
                  .budgetScale = static_cast<float>(ps.getParameter<double>("budgetScale")),
                  .deposit = {static_cast<float>(ps.getParameter<double>("expectedDepositResponse")),
                              static_cast<float>(ps.getParameter<double>("expectedDepositLogSlope")),
                              0.f,
                              1.f}} {
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

  // The HGCAL impact of the track of each candidate, computed when a shared layer cluster needs it.
  std::vector<std::optional<TrackImpact>> impacts(candidates.size());
  auto impactOf = [&](unsigned int c) -> const TrackImpact & {
    if (!impacts[c])
      impacts[c] = propagateToHGCalFront(*candidates[c].trackPtr(), bfield_.product(), *propagator_, frontDisks_);
    return *impacts[c];
  };
  const auto fractions = candidateLayerClusterFractions(
      candidates, tracksters_h.id(), tracksters, trackToTrackster, trackMode, layerClusters, impactOf, parameters_);
  auto candidateIndex = std::make_unique<std::vector<int>>();
  auto result = std::make_unique<std::vector<Trackster>>(tracksterOfFractions(fractions, *candidateIndex));
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
  desc.add<double>("budgetScale", 0.)
      ->setComment(
          "A charged hadron takes shared layer clusters up to this times its expected raw deposit; 0: no limit.");
  desc.add<double>("expectedDepositResponse", 0.67)->setComment("Expected raw deposit (response + logSlope ln p) p.");
  desc.add<double>("expectedDepositLogSlope", 0.04);
  desc.add<std::string>("propagator", "PropagatorWithMaterial");
  descriptions.add("ticlCandidateTrackstersProducer", desc);
}

DEFINE_FWK_MODULE(TICLCandidateTrackstersProducer);
