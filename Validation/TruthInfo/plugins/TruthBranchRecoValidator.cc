// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// Truth-branch validation, one plugin template covering every reco domain: one folder
// per (collection, working point), booking only num/denom so all harvesting stays
// DQMGenericClient string config.
//
// DQMGlobalEDAnalyzer, not DQMEDAnalyzer: booking and filling are both const and the
// MonitorElements live in a per-run cache, which is the modern convention shared by
// MultiTrackValidator and HGCalValidator.
//
// What differs between domains is only (a) which association map type the associator
// wrote and (b) how to read kinematics off a reco object. Both are bound to the reco
// type by RecoValidationTraits, mirroring TruthAssociationTraits on the producer side,
// so the declared product type and the consumed one cannot drift apart.

#include <cmath>
#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/VertexData.h"

#include "Validation/TruthInfo/interface/TruthBranchHistoProducerAlgo.h"

namespace {
  using Kinematics = truth::TruthBranchHistoProducerAlgo::Kinematics;

  template <typename RECO>
  struct RecoValidationTraits;

  template <>
  struct RecoValidationTraits<reco::Track> {
    using MapType = ticl::AssociationMap<ticl::mapWithSharedHitsAndScore>;
    static constexpr const char* cfiName = "truthBranchTrackValidator";
    static constexpr const char* defaultAssociator = "allTrackToTruthBranchAssociators";
    static constexpr const char* defaultDir = "TruthInfo/Tracking/";

    static Kinematics kinematics(reco::Track const& track) {
      Kinematics kin;
      kin.pt = track.pt();
      kin.eta = track.eta();
      kin.phi = track.phi();
      kin.nhits = track.numberOfValidHits();
      kin.vertpos = std::sqrt(track.vx() * track.vx() + track.vy() * track.vy());
      kin.zpos = track.vz();
      kin.dxy = track.dxy();
      kin.dz = track.dz();
      return kin;
    }
    static bool hasDirection(reco::Track const&) { return true; }
    // The truth side of a hit-based domain iterates branch roots, which are particles.
    static constexpr bool truthIsVertex = false;
    static constexpr const char* denominatorInstance = "selectedBranchRoots";
  };

  // A vertex has no momentum, so only its position and its track multiplicity are
  // meaningful; the configuration books exactly those and nothing else.
  template <>
  struct RecoValidationTraits<reco::Vertex> {
    using MapType = ticl::AssociationMap<ticl::mapWithFractionAndScore>;
    static constexpr const char* cfiName = "truthBranchVertexValidator";
    static constexpr const char* defaultAssociator = "allVertexToTruthBranchAssociators";
    static constexpr const char* defaultDir = "TruthInfo/Vertexing/";

    static Kinematics kinematics(reco::Vertex const& vertex) {
      Kinematics kin;
      // The number of tracks the vertex was built from, which is the vertex analogue of
      // a track's hit count: the constituents its truth was aggregated from.
      kin.nhits = vertex.tracksSize();
      kin.vertpos = std::sqrt(vertex.x() * vertex.x() + vertex.y() * vertex.y());
      kin.zpos = vertex.z();
      return kin;
    }
    static bool hasDirection(reco::Vertex const&) { return false; }
    // A composite object is associated to a truth VERTEX, so the truth side iterates
    // vertices and the denominator is the set of reconstructable ones.
    static constexpr bool truthIsVertex = true;
    static constexpr const char* denominatorInstance = "selectedTruthVertices";
  };

  // A trackster carries calorimeter energy through its layer clusters. Its momentum
  // direction is the barycentre, and its hit count is the number of layer clusters.
  template <>
  struct RecoValidationTraits<ticl::Trackster> {
    using MapType = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>;
    static constexpr const char* cfiName = "truthBranchTracksterValidator";
    static constexpr const char* defaultAssociator = "truthBranchTracksterAssociators";
    static constexpr const char* defaultDir = "TruthInfo/Calorimetry/";

    static Kinematics kinematics(ticl::Trackster const& trackster) {
      Kinematics kin;
      auto const& bary = trackster.barycenter();
      const double rho = std::sqrt(bary.x() * bary.x() + bary.y() * bary.y());
      const double mag = std::sqrt(rho * rho + bary.z() * bary.z());
      // Energy shared out along the barycentre direction: a trackster has no track, so
      // its transverse momentum is the raw energy projected transversally.
      kin.pt = (mag > 0.) ? trackster.raw_energy() * rho / mag : 0.;
      kin.eta = bary.eta();
      kin.phi = bary.phi();
      kin.nhits = trackster.vertices().size();
      kin.vertpos = rho;
      kin.zpos = bary.z();
      return kin;
    }
    static bool hasDirection(ticl::Trackster const&) { return true; }
    static constexpr bool truthIsVertex = false;
    static constexpr const char* denominatorInstance = "selectedBranchRoots";
  };
}  // namespace

template <typename RECO>
class TruthBranchRecoValidator : public DQMGlobalEDAnalyzer<truth::TruthBranchHistograms> {
public:
  using Histograms = truth::TruthBranchHistograms;
  using Traits = RecoValidationTraits<RECO>;
  using MapType = typename Traits::MapType;

  explicit TruthBranchRecoValidator(edm::ParameterSet const&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, Histograms&) const override;
  void dqmAnalyze(edm::Event const&, edm::EventSetup const&, Histograms const&) const override;

  // One entry per (collection, working point), in booking order.
  struct Entry {
    std::string folder;
    edm::EDGetTokenT<std::vector<RECO>> recoToken;
    edm::EDGetTokenT<MapType> recoToSimToken;
    edm::EDGetTokenT<MapType> simToRecoToken;
  };

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  edm::EDGetTokenT<std::vector<unsigned int>> selectedRootsToken_;
  const std::string dirName_;
  std::vector<Entry> entries_;
  const truth::TruthBranchHistoProducerAlgo algo_;
};

template <typename RECO>
TruthBranchRecoValidator<RECO>::TruthBranchRecoValidator(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      dirName_(cfg.getParameter<std::string>("dirName")),
      algo_(cfg.getParameter<edm::ParameterSet>("histoProducerAlgoBlock")) {
  const auto associator = cfg.getParameter<std::string>("associator");
  // Same candidate set the associator used, so the denominator counts only branches
  // that were ever eligible to be found.
  selectedRootsToken_ = consumes<std::vector<unsigned int>>(edm::InputTag(associator, Traits::denominatorInstance));
  const auto workingPoints = cfg.getParameter<std::vector<std::string>>("workingPoints");

  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("recoCollections")) {
    // The one key rule of this package: label and instance joined by an underscore,
    // used for the product instance labels AND for the folder name.
    std::string key = tag.label();
    if (!tag.instance().empty()) {
      key += "_" + tag.instance();
    }
    for (auto const& wp : workingPoints) {
      Entry entry;
      entry.folder = key + "_" + wp;
      entry.recoToken = consumes<std::vector<RECO>>(tag);
      entry.recoToSimToken = consumes<MapType>(edm::InputTag(associator, key + "ToTruthBranch" + wp));
      entry.simToRecoToken = consumes<MapType>(edm::InputTag(associator, "TruthBranchTo" + key + wp));
      entries_.push_back(std::move(entry));
    }
  }
}

template <typename RECO>
void TruthBranchRecoValidator<RECO>::bookHistograms(DQMStore::IBooker& booker,
                                                    edm::Run const&,
                                                    edm::EventSetup const&,
                                                    Histograms& histograms) const {
  for (auto const& entry : entries_) {
    booker.setCurrentFolder(dirName_ + entry.folder);
    algo_.bookHistos(booker, histograms);
  }
}

template <typename RECO>
void TruthBranchRecoValidator<RECO>::dqmAnalyze(edm::Event const& event,
                                                edm::EventSetup const&,
                                                Histograms const& histograms) const {
  auto const& graph = event.get(graphToken_);
  auto const& hitIndex = event.get(hitIndexToken_);

  for (std::size_t i = 0; i < entries_.size(); ++i) {
    auto const& entry = entries_[i];

    edm::Handle<std::vector<RECO>> recoHandle;
    event.getByToken(entry.recoToken, recoHandle);
    edm::Handle<MapType> recoToSimHandle;
    event.getByToken(entry.recoToSimToken, recoToSimHandle);
    edm::Handle<MapType> simToRecoHandle;
    event.getByToken(entry.simToRecoToken, simToRecoHandle);
    if (!recoHandle.isValid() || !recoToSimHandle.isValid() || !simToRecoHandle.isValid()) {
      continue;
    }

    auto const& recoToSim = recoToSimHandle->getMap();
    auto const& simToReco = simToRecoHandle->getMap();

    // Reco side: every object, whether it found a branch, and whether that branch came
    // from a pileup interaction rather than the signal one.
    for (std::size_t r = 0; r < recoHandle->size(); ++r) {
      const bool associated = r < recoToSim.size() && !recoToSim[r].empty();
      const Kinematics kin = Traits::kinematics((*recoHandle)[r]);

      bool pileup = false;
      if (associated) {
        // eventId 0 is the signal interaction; anything else is overlaid pileup. The
        // row index means a truth vertex for a composite domain and a particle for a
        // hit-based one, so the lookup follows the same split.
        const unsigned int matched = recoToSim[r][0].index();
        if constexpr (Traits::truthIsVertex) {
          if (matched < graph.nVertices()) {
            pileup = graph.vertices()[matched].eventId != 0;
          }
        } else {
          if (matched < graph.nParticles()) {
            pileup = graph.particles()[matched].eventId != 0;
          }
        }
      }
      // For a composite object the association always finds something, so counting the
      // match tells nothing; what it is worth is the leading truth vertex's share of the
      // object's constituents. Constituents whose particles were produced at an
      // unrelated vertex are the remainder.
      double matchQuality = 1.;
      if constexpr (Traits::truthIsVertex) {
        matchQuality = associated ? recoToSim[r][0].value() : 0.;
      }
      algo_.fill_reco(histograms, i, kin, associated, pileup, matchQuality);
      if (associated) {
        algo_.fill_match(histograms, i, recoToSim[r][0].score(), recoToSim[r][0].value());
      }
    }

    // Truth side: ONLY the branches the associator selected as candidates. Iterating
    // every particle instead would put the rejected ones in the denominator as
    // guaranteed misses and scale every efficiency down by the rejection factor.
    edm::Handle<std::vector<unsigned int>> rootsHandle;
    event.getByToken(selectedRootsToken_, rootsHandle);
    if (!rootsHandle.isValid()) {
      continue;
    }
    for (unsigned int b : *rootsHandle) {
      if (b >= simToReco.size()) {
        continue;
      }
      Kinematics kin;
      auto reason = static_cast<unsigned int>(truth::VertexReason::Unknown);

      if constexpr (Traits::truthIsVertex) {
        // The truth object IS a vertex: its position, how many selected particles were
        // produced there, and the Geant4 process that made it. depth and rootfrac are
        // properties of a particle branch and are not booked for this domain.
        if (b >= graph.nVertices()) {
          continue;
        }
        const truth::Vertex vertex(&graph, b);
        auto const& vdata = vertex.data();
        auto const& pos = vertex.position();
        kin.vertpos = std::sqrt(pos.x() * pos.x() + pos.y() * pos.y());
        kin.zpos = pos.z();
        kin.nhits = vertex.outgoingParticles().size();
        reason = vdata.hasSim() ? static_cast<unsigned int>(vdata.reason)
                                : static_cast<unsigned int>(truth::VertexReason::Other) + 1;
      } else {
        if (b >= graph.nParticles()) {
          continue;
        }
        auto const& particle = graph.particles()[b];
        const auto& p4 = particle.momentum;
        if (p4.pt() <= 0.) {
          continue;
        }
        kin.pt = p4.pt();
        kin.eta = p4.eta();
        kin.phi = p4.phi();
        // The branch's own detector footprint, which is the truth analogue of a track's
        // hit count.
        const auto subgraph = hitIndex.subgraphHits(truth::HitChannel::Tracker, b);
        kin.nhits = subgraph.size();
        const truth::Particle branchRoot(&graph, b);
        // How deep in the graph the branch root sits. A frozen truth object has one
        // fixed level and no such axis.
        kin.depth = branchRoot.ancestors().size();
        // How much of the branch footprint is the root particle's own hits rather than
        // its descendants'. Near 1 is a clean single particle, near 0 a branch whose
        // hits all come from what it produced.
        const auto direct = hitIndex.directHits(truth::HitChannel::Tracker, b);
        kin.rootfrac = subgraph.empty() ? 0. : static_cast<double>(direct.size()) / subgraph.size();

        const auto vertices = branchRoot.productionVertices();
        if (!vertices.empty()) {
          // A GEN-only production vertex has no Geant4 creator process, so its reason is
          // Unknown by construction rather than by failure to classify. It gets its own
          // bin, one past the enum, so the two do not get read as the same thing.
          auto const& vdata = vertices.front().data();
          reason = vdata.hasSim() ? static_cast<unsigned int>(vdata.reason)
                                  : static_cast<unsigned int>(truth::VertexReason::Other) + 1;
          const auto& pos = vertices.front().position();
          kin.vertpos = std::sqrt(pos.x() * pos.x() + pos.y() * pos.y());
          kin.zpos = pos.z();
          // Transverse and longitudinal impact parameter of the branch direction with
          // respect to the origin, the truth counterpart of the track dxy and dz.
          kin.dxy = (-pos.x() * p4.py() + pos.y() * p4.px()) / p4.pt();
          kin.dz = pos.z() - (pos.x() * p4.px() + pos.y() * p4.py()) / p4.pt() * (p4.pz() / p4.pt());
        }
      }

      const bool associated = !simToReco[b].empty();
      const bool duplicate = simToReco[b].size() > 1;
      algo_.fill_simul(histograms, i, kin, associated, duplicate);
      algo_.fill_reason(histograms, i, reason, associated, duplicate);
      if (associated) {
        const unsigned int r = simToReco[b][0].index();
        if (r < recoHandle->size()) {
          auto const& matched = (*recoHandle)[r];
          // A vertex has no direction, so a pt or angular residual against it would be
          // a residual against zero; only domains with a direction fill these.
          if (Traits::hasDirection(matched)) {
            const Kinematics recoKin = Traits::kinematics(matched);
            algo_.fill_resolution(histograms, i, kin, recoKin.pt, recoKin.eta, recoKin.phi);
          }
        }
      }
    }
  }
}

template <typename RECO>
void TruthBranchRecoValidator<RECO>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<std::string>("dirName", Traits::defaultDir);
  desc.add<std::string>("associator", Traits::defaultAssociator);
  desc.add<std::vector<edm::InputTag>>("recoCollections", {});
  desc.add<std::vector<std::string>>("workingPoints", {"Fixed"});

  edm::ParameterSetDescription algo;
  // Every axis is declared here; which of them a domain books is chosen by the two
  // variable lists, so adding a domain needs no new axis parameter.
  const std::vector<std::tuple<std::string, int, double, double>> axes = {{"pt", 50, 0., 100.},
                                                                          {"eta", 50, -4., 4.},
                                                                          {"phi", 36, -3.2, 3.2},
                                                                          {"nhits", 40, 0., 40.},
                                                                          {"vertpos", 40, 0., 60.},
                                                                          {"zpos", 40, -30., 30.},
                                                                          {"dxy", 40, -5., 5.},
                                                                          {"dz", 40, -20., 20.},
                                                                          {"depth", 15, 0., 15.},
                                                                          {"rootfrac", 20, 0., 1.}};
  for (auto const& [name, nbins, lo, hi] : axes) {
    algo.add<int>("nint_" + name, nbins);
    algo.add<double>("min_" + name, lo);
    algo.add<double>("max_" + name, hi);
  }
  algo.add<std::vector<std::string>>("truthVariables", {"pt", "eta", "phi"});
  algo.add<std::vector<std::string>>("recoVariables", {"pt", "eta", "phi"});
  algo.add<int>("nintScore", 50);
  algo.add<double>("minScore", 0.);
  algo.add<double>("maxScore", 1.);
  algo.add<int>("nintShared", 50);
  algo.add<double>("minShared", 0.);
  algo.add<double>("maxShared", 50.);
  algo.add<int>("nintRes", 120);
  algo.add<double>("minRes", -1.5);
  algo.add<double>("maxRes", 1.5);
  algo.add<int>("nint_res_eta", 20);
  algo.add<double>("min_res_eta", -4.);
  algo.add<double>("max_res_eta", 4.);
  algo.add<int>("nint_res_pt", 15);
  algo.add<double>("min_res_pt", 0.);
  algo.add<double>("max_res_pt", 100.);
  desc.add<edm::ParameterSetDescription>("histoProducerAlgoBlock", algo);

  descriptions.add(Traits::cfiName, desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using TruthBranchTrackValidator = TruthBranchRecoValidator<reco::Track>;
DEFINE_FWK_MODULE(TruthBranchTrackValidator);
using TruthBranchVertexValidator = TruthBranchRecoValidator<reco::Vertex>;
DEFINE_FWK_MODULE(TruthBranchVertexValidator);
using TruthBranchTracksterValidator = TruthBranchRecoValidator<ticl::Trackster>;
DEFINE_FWK_MODULE(TruthBranchTracksterValidator);
