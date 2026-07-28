// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// Truth-branch validation for the track-like domains: one folder per (collection,
// working point), booking only num/denom so all harvesting is DQMGenericClient config.
//
// DQMGlobalEDAnalyzer, not DQMEDAnalyzer: booking and filling are both const and the
// MonitorElements live in a per-run cache, which is the modern convention shared by
// MultiTrackValidator and HGCalValidator.

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/VertexData.h"

#include "Validation/TruthInfo/interface/TruthBranchHistoProducerAlgo.h"

namespace {
  using SharedHitsMap = ticl::AssociationMap<ticl::mapWithSharedHitsAndScore>;

  struct Histograms {
    truth::TruthBranchHistograms histos;
  };
}  // namespace

class TruthBranchTrackValidator : public DQMGlobalEDAnalyzer<Histograms> {
public:
  explicit TruthBranchTrackValidator(edm::ParameterSet const&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, Histograms&) const override;
  void dqmAnalyze(edm::Event const&, edm::EventSetup const&, Histograms const&) const override;

  // One entry per (collection, working point), in booking order.
  struct Entry {
    std::string folder;
    edm::EDGetTokenT<std::vector<reco::Track>> recoToken;
    edm::EDGetTokenT<SharedHitsMap> recoToSimToken;
    edm::EDGetTokenT<SharedHitsMap> simToRecoToken;
  };

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  edm::EDGetTokenT<std::vector<unsigned int>> selectedRootsToken_;
  const std::string dirName_;
  std::vector<Entry> entries_;
  const truth::TruthBranchHistoProducerAlgo algo_;
};

TruthBranchTrackValidator::TruthBranchTrackValidator(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      dirName_(cfg.getParameter<std::string>("dirName")),
      algo_(cfg.getParameter<edm::ParameterSet>("histoProducerAlgoBlock")) {
  const auto associator = cfg.getParameter<std::string>("associator");
  // Same candidate set the associator used, so the denominator counts only branches
  // that were ever eligible to be found.
  selectedRootsToken_ = consumes<std::vector<unsigned int>>(edm::InputTag(associator, "selectedBranchRoots"));
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
      entry.recoToken = consumes<std::vector<reco::Track>>(tag);
      entry.recoToSimToken = consumes<SharedHitsMap>(edm::InputTag(associator, key + "ToTruthBranch" + wp));
      entry.simToRecoToken = consumes<SharedHitsMap>(edm::InputTag(associator, "TruthBranchTo" + key + wp));
      entries_.push_back(std::move(entry));
    }
  }
}

void TruthBranchTrackValidator::bookHistograms(DQMStore::IBooker& booker,
                                               edm::Run const&,
                                               edm::EventSetup const&,
                                               Histograms& histograms) const {
  for (auto const& entry : entries_) {
    booker.setCurrentFolder(dirName_ + entry.folder);
    algo_.bookHistos(booker, histograms.histos);
  }
}

void TruthBranchTrackValidator::dqmAnalyze(edm::Event const& event,
                                           edm::EventSetup const&,
                                           Histograms const& histograms) const {
  auto const& graph = event.get(graphToken_);

  for (std::size_t i = 0; i < entries_.size(); ++i) {
    auto const& entry = entries_[i];

    edm::Handle<std::vector<reco::Track>> recoHandle;
    event.getByToken(entry.recoToken, recoHandle);
    edm::Handle<SharedHitsMap> recoToSimHandle;
    event.getByToken(entry.recoToSimToken, recoToSimHandle);
    edm::Handle<SharedHitsMap> simToRecoHandle;
    event.getByToken(entry.simToRecoToken, simToRecoHandle);
    if (!recoHandle.isValid() || !recoToSimHandle.isValid() || !simToRecoHandle.isValid()) {
      continue;
    }

    auto const& recoToSim = recoToSimHandle->getMap();
    auto const& simToReco = simToRecoHandle->getMap();

    auto const& hitIndex = event.get(hitIndexToken_);

    // Reco side: every object, whether it found a branch, and whether that branch came
    // from a pileup interaction rather than the signal one.
    for (std::size_t r = 0; r < recoHandle->size(); ++r) {
      auto const& track = (*recoHandle)[r];
      const bool associated = r < recoToSim.size() && !recoToSim[r].empty();
      truth::TruthBranchHistoProducerAlgo::Kinematics kin;
      kin.pt = track.pt();
      kin.eta = track.eta();
      kin.phi = track.phi();
      kin.nhits = track.numberOfValidHits();
      kin.vertpos = std::sqrt(track.vx() * track.vx() + track.vy() * track.vy());
      kin.zpos = track.vz();
      kin.dxy = track.dxy();
      kin.dz = track.dz();

      bool pileup = false;
      if (associated) {
        const unsigned int branch = recoToSim[r][0].index();
        if (branch < graph.nParticles()) {
          // eventId 0 is the signal interaction; anything else is overlaid pileup.
          pileup = graph.particles()[branch].eventId != 0;
        }
      }
      algo_.fill_reco(histograms.histos, i, kin, associated, pileup);
      if (associated) {
        algo_.fill_match(histograms.histos, i, recoToSim[r][0].score(), recoToSim[r][0].sharedHits());
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
      if (b >= graph.nParticles() || b >= simToReco.size()) {
        continue;
      }
      auto const& particle = graph.particles()[b];
      const auto& p4 = particle.momentum;
      if (p4.pt() <= 0.) {
        continue;
      }
      truth::TruthBranchHistoProducerAlgo::Kinematics kin;
      kin.pt = p4.pt();
      kin.eta = p4.eta();
      kin.phi = p4.phi();
      // The branch's own detector footprint, which is the truth analogue of a track's
      // hit count.
      const auto subgraph = hitIndex.subgraphHits(truth::HitChannel::Tracker, b);
      kin.nhits = subgraph.size();
      // How deep in the graph the branch root sits. A frozen truth object has one fixed
      // level and no such axis.
      const truth::Particle branchRoot(&graph, b);
      kin.depth = branchRoot.ancestors().size();
      // How much of the branch footprint is the root particle's own hits rather than its
      // descendants'. Near 1 is a clean single particle, near 0 a branch whose hits all
      // come from what it produced.
      const auto direct = hitIndex.directHits(truth::HitChannel::Tracker, b);
      kin.rootfrac = subgraph.empty() ? 0. : static_cast<double>(direct.size()) / subgraph.size();
      // Position from the production vertex of the root particle.
      const auto vertices = branchRoot.productionVertices();
      // Why this particle exists, taken from the Geant4 creator process of its
      // production vertex. A branch with no production vertex is a beam-level object.
      auto reason = static_cast<unsigned int>(truth::VertexReason::Unknown);
      if (!vertices.empty()) {
        reason = vertices.front().data().reason;
        const auto& pos = vertices.front().position();
        kin.vertpos = std::sqrt(pos.x() * pos.x() + pos.y() * pos.y());
        kin.zpos = pos.z();
        // Transverse and longitudinal impact parameter of the branch direction with
        // respect to the origin, the truth counterpart of the track dxy and dz.
        kin.dxy = (p4.pt() > 0.) ? (-pos.x() * p4.py() + pos.y() * p4.px()) / p4.pt() : 0.;
        kin.dz =
            (p4.pt() > 0.) ? pos.z() - (pos.x() * p4.px() + pos.y() * p4.py()) / p4.pt() * (p4.pz() / p4.pt()) : 0.;
      }
      const bool associated = !simToReco[b].empty();
      const bool duplicate = simToReco[b].size() > 1;
      algo_.fill_simul(histograms.histos, i, kin, associated, duplicate);
      algo_.fill_reason(histograms.histos, i, reason, associated, duplicate);
      if (associated) {
        const unsigned int r = simToReco[b][0].index();
        if (r < recoHandle->size()) {
          auto const& matched = (*recoHandle)[r];
          algo_.fill_resolution(histograms.histos, i, kin, matched.pt(), matched.eta(), matched.phi());
        }
      }
    }
  }
}

void TruthBranchTrackValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<std::string>("dirName", "TruthInfo/Tracking/");
  desc.add<std::string>("associator", "allTrackToTruthBranchAssociators");
  desc.add<std::vector<edm::InputTag>>("recoCollections", {edm::InputTag("generalTracks")});
  desc.add<std::vector<std::string>>("workingPoints", {"Fixed"});

  edm::ParameterSetDescription algo;
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
  algo.add<int>("nintScore", 50);
  algo.add<double>("minScore", 0.);
  algo.add<double>("maxScore", 1.);
  algo.add<int>("nintShared", 50);
  algo.add<double>("minShared", 0.);
  algo.add<double>("maxShared", 50.);
  algo.add<int>("nint_res_eta", 20);
  algo.add<double>("min_res_eta", -4.);
  algo.add<double>("max_res_eta", 4.);
  algo.add<int>("nint_res_pt", 15);
  algo.add<double>("min_res_pt", 0.);
  algo.add<double>("max_res_pt", 100.);
  algo.add<int>("nintRes", 120);
  algo.add<double>("minRes", -1.5);
  algo.add<double>("maxRes", 1.5);
  desc.add<edm::ParameterSetDescription>("histoProducerAlgoBlock", algo);

  descriptions.add("truthBranchTrackValidator", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TruthBranchTrackValidator);
