// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// Truth-branch validation for the track-like domains: one folder per (collection,
// working point), booking only num/denom so all harvesting is DQMGenericClient config.
//
// DQMGlobalEDAnalyzer, not DQMEDAnalyzer: booking and filling are both const and the
// MonitorElements live in a per-run cache, which is the modern convention shared by
// MultiTrackValidator and HGCalValidator.

#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"

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
  edm::EDGetTokenT<std::vector<unsigned int>> selectedRootsToken_;
  const std::string dirName_;
  std::vector<Entry> entries_;
  const truth::TruthBranchHistoProducerAlgo algo_;
};

TruthBranchTrackValidator::TruthBranchTrackValidator(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
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

    // Reco side: every object, and whether it found a branch.
    for (std::size_t r = 0; r < recoHandle->size(); ++r) {
      auto const& track = (*recoHandle)[r];
      const bool associated = r < recoToSim.size() && !recoToSim[r].empty();
      algo_.fill_reco(histograms.histos, i, track.pt(), track.eta(), track.phi(), associated);
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
      const bool associated = !simToReco[b].empty();
      const bool duplicate = simToReco[b].size() > 1;
      algo_.fill_simul(histograms.histos, i, p4.pt(), p4.eta(), p4.phi(), associated, duplicate);
    }
  }
}

void TruthBranchTrackValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<std::string>("dirName", "TruthInfo/Tracking/");
  desc.add<std::string>("associator", "allTrackToTruthBranchAssociators");
  desc.add<std::vector<edm::InputTag>>("recoCollections", {edm::InputTag("generalTracks")});
  desc.add<std::vector<std::string>>("workingPoints", {"Fixed"});

  edm::ParameterSetDescription algo;
  algo.add<int>("nintPt", 50);
  algo.add<double>("minPt", 0.);
  algo.add<double>("maxPt", 100.);
  algo.add<int>("nintEta", 50);
  algo.add<double>("minEta", -4.);
  algo.add<double>("maxEta", 4.);
  algo.add<int>("nintPhi", 36);
  algo.add<double>("minPhi", -3.2);
  algo.add<double>("maxPhi", 3.2);
  algo.add<int>("nintScore", 50);
  algo.add<double>("minScore", 0.);
  algo.add<double>("maxScore", 1.);
  algo.add<int>("nintShared", 50);
  algo.add<double>("minShared", 0.);
  algo.add<double>("maxShared", 50.);
  desc.add<edm::ParameterSetDescription>("histoProducerAlgoBlock", algo);

  descriptions.add("truthBranchTrackValidator", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TruthBranchTrackValidator);
