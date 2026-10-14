// The electron seeds of the GSF tracking with the HGCAL superclusters selected by TICLElectronSuperClustersProducer:
// the seeds of the ECAL-driven seeding, where a seed of a copied supercluster points to the original supercluster, and
// a seed from the KF track of each selected pair of supercluster and track, made from the seed of the track.

#include <string>
#include <utility>
#include <vector>

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

class TICLElectronSeedsProducer : public edm::global::EDProducer<> {
public:
  explicit TICLElectronSeedsProducer(const edm::ParameterSet &ps);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void produce(edm::StreamID, edm::Event &evt, const edm::EventSetup &) const override;

private:
  const edm::EDGetTokenT<reco::ElectronSeedCollection> seeds_token_;
  const edm::EDGetTokenT<reco::SuperClusterCollection> seedingSuperClusters_token_;
  const edm::EDGetTokenT<std::vector<int>> superClusterIndex_token_;
  const edm::EDGetTokenT<std::vector<int>> trackSeedSuperCluster_token_;
  const edm::EDGetTokenT<std::vector<int>> trackSeedTrack_token_;
  const edm::EDGetTokenT<reco::SuperClusterCollection> superClusters_token_;
  const edm::EDGetTokenT<reco::TrackCollection> tracks_token_;
  const edm::EDPutTokenT<reco::ElectronSeedCollection> put_token_;
};

namespace {
  edm::InputTag instanceTag(const edm::ParameterSet &ps, const std::string &name, const std::string &instance) {
    const auto &tag = ps.getParameter<edm::InputTag>(name);
    return edm::InputTag(tag.label(), instance, tag.process());
  }
}  // namespace

TICLElectronSeedsProducer::TICLElectronSeedsProducer(const edm::ParameterSet &ps)
    : seeds_token_(consumes(ps.getParameter<edm::InputTag>("seeds"))),
      seedingSuperClusters_token_(consumes(ps.getParameter<edm::InputTag>("seedingSuperClusters"))),
      superClusterIndex_token_(consumes(instanceTag(ps, "seedingSuperClusters", "superClusterIndex"))),
      trackSeedSuperCluster_token_(consumes(instanceTag(ps, "seedingSuperClusters", "trackSeedSuperCluster"))),
      trackSeedTrack_token_(consumes(instanceTag(ps, "seedingSuperClusters", "trackSeedTrack"))),
      superClusters_token_(consumes(ps.getParameter<edm::InputTag>("superClusters"))),
      tracks_token_(consumes(ps.getParameter<edm::InputTag>("tracks"))),
      put_token_(produces()) {}

void TICLElectronSeedsProducer::produce(edm::StreamID, edm::Event &evt, const edm::EventSetup &) const {
  const auto &seeds = evt.get(seeds_token_);
  const auto seedingSuperClustersH = evt.getHandle(seedingSuperClusters_token_);
  const auto &superClusterIndex = evt.get(superClusterIndex_token_);
  const auto &trackSeedSuperCluster = evt.get(trackSeedSuperCluster_token_);
  const auto &trackSeedTrack = evt.get(trackSeedTrack_token_);
  const auto superClustersH = evt.getHandle(superClusters_token_);
  const auto tracksH = evt.getHandle(tracks_token_);
  if (superClusterIndex.size() != seedingSuperClustersH->size() ||
      trackSeedSuperCluster.size() != trackSeedTrack.size())
    throw cms::Exception("LogicError") << "TICLElectronSeedsProducer: the supercluster selection differs in size";

  reco::ElectronSeedCollection result;
  result.reserve(seeds.size() + trackSeedTrack.size());
  auto checkIndex = [](std::size_t index, std::size_t size, const char *what) {
    if (index >= size)
      throw cms::Exception("LogicError") << "TICLElectronSeedsProducer: " << what << " index " << index << " for "
                                         << size << " entries";
  };
  for (const auto &seed : seeds) {
    result.push_back(seed);
    const auto &cluster = seed.caloCluster();
    if (cluster.isNonnull() && cluster.id() == seedingSuperClustersH.id()) {
      checkIndex(superClusterIndex[cluster.key()], superClustersH->size(), "supercluster");
      result.back().setCaloCluster(
          reco::ElectronSeed::CaloClusterRef(reco::SuperClusterRef(superClustersH, superClusterIndex[cluster.key()])));
    }
  }
  for (std::size_t i = 0; i < trackSeedTrack.size(); ++i) {
    checkIndex(trackSeedTrack[i], tracksH->size(), "track");
    checkIndex(trackSeedSuperCluster[i], superClustersH->size(), "supercluster");
    const reco::TrackRef track(tracksH, trackSeedTrack[i]);
    result.emplace_back(*track->seedRef());
    result.back().setCtfTrack(track);
    result.back().setCaloCluster(
        reco::ElectronSeed::CaloClusterRef(reco::SuperClusterRef(superClustersH, trackSeedSuperCluster[i])));
  }
  evt.emplace(put_token_, std::move(result));
}

void TICLElectronSeedsProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("seeds", edm::InputTag("ecalDrivenElectronSeeds"))
      ->setComment("Seeds of the ECAL-driven seeding, on the selected HGCAL superclusters and the barrel ones.");
  desc.add<edm::InputTag>("seedingSuperClusters", edm::InputTag("ticlElectronSuperClusters"))
      ->setComment("Supercluster selection; its index and track-seed lists are the instances of the same label.");
  desc.add<edm::InputTag>("superClusters", edm::InputTag("ticlEGammaSuperClusters"))
      ->setComment("HGCAL superclusters that the selection indexes.");
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"))
      ->setComment("Tracks of the interpretation stage; the track indices of the selection refer to them.");
  descriptions.add("ticlElectronSeedsProducer", desc);
}

DEFINE_FWK_MODULE(TICLElectronSeedsProducer);
