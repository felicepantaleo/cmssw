// The electron seeding of the HGCAL superclusters from the interpretation type of their e/gamma trackster
// (ticl::superClusterSeeding): a copy of each supercluster that gets seeds from the supercluster itself, the input of
// ecalDrivenElectronSeeds, with its index in the input collection (superClusterIndex); and the pairs of supercluster
// and KF track that get a seed from the track (trackSeedSuperCluster, trackSeedTrack).

#include <string>
#include <utility>
#include <vector>

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTICL/Interpretation/interface/AssignmentMaps.h"

class TICLElectronSuperClustersProducer : public edm::global::EDProducer<> {
public:
  explicit TICLElectronSuperClustersProducer(const edm::ParameterSet &ps);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void produce(edm::StreamID, edm::Event &evt, const edm::EventSetup &) const override;

private:
  const edm::EDGetTokenT<reco::SuperClusterCollection> superClusters_token_;
  const edm::EDGetTokenT<std::vector<int>> tracksterIndex_token_;
  const edm::EDGetTokenT<std::vector<int>> eGammaIndex_token_;
  const edm::EDGetTokenT<std::vector<int>> trackToTrackster_token_;
  const edm::EDGetTokenT<std::vector<int>> trackMode_token_;
  const edm::EDGetTokenT<std::vector<int>> electronTrackster_token_;
  const edm::EDPutTokenT<reco::SuperClusterCollection> superClusters_put_token_;
  const edm::EDPutTokenT<std::vector<int>> superClusterIndex_put_token_;
  const edm::EDPutTokenT<std::vector<int>> trackSeedSuperCluster_put_token_;
  const edm::EDPutTokenT<std::vector<int>> trackSeedTrack_put_token_;
};

namespace {
  edm::InputTag instanceTag(const edm::ParameterSet &ps, const std::string &name, const std::string &instance) {
    const auto &tag = ps.getParameter<edm::InputTag>(name);
    return edm::InputTag(tag.label(), instance, tag.process());
  }
}  // namespace

TICLElectronSuperClustersProducer::TICLElectronSuperClustersProducer(const edm::ParameterSet &ps)
    : superClusters_token_(consumes(ps.getParameter<edm::InputTag>("superClusters"))),
      tracksterIndex_token_(consumes(instanceTag(ps, "superClusters", "tracksterIndex"))),
      eGammaIndex_token_(consumes(instanceTag(ps, "eGammaTracksters", "interpretationIndex"))),
      trackToTrackster_token_(consumes(instanceTag(ps, "interpretations", "trackToTrackster"))),
      trackMode_token_(consumes(instanceTag(ps, "interpretations", "trackMode"))),
      electronTrackster_token_(consumes(instanceTag(ps, "interpretations", "electronTrackster"))),
      superClusters_put_token_(produces()),
      superClusterIndex_put_token_(produces<std::vector<int>>("superClusterIndex")),
      trackSeedSuperCluster_put_token_(produces<std::vector<int>>("trackSeedSuperCluster")),
      trackSeedTrack_put_token_(produces<std::vector<int>>("trackSeedTrack")) {}

void TICLElectronSuperClustersProducer::produce(edm::StreamID, edm::Event &evt, const edm::EventSetup &) const {
  const auto superClustersH = evt.getHandle(superClusters_token_);
  const auto &tracksterIndex = evt.get(tracksterIndex_token_);
  const auto &eGammaIndex = evt.get(eGammaIndex_token_);
  const auto &trackToTrackster = evt.get(trackToTrackster_token_);
  const auto &trackMode = evt.get(trackMode_token_);
  const auto &electronTrackster = evt.get(electronTrackster_token_);
  if (tracksterIndex.size() != superClustersH->size())
    throw cms::Exception("LogicError") << "TICLElectronSuperClustersProducer: " << tracksterIndex.size()
                                       << " trackster indices for " << superClustersH->size() << " superclusters";
  if (trackToTrackster.size() != trackMode.size() || electronTrackster.size() != trackMode.size())
    throw cms::Exception("LogicError") << "TICLElectronSuperClustersProducer: the interpretation maps differ in size";
  const auto electrons = ticl::eGammaElectrons(eGammaIndex, trackToTrackster, trackMode, electronTrackster);

  reco::SuperClusterCollection seedingSuperClusters;
  std::vector<int> superClusterIndex, trackSeedSuperCluster, trackSeedTrack;
  for (unsigned int iSC = 0; iSC < superClustersH->size(); ++iSC) {
    const auto &superCluster = (*superClustersH)[iSC];
    if (tracksterIndex[iSC] < 0 || tracksterIndex[iSC] >= static_cast<int>(eGammaIndex.size()))
      throw cms::Exception("LogicError") << "TICLElectronSuperClustersProducer: e/gamma trackster index "
                                         << tracksterIndex[iSC] << " for " << eGammaIndex.size() << " tracksters";
    const auto seeding = ticl::superClusterSeeding({tracksterIndex[iSC]}, electrons);
    for (int iTrack : seeding.tracks) {
      trackSeedSuperCluster.push_back(static_cast<int>(iSC));
      trackSeedTrack.push_back(iTrack);
    }
    if (seeding.fromSuperCluster) {
      seedingSuperClusters.push_back(superCluster);
      superClusterIndex.push_back(static_cast<int>(iSC));
    }
  }
  evt.emplace(superClusters_put_token_, std::move(seedingSuperClusters));
  evt.emplace(superClusterIndex_put_token_, std::move(superClusterIndex));
  evt.emplace(trackSeedSuperCluster_put_token_, std::move(trackSeedSuperCluster));
  evt.emplace(trackSeedTrack_put_token_, std::move(trackSeedTrack));
}

void TICLElectronSuperClustersProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("superClusters", edm::InputTag("ticlEGammaSuperClusters"))
      ->setComment(
          "HGCAL superclusters of the e/gamma tracksters; the tracksterIndex instance of the same label gives "
          "the e/gamma trackster of each one.");
  desc.add<edm::InputTag>("eGammaTracksters", edm::InputTag("ticlEGammaTracksters"))
      ->setComment("E/gamma tracksters; their interpretationIndex instance gives their final trackster.");
  desc.add<edm::InputTag>("interpretations", edm::InputTag("ticlTracksterInterpretations"))
      ->setComment("Interpretation stage; its maps are the instances of the same label.");
  descriptions.add("ticlElectronSuperClustersProducer", desc);
}

DEFINE_FWK_MODULE(TICLElectronSuperClustersProducer);
