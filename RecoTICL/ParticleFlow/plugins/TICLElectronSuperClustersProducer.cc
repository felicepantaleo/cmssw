// The electron seeding of the HGCAL superclusters from the interpretation types of the e/gamma tracksters of their PF
// clusters (ticl::superClusterSeeding): a copy of each supercluster that gets seeds from the supercluster itself, the
// input of ecalDrivenElectronSeeds, with its index in the input collection (superClusterIndex); and the pairs of
// supercluster and KF track that get a seed from the track (trackSeedSuperCluster, trackSeedTrack).

#include <string>
#include <utility>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
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
  const edm::EDGetTokenT<edm::ValueMap<reco::CaloClusterPtr>> pfClusterAssociation_token_;
  const edm::EDGetTokenT<reco::PFClusterCollection> pfClusters_token_;
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
      pfClusterAssociation_token_(consumes(ps.getParameter<edm::InputTag>("pfClusterAssociation"))),
      pfClusters_token_(consumes(ps.getParameter<edm::InputTag>("pfClusters"))),
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
  const auto &association = evt.get(pfClusterAssociation_token_);
  const auto pfClustersH = evt.getHandle(pfClusters_token_);
  const auto &eGammaIndex = evt.get(eGammaIndex_token_);
  const auto &trackToTrackster = evt.get(trackToTrackster_token_);
  const auto &trackMode = evt.get(trackMode_token_);
  const auto &electronTrackster = evt.get(electronTrackster_token_);
  // PF cluster k is made from e/gamma trackster k.
  if (pfClustersH->size() != eGammaIndex.size())
    throw cms::Exception("LogicError") << "TICLElectronSuperClustersProducer: " << pfClustersH->size()
                                       << " PF clusters for " << eGammaIndex.size() << " e/gamma tracksters";
  if (trackToTrackster.size() != trackMode.size() || electronTrackster.size() != trackMode.size())
    throw cms::Exception("LogicError") << "TICLElectronSuperClustersProducer: the interpretation maps differ in size";
  const auto electrons = ticl::eGammaElectrons(eGammaIndex, trackToTrackster, trackMode, electronTrackster);

  reco::SuperClusterCollection seedingSuperClusters;
  std::vector<int> superClusterIndex, trackSeedSuperCluster, trackSeedTrack;
  std::vector<int> clusterTracksters;
  for (unsigned int iSC = 0; iSC < superClustersH->size(); ++iSC) {
    const auto &superCluster = (*superClustersH)[iSC];
    clusterTracksters.clear();
    for (const auto &cluster : superCluster.clusters()) {
      const auto &pfCluster = association[cluster];
      if (pfCluster.id() != pfClustersH.id())
        throw cms::Exception("LogicError")
            << "TICLElectronSuperClustersProducer: a supercluster has a cluster that is not from the pfClusters";
      clusterTracksters.push_back(static_cast<int>(pfCluster.key()));
    }
    const auto seeding = ticl::superClusterSeeding(clusterTracksters, electrons);
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
  desc.add<edm::InputTag>("superClusters", edm::InputTag("particleFlowSuperClusterHGCal"))
      ->setComment("HGCAL superclusters of the PF clusters of the e/gamma tracksters.");
  desc.add<edm::InputTag>("pfClusterAssociation",
                          edm::InputTag("particleFlowSuperClusterHGCal", "PFClusterAssociationEBEE"))
      ->setComment("PF cluster of each supercluster cluster.");
  desc.add<edm::InputTag>("pfClusters", edm::InputTag("particleFlowClusterTICL"))
      ->setComment("PF clusters of the e/gamma tracksters, one per trackster, in the same order.");
  desc.add<edm::InputTag>("eGammaTracksters", edm::InputTag("ticlEGammaTracksters"))
      ->setComment("E/gamma tracksters; their interpretationIndex instance gives their final trackster.");
  desc.add<edm::InputTag>("interpretations", edm::InputTag("ticlTracksterInterpretations"))
      ->setComment("Interpretation stage; its maps are the instances of the same label.");
  descriptions.add("ticlElectronSuperClustersProducer", desc);
}

DEFINE_FWK_MODULE(TICLElectronSuperClustersProducer);
