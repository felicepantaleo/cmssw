// Author: Felice Pantaleo, felice.pantaleo@cern.ch 08/2024

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "CommonTools/RecoAlgos/interface/MultiVectorManager.h"

template <typename HIT>
class AllHitToTracksterAssociatorsProducer : public edm::global::EDProducer<> {
public:
  explicit AllHitToTracksterAssociatorsProducer(const edm::ParameterSet&);
  ~AllHitToTracksterAssociatorsProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  std::vector<std::pair<std::string, edm::EDGetTokenT<std::vector<ticl::Trackster>>>> tracksterCollectionTokens_;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClustersToken_;
  edm::EDGetTokenT<std::unordered_map<DetId, const unsigned int>> hitMapToken_;
  std::vector<edm::EDGetTokenT<std::vector<HIT>>> hitsTokens_;
};

template <typename HIT>
AllHitToTracksterAssociatorsProducer<HIT>::AllHitToTracksterAssociatorsProducer(const edm::ParameterSet& pset)
    : layerClustersToken_(consumes<std::vector<reco::CaloCluster>>(pset.getParameter<edm::InputTag>("layerClusters"))),
      hitMapToken_(
          consumes<std::unordered_map<DetId, const unsigned int>>(pset.getParameter<edm::InputTag>("hitMapTag"))) {
  const auto& tracksterCollections = pset.getParameter<std::vector<edm::InputTag>>("tracksterCollections");
  for (const auto& tag : tracksterCollections) {
    tracksterCollectionTokens_.emplace_back(tag.label() + tag.instance(), consumes<std::vector<ticl::Trackster>>(tag));
  }

  for (const auto& tag : pset.getParameter<std::vector<edm::InputTag>>("hits")) {
    hitsTokens_.emplace_back(consumes<std::vector<HIT>>(tag));
  }

  for (const auto& tracksterToken : tracksterCollectionTokens_) {
    produces<ticl::AssociationMap<ticl::mapWithFraction>>("hitTo" + tracksterToken.first);
    produces<ticl::AssociationMap<ticl::mapWithFraction>>(tracksterToken.first + "ToHit");
  }
}

template <typename HIT>
void AllHitToTracksterAssociatorsProducer<HIT>::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  using namespace edm;

  Handle<std::vector<reco::CaloCluster>> layer_clusters;
  iEvent.getByToken(layerClustersToken_, layer_clusters);

  Handle<std::unordered_map<DetId, const unsigned int>> hitMap;
  iEvent.getByToken(hitMapToken_, hitMap);

  MultiVectorManager<HIT> rechitManager;
  for (const auto& token : hitsTokens_) {
    Handle<std::vector<HIT>> hitsHandle;
    iEvent.getByToken(token, hitsHandle);
    rechitManager.addVector(*hitsHandle);
  }

  for (const auto& tracksterToken : tracksterCollectionTokens_) {
    Handle<std::vector<ticl::Trackster>> tracksters;
    iEvent.getByToken(tracksterToken.second, tracksters);

    auto hitToTracksterMap = std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(rechitManager.size());
    auto tracksterToHitMap = std::make_unique<ticl::AssociationMap<ticl::mapWithFraction>>(tracksters->size());

    for (unsigned int tracksterId = 0; tracksterId < tracksters->size(); ++tracksterId) {
      const auto& trackster = (*tracksters)[tracksterId];
      for (unsigned int j = 0; j < trackster.vertices().size(); ++j) {
        const auto& lc = (*layer_clusters)[trackster.vertices()[j]];
        float invMultiplicity = 1.0f / trackster.vertex_multiplicity()[j];

        for (const auto& hitAndFraction : lc.hitsAndFractions()) {
          auto hitMapIter = hitMap->find(hitAndFraction.first);
          if (hitMapIter != hitMap->end()) {
            unsigned int rechitIndex = hitMapIter->second;
            float fraction = hitAndFraction.second * invMultiplicity;
            hitToTracksterMap->insert(rechitIndex, tracksterId, fraction);
            tracksterToHitMap->insert(tracksterId, rechitIndex, fraction);
          }
        }
      }
    }

    iEvent.put(std::move(hitToTracksterMap), "hitTo" + tracksterToken.first);
    iEvent.put(std::move(tracksterToHitMap), tracksterToken.first + "ToHit");
  }
}

template <typename HIT>
void AllHitToTracksterAssociatorsProducer<HIT>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  if constexpr (std::is_same_v<HIT, HGCRecHit>) {
    desc.add<std::vector<edm::InputTag>>("tracksterCollections",
                                         {edm::InputTag("ticlTrackstersCLUE3DHigh"),
                                          edm::InputTag("ticlTrackstersLinks"),
                                          edm::InputTag("ticlCandidate")});
    desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
    desc.add<edm::InputTag>("hitMapTag", edm::InputTag("recHitMapProducer", "hgcalRecHitMap"));
    desc.add<std::vector<edm::InputTag>>("hits",
                                         {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
    descriptions.add("AllHitToHGCalTracksterAssociatorsProducer", desc);
  } else if constexpr (std::is_same_v<HIT, reco::PFRecHit>) {
    desc.add<std::vector<edm::InputTag>>("tracksterCollections",
                                         {edm::InputTag("ticlBarrelTracksters")});
    desc.add<edm::InputTag>("layerClusters", edm::InputTag("barrelLayerClusters"));
    desc.add<edm::InputTag>("hitMapTag", edm::InputTag("recHitMapProducer", "barrelRecHitMap"));
    desc.add<std::vector<edm::InputTag>>("hits",                                                 
                                         {edm::InputTag("particleFlowRecHitECAL"),
                                          edm::InputTag("particleFlowRecHitHBHE")});
    descriptions.add("AllHitToBarrelTracksterAssociatorsProducer", desc);
  }
}

template class AllHitToTracksterAssociatorsProducer<HGCRecHit>;
template class AllHitToTracksterAssociatorsProducer<reco::PFRecHit>;

// Define this as a plug-in
using AllHitToHGCalTracksterAssociatorsProducer = AllHitToTracksterAssociatorsProducer<HGCRecHit>;
DEFINE_FWK_MODULE(AllHitToHGCalTracksterAssociatorsProducer);
using AllHitToBarrelTracksterAssociatorsProducer = AllHitToTracksterAssociatorsProducer<reco::PFRecHit>;
DEFINE_FWK_MODULE(AllHitToBarrelTracksterAssociatorsProducer);
