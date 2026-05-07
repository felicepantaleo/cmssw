// Author: Felice Pantaleo, felice.pantaleo@cern.ch 03/2026

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/Common/interface/MultiSpan.h"
#include "DataFormats/Common/interface/RefProdVector.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

namespace {

  using DetIdRecHitMap = std::unordered_map<DetId, const unsigned int>;

  // Traits used to select the correct rechit container and default products
  // from RecHitMapProducer for each hit type.
  template <typename HIT>
  struct HitCollectionTraits;

  template <>
  struct HitCollectionTraits<HGCRecHit> {
    using Collection = HGCRecHitCollection;
    using MultiCollection = edm::RefProdVector<HGCRecHitCollection>;

    static constexpr const char* defaultHitMapInstance = "hgcalRecHitMap";
    static constexpr const char* defaultHitsInstance = "RefProdVectorHGCRecHitCollection";
  };

  template <>
  struct HitCollectionTraits<reco::PFRecHit> {
    using Collection = reco::PFRecHitCollection;
    using MultiCollection = edm::RefProdVector<reco::PFRecHitCollection>;

    static constexpr const char* defaultHitMapInstance = "pfRecHitMap";
    static constexpr const char* defaultHitsInstance = "RefProdVectorPFRecHitCollection";
  };

  // Compact representation of a rechit contribution to a reconstructed cluster.
  // The hit index is the index in the MultiSpan.
  struct HitContribution {
    DetId detId;
    unsigned int hitIndex = 0;
    float fraction = 0.f;
    float energy = 0.f;
  };

  // Temporary representation of the part of a SimCluster or CaloParticle
  // restricted to one detector layer.
  //
  // fractionByDetId is used for fast lookup when computing reco -> sim scores.
  // hitsAndFractions is used when looping over the sim object itself for
  // sim -> reco scores.
  struct ObjectOnLayer {
    std::vector<std::pair<DetId, float>> hitsAndFractions;
    std::unordered_map<DetId, float> fractionByDetId;
    float denominator = 0.f;

    void add(DetId detId, float fraction) { fractionByDetId[detId] += fraction; }

    template <typename HIT>
    void finalize(const DetIdRecHitMap& hitMap, const edm::MultiSpan<HIT>& hits) {
      hitsAndFractions.reserve(fractionByDetId.size());

      for (const auto& [detId, fraction] : fractionByDetId) {
        const auto hitIt = hitMap.find(detId);
        if (hitIt == hitMap.end()) {
          continue;
        }

        const float energy = hits[hitIt->second].energy();
        denominator += fraction * fraction * energy * energy;
        hitsAndFractions.emplace_back(detId, fraction);
      }
    }

    float fraction(DetId detId) const {
      const auto it = fractionByDetId.find(detId);
      return it == fractionByDetId.end() ? 0.f : it->second;
    }

    bool empty() const { return hitsAndFractions.empty(); }
  };

  // Scores are distances, therefore smaller is better. If two scores are equal,
  // prefer the association with larger shared energy.
  template <typename Map>
  void sortByIncreasingScoreThenDecreasingSharedEnergy(Map& map) {
    map.sort([](const auto& a, const auto& b) {
      if (a.score() != b.score()) {
        return a.score() < b.score();
      }
      if (a.sharedEnergy() != b.sharedEnergy()) {
        return a.sharedEnergy() > b.sharedEnergy();
      }
      return a.index() < b.index();
    });
  }

}  // namespace

template <typename HIT, typename CLUSTER>
class LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT : public edm::global::EDProducer<> {
public:
  using MultiCollection = typename HitCollectionTraits<HIT>::MultiCollection;

  using SimClusterCollection = std::vector<SimCluster>;
  using CaloParticleCollection = std::vector<CaloParticle>;

  using LCToSCMap = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore, CLUSTER, SimClusterCollection>;
  using SCToLCMap = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore, SimClusterCollection, CLUSTER>;
  using LCToCPMap = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore, CLUSTER, CaloParticleCollection>;
  using CPToLCMap = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore, CaloParticleCollection, CLUSTER>;

  explicit LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT(const edm::ParameterSet&);
  ~LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // Returns a layer index. For HGCAL, the z-side is folded into the index,
  // so the same layer number on +z and -z is kept distinct.
  unsigned int layerIndex(DetId detId, const hgcal::RecHitTools& recHitTools, unsigned int layers) const;

  // Selects only the hits relevant for the current HIT type.
  bool acceptSimHit(DetId detId, const hgcal::RecHitTools& recHitTools) const;

  bool passHardScatter(const SimCluster& simCluster) const;

  // Extracts the rechits of a reconstructed cluster and decorates them with
  // their MultiSpan index and rechit energy.
  std::vector<HitContribution> layerClusterHits(const typename CLUSTER::value_type& layerCluster,
                                                const DetIdRecHitMap& hitMap,
                                                const edm::MultiSpan<HIT>& hits) const;

  // Denominator for the reco -> sim directional score:
  // sum_i (recoFraction_i * E_i)^2.
  float layerClusterDenominator(const std::vector<HitContribution>& hits) const;

  // Build SimCluster-on-layer objects. These are needed because a LayerCluster
  // lives on a single layer and should not be compared to the full SimCluster.
  std::vector<std::vector<ObjectOnLayer>> buildSimClustersOnLayer(const SimClusterCollection& simClusters,
                                                                  const DetIdRecHitMap& hitMap,
                                                                  const edm::MultiSpan<HIT>& hits,
                                                                  const hgcal::RecHitTools& recHitTools,
                                                                  unsigned int layers,
                                                                  unsigned int nLayers) const;

  // Build CaloParticle-on-layer objects by aggregating the fractions of all
  // SimClusters belonging to the same CaloParticle on each DetId.
  std::vector<std::vector<ObjectOnLayer>> buildCaloParticlesOnLayer(const CaloParticleCollection& caloParticles,
                                                                    const DetIdRecHitMap& hitMap,
                                                                    const edm::MultiSpan<HIT>& hits,
                                                                    const hgcal::RecHitTools& recHitTools,
                                                                    unsigned int layers,
                                                                    unsigned int nLayers) const;

  void putEmptyProducts(edm::Event& iEvent) const;

  edm::EDGetTokenT<CLUSTER> layerClustersToken_;
  edm::EDGetTokenT<SimClusterCollection> simClustersToken_;
  edm::EDGetTokenT<CaloParticleCollection> caloParticlesToken_;

  edm::EDGetTokenT<DetIdRecHitMap> hitMapToken_;
  edm::EDGetTokenT<MultiCollection> hitsToken_;

  edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>> hitToLayerClusterMapToken_;
  edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>> hitToSimClusterMapToken_;
  edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>> hitToCaloParticleMapToken_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;

  bool hardScatterOnly_;
};

template <typename HIT, typename CLUSTER>
LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT<HIT, CLUSTER>::
    LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT(const edm::ParameterSet& pset)
    : layerClustersToken_(consumes<CLUSTER>(pset.getParameter<edm::InputTag>("layerClusters"))),
      simClustersToken_(consumes<SimClusterCollection>(pset.getParameter<edm::InputTag>("simClusters"))),
      caloParticlesToken_(consumes<CaloParticleCollection>(pset.getParameter<edm::InputTag>("caloParticles"))),
      hitMapToken_(consumes<DetIdRecHitMap>(pset.getParameter<edm::InputTag>("hitMap"))),
      hitsToken_(consumes<MultiCollection>(pset.getParameter<edm::InputTag>("hits"))),
      hitToLayerClusterMapToken_(consumes<ticl::AssociationMap<ticl::mapWithFraction>>(
          pset.getParameter<edm::InputTag>("hitToLayerClusterMap"))),
      hitToSimClusterMapToken_(consumes<ticl::AssociationMap<ticl::mapWithFraction>>(
          pset.getParameter<edm::InputTag>("hitToSimClusterMap"))),
      hitToCaloParticleMapToken_(consumes<ticl::AssociationMap<ticl::mapWithFraction>>(
          pset.getParameter<edm::InputTag>("hitToCaloParticleMap"))),
      caloGeometryToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      hardScatterOnly_(pset.getParameter<bool>("hardScatterOnly")) {
  produces<LCToSCMap>("layerClusterToSimClusterMap");
  produces<SCToLCMap>("simClusterToLayerClusterMap");
  produces<LCToCPMap>("layerClusterToCaloParticleMap");
  produces<CPToLCMap>("caloParticleToLayerClusterMap");
}

template <typename HIT, typename CLUSTER>
unsigned int LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT<HIT, CLUSTER>::layerIndex(
    DetId detId, const hgcal::RecHitTools& recHitTools, unsigned int layers) const {
  unsigned int layer = recHitTools.getLayer(detId);

  if constexpr (std::is_same_v<HIT, HGCRecHit>) {
    layer += layers * ((recHitTools.zside(detId) + 1) >> 1) - 1;
  }

  return layer;
}

template <typename HIT, typename CLUSTER>
bool LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT<HIT, CLUSTER>::acceptSimHit(
    DetId detId, const hgcal::RecHitTools& recHitTools) const {
  if constexpr (std::is_same_v<HIT, HGCRecHit>) {
    return !recHitTools.isBarrel(detId);
  } else {
    return recHitTools.isBarrel(detId);
  }
}

template <typename HIT, typename CLUSTER>
bool LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT<HIT, CLUSTER>::passHardScatter(
    const SimCluster& simCluster) const {
  if (!hardScatterOnly_) {
    return true;
  }

  if (simCluster.g4Tracks().empty()) {
    return false;
  }

  const auto& eventId = simCluster.g4Tracks()[0].eventId();
  return eventId.event() == 0 && eventId.bunchCrossing() == 0;
}

template <typename HIT, typename CLUSTER>
std::vector<HitContribution>
LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT<HIT, CLUSTER>::layerClusterHits(
    const typename CLUSTER::value_type& layerCluster,
    const DetIdRecHitMap& hitMap,
    const edm::MultiSpan<HIT>& hits) const {
  std::vector<HitContribution> result;
  const auto& hitsAndFractions = layerCluster.hitsAndFractions();
  result.reserve(hitsAndFractions.size());

  for (const auto& [detId, fraction] : hitsAndFractions) {
    const auto hitIt = hitMap.find(detId);
    if (hitIt == hitMap.end()) {
      continue;
    }

    const unsigned int hitIndex = hitIt->second;
    result.push_back({detId, hitIndex, fraction, hits[hitIndex].energy()});
  }

  return result;
}

template <typename HIT, typename CLUSTER>
float LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT<HIT, CLUSTER>::layerClusterDenominator(
    const std::vector<HitContribution>& hits) const {
  float denominator = 0.f;

  for (const auto& hit : hits) {
    denominator += hit.fraction * hit.fraction * hit.energy * hit.energy;
  }

  return denominator;
}

template <typename HIT, typename CLUSTER>
std::vector<std::vector<ObjectOnLayer>>
LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT<HIT, CLUSTER>::buildSimClustersOnLayer(
    const SimClusterCollection& simClusters,
    const DetIdRecHitMap& hitMap,
    const edm::MultiSpan<HIT>& hits,
    const hgcal::RecHitTools& recHitTools,
    unsigned int layers,
    unsigned int nLayers) const {
  std::vector<std::vector<ObjectOnLayer>> simClustersOnLayer(simClusters.size(), std::vector<ObjectOnLayer>(nLayers));

  for (unsigned int scId = 0; scId < simClusters.size(); ++scId) {
    const auto& simCluster = simClusters[scId];

    if (!passHardScatter(simCluster)) {
      continue;
    }

    for (const auto& [detIdRaw, fraction] : simCluster.hits_and_fractions()) {
      const DetId detId(detIdRaw);

      // Keep only hits compatible with the hit type used by this producer.
      if (!acceptSimHit(detId, recHitTools)) {
        continue;
      }

      // Ignore simulated hits that do not correspond to a reconstructed hit.
      if (hitMap.find(detId) == hitMap.end()) {
        continue;
      }

      const unsigned int layer = layerIndex(detId, recHitTools, layers);
      if (layer >= nLayers) {
        continue;
      }

      simClustersOnLayer[scId][layer].add(detId, fraction);
    }
  }

  // Build the vector view and compute the score denominator for each layer.
  for (auto& objectLayers : simClustersOnLayer) {
    for (auto& objectOnLayer : objectLayers) {
      objectOnLayer.finalize(hitMap, hits);
    }
  }

  return simClustersOnLayer;
}

template <typename HIT, typename CLUSTER>
std::vector<std::vector<ObjectOnLayer>>
LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT<HIT, CLUSTER>::buildCaloParticlesOnLayer(
    const CaloParticleCollection& caloParticles,
    const DetIdRecHitMap& hitMap,
    const edm::MultiSpan<HIT>& hits,
    const hgcal::RecHitTools& recHitTools,
    unsigned int layers,
    unsigned int nLayers) const {
  std::vector<std::vector<ObjectOnLayer>> caloParticlesOnLayer(caloParticles.size(),
                                                               std::vector<ObjectOnLayer>(nLayers));

  for (unsigned int cpId = 0; cpId < caloParticles.size(); ++cpId) {
    const auto& caloParticle = caloParticles[cpId];

    for (const auto& simClusterRef : caloParticle.simClusters()) {
      const auto& simCluster = *simClusterRef;

      if (!passHardScatter(simCluster)) {
        continue;
      }

      for (const auto& [detIdRaw, fraction] : simCluster.hits_and_fractions()) {
        const DetId detId(detIdRaw);

        if (!acceptSimHit(detId, recHitTools)) {
          continue;
        }

        if (hitMap.find(detId) == hitMap.end()) {
          continue;
        }

        const unsigned int layer = layerIndex(detId, recHitTools, layers);
        if (layer >= nLayers) {
          continue;
        }

        // Fractions from different SimClusters of the same CaloParticle are
        // aggregated per DetId before computing the CP score.
        caloParticlesOnLayer[cpId][layer].add(detId, fraction);
      }
    }
  }

  for (auto& objectLayers : caloParticlesOnLayer) {
    for (auto& objectOnLayer : objectLayers) {
      objectOnLayer.finalize(hitMap, hits);
    }
  }

  return caloParticlesOnLayer;
}

template <typename HIT, typename CLUSTER>
void LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT<HIT, CLUSTER>::putEmptyProducts(
    edm::Event& iEvent) const {
  iEvent.put(std::make_unique<LCToSCMap>(), "layerClusterToSimClusterMap");
  iEvent.put(std::make_unique<SCToLCMap>(), "simClusterToLayerClusterMap");
  iEvent.put(std::make_unique<LCToCPMap>(), "layerClusterToCaloParticleMap");
  iEvent.put(std::make_unique<CPToLCMap>(), "caloParticleToLayerClusterMap");
}

template <typename HIT, typename CLUSTER>
void LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT<HIT, CLUSTER>::produce(
    edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto layerClustersHandle = iEvent.getHandle(layerClustersToken_);
  auto simClustersHandle = iEvent.getHandle(simClustersToken_);
  auto caloParticlesHandle = iEvent.getHandle(caloParticlesToken_);
  auto hitMapHandle = iEvent.getHandle(hitMapToken_);
  auto hitsHandle = iEvent.getHandle(hitsToken_);
  auto hitToLayerClusterMapHandle = iEvent.getHandle(hitToLayerClusterMapToken_);
  auto hitToSimClusterMapHandle = iEvent.getHandle(hitToSimClusterMapToken_);
  auto hitToCaloParticleMapHandle = iEvent.getHandle(hitToCaloParticleMapToken_);

  if (!layerClustersHandle.isValid() || !simClustersHandle.isValid() || !caloParticlesHandle.isValid() ||
      !hitMapHandle.isValid() || !hitsHandle.isValid() || !hitToLayerClusterMapHandle.isValid() ||
      !hitToSimClusterMapHandle.isValid() || !hitToCaloParticleMapHandle.isValid()) {
    edm::LogWarning("LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT")
        << "Missing input collection. Producing empty association maps.";
    putEmptyProducts(iEvent);
    return;
  }

  const auto& layerClusters = *layerClustersHandle;
  const auto& simClusters = *simClustersHandle;
  const auto& caloParticles = *caloParticlesHandle;
  const auto& hitMap = *hitMapHandle;
  const auto& hitToLayerClusterMap = *hitToLayerClusterMapHandle;
  const auto& hitToSimClusterMap = *hitToSimClusterMapHandle;
  const auto& hitToCaloParticleMap = *hitToCaloParticleMapHandle;

  const auto hitsCollections = iEvent.get(hitsToken_);
  edm::MultiSpan<HIT> rechitSpan(hitsCollections);

  if (rechitSpan.size() == 0 || hitMap.empty()) {
    edm::LogWarning("LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT")
        << "No valid rechits or empty hit map. Producing empty association maps.";
    putEmptyProducts(iEvent);
    return;
  }

  hgcal::RecHitTools recHitTools;
  const auto& geometry = iSetup.getData(caloGeometryToken_);
  recHitTools.setGeometry(geometry);

  unsigned int layers = 0;
  if constexpr (std::is_same_v<HIT, HGCRecHit>) {
    layers = recHitTools.lastLayerBH();
  } else {
    layers = recHitTools.lastLayerBarrel() + 1;
  }

  const unsigned int nLayers = std::is_same_v<HIT, HGCRecHit> ? 2 * layers : layers;

  // Precompute the sim truth restricted layer-by-layer. This avoids comparing
  // a single-layer reco object with the full longitudinal extent of the truth object.
  const auto simClustersOnLayer =
      buildSimClustersOnLayer(simClusters, hitMap, rechitSpan, recHitTools, layers, nLayers);
  const auto caloParticlesOnLayer =
      buildCaloParticlesOnLayer(caloParticles, hitMap, rechitSpan, recHitTools, layers, nLayers);

  auto layerClusterToSimClusterMap = std::make_unique<LCToSCMap>(layerClustersHandle, simClustersHandle, iEvent);
  auto simClusterToLayerClusterMap = std::make_unique<SCToLCMap>(simClustersHandle, layerClustersHandle, iEvent);
  auto layerClusterToCaloParticleMap = std::make_unique<LCToCPMap>(layerClustersHandle, caloParticlesHandle, iEvent);
  auto caloParticleToLayerClusterMap = std::make_unique<CPToLCMap>(caloParticlesHandle, layerClustersHandle, iEvent);

  // Reco -> sim associations.
  //
  // For each LayerCluster, find candidate SimClusters and CaloParticles through
  // the hit -> sim maps, then compute a directional score:
  //
  //   score = sum_i max(0, recoFraction_i - simFraction_i)^2 * E_i^2
  //           / sum_i (recoFraction_i * E_i)^2
  //
  // This penalizes the part of the reco object that is not covered by the sim object.
  for (unsigned int lcId = 0; lcId < layerClusters.size(); ++lcId) {
    const auto lcHits = layerClusterHits(layerClusters[lcId], hitMap, rechitSpan);

    if (lcHits.empty()) {
      continue;
    }

    const unsigned int lcLayer = layerIndex(lcHits.front().detId, recHitTools, layers);
    if (lcLayer >= nLayers) {
      continue;
    }

    const float denominator = layerClusterDenominator(lcHits);
    if (denominator <= 0.f) {
      continue;
    }

    const float invDenominator = 1.f / denominator;

    std::vector<unsigned int> associatedSimClusters;
    std::vector<unsigned int> associatedCaloParticles;

    // Candidate truth objects are those sharing at least one rechit with the LayerCluster.
    for (const auto& hit : lcHits) {
      if (hit.hitIndex < hitToSimClusterMap.size()) {
        for (const auto& simElement : hitToSimClusterMap[hit.hitIndex]) {
          associatedSimClusters.push_back(simElement.index());
        }
      }

      if (hit.hitIndex < hitToCaloParticleMap.size()) {
        for (const auto& cpElement : hitToCaloParticleMap[hit.hitIndex]) {
          associatedCaloParticles.push_back(cpElement.index());
        }
      }
    }

    std::sort(associatedSimClusters.begin(), associatedSimClusters.end());
    associatedSimClusters.erase(std::unique(associatedSimClusters.begin(), associatedSimClusters.end()),
                                associatedSimClusters.end());

    std::sort(associatedCaloParticles.begin(), associatedCaloParticles.end());
    associatedCaloParticles.erase(std::unique(associatedCaloParticles.begin(), associatedCaloParticles.end()),
                                  associatedCaloParticles.end());

    edm::Ref<CLUSTER> lcRef(layerClustersHandle, lcId);

    for (const unsigned int scId : associatedSimClusters) {
      if (scId >= simClustersOnLayer.size()) {
        continue;
      }

      const auto& simObject = simClustersOnLayer[scId][lcLayer];
      if (simObject.empty()) {
        continue;
      }

      edm::Ref<SimClusterCollection> scRef(simClustersHandle, scId);

      for (const auto& hit : lcHits) {
        const float recoFraction = hit.fraction;
        const float simFraction = simObject.fraction(hit.detId);

        const float sharedEnergy = std::min(recoFraction, simFraction) * hit.energy;
        const float missingRecoFraction = std::max(0.f, recoFraction - simFraction);
        const float score = invDenominator * missingRecoFraction * missingRecoFraction * hit.energy * hit.energy;

        // AssociationMap accumulates shared energy and score contributions
        // for repeated insertions of the same pair.
        layerClusterToSimClusterMap->insert(lcRef, scRef, sharedEnergy, score);
      }
    }

    for (const unsigned int cpId : associatedCaloParticles) {
      if (cpId >= caloParticlesOnLayer.size()) {
        continue;
      }

      const auto& simObject = caloParticlesOnLayer[cpId][lcLayer];
      if (simObject.empty()) {
        continue;
      }

      edm::Ref<CaloParticleCollection> cpRef(caloParticlesHandle, cpId);

      for (const auto& hit : lcHits) {
        const float recoFraction = hit.fraction;
        const float simFraction = simObject.fraction(hit.detId);

        const float sharedEnergy = std::min(recoFraction, simFraction) * hit.energy;
        const float missingRecoFraction = std::max(0.f, recoFraction - simFraction);
        const float score = invDenominator * missingRecoFraction * missingRecoFraction * hit.energy * hit.energy;

        layerClusterToCaloParticleMap->insert(lcRef, cpRef, sharedEnergy, score);
      }
    }
  }

  // SimCluster -> reco associations.
  //
  // The denominator is computed from the SimCluster restricted to the layer.
  // The score penalizes the part of the sim object that is not covered by the reco object.
  for (unsigned int scId = 0; scId < simClustersOnLayer.size(); ++scId) {
    edm::Ref<SimClusterCollection> scRef(simClustersHandle, scId);

    for (unsigned int layer = 0; layer < simClustersOnLayer[scId].size(); ++layer) {
      const auto& simObject = simClustersOnLayer[scId][layer];

      if (simObject.empty() || simObject.denominator <= 0.f) {
        continue;
      }

      const float invDenominator = 1.f / simObject.denominator;

      std::vector<unsigned int> associatedLayerClusters;

      // Candidate LayerClusters are found through hit -> LayerCluster associations.
      for (const auto& [detId, simFraction] : simObject.hitsAndFractions) {
        const auto hitIt = hitMap.find(detId);
        if (hitIt == hitMap.end()) {
          continue;
        }

        const unsigned int hitIndex = hitIt->second;
        if (hitIndex >= hitToLayerClusterMap.size()) {
          continue;
        }

        for (const auto& lcElement : hitToLayerClusterMap[hitIndex]) {
          associatedLayerClusters.push_back(lcElement.index());
        }
      }

      std::sort(associatedLayerClusters.begin(), associatedLayerClusters.end());
      associatedLayerClusters.erase(std::unique(associatedLayerClusters.begin(), associatedLayerClusters.end()),
                                    associatedLayerClusters.end());

      for (const unsigned int lcId : associatedLayerClusters) {
        if (lcId >= layerClusters.size()) {
          continue;
        }

        edm::Ref<CLUSTER> lcRef(layerClustersHandle, lcId);

        for (const auto& [detId, simFraction] : simObject.hitsAndFractions) {
          const auto hitIt = hitMap.find(detId);
          if (hitIt == hitMap.end()) {
            continue;
          }

          const unsigned int hitIndex = hitIt->second;
          const float energy = rechitSpan[hitIndex].energy();

          float recoFraction = 0.f;
          if (hitIndex < hitToLayerClusterMap.size()) {
            const auto& hitToLCVec = hitToLayerClusterMap[hitIndex];
            const auto lcIt = std::find_if(
                hitToLCVec.begin(), hitToLCVec.end(), [lcId](const auto& element) { return element.index() == lcId; });

            if (lcIt != hitToLCVec.end()) {
              recoFraction = lcIt->fraction();
            }
          }

          const float sharedEnergy = std::min(recoFraction, simFraction) * energy;
          const float missingSimFraction = std::max(0.f, simFraction - recoFraction);
          const float score = invDenominator * missingSimFraction * missingSimFraction * energy * energy;

          simClusterToLayerClusterMap->insert(scRef, lcRef, sharedEnergy, score);
        }
      }
    }
  }

  // CaloParticle -> reco associations.
  //
  // Same directional score as SimCluster -> reco, but using the CaloParticle
  // fractions aggregated on each DetId and layer.
  for (unsigned int cpId = 0; cpId < caloParticlesOnLayer.size(); ++cpId) {
    edm::Ref<CaloParticleCollection> cpRef(caloParticlesHandle, cpId);

    for (unsigned int layer = 0; layer < caloParticlesOnLayer[cpId].size(); ++layer) {
      const auto& simObject = caloParticlesOnLayer[cpId][layer];

      if (simObject.empty() || simObject.denominator <= 0.f) {
        continue;
      }

      const float invDenominator = 1.f / simObject.denominator;

      std::vector<unsigned int> associatedLayerClusters;

      for (const auto& [detId, simFraction] : simObject.hitsAndFractions) {
        const auto hitIt = hitMap.find(detId);
        if (hitIt == hitMap.end()) {
          continue;
        }

        const unsigned int hitIndex = hitIt->second;
        if (hitIndex >= hitToLayerClusterMap.size()) {
          continue;
        }

        for (const auto& lcElement : hitToLayerClusterMap[hitIndex]) {
          associatedLayerClusters.push_back(lcElement.index());
        }
      }

      std::sort(associatedLayerClusters.begin(), associatedLayerClusters.end());
      associatedLayerClusters.erase(std::unique(associatedLayerClusters.begin(), associatedLayerClusters.end()),
                                    associatedLayerClusters.end());

      for (const unsigned int lcId : associatedLayerClusters) {
        if (lcId >= layerClusters.size()) {
          continue;
        }

        edm::Ref<CLUSTER> lcRef(layerClustersHandle, lcId);

        for (const auto& [detId, simFraction] : simObject.hitsAndFractions) {
          const auto hitIt = hitMap.find(detId);
          if (hitIt == hitMap.end()) {
            continue;
          }

          const unsigned int hitIndex = hitIt->second;
          const float energy = rechitSpan[hitIndex].energy();

          float recoFraction = 0.f;
          if (hitIndex < hitToLayerClusterMap.size()) {
            const auto& hitToLCVec = hitToLayerClusterMap[hitIndex];
            const auto lcIt = std::find_if(
                hitToLCVec.begin(), hitToLCVec.end(), [lcId](const auto& element) { return element.index() == lcId; });

            if (lcIt != hitToLCVec.end()) {
              recoFraction = lcIt->fraction();
            }
          }

          const float sharedEnergy = std::min(recoFraction, simFraction) * energy;
          const float missingSimFraction = std::max(0.f, simFraction - recoFraction);
          const float score = invDenominator * missingSimFraction * missingSimFraction * energy * energy;

          caloParticleToLayerClusterMap->insert(cpRef, lcRef, sharedEnergy, score);
        }
      }
    }
  }

  sortByIncreasingScoreThenDecreasingSharedEnergy(*layerClusterToSimClusterMap);
  sortByIncreasingScoreThenDecreasingSharedEnergy(*simClusterToLayerClusterMap);
  sortByIncreasingScoreThenDecreasingSharedEnergy(*layerClusterToCaloParticleMap);
  sortByIncreasingScoreThenDecreasingSharedEnergy(*caloParticleToLayerClusterMap);

  iEvent.put(std::move(layerClusterToSimClusterMap), "layerClusterToSimClusterMap");
  iEvent.put(std::move(simClusterToLayerClusterMap), "simClusterToLayerClusterMap");
  iEvent.put(std::move(layerClusterToCaloParticleMap), "layerClusterToCaloParticleMap");
  iEvent.put(std::move(caloParticleToLayerClusterMap), "caloParticleToLayerClusterMap");
}

template <typename HIT, typename CLUSTER>
void LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT<HIT, CLUSTER>::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("hardScatterOnly", true);

  desc.add<edm::InputTag>("simClusters", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));

  desc.add<edm::InputTag>("hitMap",
                          edm::InputTag("recHitMapProducer", HitCollectionTraits<HIT>::defaultHitMapInstance));

  desc.add<edm::InputTag>("hits", edm::InputTag("recHitMapProducer", HitCollectionTraits<HIT>::defaultHitsInstance));

  desc.add<edm::InputTag>("hitToSimClusterMap",
                          edm::InputTag("hitToSimClusterCaloParticleAssociator", "hitToSimClusterMap"));
  desc.add<edm::InputTag>("hitToCaloParticleMap",
                          edm::InputTag("hitToSimClusterCaloParticleAssociator", "hitToCaloParticleMap"));

  if constexpr (std::is_same_v<CLUSTER, reco::CaloClusterCollection>) {
    desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
    desc.add<edm::InputTag>("hitToLayerClusterMap",
                            edm::InputTag("hitToLayerClusterAssociator", "hitToLayerClusterMap"));
  } else {
    desc.add<edm::InputTag>("layerClusters", edm::InputTag("particleFlowClusterHGCal"));
    desc.add<edm::InputTag>("hitToLayerClusterMap", edm::InputTag("hitToPFClusterAssociator", "hitToLayerClusterMap"));
  }

  descriptions.addWithDefaultLabel(desc);
}

using HGCRecHitCaloClusterToSimClusterCaloParticleAssociatorByHitsProducer =
    LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT<HGCRecHit, reco::CaloClusterCollection>;

using HGCRecHitPFClusterToSimClusterCaloParticleAssociatorByHitsProducer =
    LayerClusterToSimClusterCaloParticleAssociatorByHitsProducerT<HGCRecHit, reco::PFClusterCollection>;

DEFINE_FWK_MODULE(HGCRecHitCaloClusterToSimClusterCaloParticleAssociatorByHitsProducer);
DEFINE_FWK_MODULE(HGCRecHitPFClusterToSimClusterCaloParticleAssociatorByHitsProducer);
