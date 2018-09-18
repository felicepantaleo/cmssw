// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCal3DClustering.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalDepthPreClusterer.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalImagingAlgo.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "MaskedLayerClustersProducer.h"

DEFINE_FWK_MODULE(MaskedLayerClustersProducer);

MaskedLayerClustersProducer::MaskedLayerClustersProducer(
    const edm::ParameterSet& ps)
    : isFirstIteration_(ps.getParameter<bool>("isFirstIteration")) {
  clusters_token = consumes<std::vector<reco::BasicCluster>>(
      ps.getParameter<edm::InputTag>("HGCLayerClusters"));
  clustersMask_token = consumes<std::vector<float>>(
      ps.getParameter<edm::InputTag>("LayerClustersInputMask"));
  produces<std::vector<std::pair<unsigned int, float>>>("maskedLayerClusters");
}

void MaskedLayerClustersProducer::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  // hgcalMultiClusters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("HGCLayerClusters",
                          edm::InputTag("hgcalLayerClusters"));
  desc.add<edm::InputTag>(
      "LayerClustersInputMask",
      edm::InputTag("hgcalLayerClusters", "InitialLayerClustersMask"));
  desc.add<bool>("isFirstIteration", true);
  descriptions.add("ClustersInSeedingRegion", desc);
}

std::vector<std::pair<unsigned int, float>>
MaskedLayerClustersProducer::maskLayerClusters(
    const std::vector<reco::BasicCluster>& layerClusters,
    const std::vector<float>& clustersMask, bool isFirstIteration) {
  std::vector<std::pair<unsigned int, float>> v;
  unsigned int numLayerClusters = layerClusters.size();

  if (isFirstIteration) {
    v.reserve(numLayerClusters);

    for (unsigned int i = 0; i < numLayerClusters; ++i) {
      v.emplace_back(std::make_pair(i, 1.f));
    }
  } else {
    for (unsigned int i = 0; i < numLayerClusters; ++i) {
      if (clustersMask[i] > 0.f) {
        v.emplace_back(std::make_pair(i, clustersMask[i]));
      }
    }
  }
  return v;
}

void MaskedLayerClustersProducer::produce(edm::Event& evt,
                                          const edm::EventSetup& es) {
  edm::Handle<std::vector<reco::BasicCluster>> clusterHandle;
  edm::Handle<std::vector<float>> inputClustersMaskHandle;
  evt.getByToken(clusters_token, clusterHandle);
  evt.getByToken(clustersMask_token, inputClustersMaskHandle);

  const auto& layerClusters = *clusterHandle;
  const auto& inputClusterMask = *inputClustersMaskHandle;
  auto filteredLayerClusters = std::make_unique<std::vector<std::pair<unsigned int, float> > >( maskLayerClusters(layerClusters,inputClusterMask, isFirstIteration_ ));

  evt.put(std::move(filteredLayerClusters), "iterFilteredLayerClusters");
}
