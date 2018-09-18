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

#include "ClustersInSeedingRegionProducer.h"

DEFINE_FWK_MODULE(ClustersInSeedingRegionProducer);

ClustersInSeedingRegionProducer::ClustersInSeedingRegionProducer(const edm::ParameterSet &ps)
{
  clusters_token = consumes<std::vector<reco::BasicCluster>>(
      ps.getParameter<edm::InputTag>("HGCLayerClusters"));
  clustersMask_token = consumes<std::vector<float>>(
      ps.getParameter<edm::InputTag>("LayerClustersInputMask"));
  auto sumes = consumesCollector();
  produces<std::vector<std::pair<unsigned int, float> > >("filteredLayerClusters");
}

void ClustersInSeedingRegionProducer::fillDescriptions(
    edm::ConfigurationDescriptions &descriptions) {
  // hgcalMultiClusters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("HGCLayerClusters", edm::InputTag("hgcalLayerClusters"));
  desc.add<edm::InputTag>("LayerClustersInputMask",
      edm::InputTag("hgcalLayerClusters", "InitialLayerClustersMask"));
  desc.add<bool>("isFirstIteration", true);
  descriptions.add("ClustersInSeedingRegion", desc);
}


std::vector<std::pair<unsigned int, float> > ClustersInSeedingRegionProducer::filterLayerClustersByPosition(const std::vector<std::array<math::XYZPoint, 2> >& boundaryPoints, const std::vector<reco::BasicCluster>& layerClusters, const std::vector<float>& clustersMask )
{
  std::vector<unsigned int> v;
  unsigned int numLayerClusters = layerClusters.size();

  for(unsigned int i =0; i< numLayerClusters; ++i)
  {
      if(clustersMask[i] > 0.f)
      {
          auto& lc = layerClusters[i];
          for(auto& bp : boundaryPoints)
          {
              if(lc.z() >= bp[0].z() and lc.z() <= bp[1].z() and
                 lc.x() >= bp[0].x() and lc.x() <= bp[1].x() and
                 lc.y() >= bp[0].y() and lc.y() <= bp[1].y()
                )
                {
                    v.emplace_back(i,clustersMask[i]);
                    break;
                }
          }
      }
  }
  return v;
}

void ClustersInSeedingRegionProducer::produce(edm::Event &evt,
                                       const edm::EventSetup &es) {

  edm::Handle<std::vector<reco::BasicCluster>> clusterHandle;
  edm::Handle<std::vector<float>> inputClustersMaskHandle;
  evt.getByToken(clusters_token, clusterHandle);
  evt.getByToken(clustersMask_token, inputClustersMaskHandle);

  const auto& layerClusters = *clusterHandle;
  const auto& inputClusterMask = *inputClustersMaskHandle;
  std::unique_ptr<std::vector<std::pair<unsigned int, float> > > filteredLayerClusters(new std::vector<std::pair<unsigned int, float>>);


  evt.put(std::move(filteredLayerClusters),"iterFilteredLayerClusters");
}
