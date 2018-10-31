// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 09/2018
// Copyright CERN

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"


#include "RecoHGCal/TICL/interface/PatternRecognitionAlgo.h"
#include "RecoHGCal/TICL/interface/Trackster.h"
#include "RecoHGCal/TICL/plugins/TrackstersProducer.h"

#include "RecoHGCal/TICL/interface/PatternRecognitionbyMultiClusters.h"


DEFINE_FWK_MODULE(TrackstersProducer);

TrackstersProducer::TrackstersProducer(const edm::ParameterSet& ps): theAlgo(std::make_unique<PatternRecognitionbyMultiClusters>(ps))
{
  clusters_token = consumes<std::vector<reco::CaloCluster>>(
      ps.getParameter<edm::InputTag>("HGCLayerClusters"));
  filteredClustersMask_token = consumes<std::vector<std::pair<unsigned int, float>>>(
      ps.getParameter<edm::InputTag>("filteredLayerClusters"));
  produces<std::vector<Trackster>>("TrackstersByMultiClusters");
}

void TrackstersProducer::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  // hgcalMultiClusters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("HGCLayerClusters",
                          edm::InputTag("hgcalLayerClusters"));
  desc.add<edm::InputTag>("filteredLayerClusters",
      edm::InputTag("FilteredLayerClusters","iterationLabelGoesHere"));
  descriptions.add("Tracksters", desc);
}


void TrackstersProducer::produce(edm::Event& evt,
                                          const edm::EventSetup& es) {
  edm::Handle<std::vector<reco::CaloCluster>> clusterHandle;
  edm::Handle<std::vector<std::pair<unsigned int, float>>> filteredLayerClustersHandle;

  evt.getByToken(clusters_token, clusterHandle);
  evt.getByToken(filteredClustersMask_token, filteredLayerClustersHandle);
  std::cout << "TrackstersProducer::produce" << std::endl;

  const auto& layerClusters = *clusterHandle;
  const auto& inputClusterMask = *filteredLayerClustersHandle;
  std::unique_ptr<std::vector<std::pair<unsigned int, float>>> filteredLayerClusters;
  auto result = std::make_unique<std::vector<Trackster>>();;
  theAlgo.makeTracksters(evt, es, layerClusters, inputClusterMask, *result);
  evt.put(std::move(result), "TrackstersByMultiClusters");

}
