import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import *
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import *

# patch particle flow clusters for HGC into local reco sequence
# (for now until global reco is going with some sort of clustering)
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGC_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHGC_cfi import *
from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff import hgcalLayerClusters
from RecoLocalCalo.HGCalRecProducers.hgcalMultiClusters_cfi import hgcalMultiClusters

from RecoHGCal.TICL.ticlSeedingRegionProducer_cfi import ticlSeedingRegionProducer
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer
from RecoHGCal.TICL.ticlCandidateFromTrackstersProducer_cfi import ticlCandidateFromTrackstersProducer
from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer
from Validation.HGCalValidation.ticlPFValidationDefault_cfi import ticlPFValidationDefault as ticlPFValidation

## withReco: requires full reco of the event to run this part
## i.e. collections of generalTracks can be accessed
def TICL_iterations_withReco(process):
  process.FEVTDEBUGHLTEventContent.outputCommands.extend([
    'keep *_MultiClustersFromTracksters*_*_*',
    'keep *_ticlCandidateFromTrackstersProducer*_*_*',
    'keep *_pfTICLProducer*_*_*',
  ])

  process.ticlLayerTileProducer = ticlLayerTileProducer.clone()

  process.ticlSeedingTrk = ticlSeedingRegionProducer.clone(
    algoId = 1
  )

  process.filteredLayerClustersTrk = filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgo",
    algo_number = 8,
    iteration_label = "Trk"
  )

  process.trackstersTrk = trackstersProducer.clone(
    filtered_mask = cms.InputTag("filteredLayerClustersTrk", "Trk"),
    seeding_regions = cms.InputTag("ticlSeedingTrk"),
    algo_verbosity = 0,
    missing_layers = 3,
    min_clusters_per_ntuplet = 5,
    min_cos_theta = 0.99, # ~10 degrees                                              
    min_cos_pointing = 0.9
  )

  process.multiClustersFromTrackstersTrk = multiClustersFromTrackstersProducer.clone(
      label = "TrkMultiClustersFromTracksterByCA",
      Tracksters = "trackstersTrk"
  )


  process.ticlSeedingGlobal = ticlSeedingRegionProducer.clone(
    algoId = 2
  )

  process.filteredLayerClustersMIP = filteredLayerClustersProducer.clone(
      clusterFilter = "ClusterFilterBySize",
      algo_number = 8,
      max_cluster_size = 2, # inclusive
      iteration_label = "MIP"
  )

  process.trackstersMIP = trackstersProducer.clone(
      filtered_mask = cms.InputTag("filteredLayerClustersMIP", "MIP"),
      seeding_regions = cms.InputTag("ticlSeedingGlobal"),
      missing_layers = 3,
      min_clusters_per_ntuplet = 15,
      min_cos_theta = 0.99, # ~10 degrees
      min_cos_pointing = 0.9
  )

  process.multiClustersFromTrackstersMIP = multiClustersFromTrackstersProducer.clone(
      label = "MIPMultiClustersFromTracksterByCA",
      Tracksters = "trackstersMIP"
  )

  process.filteredLayerClusters = filteredLayerClustersProducer.clone(
      clusterFilter = "ClusterFilterByAlgoAndSize",
      min_cluster_size = 2,
      algo_number = 8,
      iteration_label = "algo8",
      LayerClustersInputMask = "trackstersMIP"
  )

  process.tracksters = trackstersProducer.clone(
      original_mask = "trackstersMIP",
      filtered_mask = cms.InputTag("filteredLayerClusters", "algo8"),
      seeding_regions = cms.InputTag("ticlSeedingGlobal"),
      missing_layers = 2,
      min_clusters_per_ntuplet = 15,
      min_cos_theta = 0.94, # ~20 degrees
      min_cos_pointing = 0.7,
      eid_graph_path = cms.string("RecoHGCal/TICL/data/tf_models/energy_id_v0.pb"),
      eid_input_name = cms.string("input"),
      eid_output_name_energy = cms.string(""),
      eid_output_name_id = cms.string("output/id_probabilities"),
      eid_min_cluster_energy = cms.double(1.),
      eid_n_layers = cms.int32(50),
      eid_n_clusters = cms.int32(10),
  )

  process.ticlCandidateFromTrackstersProducer = ticlCandidateFromTrackstersProducer.clone()
  process.multiClustersFromTracksters = multiClustersFromTrackstersProducer.clone(
      Tracksters = "tracksters"
  )

  process.pfTICLProducer = pfTICLProducer.clone()

  process.hgcalMultiClusters = hgcalMultiClusters
  process.TICL_Task = cms.Task(
      process.ticlLayerTileProducer,
      process.ticlSeedingTrk,
      process.filteredLayerClustersTrk,
      process.trackstersTrk,
      process.multiClustersFromTrackstersTrk,
      process.ticlSeedingGlobal,
      process.filteredLayerClustersMIP,
      process.trackstersMIP,
      process.multiClustersFromTrackstersMIP,
      process.filteredLayerClusters,
      process.tracksters,
      process.multiClustersFromTracksters,
      process.ticlCandidateFromTrackstersProducer,
      process.pfTICLProducer)
  process.schedule.associate(process.TICL_Task)
  process.ticlPFValidation = ticlPFValidation
  process.hgcalValidation.insert(-1, process.ticlPFValidation)

  return process

def TICL_iterations(process):
  process.FEVTDEBUGHLTEventContent.outputCommands.extend(['keep *_multiClustersFromTracksters*_*_*'])

  process.ticlLayerTileProducer = ticlLayerTileProducer.clone()

  process.ticlSeedingGlobal = ticlSeedingRegionProducer.clone(
    algoId = 2
  )

  process.filteredLayerClustersMIP = filteredLayerClustersProducer.clone(
      clusterFilter = "ClusterFilterBySize",
      algo_number = 8,
      max_cluster_size = 2, # inclusive
      iteration_label = "MIP"
  )

  process.trackstersMIP = trackstersProducer.clone(
      filtered_mask = cms.InputTag("filteredLayerClustersMIP", "MIP"),
      seeding_regions = cms.InputTag("ticlSeedingGlobal"),
      missing_layers = 3,
      min_clusters_per_ntuplet = 15,
      min_cos_theta = 0.99 # ~10 degrees
  )

  process.multiClustersFromTrackstersMIP = multiClustersFromTrackstersProducer.clone(
      label = "MIPMultiClustersFromTracksterByCA",
      Tracksters = "trackstersMIP"
  )

  process.filteredLayerClusters = filteredLayerClustersProducer.clone(
      clusterFilter = "ClusterFilterByAlgoAndSize",
      min_cluster_size = 2,
      algo_number = 8,
      iteration_label = "algo8"
  )

  process.tracksters = trackstersProducer.clone(
      original_mask = "trackstersMIP",
      filtered_mask = cms.InputTag("filteredLayerClusters", "algo8"),
      seeding_regions = cms.InputTag("ticlSeedingGlobal"),
      missing_layers = 2,
      min_clusters_per_ntuplet = 15,
      min_cos_theta = 0.94, # ~20 degrees
      min_cos_pointing = 0.7,
      eid_graph_path = cms.string("RecoHGCal/TICL/data/tf_models/energy_id_v0.pb"),
      eid_input_name = cms.string("input"),
      eid_output_name_energy = cms.string(""),
      eid_output_name_id = cms.string("output/id_probabilities"),
      eid_min_cluster_energy = cms.double(1.),
      eid_n_layers = cms.int32(50),
      eid_n_clusters = cms.int32(10),
  )

  process.multiClustersFromTracksters = multiClustersFromTrackstersProducer.clone(
      Tracksters = "tracksters"
  )

  process.HGCalUncalibRecHit = HGCalUncalibRecHit
  process.HGCalRecHit = HGCalRecHit
  process.hgcalLayerClusters = hgcalLayerClusters
  process.hgcalMultiClusters = hgcalMultiClusters
  process.TICL_Task = cms.Task(process.HGCalUncalibRecHit,
      process.HGCalRecHit,
      process.hgcalLayerClusters,
      process.filteredLayerClustersMIP,
      process.ticlLayerTileProducer,
      process.ticlSeedingGlobal,
      process.trackstersMIP,
      process.multiClustersFromTrackstersMIP,
      process.filteredLayerClusters,
      process.tracksters,
      process.multiClustersFromTracksters,
      process.hgcalMultiClusters)
  process.schedule = cms.Schedule(process.raw2digi_step,process.FEVTDEBUGHLToutput_step)
  process.schedule.associate(process.TICL_Task)
  return process

