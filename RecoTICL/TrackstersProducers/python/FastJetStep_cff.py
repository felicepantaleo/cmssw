import FWCore.ParameterSet.Config as cms

from RecoTICL.LayerClustersProducers.TICLSeedingRegions_cff import ticlSeedingGlobal, ticlSeedingGlobalHFNose
from RecoTICL.TrackstersProducers.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoTICL.LayerClustersProducers.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersFastJet = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSize",
    min_cluster_size = 3, # inclusive
    iteration_label = "FastJet"
)

# PATTERN RECOGNITION

ticlTrackstersFastJet = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersFastJet:FastJet",
    seeding_regions = "ticlSeedingGlobal",
    itername = "FastJet",
    patternRecognitionBy = "FastJet",
    pluginPatternRecognitionByFastJet = dict (
        algo_verbosity = 2
    )
)

ticlFastJetStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersFastJet
    ,ticlTrackstersFastJet)

