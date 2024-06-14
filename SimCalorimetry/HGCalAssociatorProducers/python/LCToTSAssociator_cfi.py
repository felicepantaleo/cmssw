import FWCore.ParameterSet.Config as cms

layerClusterToTracksterAssociation = cms.EDProducer("LCToTSAssociatorProducer",
    layer_clusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksters = cms.InputTag("ticlTracksters"),
)
