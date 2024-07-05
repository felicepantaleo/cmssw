import FWCore.ParameterSet.Config as cms

layerClusterToTracksterAssociation = cms.EDProducer("LCToTSAssociatorProducer",
    layer_clusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksters = cms.InputTag("ticlTracksters"),
)

from SimCalorimetry.HGCalAssociatorProducers.LCToTSAssociatorProducer_cfi import LCToTSAssociatorProducer

layerClusterToCLUE3DTracksterAssociation = LCToTSAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersCLUE3DHigh")
)

layerClusterToTracksterMergeAssociation = LCToTSAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersMerge")
)

layerClusterToSimTracksterAssociation = LCToTSAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlSimTracksters")
)

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
ticl_v5.toModify(layerClusterToTracksterMergeAssociation, tracksters = cms.InputTag("ticlCandidate"))