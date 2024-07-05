import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalAssociatorProducers.LCToTSAssociator_cfi import layerClusterToCLUE3DTracksterAssociation, layerClusterToTracksterMergeAssociation, layerClusterToSimTracksterAssociation
from SimCalorimetry.HGCalAssociatorProducers.tracksterToSimTracksterAssociatorProducer_cfi import tracksterToSimTracksterAssociatorProducer

tracksterSimTracksterFromCPsAssociationLinking = tracksterToSimTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersMerge"),
    simTracksters = cms.InputTag("ticlSimTracksters", "fromCPs"),
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterMap = cms.InputTag("layerClusterToTracksterMergeAssociation"),
    simTracksterMap = cms.InputTag("layerClusterToSimTracksterAssociation")
)

tracksterSimTracksterAssociationLinking = tracksterToSimTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersMerge"),
    simTracksters = cms.InputTag("ticlSimTracksters"),
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterMap = cms.InputTag("layerClusterToTracksterMergeAssociation"),
    simTracksterMap = cms.InputTag("layerClusterToSimTracksterAssociation")
)


tracksterSimTracksterFromCPsAssociationPR = tracksterToSimTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersCLUE3DHigh"),
    simTracksters = cms.InputTag("ticlSimTracksters", "fromCPs"),
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterMap = cms.InputTag("layerClusterToCLUE3DTracksterAssociation"),
    simTracksterMap = cms.InputTag("layerClusterToSimTracksterAssociation")
)

tracksterSimTracksterAssociationPR = tracksterToSimTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersCLUE3DHigh"),
    simTracksters = cms.InputTag("ticlSimTracksters"),
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterMap = cms.InputTag("layerClusterToCLUE3DTracksterAssociation"),
    simTracksterMap = cms.InputTag("layerClusterToSimTracksterAssociation")
)


from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5

ticl_v5.toModify(tracksterSimTracksterAssociationLinking, tracksters = cms.InputTag("ticlCandidate"))
ticl_v5.toModify(tracksterSimTracksterFromCPsAssociationLinking, tracksters = cms.InputTag("ticlCandidate"))
