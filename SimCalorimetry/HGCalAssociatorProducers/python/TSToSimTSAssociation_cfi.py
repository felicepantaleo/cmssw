import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalAssociatorProducers.LCToTSAssociator_cfi import layerClusterToCLUE3DTracksterAssociation, layerClusterToTracksterMergeAssociation, layerClusterToSimTracksterAssociation2
from SimCalorimetry.HGCalAssociatorProducers.tracksterToSimTracksterAssociatorProducer_cfi import tracksterToSimTracksterAssociatorProducer

tracksterSimTracksterFromCPsAssociationLinking = tracksterToSimTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersMerge"),
    simTracksters = cms.InputTag("ticlSimTracksters", "fromCPs"),
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterMap = cms.InputTag("layerClusterToTracksterMergeAssociation"),
    simTracksterMap = cms.InputTag("layerClusterToSimTracksterAssociation2")
)

tracksterSimTracksterAssociationLinking = tracksterToSimTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersMerge"),
    simTracksters = cms.InputTag("ticlSimTracksters"),
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterMap = cms.InputTag("layerClusterToTracksterMergeAssociation"),
    simTracksterMap = cms.InputTag("layerClusterToSimTracksterAssociation2")
)


tracksterSimTracksterFromCPsAssociationPR = tracksterToSimTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersCLUE3DHigh"),
    simTracksters = cms.InputTag("ticlSimTracksters", "fromCPs"),
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterMap = cms.InputTag("layerClusterToCLUE3DTracksterAssociation"),
    simTracksterMap = cms.InputTag("layerClusterToSimTracksterAssociation2")
)

tracksterSimTracksterAssociationPR = tracksterToSimTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersCLUE3DHigh"),
    simTracksters = cms.InputTag("ticlSimTracksters"),
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterMap = cms.InputTag("layerClusterToCLUE3DTracksterAssociation"),
    simTracksterMap = cms.InputTag("layerClusterToSimTracksterAssociation2")
)


from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
''' For future separate iterations
ticl_v5.toModify(tracksterSimTracksterAssociationLinkingbyCLUE3D, tracksters = cms.InputTag("mergedTrackstersProducer"))
tracksterSimTracksterAssociationLinkingbyCLUE3DEM = tracksterSimTracksterAssociationLinkingbyCLUE3D.clone(tracksters = cms.InputTag("ticlTrackstersCLUE3DEM"))
tracksterSimTracksterAssociationLinkingbyCLUE3DHAD = tracksterSimTracksterAssociationLinkingbyCLUE3D.clone(tracksters = cms.InputTag("ticlTrackstersCLUE3DHAD"))

ticl_v5.toModify(tracksterSimTracksterAssociationPRbyCLUE3D, tracksters = cms.InputTag("mergedTrackstersProducer"))
tracksterSimTracksterAssociationPRbyCLUE3DEM = tracksterSimTracksterAssociationPRbyCLUE3D.clone(tracksters = cms.InputTag("ticlTrackstersCLUE3DEM"))
tracksterSimTracksterAssociationPRbyCLUE3DHAD = tracksterSimTracksterAssociationPRbyCLUE3D.clone(tracksters = cms.InputTag("ticlTrackstersCLUE3DHAD"))
'''

ticl_v5.toModify(tracksterSimTracksterAssociationLinking, tracksters = cms.InputTag("ticlCandidate"))
ticl_v5.toModify(tracksterSimTracksterFromCPsAssociationLinking, tracksters = cms.InputTag("ticlCandidate"))
