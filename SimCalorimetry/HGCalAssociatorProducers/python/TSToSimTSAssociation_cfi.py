import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalAssociatorProducers.LCToTSAssociator_cfi import layerClusterToCLUE3DTracksterAssociation, layerClusterToTracksterMergeAssociation, layerClusterToSimTracksterAssociation, layerClusterToSimTracksterFromCPsAssociation
from SimCalorimetry.HGCalAssociatorProducers.tracksterToSimTracksterAssociatorProducer_cfi import tracksterToSimTracksterAssociatorProducer

tracksterSimTracksterFromCPsAssociationLinking = tracksterToSimTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersMerge"),
    simTracksters = cms.InputTag("ticlSimTracksters", "fromCPs"),
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlTrackstersMerge"),
    simTracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlSimTrackstersfromCPs")
)

tracksterSimTracksterAssociationLinking = tracksterToSimTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersMerge"),
    simTracksters = cms.InputTag("ticlSimTracksters"),
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlTrackstersMerge"),
    simTracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlSimTracksters")
)


tracksterSimTracksterFromCPsAssociationPR = tracksterToSimTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersCLUE3DHigh"),
    simTracksters = cms.InputTag("ticlSimTracksters", "fromCPs"),
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlTrackstersCLUE3DHigh"),
    simTracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlSimTrackstersfromCPs")
)

tracksterSimTracksterAssociationPR = tracksterToSimTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersCLUE3DHigh"),
    simTracksters = cms.InputTag("ticlSimTracksters"),
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlTrackstersCLUE3DHigh"),
    simTracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlSimTracksters")
)

tracksterSimTracksterAssociationSkeletonsPR = tracksterToSimTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersLinks"),
    simTracksters = cms.InputTag("ticlSimTracksters"),
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlTrackstersLinks"),
    simTracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlSimTracksters")
)

tracksterSimTracksterAssociationSkeletonsLinking = tracksterToSimTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersLinks"),
    simTracksters = cms.InputTag("ticlSimTracksters", "fromCPs"),
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlTrackstersLinks"),
    simTracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlSimTrackstersfromCPs")
)

tracksterSimTracksterAssociationRecoveryPR = tracksterToSimTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersRecovery"),
    simTracksters = cms.InputTag("ticlSimTracksters"),
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlTrackstersLinks"),
    simTracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlSimTracksters")
)

tracksterSimTracksterAssociationRecoveryLinking = tracksterToSimTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersRecovery"),
    simTracksters = cms.InputTag("ticlSimTracksters", "fromCPs"),
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlTrackstersLinks"),
    simTracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlSimTrackstersfromCPs")
)


from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5

ticl_v5.toModify(tracksterSimTracksterAssociationLinking, tracksters = cms.InputTag("ticlCandidate"), tracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlCandidate"))
ticl_v5.toModify(tracksterSimTracksterFromCPsAssociationLinking, tracksters = cms.InputTag("ticlCandidate"), tracksterMap = cms.InputTag("allLayerClusterToTracksterAssociations", "ticlCandidate"))


from SimCalorimetry.HGCalAssociatorProducers.AllTracksterToSimTracksterAssociatorsByLCsProducer_cfi import AllTracksterToSimTracksterAssociatorsByLCsProducer
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels

allTrackstersToSimTrackstersAssociationsByLCs = AllTracksterToSimTracksterAssociatorsByLCsProducer.clone(    
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in ticlIterLabels]
    ),
    simTracksterCollections = cms.VInputTag(
      'ticlSimTracksters',
      'ticlSimTracksters:fromCPs'
    ),
)

