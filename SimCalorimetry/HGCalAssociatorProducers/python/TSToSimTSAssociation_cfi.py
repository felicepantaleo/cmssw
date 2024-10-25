import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalAssociatorProducers.LCToTSAssociator_cfi import layerClusterToCLUE3DTracksterAssociation, layerClusterToTracksterMergeAssociation, layerClusterToSimTracksterAssociation, layerClusterToSimTracksterFromCPsAssociation

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
from Configuration.ProcessModifiers.ticl_superclustering_mustache_ticl_cff import ticl_superclustering_mustache_ticl


from SimCalorimetry.HGCalAssociatorProducers.AllTracksterToSimTracksterAssociatorsByLCsProducer_cfi import AllTracksterToSimTracksterAssociatorsByLCsProducer
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels, associatorsInstances

allTrackstersToSimTrackstersAssociationsByLCs = AllTracksterToSimTracksterAssociatorsByLCsProducer.clone(    
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in ticlIterLabels]
    ),
    simTracksterCollections = cms.VInputTag(
      cms.InputTag('ticlSimTracksters'),
      cms.InputTag('ticlSimTracksters','fromCPs')
    ),
)

### Barrel associatord

barrelTracksterSimTracksterAssociationPR = cms.EDProducer("TSToSimTSHitLCAssociatorEDProducer",
    associator = cms.InputTag("barrelSimTracksterHitLCAssociatorByEnergyScoreProducer"),
    label_tst = cms.InputTag("ticlBarrelTracksters"),
    label_simTst = cms.InputTag("ticlBarrelSimTracksters"),
    label_lcl = cms.InputTag("barrelLayerClusters"),
    label_scl = cms.InputTag("mix", "MergedCaloTruth"),
    label_cp = cms.InputTag("mix", "MergedCaloTruth")
)

barrelTracksterSimTracksterAssociationLinkingPR = cms.EDProducer("TSToSimTSHitLCAssociatorEDProducer",
    associator = cms.InputTag("barrelSimTracksterHitLCAssociatorByEnergyScoreProducer"),
    label_tst = cms.InputTag("ticlBarrelTracksters"),
    label_simTst = cms.InputTag("ticlBarrelSimTracksters", "fromCPs"),
    label_lcl = cms.InputTag("barrelLayerClusters"),
    label_scl = cms.InputTag("mix", "MergedCaloTruth"),
    label_cp = cms.InputTag("mix", "MergedCaloTruth")
)

