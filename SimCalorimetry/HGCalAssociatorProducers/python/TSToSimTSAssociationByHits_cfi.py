import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalAssociatorProducers.HitToTracksterAssociation_cfi import *
from SimCalorimetry.HGCalAssociatorProducers.tracksterToSimTracksterAssociatorByHitsProducer_cfi import tracksterToSimTracksterAssociatorByHitsProducer



tracksterSimTracksterAssociationByHitsLinking = tracksterToSimTracksterAssociatorByHitsProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersMerge"),
    hitToTracksterMap = cms.InputTag("hitToTrackstersAssociationLinking","hitToTracksterMap"),
    tracksterToHitMap = cms.InputTag("hitToTrackstersAssociationLinking","tracksterToHitMap"),
)


tracksterSimTracksterAssociationByHitsPR = tracksterToSimTracksterAssociatorByHitsProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersCLUE3DHigh"),
    hitToTracksterMap = cms.InputTag("hitToTrackstersAssociationPR","hitToTracksterMap"),
    tracksterToHitMap = cms.InputTag("hitToTrackstersAssociationPR","tracksterToHitMap"),
)



from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5

ticl_v5.toModify(tracksterSimTracksterAssociationByHitsLinking, tracksters = cms.InputTag("ticlCandidate"))


from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2

premix_stage2.toModify(tracksterSimTracksterAssociationByHitsLinking,
    caloParticles = "mixData:MergedCaloTruth",
)

premix_stage2.toModify(tracksterSimTracksterAssociationByHitsPR,
    caloParticles = "mixData:MergedCaloTruth",
)