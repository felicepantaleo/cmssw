import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalAssociatorProducers.HitToTracksterAssociatorProducer_cfi import hitToTracksterAssociatorProducer

hitToTrackstersAssociationLinking = hitToTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersMerge"),
)


hitToTrackstersAssociationPR = hitToTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersCLUE3DHigh"),
)

hitToSimTracksterAssociation = hitToTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlSimTracksters"),
)

hitToSimTracksterFromCPsAssociation = hitToTracksterAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlSimTracksters", "fromCPs"),
)


from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5

ticl_v5.toModify(hitToTrackstersAssociationLinking, tracksters = cms.InputTag("ticlCandidate"))
