import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalAssociatorProducers.hitToTracksterAssociator_cfi import hitToTracksterAssociator

hitToTrackstersAssociationLinking = hitToTracksterAssociator.clone(
    tracksters = cms.InputTag("ticlTrackstersMerge"),
)


hitToTrackstersAssociationPR = hitToTracksterAssociator.clone(
    tracksters = cms.InputTag("ticlTrackstersCLUE3DHigh"),
)

hitToSimTracksterAssociation = hitToTracksterAssociator.clone(
    tracksters = cms.InputTag("ticlSimTracksters"),
)

hitToSimTracksterFromCPsAssociation = hitToTracksterAssociator.clone(
    tracksters = cms.InputTag("ticlSimTracksters", "fromCPs"),
)


from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5

ticl_v5.toModify(hitToTrackstersAssociationLinking, tracksters = cms.InputTag("ticlCandidate"))
