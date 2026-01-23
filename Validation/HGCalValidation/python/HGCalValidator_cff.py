import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQM_cfg import *
from Validation.HGCalValidation.hgcalValidator_cfi import hgcalValidator as _hgcalValidator
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels, associatorsInstances
from Configuration.ProcessModifiers.ticl_v4_cff import ticl_v4

# Default configuration is now TICLv5
hgcalValidator = _hgcalValidator.clone(
    label_tst = cms.VInputTag(*[cms.InputTag(label) for label in ticlIterLabels] + [cms.InputTag("ticlSimTracksters", "fromCPs"), cms.InputTag("ticlSimTracksters")]),
    allTracksterTracksterAssociatorsLabels = cms.VInputTag( *[cms.InputTag('allTrackstersToSimTrackstersAssociationsByLCs:'+associator) for associator in associatorsInstances] ),
    allTracksterTracksterByHitsAssociatorsLabels = cms.VInputTag( *[cms.InputTag('allTrackstersToSimTrackstersAssociationsByHits:'+associator) for associator in associatorsInstances] ),
    # v5 defaults
    LayerClustersInputMask = cms.VInputTag(
        cms.InputTag("ticlTrackstersCLUE3DHigh"),
        cms.InputTag("ticlSimTracksters", "fromCPs"),
        cms.InputTag("ticlSimTracksters")
    ),
    ticlTrackstersMerge = cms.InputTag("ticlCandidate"),
    isticlv5 = cms.untracked.bool(True),
    mergeSimToRecoAssociator = cms.InputTag("allTrackstersToSimTrackstersAssociationsByLCs:ticlSimTrackstersfromCPsToticlCandidate"),
    mergeRecoToSimAssociator = cms.InputTag("allTrackstersToSimTrackstersAssociationsByLCs:ticlCandidateToticlSimTrackstersfromCPs"),
)

# Revert logic for TICLv4
ticl_v4.toModify(hgcalValidator,
    LayerClustersInputMask = cms.VInputTag(
        cms.InputTag("ticlTrackstersCLUE3DHigh"),
        cms.InputTag("ticlSimTracksters", "fromCPs"),
        cms.InputTag("ticlSimTracksters")
    ),
    ticlTrackstersMerge = cms.InputTag("ticlTrackstersMerge"),
    isticlv5 = cms.untracked.bool(False),
    mergeSimToRecoAssociator = cms.InputTag("allTrackstersToSimTrackstersAssociationsByLCs:ticlSimTrackstersfromCPsToticlTrackstersMerge"),
    mergeRecoToSimAssociator = cms.InputTag("allTrackstersToSimTrackstersAssociationsByLCs:ticlTrackstersMergeToticlSimTrackstersfromCPs"),
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(hgcalValidator,
    label_cp_fake = "mixData:MergedCaloTruth",
    label_cp_effic = "mixData:MergedCaloTruth",
    label_scl = "mixData:MergedCaloTruth",
)

from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
phase2_hgcalV10.toModify(hgcalValidator, totallayers_to_monitor = cms.int32(50))

from Configuration.Eras.Modifier_phase2_hgcalV16_cff import phase2_hgcalV16
phase2_hgcalV16.toModify(hgcalValidator, totallayers_to_monitor = cms.int32(47))