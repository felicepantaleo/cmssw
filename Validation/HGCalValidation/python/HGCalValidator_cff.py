import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQM_cfg import *
from Validation.HGCalValidation.hgcalValidator_cfi import hgcalValidator as _hgcalValidator
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels_v5


hgcalValidator = _hgcalValidator.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(hgcalValidator,
    label_cp_fake = "mixData:MergedCaloTruth"
)

from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
phase2_hgcalV10.toModify(hgcalValidator, totallayers_to_monitor = cms.int32(50))

from Configuration.Eras.Modifier_phase2_hgcalV16_cff import phase2_hgcalV16
phase2_hgcalV16.toModify(hgcalValidator, totallayers_to_monitor = cms.int32(47))

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5

lcInputMask_v5  = ["ticlTrackstersCLUE3DHigh"]
lcInputMask_v5.extend([cms.InputTag("ticlSimTracksters", "fromCPs"), cms.InputTag("ticlSimTracksters")])
labelsValidator = ticlIterLabels_v5.copy()
labelsValidator.extend([cms.InputTag("ticlSimTracksters", "fromCPs"), cms.InputTag("ticlSimTracksters")])
ticl_v5.toModify(hgcalValidator,
    label_tst = cms.VInputTag(labelsValidator),
    LayerClustersInputMask = cms.VInputTag(lcInputMask_v5),
    ticlTrackstersMerge = cms.InputTag("ticlCandidate"),
    isticlv5 = cms.untracked.bool(True)
)
