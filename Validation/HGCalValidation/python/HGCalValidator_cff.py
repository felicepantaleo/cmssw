import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQM_cfg import *
from Validation.HGCalValidation.hgcalValidator_cfi import hgcalValidator as _hgcalValidator


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
# labelTst_v5 = ["ticlTrackstersCLUE3DEM", "ticlTrackstersCLUE3DHAD", "ticlTracksterLinks"] # for separate CLUE3D iterations
labelTst_v5 = ["ticlTrackstersCLUE3DHigh", "ticlTracksterLinks"]
labelTst_v5.extend([cms.InputTag("ticlSimTracksters", "fromCPs"), cms.InputTag("ticlSimTracksters")])
# lcInputMask_v5  = ["ticlTrackstersCLUE3DEM", "ticlTrackstersCLUE3DHAD", "ticlTracksterLinks"] # for separate CLUE3D iterations
lcInputMask_v5  = ["ticlTrackstersCLUE3DHigh", "ticlTracksterLinks"]
lcInputMask_v5.extend([cms.InputTag("ticlSimTracksters", "fromCPs"), cms.InputTag("ticlSimTracksters")])

ticl_v5.toModify(hgcalValidator,
    label_tst = cms.VInputTag(labelTst_v5),
    LayerClustersInputMask = cms.VInputTag(lcInputMask_v5),
    ticlTrackstersMerge = cms.InputTag("ticlCandidate"),
    isticlv5 = cms.untracked.bool(True)
)
