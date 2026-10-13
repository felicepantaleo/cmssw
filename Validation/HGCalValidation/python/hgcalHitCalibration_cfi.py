import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalHitCalibrationDefault_cfi import hgcalHitCalibrationDefault as _hgcalHitCalibrationDefault
hgcalHitCalibration = _hgcalHitCalibrationDefault.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(hgcalHitCalibration, caloParticles = "mixData:MergedCaloTruth")

# TICLv6 runs no particleFlowClusterHGCal: its HGCAL PF clusters come from the typed tracksters.
from Configuration.ProcessModifiers.ticl_v6_cff import ticl_v6
ticl_v6.toModify(hgcalHitCalibration, hgcalMultiClusters = "particleFlowClusterTICL")
