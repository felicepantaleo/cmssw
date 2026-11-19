# Builds TICLCandidateArbitrationProducer and its global cache with the default and with other valid configurations.
# With an argument of _bad, the module has that configuration and its construction must throw.
import sys

import FWCore.ParameterSet.Config as cms
from RecoTICL.Interpretation.ticlCandidateArbitrationProducer_cfi import ticlCandidateArbitrationProducer as _arbitration

process = cms.Process("TEST")
process.source = cms.Source("EmptySource")
process.maxEvents.input = 0
process.load("PhysicsTools.ONNXRuntime.ONNXService_cfi")

_bad = {
    "claimEnergyWithoutClaims": dict(attachClaimTracksters=False, useClaimTracksterEnergy=True),
    "negativeDepositFraction": dict(claimTracksterMinDepositFraction=-0.1),
    "badAssignment": dict(candidateTracksters=dict(assignment="nearestTrack")),
    "photonScoreWithoutProducts": dict(photonMinScore=0.35),
}
_args = [a for a in sys.argv if a in _bad]
if _args:
    modules = {"arbitration": _arbitration.clone(**_bad[_args[0]])}
else:
    modules = {
        "arbitration": _arbitration.clone(),
        "arbitrationNoCalibration": _arbitration.clone(hadronicTracksterCalibration=""),
        "arbitrationEnergyShare": _arbitration.clone(candidateTracksters=dict(assignment="energyShare")),
        "arbitrationNone": _arbitration.clone(candidateTracksters=dict(assignment="none"),
                                              useClaimTracksterEnergy=True),
        "arbitrationPhotonScore": _arbitration.clone(photonMinScore=0.35,
                                                     eGammaScore=cms.InputTag("ticlEGammaSuperClusters", "tracksterScore"),
                                                     eGammaTracksters="ticlEGammaTracksters"),
    }
process.p = cms.Path()
for label, module in modules.items():
    setattr(process, label, module)
    process.p += module
