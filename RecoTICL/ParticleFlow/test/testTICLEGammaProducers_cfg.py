# Builds the four e/gamma producers on a Path with their defaults, so TICLEGammaSuperClustersProducer loads its three
# models; and without models. With an argument of _bad, the supercluster producer has that configuration.
import sys

import FWCore.ParameterSet.Config as cms
from RecoTICL.ParticleFlow.ticlEGammaTrackstersProducer_cfi import ticlEGammaTrackstersProducer as _tracksters
from RecoTICL.ParticleFlow.ticlEGammaSuperClustersProducer_cfi import ticlEGammaSuperClustersProducer as _superClusters
from RecoTICL.ParticleFlow.ticlElectronSuperClustersProducer_cfi import ticlElectronSuperClustersProducer as _selection
from RecoTICL.ParticleFlow.ticlElectronSeedsProducer_cfi import ticlElectronSeedsProducer as _seeds

process = cms.Process("TEST")
process.source = cms.Source("EmptySource")
process.maxEvents.input = 0
process.load("PhysicsTools.ONNXRuntime.ONNXService_cfi")

_bad = {
    "missingRegression": dict(regressionModel="RecoTICL/ParticleFlow/data/egammaSuperClusters/regression_v0.onnx"),
    "missingScore": dict(scoreModel="RecoTICL/ParticleFlow/data/egammaSuperClusters/score_v0.onnx"),
    "missingTypeScore": dict(typeScoreModel="RecoTICL/ParticleFlow/data/egammaSuperClusters/type_v0.onnx"),
}
_args = [a for a in sys.argv if a in _bad]
process.ticlEGammaTracksters = _tracksters.clone()
process.ticlEGammaSuperClusters = _superClusters.clone(**(_bad[_args[0]] if _args else {}))
process.ticlElectronSuperClusters = _selection.clone()
process.ticlElectronSeeds = _seeds.clone()
process.p = cms.Path(process.ticlEGammaTracksters + process.ticlEGammaSuperClusters +
                     process.ticlElectronSuperClusters + process.ticlElectronSeeds)
if not _args:
    process.ticlEGammaSuperClustersNoModels = _superClusters.clone(regressionModel="", scoreModel="",
                                                                   typeScoreModel="")
    process.p += process.ticlEGammaSuperClustersNoModels
