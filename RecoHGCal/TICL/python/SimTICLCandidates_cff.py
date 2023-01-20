import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.simTICLCandidatesProducer_cfi import simTICLCandidatesProducer as _simTICLCandidatesProducer

ticlSimTICLCandidates = _simTICLCandidatesProducer.clone(
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(ticlSimTICLCandidates,
    simclusters = "mixData:MergedCaloTruth",
    caloparticles = "mixData:MergedCaloTruth",
)

ticlSimTICLCandidatesTask = cms.Task(ticlSimTICLCandidates)
