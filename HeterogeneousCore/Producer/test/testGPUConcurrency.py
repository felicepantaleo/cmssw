import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECO',eras.Run2_2018)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('HeterogeneousCore.CUDAServices.CUDAService_cfi')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("EmptySource",)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32( 4 ),
    numberOfStreams = cms.untracked.uint32( 4 ),
)

# Path and EndPath definitions
from HeterogeneousCore.Producer.testHeterogeneousEDProducerGPU_cfi import testHeterogeneousEDProducerGPU
process.testGPU = testHeterogeneousEDProducerGPU.clone()
process.testGPU.sleep =1000000
process.testGPU.blocks =100000
process.testGPU.threads =256

process.p = cms.Path(process.testGPU)
process.schedule = cms.Schedule(process.p)
