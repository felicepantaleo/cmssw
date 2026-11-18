# Writes the HGCalReco SoA host collections to a TTree file and to an RNTuple file.
import FWCore.ParameterSet.Config as cms

process = cms.Process("WRITE")
process.source = cms.Source("EmptySource")
process.maxEvents.input = 8

process.writer = cms.EDProducer("TestWriteHGCalRecoSoA")

process.outputTTree = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("testHGCalRecoSoA.root")
)
process.outputRNTuple = cms.OutputModule("RNTupleTempOutputModule",
    fileName = cms.untracked.string("testHGCalRecoSoA_rntuple.root")
)

process.path = cms.Path(process.writer)
process.endPath = cms.EndPath(process.outputTTree + process.outputRNTuple)
