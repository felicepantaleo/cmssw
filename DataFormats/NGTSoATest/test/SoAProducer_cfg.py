import FWCore.ParameterSet.Config as cms
from DataFormats.NGTSoATest.soAProducer4_cfi import soAProducer4

process = cms.Process("SoAProducer")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Accelerators_cff")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# Number of events
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

# Empty source
process.source = cms.Source("EmptySource")

# Add the producer
process.soaproducer = cms.EDProducer("SoAProducer", soaParameter = cms.int32(42))
process.soaproducer2 = cms.EDProducer("SoAProducer2", soaInput = cms.InputTag("soaproducer", "SoAProduct"))
process.soaproducer3 = cms.EDProducer("SoAProducer3", soaInput_0 = cms.InputTag("soaproducer", "SoAProduct"), soaInput_1 = cms.InputTag("soaproducer2", "SoAProduct2"))
#process.soaproducer4 = soAProducer4.clone(soaInput_2 = cms.InputTag("soaproducer3", "SoAProduct3"))
process.soaproducer4 = cms.EDProducer("SoAProducer4@alpaka",
    soaInput_2 = cms.InputTag("soaproducer3", "SoAProduct3"),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string("serial_sync")
    )
)

# Add to process path
process.p = cms.Path(process.soaproducer * process.soaproducer2 * process.soaproducer3 
                     * process.soaproducer4
                     )