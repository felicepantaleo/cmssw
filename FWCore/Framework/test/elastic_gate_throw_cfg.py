import FWCore.ParameterSet.Config as cms
# An ExternalWork module that fails in acquire must still give its slot back, or the
# pool shrinks by one per failure and eventually stops admitting events.
process = cms.Process("GATETHROW")
process.source = cms.Source("EmptySource")
process.maxEvents.input = 60
process.options.numberOfThreads = 4
process.options.numberOfStreams = 4
process.options.TryToContinue = cms.untracked.vstring('ElasticGateTestFailure')
process.flaky = cms.EDProducer("ElasticGateTestAcquirer",
                               asyncMicros=cms.uint32(500),
                               throwEvery=cms.uint32(5))
process.p = cms.Path(process.flaky)
