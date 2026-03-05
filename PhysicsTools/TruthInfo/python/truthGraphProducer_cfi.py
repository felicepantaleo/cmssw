import FWCore.ParameterSet.Config as cms

truthGraphProducer = cms.EDProducer(
    "TruthGraphProducer",
    # genParticles = cms.InputTag("genParticles"),
    # simTracks    = cms.InputTag("g4SimHits"),
)
