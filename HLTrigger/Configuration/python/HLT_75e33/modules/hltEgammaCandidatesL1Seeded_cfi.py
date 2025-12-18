import FWCore.ParameterSet.Config as cms

hltEgammaCandidatesL1Seeded = cms.EDProducer("EgammaHLTRecoEcalCandidateProducers",
    recoEcalCandidateCollection = cms.string(''),
    scHybridBarrelProducer = cms.InputTag("hltParticleFlowSuperClusterECALL1Seeded","particleFlowSuperClusterECALBarrel"),
    scIslandEndcapProducer = cms.InputTag("hltTiclEGammaSuperClusterProducerL1Seeded")
)


from Configuration.ProcessModifiers.ticl_v4_cff import ticl_v4
ticl_v4.toModify(hltEgammaCandidatesL1Seeded, scIslandEndcapProducer = cms.InputTag("hltParticleFlowSuperClusterHGCalFromTICLL1Seeded"))