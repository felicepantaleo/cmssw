import FWCore.ParameterSet.Config as cms

hltEgammaCandidatesUnseeded = cms.EDProducer("EgammaHLTRecoEcalCandidateProducers",
    recoEcalCandidateCollection = cms.string(''),
    scHybridBarrelProducer = cms.InputTag("hltParticleFlowSuperClusterECALUnseeded","particleFlowSuperClusterECALBarrel"),
    scIslandEndcapProducer = cms.InputTag("hltParticleFlowSuperClusterHGCalFromTICLUnseeded")
)

from Configuration.ProcessModifiers.ticl_v4_cff import ticl_v4
ticl_v4.toModify(hltEgammaCandidatesUnseeded, scIslandEndcapProducer = cms.InputTag("hltParticleFlowSuperClusterHGCalFromTICLUnseeded"))

