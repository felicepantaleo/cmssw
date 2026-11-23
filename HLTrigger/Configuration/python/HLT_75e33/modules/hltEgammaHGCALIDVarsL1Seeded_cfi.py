import FWCore.ParameterSet.Config as cms

hltEgammaHGCALIDVarsL1Seeded = cms.EDProducer("EgammaHLTHGCalIDVarProducer",
    hgcalRecHits = cms.InputTag(""),
    hgcalHitMap = cms.InputTag("hltHGCalRecHitMapL1Seeded","hgcalRecHitMap"),
    hgcalHits = cms.InputTag("hltHGCalRecHitMapL1Seeded","RefProdVectorHGCRecHitCollection"),
    minSignalOverNoise = cms.double(5.0),
    layerClusters = cms.InputTag("hltMergeLayerClustersL1Seeded"),
    rCylinder = cms.double(2.8),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesL1Seeded")
)
