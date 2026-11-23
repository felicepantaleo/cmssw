import FWCore.ParameterSet.Config as cms

hltEgammaHGCALIDVarsUnseeded = cms.EDProducer("EgammaHLTHGCalIDVarProducer",
    hgcalHitMap = cms.InputTag("hltHGCalRecHitMap","hgcalRecHitMap"),
    hgcalHits = cms.InputTag("hltHGCalRecHitMap","RefProdVectorHGCRecHitCollection"),
    minSignalOverNoise = cms.double(5.0),
    layerClusters = cms.InputTag("hltMergeLayerClusters"),
    rCylinder = cms.double(2.8),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesUnseeded")
)
