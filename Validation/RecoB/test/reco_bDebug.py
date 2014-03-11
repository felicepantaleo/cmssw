# The following comments couldn't be translated into the new config version:

#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("validationdebug")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("DQMServices.Core.DQM_cfg")

# load the full reconstraction configuration, to make sure we're getting all needed dependencies
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi")  

process.load("Validation.RecoB.bTagAnalysis_cfi")
process.bTagValidation.jetMCSrc = 'AK5byValAlgo'
process.bTagValidation.allHistograms = True 
#process.bTagValidation.fastMC = True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.GlobalTag.globaltag = 'START53_V27::All'

process.load("SimTracker.TrackHistory.TrackClassifier_cff")

process.allBParticles  = cms.EDFilter("GenParticleSelector",
     filter = cms.bool(False),
     src = cms.InputTag("genParticles"),
    cut = cms.string('((abs(pdgId) == 5) || (abs(pdgId) > 500  && abs(pdgId) < 600) || (abs(pdgId) > 5000  && abs(pdgId) < 6000))'),
     stableOnly = cms.bool(False)
)
process.bParticlesFilter  = cms.EDFilter("GenParticleSelector",
     filter = cms.bool(True),
     src = cms.InputTag("genParticles"),
    cut = cms.string('((abs(pdgId) == 5) || (abs(pdgId) > 500  && abs(pdgId) < 600) || (abs(pdgId) > 5000  && abs(pdgId) < 6000)) && pt > 100'),
     stableOnly = cms.bool(False)
)
process.cParticlesFilter  = cms.EDFilter("GenParticleSelector",
     filter = cms.bool(False),
     src = cms.InputTag("genParticles"),
    cut = cms.string('(abs(pdgId) == 4) || (abs(pdgId) > 400  && abs(pdgId) < 500) || (abs(pdgId) > 4000  && abs(pdgId) < 5000)'),
     stableOnly = cms.bool(False)
)

process.bTracksProducer = cms.EDProducer("BTracksProducer",
    trackConfig = cms.PSet(process.trackClassifier),
    simG4 = cms.InputTag("g4SimHits"),
    trackingTruth = cms.untracked.InputTag("mergedtruth","MergedTrackTruth"),
    trackInputTag = cms.untracked.InputTag("generalTracks"),
    allSim = cms.untracked.bool(False)
)
#process.bTracksProducer.trackConfig.trackProducer = cms.untracked.InputTag("generalTracks")
process.bTracksProducer.trackConfig.enableSimToReco = cms.untracked.bool(True)
#process.bTracksProducer.trackConfig.bestMatchByMaxValue = cms.untracked.bool(True)
#process.bTracksProducer.trackingTruth.signalOnlyTP  = cms.untracked.bool(True)
process.load("RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff")

process.btagNonBFilter = cms.EDFilter("BTagAndFlavourFilter",
     minDiscr = cms.double(0.90),
     minPt = cms.double(30),
     jetMCSrc =  cms.InputTag('AK5byValAlgo'),
     jetTag  = cms.InputTag("combinedInclusiveSecondaryVertexBJetTags"),
     isSignal = cms.bool(False)
)
process.vertexSplitter = cms.EDProducer("VertexTrackSplitter",
vertexInputTag = cms.InputTag("inclusiveMergedVertices"),
max = cms.untracked.uint32(5)
)
process.bTracksProducerSV0 = cms.EDProducer("BTracksProducer",
    trackConfig = cms.PSet(process.trackClassifier),
    simG4 = cms.InputTag("g4SimHits"),
    trackingTruth = cms.untracked.InputTag("mergedtruth","MergedTrackTruth"),
    trackInputTag = cms.untracked.InputTag("vertexSplitter","vertex0"),
    allSim = cms.untracked.bool(True)
)
process.bTracksProducerSV0.trackConfig.enableSimToReco = cms.untracked.bool(True)
process.bTracksProducerSV0.trackConfig.trackProducer = cms.untracked.InputTag("generalTracks")
process.bTracksProducerSV1 = process.bTracksProducerSV0.clone(trackInputTag = cms.untracked.InputTag("vertexSplitter","vertex1"))
process.bTracksProducerSV2 = process.bTracksProducerSV0.clone(trackInputTag = cms.untracked.InputTag("vertexSplitter","vertex2"))
process.bTracksProducerSV3 = process.bTracksProducerSV0.clone(trackInputTag = cms.untracked.InputTag("vertexSplitter","vertex3"))
process.bTracksProducerSV4 = process.bTracksProducerSV0.clone(trackInputTag = cms.untracked.InputTag("vertexSplitter","vertex4"))

process.bTracksProducerSV0.trackConfig.trackProducer = cms.untracked.InputTag("vertexSplitter","vertex0")
process.bTracksProducerSV1.trackConfig.trackProducer = cms.untracked.InputTag("vertexSplitter","vertex1")
process.bTracksProducerSV2.trackConfig.trackProducer = cms.untracked.InputTag("vertexSplitter","vertex2")
process.bTracksProducerSV3.trackConfig.trackProducer = cms.untracked.InputTag("vertexSplitter","vertex3")
process.bTracksProducerSV4.trackConfig.trackProducer = cms.untracked.InputTag("vertexSplitter","vertex4")


#process.p = cms.Path(process.allBParticles * process.bParticlesFilter* process.cParticlesFilter* process.bTracksProducer * process.myPartons* process.AK5Flavour * process.btagging * process.inclusiveVertexing * process.inclusiveSecondaryVertexFinderTagInfos * process.combinedInclusiveSecondaryVertexBJetTags)
#process.p = cms.Path(process.allBParticles * process.bParticlesFilter * process.cParticlesFilter * process.myPartons* process.AK5Flavour * process.btagging * process.inclusiveVertexing * process.inclusiveSecondaryVertexFinderTagInfos * process.combinedInclusiveSecondaryVertexBJetTags *  process.bTracksProducer * process.vertexSplitter * process.bTracksProducerSV0 * process.bTracksProducerSV1 * process.bTracksProducerSV2 * process.bTracksProducerSV3 * process.bTracksProducerSV4 )
process.pfake = cms.Path(process.allBParticles *  process.cParticlesFilter * process.myPartons* process.AK5Flavour * process.btagging * process.inclusiveVertexing * process.inclusiveSecondaryVertexFinderTagInfos * process.combinedInclusiveSecondaryVertexBJetTags  *  process.bTracksProducer * process.vertexSplitter * process.bTracksProducerSV0 * process.bTracksProducerSV1 * process.bTracksProducerSV2 * process.bTracksProducerSV3 * process.bTracksProducerSV4 )

process.PoolSource.fileNames = [
"file:trk.root"
]
aa=[
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/00020B8A-DCF3-E111-B63E-00266CF33340.root",
#"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/0003943E-DFF3-E111-A054-0025904B1446.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/0055C993-E4F3-E111-9674-003048F0E184.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/00EBF575-E0F3-E111-A6C7-0030487D5E9D.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/0212A580-DFF3-E111-8389-003048D4DFA6.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/021AA457-E3F3-E111-930E-0025901D4C46.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/02730427-E2F3-E111-A21E-00266CFFA2B8.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/02CF08B4-E0F3-E111-9AF4-00266CF2AE10.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/02E06584-E3F3-E111-8BFF-002481E0D714.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/046C5A9C-E4F3-E111-B2AC-00266CFFA23C.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/0471AE09-E2F3-E111-99D3-00266CFF9FFC.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/047F17FC-DCF3-E111-9E4E-0025901D4D54.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/048739FC-E4F3-E111-B79C-0025901D4C3C.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/04E7C172-E0F3-E111-B302-0025904B12E0.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/04F6AA7F-E0F3-E111-A45F-002481E94BFE.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/FEF5C099-DEF3-E111-AFE3-00266CF32E70.root",
#"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/1E903A12-E7F3-E111-8EC4-003048D43958.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/E680E1B5-E3F3-E111-BC5C-00215AD4D670.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/1E9A16ED-DFF3-E111-AB70-0030487F933D.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/1CC6AAD3-E0F3-E111-89A5-0030487E4B8D.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/E63A622D-40F4-E111-94BA-0025904B578E.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/1CC703BF-E1F3-E111-BFA0-002481E0D144.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/E618D7B3-E4F3-E111-86C6-003048C693D6.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/1CE1A35A-E1F3-E111-B695-0025901D4090.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/E4F0D3E5-DFF3-E111-ACB5-003048C69310.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/1E8C90AF-DEF3-E111-9633-00266CFFA2D0.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/E4DD182D-DFF3-E111-ADF4-0025901D4C3C.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/1EB621B8-DEF3-E111-B521-003048C69310.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/1EFCFA15-E4F3-E111-8D9A-0030487E5179.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/1EBB8017-DFF3-E111-93C7-0025901D4C18.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/200C3B02-E2F3-E111-A0C5-0025904B1446.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/200E4D0A-DFF3-E111-A932-0025904B5FBA.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/201D1ED7-40F4-E111-AB62-003048D4DF90.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/20281D46-E1F3-E111-99A9-0025904B12B2.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/1ED84632-E1F3-E111-A1BB-003048C68A9E.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/2061FF3B-E4F3-E111-8F86-003048F0E18E.root",
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/1EEC3299-E3F3-E111-AF4B-00266CF32A20.root"
#file:/networkdata/arizzi/QCD470600/reco/001793DE-DB07-E211-B4CE-003048679162.root'
]

process.outfake = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('btag004-fakefromttbar.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring("pfake"))
)    
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('btag003.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring("p"))
)
process.endpath= cms.EndPath(process.outfake)


process.load('CondCore.DBCommon.CondDBSetup_cfi')
process.BTauMVAJetTagComputerRecord = cms.ESSource('PoolDBESSource',
    process.CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('BTauGenericMVAJetTagComputerRcd'),
        tag = cms.string('MVAComputerContainer_Retrained53X_JetTags_v2')
    )),
    connect = cms.string('frontier://FrontierProd/CMS_COND_PAT_000'),
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
)
process.es_prefer_BTauMVAJetTagComputerRecord = cms.ESPrefer('PoolDBESSource','BTauMVAJetTagComputerRecord')


