# The following comments couldn't be translated into the new config version:

#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("validation")
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


process.bTagValidation.tagConfig.append(cms.PSet(
            parameters = cms.PSet(
        			discriminatorStart = cms.double(-0.05),
        			discriminatorEnd = cms.double(1.05),
        			nBinEffPur = cms.int32(200),
        			# the constant b-efficiency for the differential plots versus pt and eta
        			effBConst = cms.double(0.5),
        			endEffPur = cms.double(1.005),
        			startEffPur = cms.double(-0.005)
  				),
            label = cms.InputTag("combinedInclusiveSecondaryVertexBJetTags"),
            folder = cms.string("CSVIVF")
)
)

process.CustombTagValidation = process.bTagValidation.clone(
    tagConfig = cms.VPSet(
        cms.PSet(
				    parameters = cms.PSet(
        			discriminatorStart = cms.double(-0.05),
        			discriminatorEnd = cms.double(1.05),
        			nBinEffPur = cms.int32(200),
        			# the constant b-efficiency for the differential plots versus pt and eta
        			effBConst = cms.double(0.5),
        			endEffPur = cms.double(1.005),
        			startEffPur = cms.double(-0.005)
    				),
            label = cms.InputTag("combinedSecondaryVertexBJetTags"),
            folder = cms.string("CSV")
        ),  
        cms.PSet(
            parameters = cms.PSet(
                                discriminatorStart = cms.double(-0.05),
                                discriminatorEnd = cms.double(1.05),
                                nBinEffPur = cms.int32(200),
                                # the constant b-efficiency for the differential plots versus pt and eta
                                effBConst = cms.double(0.5),
                                endEffPur = cms.double(1.005),
                                startEffPur = cms.double(-0.005)
                                ),
            label = cms.InputTag("combinedInclusiveSecondaryVertexBJetTags"),
            folder = cms.string("CSVIVF")
           ),  
        cms.PSet(
            parameters = cms.PSet(
                                discriminatorStart = cms.double(-0.05),
                                discriminatorEnd = cms.double(1.05),
                                nBinEffPur = cms.int32(200),
                                # the constant b-efficiency for the differential plots versus pt and eta
                                effBConst = cms.double(0.5),
                                endEffPur = cms.double(1.005),
                                startEffPur = cms.double(-0.005)
                                ),
            label = cms.InputTag("cleanedCombinedInclusiveSecondaryVertexBJetTags"),
            folder = cms.string("CSVIVF-NI")
           ),
        cms.PSet(
                                    parameters = cms.PSet(
                                discriminatorStart = cms.double(-0.05),
                                discriminatorEnd = cms.double(1.05),
                                nBinEffPur = cms.int32(200),
                                # the constant b-efficiency for the differential plots versus pt and eta
                                effBConst = cms.double(0.5),
                                endEffPur = cms.double(1.005),
                                startEffPur = cms.double(-0.005)
                                ),
            label = cms.InputTag("combinedSecondaryVertexBJetTags","","RECO"),
            folder = cms.string("CSV09")
        ),


       )


)
process.CustombTagValidation.ptRecJetMin = cms.double(20.0)
process.CustombTagValidation.ptRanges = cms.vdouble(20, 30, 50.0, 80.0, 120.0,300, 600)

process.load("RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff")



process.nuclearInteractionIdentifier = cms.EDProducer("NuclearInteractionIdentifier",
     primaryVertices = cms.InputTag("offlinePrimaryVertices"),
     secondaryVertices = cms.InputTag("inclusiveMergedVertices"),
     beamSpot = cms.InputTag("offlineBeamSpot")
)

process.cleanedInclusiveMergedVertices = cms.EDProducer("VertexCleaner",
        primaryVertices= cms.InputTag("nuclearInteractionIdentifier"),
        secondaryVertices = cms.InputTag("inclusiveMergedVertices"),
        maxFraction = cms.double(0.0)
)

process.trackCollectionCleaner = cms.EDProducer("TrackCollectionCleaner",
        vertices= cms.InputTag("nuclearInteractionIdentifier"),
        tracks = cms.InputTag("generalTracks")
)
process.ak5JetCleanedTracksAssociatorAtVertex = process.ak5JetTracksAssociatorAtVertex.clone()
process.ak5JetCleanedTracksAssociatorAtVertex.tracks = cms.InputTag("trackCollectionCleaner")


#Redo everyting after NI cleaning 
process.inclusiveVertexFinder2 = process.inclusiveVertexFinder.clone(tracks = cms.InputTag("trackCollectionCleaner"))
process.vertexMerger2 = process.vertexMerger.clone(secondaryVertices = cms.InputTag("inclusiveVertexFinder2"))
process.trackVertexArbitrator2=process.trackVertexArbitrator.clone(tracks = cms.InputTag("trackCollectionCleaner"),secondaryVertices = cms.InputTag("vertexMerger2"))
process.inclusiveMergedVertices2= process.inclusiveMergedVertices.clone(secondaryVertices = cms.InputTag("trackVertexArbitrator2"))

process.inclusiveVertexing2 = cms.Sequence(process.inclusiveVertexFinder2*process.vertexMerger2*process.trackVertexArbitrator2*process.inclusiveMergedVertices2)

process.offlinePrimaryVertices2 = process.offlinePrimaryVertices.clone(TrackLabel=cms.InputTag("trackCollectionCleaner"))
process.inclusiveVertexFinder2.primaryVertices = cms.InputTag("offlinePrimaryVertices2")
process.trackVertexArbitrator2.primaryVertices = cms.InputTag("offlinePrimaryVertices2")

process.cleanedImpactParameterTagInfos = process.impactParameterTagInfos.clone()
process.cleanedImpactParameterTagInfos.jetTracks = cms.InputTag("ak5JetCleanedTracksAssociatorAtVertex")
process.cleanedImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices2")


process.cleanedInclusiveSecondaryVertexFinderTagInfos = process.inclusiveSecondaryVertexFinderTagInfos.clone(
        extSVCollection = cms.InputTag("inclusiveMergedVertices2"),
        trackIPTagInfos = cms.InputTag("cleanedImpactParameterTagInfos")
)
process.cleanedCombinedInclusiveSecondaryVertexBJetTags = process.combinedInclusiveSecondaryVertexBJetTags.clone(
        tagInfos = cms.VInputTag(cms.InputTag("cleanedImpactParameterTagInfos"),
                                 cms.InputTag("cleanedInclusiveSecondaryVertexFinderTagInfos"))
)



#feed IVF vertices to IPTagInfo in order to let IVF tracks be selected 
process.impactParameterTagInfos.extSVCollection = cms.InputTag("inclusiveMergedVertices")
process.impactParameterTagInfos.selectTracksFromExternalSV = cms.bool(False)
process.cleanedImpactParameterTagInfos.extSVCollection = cms.InputTag("inclusiveMergedVertices2")
process.cleanedImpactParameterTagInfos.selectTracksFromExternalSV = cms.bool(False)

process.inclusiveSecondaryVertexFinderTagInfos.vertexCuts.distVal2dMax = 2.5
process.cleanedInclusiveSecondaryVertexFinderTagInfos.vertexCuts.distVal2dMax = 2.5

process.siPixelClusters = cms.EDProducer("JetCoreClusterSplitter",
    pixelClusters         = cms.InputTag("siPixelClusters"),
    vertices              = cms.InputTag('offlinePrimaryVertices'),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
 
    )

process.GroupedCkfTrajectoryBuilder.maxCand=25
process.GroupedCkfTrajectoryBuilderP5.maxCand=25
#process.convCkfTrajectoryBuilder.maxCand=25
#process.detachedTripletStepTrajectoryBuilder.maxCand=25
process.initialStepTrajectoryBuilder.maxCand=25
#process.lowPtTripletStepTrajectoryBuilder.maxCand=25
process.mixedTripletStepTrajectoryBuilder.maxCand=25
process.pixelLessStepTrajectoryBuilder.maxCand=25
process.tobTecStepTrajectoryBuilder.maxCand=200

#redo tracking + nominal btagging (with IVF used in IP TagInfo too) + NI-cleaned btagging
process.reco = cms.Sequence(process.siPixelClusters+process.siPixelRecHits+process.siStripMatchedRecHits+process.pixelTracks+process.ckftracks_wodEdX+process.offlinePrimaryVertices+process.ak5JetTracksAssociatorAtVertex+process.inclusiveVertexing+process.btagging  * process.inclusiveSecondaryVertexFinderTagInfos * process.combinedInclusiveSecondaryVertexBJetTags * process.nuclearInteractionIdentifier * process.cleanedInclusiveMergedVertices * process.trackCollectionCleaner * process.offlinePrimaryVertices2 * process.inclusiveVertexing2 * process.ak5JetCleanedTracksAssociatorAtVertex * process.cleanedImpactParameterTagInfos * process.cleanedInclusiveSecondaryVertexFinderTagInfos * process.cleanedCombinedInclusiveSecondaryVertexBJetTags)

#DQM/Validation stuff
process.plots = cms.Sequence(process.myPartons* process.AK5Flavour * process.CustombTagValidation * process.dqmSaver)
process.dqmEnv.subSystemFolder = 'BTAG'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = '/POG/BTAG/BJET470600splitClusterNONEWDEVEL'
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd =cms.untracked.bool(True) 
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)

process.p = cms.Path(process.reco*process.plots)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('trk.root'),
)
process.endpath= cms.EndPath(process.out)


process.PoolSource.fileNames = [
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
"file:/networkdata/cvernier/validation_533/CMSSW_5_3_3_patch2/src/Validation/RecoB/FEF5C099-DEF3-E111-AFE3-00266CF32E70.root"
]



