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
           )

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


process.inclusiveVertexFinder2 = process.inclusiveVertexFinder.clone(tracks = cms.InputTag("trackCollectionCleaner"))
process.vertexMerger2 = process.vertexMerger.clone(secondaryVertices = cms.InputTag("inclusiveVertexFinder2"))
process.trackVertexArbitrator2=process.trackVertexArbitrator.clone(tracks = cms.InputTag("trackCollectionCleaner"),secondaryVertices = cms.InputTag("vertexMerger2"))
process.inclusiveMergedVertices2= process.inclusiveMergedVertices.clone(secondaryVertices = cms.InputTag("trackVertexArbitrator2"))

process.inclusiveVertexing2 = cms.Sequence(process.inclusiveVertexFinder2*process.vertexMerger2*process.trackVertexArbitrator2*process.inclusiveMergedVertices2)

#new
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




#process.plots = cms.Path(process.nuclearInteractionIdentifier * process.cleanedInclusiveMergedVertices * process.trackCollectionCleaner * process.inclusiveVertexing2 * process.ak5JetCleanedTracksAssociatorAtVertex * process.cleanedImpactParameterTagInfos * process.cleanedInclusiveSecondaryVertexFinderTagInfos * process.cleanedCombinedInclusiveSecondaryVertexBJetTags * process.cleaned1InclusiveSecondaryVertexFinderTagInfos * process.cleaned1CombinedInclusiveSecondaryVertexBJetTags)

process.plots = cms.Path(process.myPartons* process.AK5Flavour * process.btagging  * process.inclusiveVertexing * process.inclusiveSecondaryVertexFinderTagInfos * process.combinedInclusiveSecondaryVertexBJetTags * process.nuclearInteractionIdentifier * process.cleanedInclusiveMergedVertices * process.trackCollectionCleaner * process.offlinePrimaryVertices2 * process.inclusiveVertexing2 * process.ak5JetCleanedTracksAssociatorAtVertex * process.cleanedImpactParameterTagInfos * process.cleanedInclusiveSecondaryVertexFinderTagInfos * process.cleanedCombinedInclusiveSecondaryVertexBJetTags * process.CustombTagValidation * process.dqmSaver)

process.dqmEnv.subSystemFolder = 'BTAG'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = '/POG/BTAG/BJETALL'
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd =cms.untracked.bool(True) 
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
process.PoolSource.fileNames = [
'file:/networkdata/arizzi/QCD470600/00256BFE-F307-E211-A264-003048FFCB96.root',
'file:/networkdata/arizzi/QCD470600/00490DE0-DD07-E211-B197-001A928116B2.root',
'file:/networkdata/arizzi/QCD470600/00755239-F507-E211-A7BD-00261894393F.root'
]



