# The following comments couldn't be translated into the new config version:

#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("rereco")

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")


# load the full reconstraction configuration, to make sure we're getting all needed dependencies
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

#parallel processing
process.Tracer = cms.Service("Tracer")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.GlobalTag.globaltag = 'START53_V27::All'

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
process.impactParameterTagInfos.selectTracksFromExternalSV = cms.bool(True)
process.cleanedImpactParameterTagInfos.extSVCollection = cms.InputTag("inclusiveMergedVertices2")
process.cleanedImpactParameterTagInfos.selectTracksFromExternalSV = cms.bool(True)

process.inclusiveSecondaryVertexFinderTagInfos.vertexCuts.distVal2dMax = 8
process.cleanedInclusiveSecondaryVertexFinderTagInfos.vertexCuts.distVal2dMax = 8

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

process.p = cms.Path(process.reco)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('trk.root'),
)
process.endpath= cms.EndPath(process.out)


process.PoolSource.fileNames = [
"file:/data/arizzi/TomoGerrit/CMSSW_5_3_12_patch1/src/Validation/RecoB/test/qcdquick/btag004-unreconstructedFromttbar30-80angle.root"
]



