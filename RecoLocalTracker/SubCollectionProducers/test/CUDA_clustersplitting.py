# The following comments couldn't be translated into the new config version:

#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("rereco2")

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")




# load the full reconstraction configuration, to make sure we're getting all needed dependencies
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

#parallel processing

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(400)
#    input = cms.untracked.int32(23)
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
#    skipEvents =  cms.untracked.uint32(58),
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
#process.impactParameterTagInfos.extSVCollection = cms.InputTag("inclusiveMergedVertices")
#process.impactParameterTagInfos.selectTracksFromExternalSV = cms.bool(True)
#process.cleanedImpactParameterTagInfos.extSVCollection = cms.InputTag("inclusiveMergedVertices2")
#process.cleanedImpactParameterTagInfos.selectTracksFromExternalSV = cms.bool(True)

#process.inclusiveSecondaryVertexFinderTagInfos.vertexCuts.distVal2dMax = 8
#process.cleanedInclusiveSecondaryVertexFinderTagInfos.vertexCuts.distVal2dMax = 8

process.compareOldSplit = cms.EDProducer("CompareWithIdealClustering",
    pixelClusters         = cms.InputTag("siPixelClusters","","rereco"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
)
process.compareRECO = cms.EDProducer("CompareWithIdealClustering",
    pixelClusters         = cms.InputTag("siPixelClusters","","RECO"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
)

process.compare1 = cms.EDProducer("CompareWithIdealClustering",
    pixelClusters         = cms.InputTag("siPixelClusters1"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
)
process.compare2 = cms.EDProducer("CompareWithIdealClustering",
    pixelClusters         = cms.InputTag("siPixelClusters2"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
)
process.compare3 = cms.EDProducer("CompareWithIdealClustering",
    pixelClusters         = cms.InputTag("siPixelClusters3"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
)
process.compare4 = cms.EDProducer("CompareWithIdealClustering",
    pixelClusters         = cms.InputTag("siPixelClusters4"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
)
process.compare5 = cms.EDProducer("CompareWithIdealClustering",
    pixelClusters         = cms.InputTag("siPixelClusters5"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
)
process.compare6 = cms.EDProducer("CompareWithIdealClustering",
    pixelClusters         = cms.InputTag("siPixelClusters6"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
)
process.compare7 = cms.EDProducer("CompareWithIdealClustering",
    pixelClusters         = cms.InputTag("siPixelClusters7"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
)
process.compare8 = cms.EDProducer("CompareWithIdealClustering",
    pixelClusters         = cms.InputTag("siPixelClusters8"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
)
process.compare9 = cms.EDProducer("CompareWithIdealClustering",
    pixelClusters         = cms.InputTag("siPixelClusters9"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
)
process.compare10 = cms.EDProducer("CompareWithIdealClustering",
    pixelClusters         = cms.InputTag("siPixelClusters10"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
)
process.compare = cms.EDProducer("CompareWithIdealClustering",
    pixelClusters         = cms.InputTag("siPixelClusters"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
    )

process.siPixelClusters = cms.EDProducer("JetCoreClusterSplitter2",
    pixelClusters         = cms.InputTag("siPixelClusters","","RECO"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
    )
process.siPixelClusters1 = cms.EDProducer("JetCoreClusterSplitter2",
    pixelClusters         = cms.InputTag("siPixelClusters","","RECO"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
    param1 = cms.double( 1.8 ),
    param2 = cms.double( 0.2 ),
    )
process.siPixelClusters2 = cms.EDProducer("JetCoreClusterSplitter2",
    pixelClusters         = cms.InputTag("siPixelClusters","","RECO"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
    param1 = cms.double( 1.8 ),
    param2 = cms.double( 0.3 ),
    )
process.siPixelClusters3 = cms.EDProducer("JetCoreClusterSplitter2",
    pixelClusters         = cms.InputTag("siPixelClusters","","RECO"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
    param1 = cms.double( 1.8 ),
    param2 = cms.double( 0.5 ),
    )
process.siPixelClusters4 = cms.EDProducer("JetCoreClusterSplitter2",
    pixelClusters         = cms.InputTag("siPixelClusters","","RECO"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
    param1 = cms.double( 1.8 ),
    param2 = cms.double( 0.7 ),
    )
process.siPixelClusters5 = cms.EDProducer("JetCoreClusterSplitter2",
    pixelClusters         = cms.InputTag("siPixelClusters","","RECO"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
    param1 = cms.double( 1.8 ),
    param2 = cms.double( 0.9 ),
    )
process.siPixelClusters6 = cms.EDProducer("JetCoreClusterSplitter2",
    pixelClusters         = cms.InputTag("siPixelClusters","","RECO"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
    param1 = cms.double( 1.8 ),
    param2 = cms.double( 0.5 ),
    )
process.siPixelClusters7 = cms.EDProducer("JetCoreClusterSplitter2",
    pixelClusters         = cms.InputTag("siPixelClusters","","RECO"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
    param1 = cms.double( 1.75 ),
    param2 = cms.double( 0.5 ),
    )
process.siPixelClusters8 = cms.EDProducer("JetCoreClusterSplitter2",
    pixelClusters         = cms.InputTag("siPixelClusters","","RECO"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
    param1 = cms.double( 1.7 ),
    param2 = cms.double( 0.5 ),
    )
process.siPixelClusters9 = cms.EDProducer("JetCoreClusterSplitter2",
    pixelClusters         = cms.InputTag("siPixelClusters","","RECO"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
    param1 = cms.double( 1.85 ),
    param2 = cms.double( 0.5 ),
    )
process.siPixelClusters10 = cms.EDProducer("JetCoreClusterSplitter2",
    pixelClusters         = cms.InputTag("siPixelClusters","","RECO"),
    vertices              = cms.InputTag('offlinePrimaryVertices',"","RECO"),
    pixelCPE = cms.string( "PixelCPEGeneric" ),
    param1 = cms.double( 1.9 ),
    param2 = cms.double( 0.5 ),
    )

process.IdealsiPixelClusters = cms.EDProducer(
    "TrackClusterSplitter",
    stripClusters         = cms.InputTag("siStripClusters"),
    pixelClusters         = cms.InputTag("siPixelClusters","","RECO"),
    useTrajectories       = cms.bool(False),
    trajTrackAssociations = cms.InputTag('generalTracks'),
    tracks                = cms.InputTag('pixelTracks'),
    propagator            = cms.string('AnalyticalPropagator'),
    vertices              = cms.InputTag('pixelVertices'),
    simSplitPixel         = cms.bool(True), # ideal pixel splitting turned OFF
    simSplitStrip         = cms.bool(False), # ideal strip splitting turned OFF
    tmpSplitPixel         = cms.bool(False), # template pixel spliting
    tmpSplitStrip         = cms.bool(False), # template strip splitting
    useStraightTracks     = cms.bool(True),
    test     = cms.bool(True),
#    SkipEvent             = cms.untracked.vstring('ProductNotFound'),
    )


process.GroupedCkfTrajectoryBuilder.maxCand=50
process.GroupedCkfTrajectoryBuilderP5.maxCand=50
#process.convCkfTrajectoryBuilder.maxCand=25
#process.detachedTripletStepTrajectoryBuilder.maxCand=25
process.initialStepTrajectoryBuilder.maxCand=50
#process.lowPtTripletStepTrajectoryBuilder.maxCand=25
process.mixedTripletStepTrajectoryBuilder.maxCand=50
process.pixelLessStepTrajectoryBuilder.maxCand=50
process.tobTecStepTrajectoryBuilder.maxCand=400
#process.jetCoreRegionalStepTrajectoryBuilder.maxCand=200 #default is 50


#redo tracking + nominal btagging (with IVF used in IP TagInfo too) + NI-cleaned btagging
#process.reco = cms.Sequence(process.siPixelClusters+process.siPixelRecHits+process.siStripMatchedRecHits+process.pixelTracks+process.ckftracks_wodEdX+process.offlinePrimaryVertices+process.ak5JetTracksAssociatorAtVertex+process.inclusiveVertexing+process.btagging  * process.inclusiveSecondaryVertexFinderTagInfos * process.combinedInclusiveSecondaryVertexBJetTags * process.nuclearInteractionIdentifier * process.cleanedInclusiveMergedVertices * process.trackCollectionCleaner * process.offlinePrimaryVertices2 * process.inclusiveVertexing2 * process.ak5JetCleanedTracksAssociatorAtVertex * process.cleanedImpactParameterTagInfos * process.cleanedInclusiveSecondaryVertexFinderTagInfos * process.cleanedCombinedInclusiveSecondaryVertexBJetTags)

#do IVF and btag on OLD reco
#process.reco = cms.Path(process.IdealsiPixelClusters + process.siPixelClusters + process.compareRECO + process.compareOldSplit + process.compare + process.siPixelClusters2 + process.compare2)
process.reco = cms.Path(process.IdealsiPixelClusters 
#+ process.siPixelClusters + process.compare
+ process.siPixelClusters1 + process.compare1
#+ process.siPixelClusters2 + process.compare2
#+ process.siPixelClusters3 + process.compare3
#+ process.siPixelClusters4 + process.compare4
#+ process.siPixelClusters5 + process.compare5
#+ process.siPixelClusters6 + process.compare6
#+ process.siPixelClusters7 + process.compare7
#+ process.siPixelClusters8 + process.compare8
#+ process.siPixelClusters9 + process.compare9
#+ process.siPixelClusters10 + process.compare10
)


process.PoolSource.fileNames =[
#"file:skimmed30ev.root",
"file:/afs/cern.ch/work/f/fpantale/CMSSW/voltabuona_RAW2DIGI_RECO.root",
]
