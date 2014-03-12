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


#redo tracking + nominal btagging (with IVF used in IP TagInfo too) + NI-cleaned btagging
process.reco = cms.Sequence(process.siPixelRecHits+process.siStripMatchedRecHits+process.pixelTracks+process.ckftracks_wodEdX+process.offlinePrimaryVertices+process.ak5JetTracksAssociatorAtVertex+process.inclusiveVertexing+process.btagging  * process.inclusiveSecondaryVertexFinderTagInfos * process.combinedInclusiveSecondaryVertexBJetTags * process.nuclearInteractionIdentifier * process.cleanedInclusiveMergedVertices * process.trackCollectionCleaner * process.offlinePrimaryVertices2 * process.inclusiveVertexing2 * process.ak5JetCleanedTracksAssociatorAtVertex * process.cleanedImpactParameterTagInfos * process.cleanedInclusiveSecondaryVertexFinderTagInfos * process.cleanedCombinedInclusiveSecondaryVertexBJetTags)

process.p = cms.Path(process.reco)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('trk.root'),
)
process.endpath= cms.EndPath(process.out)


process.PoolSource.fileNames = [
"file:/networkdata/arizzi/RECOEVENTSQCD/ECCD6993-DA07-E211-B098-003048678B3C.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/EEB8D157-DF07-E211-9F23-0030486790BE.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/EECF8431-DC07-E211-B6FF-002618B27F8A.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/EEF33C2B-CF07-E211-A0F5-00261894398C.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/F0173612-D807-E211-A32F-00304867BEDE.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/F034D8C2-EA07-E211-BE25-001BFCDBD190.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/F07E20FF-0708-E211-A642-001A928116D0.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/F081CF5C-D507-E211-B8B0-003048678A7E.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/F0FE8534-DE07-E211-924B-0026189438D8.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/F224C59A-DE07-E211-B858-0018F3D095F6.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/F2368E08-3608-E211-A1B6-00261894393A.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/F4045994-D307-E211-8C6C-0030486790B8.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/F43C4D0F-D207-E211-A95E-00261894386E.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/F47515E7-EE07-E211-A0D7-0018F3D0963C.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/F4948930-3A08-E211-9E69-002618943904.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/F4AA56D8-FA07-E211-9C45-00261894391B.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/F4FBFD2F-D807-E211-9D2F-002618943845.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/F65A8F31-D507-E211-B44E-003048679000.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/F897599F-D107-E211-99EA-002618943910.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/F8B7C0B3-D607-E211-8EF7-002618943939.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/F8E05C25-DC07-E211-B502-002618943842.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/FA222B1F-DF07-E211-BBB0-001A928116E8.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/FA6523FD-D407-E211-884A-002618943858.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/FA96E3C1-4108-E211-9DFA-003048FFCB96.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/FAFA2B74-D907-E211-956A-002618FDA287.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/FC089373-D107-E211-9124-00261894390E.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/FC3D42FF-D907-E211-A413-002618943944.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/FC7C83EC-D507-E211-BC80-002618943885.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/FC86EA4D-D107-E211-9FDD-002618943963.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/FCF09F81-D307-E211-A029-00304867918A.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/FEC2E98A-D407-E211-90FC-002618FDA263.root",
"file:/networkdata/arizzi/RECOEVENTSQCD/FEE29090-D007-E211-A159-003048678C3A.root"
]



