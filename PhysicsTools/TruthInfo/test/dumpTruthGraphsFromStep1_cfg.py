import FWCore.ParameterSet.Config as cms

process = cms.Process("TRUTHGRAPH")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(
        "file:step1.root"
    )
)

process.options = cms.untracked.PSet(
    wantSummary=cms.untracked.bool(True)
)

process.truthGraphProducer = cms.EDProducer(
    "TruthGraphProducer",
    genEventHepMC3=cms.InputTag("generatorSmeared"),
    genEventHepMC=cms.InputTag("generatorSmeared"),
    simTracks=cms.InputTag("g4SimHits"),
    simVertices=cms.InputTag("g4SimHits"),
    addGenToSimEdges=cms.bool(True),
)

process.truthGraphDumper = cms.EDAnalyzer(
    "TruthGraphDumper",
    src=cms.InputTag("truthGraphProducer"),
    dotFile=cms.string("truthgraph.dot"),
    maxNodes=cms.uint32(20000),
    maxEdgesPerNode=cms.uint32(50),
    simTracks=cms.InputTag("g4SimHits"),
    simVertices=cms.InputTag("g4SimHits"),
    genEventHepMC=cms.InputTag("generatorSmeared"),
    genEventHepMC3=cms.InputTag("generatorSmeared"),
)

process.truthLogicalGraphProducer = cms.EDProducer(
    "TruthLogicalGraphProducer",
    src=cms.InputTag("truthGraphProducer"),
    simTracks=cms.InputTag("g4SimHits"),
    simVertices=cms.InputTag("g4SimHits"),
    genEventHepMC3=cms.InputTag("generatorSmeared"),
    genEventHepMC=cms.InputTag("generatorSmeared"),
    motherPdgId=cms.int32(0),
    mergeGenSimVertices=cms.bool(True),
    collapseIntermediateGenParticles=cms.bool(True),
)

process.truthLogicalGraphHitIndexProducer = cms.EDProducer(
    "TruthLogicalGraphHitIndexProducer",
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("truthGraphProducer"),
    simTracks=cms.InputTag("g4SimHits"),

    simHitCollections=cms.VInputTag(
        cms.InputTag("g4SimHits", "HGCHitsEE"),
        cms.InputTag("g4SimHits", "HGCHitsHEfront"),
        cms.InputTag("g4SimHits", "HGCHitsHEback"),
    ),

    doHGCal=cms.bool(True),
    doHGCalRelabelling=cms.bool(False),
)

process.truthLogicalGraphDumper = cms.EDAnalyzer(
    "TruthLogicalGraphDumper",
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("truthGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),

    dotFile=cms.string("truthlogicalgraph.dot"),

    maxParticles=cms.uint32(20000),
    maxVertices=cms.uint32(20000),
    maxEdgesPerNode=cms.uint32(300),
)

process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.cerr.default = cms.untracked.PSet(
    limit=cms.untracked.int32(0)
)
process.MessageLogger.cerr.TruthGraphProducer = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)
)
process.MessageLogger.cerr.TruthLogicalGraphProducer = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)
)

process.truthGraph_step = cms.Path(
    process.truthGraphProducer
    + process.truthGraphDumper
    + process.truthLogicalGraphProducer
    + process.truthLogicalGraphHitIndexProducer
    + process.truthLogicalGraphDumper
)