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

    # The dumper can append run/lumi/event to this base name if the code was updated.
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

    # 0 means: keep the full truth graph.
    # Example: 13 keeps only muons with pdgId == 13 and their descendants.
    motherPdgId=cms.int32(0),

    # This is harmless when there are no GEN-SIM particle associations.
    # It will only do something once robust GEN-SIM links exist.
    mergeGenSimVertices=cms.bool(True),

    # Collapse GEN-only chains P -> V -> C where P has status != 1,
    # C is the only daughter, and P and C have the same PDG id.
    collapseIntermediateGenParticles=cms.bool(True),
)

process.truthLogicalGraphDumper = cms.EDAnalyzer(
    "TruthLogicalGraphDumper",
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("truthGraphProducer"),

    # The dumper can append run/lumi/event to this base name if the code was updated.
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
    + process.truthLogicalGraphDumper
)