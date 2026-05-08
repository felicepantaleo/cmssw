import FWCore.ParameterSet.Config as cms

truthLogicalGraphDumper = cms.EDAnalyzer(
    "TruthLogicalGraphDumper",
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("truthGraphProducer"),
    dotFile=cms.string("truthlogicalgraph.dot"),
    maxParticles=cms.uint32(5000),
    maxVertices=cms.uint32(5000),
    maxEdgesPerNode=cms.uint32(200),
)