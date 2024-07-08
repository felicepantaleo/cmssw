import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process("TICLGeomAnalyze", Phase2C17I13M9)


process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False),
    numberOfThreads = cms.untracked.uint32(1),
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.Geometry.GeometryExtended2026D98Reco_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("Geometry.HGCalGeometry.TICLGeom_cff")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:step3.root'
    )
)

# Analyzers
process.TICLGeomAnalyzerECAL = cms.EDAnalyzer("TICLGeomAnalyzer",
    label = cms.string("ECAL")
)

process.TICLGeomAnalyzerHCAL = cms.EDAnalyzer("TICLGeomAnalyzer",
    label = cms.string("HCAL")
)

process.TICLGeomAnalyzerHGCal = cms.EDAnalyzer("TICLGeomAnalyzer",
    label = cms.string("HGCal")
)

process.TICLGeomAnalyzerHFNose = cms.EDAnalyzer("TICLGeomAnalyzer",
    label = cms.string("HFNose")
)

# Tasks and Paths
ticlGeometryTask = cms.Task(
    process.ticlGeomESProducerECAL,
    process.ticlGeomESProducerHCAL,
    process.ticlGeomESProducerHGCal,
    process.ticlGeomESProducerHFNose
)

process.ticlGeometry = cms.Path(ticlGeometryTask)

process.TICLGeomAnalyzePath = cms.Path(
    process.TICLGeomAnalyzerECAL +
    process.TICLGeomAnalyzerHCAL +
    process.TICLGeomAnalyzerHGCal +
    process.TICLGeomAnalyzerHFNose
)

process.schedule = cms.Schedule(process.ticlGeometry, process.TICLGeomAnalyzePath)

