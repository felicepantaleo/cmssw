import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process("TICLGeomAnalyze", Phase2C17I13M9)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.Geometry.GeometryExtended2026D98Reco_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:step3.root'
    )
)

from Geometry.HGCalGeometry.TICLGeomESProducer_cfi import TICLGeomESProducer
process.TICLGeomESProducer = TICLGeomESProducer.clone()

process.TICLGeomAnalyzer = cms.EDAnalyzer("TICLGeomAnalyzer",
    label = cms.string("all")
)

process.TICLGeomESTask = cms.Task(process.TICLGeomESProducer)

process.TICLGeomAnalyzePath = cms.Path(process.TICLGeomAnalyzer, process.TICLGeomESTask)
process.schedule = cms.Schedule(process.TICLGeomAnalyzePath)
