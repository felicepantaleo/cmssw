"""Run the RecHit time harvester on a saved Run 1 histogram file."""
import os
import ROOT
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

options = VarParsing("analysis")
options.register("inputFile", "DQM_V0001_HGCAL_R000000001.root",
                 VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Saved Run 1 DQM histogram file (local path)")
options.parseArguments()
input_file = os.path.abspath(options.inputFile.removeprefix("file:"))

# Check the actual input before running the framework: booked plots can be empty.
source = ROOT.TFile.Open(input_file, "READ")
if not source or source.IsZombie():
    raise RuntimeError(f"Cannot open {input_file}")
found = 0
for endcap in ("Minus", "Plus"):
    path = f"DQMData/Run 1/HGCAL/Run summary/EndCap_{endcap}/rechittimevscrateProfile"
    profile = source.Get(path)
    if profile and profile.InheritsFrom("TProfile"):
        found += 1
        print(f"{path}: {profile.GetEntries():g} entries", flush=True)
    else:
        print(f"Missing: {path}", flush=True)
source.Close()
if not found:
    raise RuntimeError("No RecHit crate profiles found. Check whether RecHits were produced and monitored.")

process = cms.Process("HARVESTING")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.DQMStore = cms.Service("DQMStore")
process.source = cms.Source("EmptySource", firstRun=cms.untracked.uint32(1))
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(1))
process.options = cms.untracked.PSet(
    numberOfThreads=cms.untracked.uint32(1),
    numberOfStreams=cms.untracked.uint32(1),
)
process.fit = DQMEDHarvester(
    "HGCalRecHitTimeHarvester", inputFile=cms.string(input_file)
)
process.fitPath = cms.Path(process.fit)
process.load("DQMServices.Components.DQMFileSaver_cfi")
process.dqmSaver.workflow = "/HGCal/RecHitTime/Harvesting"
process.dqmSaver.saveByRun = -1
process.dqmSaver.saveAtJobEnd = True
process.dqmSaver.forceRunNumber = 1
process.savePath = cms.EndPath(process.dqmSaver)
process.schedule = cms.Schedule(process.fitPath, process.savePath)

