# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

# Standalone driver for the dedicated two-channel TICLCandidate Branch validator
# (BranchTICLCandidateValidator). It rebuilds the truth-graph chain from a
# GEN-SIM-RECO file and runs the candidate validator, writing a DQMIO file (inspect
# under DQMData/Run 1/HGCAL/Run summary/BranchValidator/TICLCandidate).
#
# For a single-particle gun pass --pdgId: the branch reference is then restricted to
# the fired GEN species (onlyGenPrimaries=True), i.e. a clean antichain, so the
# candidate efficiency / PID / energy-response numbers are meaningful (against the
# full graph the reference degenerates - see truthGraphValidation_cff).

import FWCore.ParameterSet.Config as cms
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("inputFile", nargs='?', default="step3.root", metavar='FILE')
parser.add_argument('-n', "--maxevts", type=int, default=-1)
parser.add_argument('-o', "--out", default="branch_candidate_dqm.root")
parser.add_argument("--pdgId", type=int, default=0,
                    help="Fired gun PDG id; restricts the branch reference to that GEN species (0 = all).")
args = parser.parse_args()
if '/' not in args.inputFile and ':' not in args.inputFile:
    args.inputFile = 'file:' + args.inputFile

process = cms.Process("BRANCHCANDDQM")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.Geometry.GeometryExtendedRun4D120Reco_cff")
process.load("DQMServices.Core.DQMStore_cfi")
process.trackerGeometry.applyAlignment = cms.bool(False)

process.load("PhysicsTools.TruthInfo.truthGraphValidation_cff")

# Per-gun override: fire-species antichain reference.
if args.pdgId != 0:
    process.branchTICLCandidateValidator.interestingPdgIds = cms.vint32(args.pdgId, -args.pdgId)
    process.branchTICLCandidateValidator.onlyGenPrimaries = cms.bool(True)

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(args.maxevts))
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(args.inputFile))
process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(False))

process.dqmOut = cms.OutputModule("DQMRootOutputModule", fileName=cms.untracked.string(args.out))

process.p = cms.Path(
    process.truthGraphProducer
    + process.truthLogicalGraphProducer
    + process.detIdToRecHitMapProducer
    + process.truthLogicalGraphHitIndexProducer
    + process.branchTICLCandidateValidator
)
process.e = cms.EndPath(process.dqmOut)
