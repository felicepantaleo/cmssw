# Reads the HGCalReco SoA host collections from a file of writeHGCalRecoSoA_cfg.py and checks their content.
import argparse
import sys

import FWCore.ParameterSet.Config as cms

parser = argparse.ArgumentParser(prog=sys.argv[0], description="Read and check the HGCalReco SoA host collections")
parser.add_argument("--rntuple", action="store_true", help="read the RNTuple file instead of the TTree file")
parser.add_argument("--products", nargs="+", default=["extra", "mask", "tracksters", "links", "pid"],
                    help="the collections to check")
args = parser.parse_args()

process = cms.Process("READ")
if args.rntuple:
    process.source = cms.Source("RNTupleTempSource", fileNames = cms.untracked.vstring("file:testHGCalRecoSoA_rntuple.root"))
else:
    process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:testHGCalRecoSoA.root"))

process.reader = cms.EDAnalyzer("TestReadHGCalRecoSoA",
    source = cms.InputTag("writer"),
    products = cms.vstring(args.products)
)

process.path = cms.Path(process.reader)
