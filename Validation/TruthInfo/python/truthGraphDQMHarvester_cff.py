# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# DQM harvesting for the Branch validators: DQMGenericClient divides the numerator
# and denominator histograms into reproduction efficiencies vs eta/pt/energy.
# The analyzers fill the profiles (purity/completeness/response) directly.

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

_branchEfficiency = cms.vstring(
    "efficiency_eta 'Branch reproduction efficiency vs #eta;#eta;efficiency' effnum_eta denom_eta",
    "efficiency_pt 'Branch reproduction efficiency vs p_{T};p_{T} [GeV];efficiency' effnum_pt denom_pt",
    "efficiency_energy 'Branch reproduction efficiency vs E;E [GeV];efficiency' effnum_energy denom_energy",
    # Fraction of objects whose best hit-matched Branch is the natural (trackId-seeded) one.
    "selfmatchrate_eta 'Best Branch is the natural one vs #eta;#eta;self-match rate' selfmatch_eta denom_eta",
    "selfmatchrate_pt 'Best Branch is the natural one vs p_{T};p_{T} [GeV];self-match rate' selfmatch_pt denom_pt",
)

branchHGCalPostProcessor = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring(
        "HGCAL/BranchValidator/CaloParticle",
        "HGCAL/BranchValidator/SimCluster",
    ),
    efficiency=_branchEfficiency,
    resolution=cms.vstring(),
    verbose=cms.untracked.uint32(0),
    outputFileName=cms.untracked.string(""),
)

# Tracking: efficiency for the Branch to reproduce the TrackingParticle assignment of a track, vs eta/pt.
_branchTrackingEfficiency = cms.vstring(
    "efficiency_eta 'Branch reproduction efficiency vs #eta;#eta;efficiency' effnum_eta denom_eta",
    "efficiency_pt 'Branch reproduction efficiency vs p_{T};p_{T} [GeV];efficiency' effnum_pt denom_pt",
)

branchTrackingPostProcessor = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring(
        "Tracking/BranchValidator/TrackingParticle",
    ),
    efficiency=_branchTrackingEfficiency,
    resolution=cms.vstring(),
    verbose=cms.untracked.uint32(0),
    outputFileName=cms.untracked.string(""),
)

truthGraphDQMHarvesting = cms.Sequence(branchHGCalPostProcessor + branchTrackingPostProcessor)

