# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# The DQM analyzers and their harvesting, both generated from the same label and
# working-point lists the associators use, so the folder names, the ME names and the
# harvester subDirs cannot drift apart.

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociationLabels_cff import (
    truthBranchWorkingPointsPSet,
    recoLabels,
    instanceKey,
)

truthPlotVariables = ["pt", "eta", "phi", "nhits", "vertpos", "zpos", "dxy", "dz", "depth", "rootfrac"]

_wps = list(truthBranchWorkingPointsPSet.names)
truthInfoDqmDir = "TruthInfo/Tracking/"

# Axis definition per x variable, in the same order as the C++ Variable enum. Built
# here so the booking, the harvester strings and the plot script all read one list.
_axes = {
    "pt": (50, 0.0, 100.0),
    "eta": (50, -4.0, 4.0),
    "phi": (36, -3.2, 3.2),
    "nhits": (40, 0.0, 40.0),
    "vertpos": (40, 0.0, 60.0),
    "zpos": (40, -30.0, 30.0),
    "dxy": (40, -5.0, 5.0),
    "dz": (40, -20.0, 20.0),
    # Graph-only axes: depth of the branch root in the graph, and the fraction of the
    # branch footprint that belongs to the root particle itself.
    "depth": (15, 0.0, 15.0),
    "rootfrac": (20, 0.0, 1.0),
}
_algoBlockArgs = {}
for _name, (_n, _lo, _hi) in _axes.items():
    _algoBlockArgs["nint_" + _name] = cms.int32(_n)
    _algoBlockArgs["min_" + _name] = cms.double(_lo)
    _algoBlockArgs["max_" + _name] = cms.double(_hi)
_algoBlockArgs.update(
    nintScore=cms.int32(50), minScore=cms.double(0.0), maxScore=cms.double(1.0),
    nintShared=cms.int32(50), minShared=cms.double(0.0), maxShared=cms.double(50.0),
    # Wide on purpose. The truth reference is the BRANCH ROOT, and a track matched to a
    # branch by shared hits can belong to a descendant of that root, so (reco - truth)/truth
    # has a long tail that a +-0.6 window pushes into the overflow, leaving the slice fit
    # with a nearly flat in-range distribution and no convergence.
    nintRes=cms.int32(120), minRes=cms.double(-1.5), maxRes=cms.double(1.5),
    # Coarser than the efficiency axes on purpose: every x slice of these 2D gets a
    # Gaussian fit, and a slice with a handful of entries returns a meaningless width.
    nint_res_eta=cms.int32(20), min_res_eta=cms.double(-4.0), max_res_eta=cms.double(4.0),
    nint_res_pt=cms.int32(15), min_res_pt=cms.double(0.0), max_res_pt=cms.double(100.0),
)
_algoBlock = cms.PSet(**_algoBlockArgs)

truthBranchTrackValidator = cms.EDProducer(
    "TruthBranchTrackValidator",
    src=cms.InputTag("truthLogicalGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    dirName=cms.string(truthInfoDqmDir),
    associator=cms.string("allTrackToTruthBranchAssociators"),
    recoCollections=cms.VInputTag(*[cms.InputTag(*l.split(":")) for l in recoLabels("tracks")]),
    workingPoints=cms.vstring(*_wps),
    histoProducerAlgoBlock=_algoBlock,
)

truthBranchValidationSequence = cms.Sequence(truthBranchTrackValidator)

# One folder per (collection, working point); the same string the validator books into.
_subDirs = [
    truthInfoDqmDir + instanceKey(label) + "_" + wp
    for label in recoLabels("tracks")
    for wp in _wps
]

# The x variables the validator books, in the same order as the C++ enum. Kept here so
# the harvester strings and the plot script cannot drift from the booking.

# Every ratio is formed by DQMGenericClient from the num/denom names, so this package
# ships no harvesting C++. The metric set follows MultiTrackValidator (efficiency, fake,
# duplicate, pileup) plus purity from the TICL trackster validation, which asks the
# complementary question: how much of the reco object belongs to the branch it matched.
_efficiencies = []
for var in truthPlotVariables:
    _efficiencies.append(
        f"efficiency_vs_{var} 'Branch efficiency vs {var}' num_assoc(simToReco)_{var} num_simul_{var}"
    )
    _efficiencies.append(
        f"fakerate_vs_{var} 'Fake rate vs {var}' num_assoc(recoToSim)_{var} num_reco_{var} fake"
    )
    _efficiencies.append(
        f"duplicate_vs_{var} 'Duplicate rate vs {var}' num_duplicate_{var} num_simul_{var}"
    )
    _efficiencies.append(
        f"pileuprate_vs_{var} 'Pileup rate vs {var}' num_pileup_{var} num_reco_{var}"
    )
    _efficiencies.append(
        f"purity_vs_{var} 'Purity vs {var}' num_assoc(recoToSim)_{var} num_reco_{var}"
    )

# Efficiency and duplicate rate against the Geant4 creation process of the branch. The
# axis is categorical, one bin per truth::VertexReason, and it exists only because the
# graph keeps the process that made each particle; a frozen truth object does not.
_efficiencies.append(
    "efficiency_vs_reason 'Branch efficiency vs creation process' "
    "num_assoc(simToReco)_reason num_simul_reason"
)
_efficiencies.append(
    "duplicate_vs_reason 'Duplicate rate vs creation process' num_duplicate_reason num_simul_reason"
)

# Gaussian slice fits, the same mechanism MTV uses: DQMGenericClient books
# <prefix>_Mean and <prefix>_Sigma from each 2D. The string is three tokens,
# "<outputPrefix> '<title>' <sourceHistogram>"; a two-token form parses without an
# error and silently produces nothing.
_resolutions = [
    "ptres_vs_eta 'Relative p_{T} residual vs #eta' ptres_vs_eta",
    "ptres_vs_pt 'Relative p_{T} residual vs p_{T}' ptres_vs_pt",
    "etares_vs_eta '#eta residual vs #eta' etares_vs_eta",
    "phires_vs_eta '#phi residual vs #eta' phires_vs_eta",
]

truthBranchPostProcessor = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring(*_subDirs),
    efficiency=cms.vstring(*_efficiencies),
    resolution=cms.vstring(*_resolutions),
    # Fit the core, not the tail: the slice fit is restricted to a window around the peak,
    # which is what makes Sigma a resolution rather than the width of the axis.
    resolutionLimitedFit=cms.untracked.bool(True),
    verbose=cms.untracked.uint32(0),
    outputFileName=cms.untracked.string(""),
)

truthBranchHarvestingSequence = cms.Sequence(truthBranchPostProcessor)
