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

truthPlotVariables = ["pt", "eta", "phi", "nhits", "vertpos", "zpos", "dxy", "dz"]

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
}
_algoBlockArgs = {}
for _name, (_n, _lo, _hi) in _axes.items():
    _algoBlockArgs["nint_" + _name] = cms.int32(_n)
    _algoBlockArgs["min_" + _name] = cms.double(_lo)
    _algoBlockArgs["max_" + _name] = cms.double(_hi)
_algoBlockArgs.update(
    nintScore=cms.int32(50), minScore=cms.double(0.0), maxScore=cms.double(1.0),
    nintShared=cms.int32(50), minShared=cms.double(0.0), maxShared=cms.double(50.0),
    nintRes=cms.int32(60), minRes=cms.double(-0.6), maxRes=cms.double(0.6),
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

# Gaussian slice fits, the same mechanism MTV uses: DQMGenericClient books
# <prefix>_Mean and <prefix>_Sigma from each 2D.
_resolutions = [
    "ptres_vs_eta 'Relative p_{T} residual vs #eta'",
    "ptres_vs_pt 'Relative p_{T} residual vs p_{T}'",
    "etares_vs_eta '#eta residual vs #eta'",
    "phires_vs_eta '#phi residual vs #eta'",
]

truthBranchPostProcessor = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring(*_subDirs),
    efficiency=cms.vstring(*_efficiencies),
    resolution=cms.vstring(*_resolutions),
    verbose=cms.untracked.uint32(0),
    outputFileName=cms.untracked.string(""),
)

truthBranchHarvestingSequence = cms.Sequence(truthBranchPostProcessor)
