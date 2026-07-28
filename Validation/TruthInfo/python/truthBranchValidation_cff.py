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

_wps = list(truthBranchWorkingPointsPSet.names)
truthInfoDqmDir = "TruthInfo/Tracking/"

truthBranchTrackValidator = cms.EDProducer(
    "TruthBranchTrackValidator",
    src=cms.InputTag("truthLogicalGraphProducer"),
    dirName=cms.string(truthInfoDqmDir),
    associator=cms.string("allTrackToTruthBranchAssociators"),
    recoCollections=cms.VInputTag(*[cms.InputTag(*l.split(":")) for l in recoLabels("tracks")]),
    workingPoints=cms.vstring(*_wps),
    histoProducerAlgoBlock=cms.PSet(
        nintPt=cms.int32(50), minPt=cms.double(0.0), maxPt=cms.double(100.0),
        nintEta=cms.int32(50), minEta=cms.double(-4.0), maxEta=cms.double(4.0),
        nintPhi=cms.int32(36), minPhi=cms.double(-3.2), maxPhi=cms.double(3.2),
        nintScore=cms.int32(50), minScore=cms.double(0.0), maxScore=cms.double(1.0),
        nintShared=cms.int32(50), minShared=cms.double(0.0), maxShared=cms.double(50.0),
    ),
)

truthBranchValidationSequence = cms.Sequence(truthBranchTrackValidator)

# One folder per (collection, working point); the same string the validator books into.
_subDirs = [
    truthInfoDqmDir + instanceKey(label) + "_" + wp
    for label in recoLabels("tracks")
    for wp in _wps
]

# Efficiency and fake rate are formed by DQMGenericClient from the num/denom names, so
# this package ships no harvesting C++.
_efficiencies = []
for var in ["pt", "eta", "phi"]:
    _efficiencies.append(
        f"efficiency_vs_{var} 'Branch efficiency vs {var}' num_assoc(simToReco)_{var} num_simul_{var}"
    )
    _efficiencies.append(
        f"fakerate_vs_{var} 'Fake rate vs {var}' num_assoc(recoToSim)_{var} num_reco_{var} fake"
    )
for var in ["pt", "eta"]:
    _efficiencies.append(
        f"duplicate_vs_{var} 'Duplicate rate vs {var}' num_duplicate_{var} num_simul_{var}"
    )

truthBranchPostProcessor = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring(*_subDirs),
    efficiency=cms.vstring(*_efficiencies),
    resolution=cms.vstring(),
    verbose=cms.untracked.uint32(0),
    outputFileName=cms.untracked.string(""),
)

truthBranchHarvestingSequence = cms.Sequence(truthBranchPostProcessor)
