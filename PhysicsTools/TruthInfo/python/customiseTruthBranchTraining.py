"""Step3 customisation for the truth-branch training dataset: run the truth graph
chain plus the trackster-to-branch associator during RECO, and persist the truth
graph and the association maps so a downstream NanoAOD step (autoNANO @HGCALTruth)
can build the training tables without access to the sim hits."""

import FWCore.ParameterSet.Config as cms


def customise(process):
    from PhysicsTools.TruthInfo.truthGraphValidation_cff import (
        truthGraphProducer,
        truthLogicalGraphProducer,
        detIdToRecHitMapProducer,
        truthLogicalGraphHitIndexProducer,
    )
    from PhysicsTools.TruthInfo.allTrackstersToTruthBranchAssociations_cfi import allTrackstersToTruthBranchAssociations

    process.truthGraphProducer = truthGraphProducer
    process.truthLogicalGraphProducer = truthLogicalGraphProducer
    process.detIdToRecHitMapProducer = detIdToRecHitMapProducer
    process.truthLogicalGraphHitIndexProducer = truthLogicalGraphHitIndexProducer
    process.allTrackstersToTruthBranchAssociations = allTrackstersToTruthBranchAssociations

    process.truthBranchTrainingPath = cms.Path(
        process.truthGraphProducer
        + process.truthLogicalGraphProducer
        + process.detIdToRecHitMapProducer
        + process.truthLogicalGraphHitIndexProducer
        + process.allTrackstersToTruthBranchAssociations
    )
    process.schedule.append(process.truthBranchTrainingPath)

    keeps = [
        "keep *_truthLogicalGraphProducer_*_*",
        "keep *_allTrackstersToTruthBranchAssociations_*_*",
    ]
    for outName in ("FEVTDEBUGHLToutput", "RECOSIMoutput", "output"):
        if hasattr(process, outName):
            getattr(process, outName).outputCommands.extend(keeps)
    return process
