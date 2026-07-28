# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# Schedule the truth-graph association producers at RECO and keep their products.
# The producers form a cms.Task, so the framework runs only what is actually consumed
# and resolves the constituent dependency (vertices need the track maps) itself.

import FWCore.ParameterSet.Config as cms


def customiseTruthGraphAssociators(process):
    from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociators_cff import (
        allTrackToTruthBranchAssociators,
        allVertexToTruthBranchAssociators,
        allSecondaryVertexToTruthBranchAssociators,
    )

    # Attach each producer to the process FIRST: a cms.Task imported by name carries
    # modules that have no label yet, and adding it directly fails with "an entry in
    # task ... has not been attached to the process".
    process.allTrackToTruthBranchAssociators = allTrackToTruthBranchAssociators
    process.allVertexToTruthBranchAssociators = allVertexToTruthBranchAssociators
    process.allSecondaryVertexToTruthBranchAssociators = allSecondaryVertexToTruthBranchAssociators

    process.truthGraphAssociatorsTask = cms.Task(
        process.allTrackToTruthBranchAssociators,
        process.allVertexToTruthBranchAssociators,
        process.allSecondaryVertexToTruthBranchAssociators,
    )
    process.truthGraphAssociatorsPath = cms.Path(process.truthGraphAssociatorsTask)
    if process.schedule is not None:
        process.schedule.append(process.truthGraphAssociatorsPath)

    for out in process.outputModules_().values():
        out.outputCommands.extend(
            [
                "keep *_allTrackToTruthBranchAssociators_*_*",
                "keep *_allVertexToTruthBranchAssociators_*_*",
                "keep *_allSecondaryVertexToTruthBranchAssociators_*_*",
            ]
        )
    return process
