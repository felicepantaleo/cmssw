# Schedule the truth-branch DQM validation on a RECO job.
#
# The validators associate the reco collections to the truth graph themselves (the
# associator is named by a string), so beyond the reco they need only the graph and the
# hit index. Those are built at DIGI by mixedTruthGraphCustomize.buildCompactTruthAtDigi
# and must be KEPT in the DIGI output (the compact truth event content) for this to have
# anything to read: a step2 without them leaves every validator silently unfilled, with
# no ProductNotFound to show for it.

import FWCore.ParameterSet.Config as cms


def customise(process):
    from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociationLabels_cff import (
        setTracksterLabelsFromProcess,
    )

    # Retarget the collection registry at what this process produces or reads, and do it
    # BEFORE the validation cff is imported: that cff builds its modules, its DQM folder
    # names and its harvester subDirs from the registry at import time, so the registry is
    # the only place a correction reaches all three. Patching the built modules instead
    # fixes the validators and leaves the harvester asking for folders that moved.
    setTracksterLabelsFromProcess(process)

    # process.load, not a plain import: the cff creates its analyzers with
    # globals()[label] = module, so they only acquire EDM labels when the whole cff is
    # loaded into the process. Importing just the sequence gives unlabelled modules.
    # The ASSOCIATORS, then the validators. TruthBranchRecoValidator consumes an
    # association map per (collection, working point) and does nothing but `continue`
    # when that map is absent, so scheduling the validators alone yields a full folder
    # tree in which every measurement element is empty and no exception is raised.
    process.load("SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociators_cff")
    process.load("Validation.TruthInfo.truthBranchValidation_cff")

    process.truthBranchValidationPath = cms.Path(
        process.truthBranchValidationSequence, process.truthGraphAssociatorsTask
    )
    process.schedule.append(process.truthBranchValidationPath)

    if not hasattr(process, "DQMoutput"):
        process.DQMoutput = cms.OutputModule(
            "DQMRootOutputModule",
            fileName=cms.untracked.string("file:step3_inDQM.root"),
            outputCommands=cms.untracked.vstring("drop *", "keep *_MEtoEDMConverter_*_*"),
            splitLevel=cms.untracked.int32(0),
        )
        process.DQMoutput_step = cms.EndPath(process.DQMoutput)
        process.schedule.append(process.DQMoutput_step)

    process.add_(cms.Service("DQMStore"))
    return process
