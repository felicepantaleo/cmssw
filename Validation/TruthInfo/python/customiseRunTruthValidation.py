# Schedule the truth-branch DQM validation on a RECO job.
#
# The associators, and the collection discovery they depend on, come from the canonical
# customiseTruthGraphAssociators: this file only adds the validators and a DQM output on
# top. Doing the association wiring here instead is how the validators once ran with no
# association maps to read, which produces a complete DQM folder tree in which every
# measurement element is empty, with no exception raised.
#
# The truth graph and the hit index are built and persisted at DIGI by default, because
# enableTruth is in the era every Phase-2 geometry inherits, so a plain step2 already
# carries them and nothing truth-side is rebuilt here.

import FWCore.ParameterSet.Config as cms


def customise(process):
    from SimGeneral.TruthGraphAssociatorProducers.customiseTruthGraphAssociators import (
        customiseTruthGraphAssociators,
    )

    process = customiseTruthGraphAssociators(process)

    # process.load, not a plain import: the cff creates its analyzers with
    # globals()[label] = module, so they only acquire EDM labels when the whole cff is
    # loaded into the process. Importing just the sequence gives unlabelled modules.
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
