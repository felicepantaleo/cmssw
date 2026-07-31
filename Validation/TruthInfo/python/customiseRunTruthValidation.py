# Schedule the truth-branch DQM validation on a RECO job.
#
# The validators associate the reco collections to the truth graph themselves (the
# associator is named by a string), so the only inputs they need beyond the reco are the
# graph and the hit index, which enableTruth already builds at DIGI. Nothing truth-side
# is rebuilt here, so every leg of an A/B is scored against the identical truth.

import FWCore.ParameterSet.Config as cms


def customise(process):
    # process.load, not a plain import: the cff creates its analyzers with
    # globals()[label] = module, so they only acquire EDM labels when the whole cff is
    # loaded into the process. Importing just the sequence gives unlabelled modules.
    process.load("Validation.TruthInfo.truthBranchValidation_cff")

    process.truthBranchValidationPath = cms.Path(process.truthBranchValidationSequence)
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
