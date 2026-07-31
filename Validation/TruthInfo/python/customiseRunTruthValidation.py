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

    # Retarget the calorimetry validators (and their harvesters) at the trackster
    # collections this process actually schedules. The cff builds its recoCollections
    # from the static ticlIterLabelsPSet, which does not mention ticlTracksterInterpretations
    # even though both pyTICL presets assemble it, so without this the final tracksters of
    # the two-stage chain are validated by nobody while the job still runs green.
    from RecoTICL.Configuration.labels import tracksterLabelsInProcess

    present = tracksterLabelsInProcess(process, includeCandidate=False)
    for name, mod in process.producers_().items():
        if not hasattr(mod, "recoCollections"):
            continue
        configured = [t.getModuleLabel() for t in mod.recoCollections]
        # only the calorimetry domain: leave tracks, vertices and jets alone
        if not any(l.startswith("ticl") or l.startswith("hltTicl") for l in configured):
            continue
        missing = [l for l in present if l not in configured and not l.startswith("hlt")]
        if missing:
            mod.recoCollections = cms.VInputTag(*[cms.InputTag(l) for l in configured + missing])

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


def customiseHarvest(process, collections=("ticlTracksterInterpretations",)):
    """Extend the truth-branch harvesters to the collections the RECO step validated.

    The harvester subDirs are built from the same static list as the validators, so a
    collection added at RECO gets its folders booked but no ratios computed: the plots
    exist and are empty, which is the failure this whole wiring exists to avoid. The
    harvesting job cannot introspect the reco chain (those modules are not there), so the
    collections are passed in; they default to the two-stage chain's final tracksters,
    which is the one the static list omits.
    """
    from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociationLabels_cff import (
        instanceKey,
        workingPoints,
    )

    wps = [wp for wp, _, _ in workingPoints()]
    for name in list(process.analyzers_()) + list(process.producers_()):
        mod = getattr(process, name)
        if not hasattr(mod, "subDirs"):
            continue
        subDirs = list(mod.subDirs)
        calo = [d for d in subDirs if "Calorimetry" in d]
        if not calo:
            continue
        # Mirror the folder shape already in use for this harvester, per new collection.
        prefix = calo[0].rsplit("/", 1)[0]
        suffixes = sorted({d.rsplit("/", 1)[1].split("_", 1)[1] for d in calo if "_" in d.rsplit("/", 1)[1]})
        added = [f"{prefix}/{instanceKey(c)}_{sfx}" for c in collections for sfx in suffixes]
        missing = [d for d in added if d not in subDirs]
        if missing:
            mod.subDirs = cms.untracked.vstring(*(subDirs + missing))
    return process
