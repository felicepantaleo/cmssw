# RECO-step customise for a linking A/B: associate the FINAL tracksters of whichever
# TICL version is configured to the truth-graph branches, and dump the per-trackster
# features (including the validator-style best score) to a flat table.
#
# It builds on the canonical addAdaptiveAssociator, so the truth graph and hit index
# are read from the products built at DIGI; nothing truth-side is rebuilt here and all
# legs of an A/B therefore score against the identical truth.

import FWCore.ParameterSet.Config as cms

from PhysicsTools.TruthInfo.addAdaptiveAssociator import addAdaptiveAssociator


def _finalTracksterLabel(process):
    # v6 produces the final tracksters in ticlTracksterInterpretations; v5 ends at
    # ticlTracksterLinks. Pick whichever this configuration actually schedules.
    if hasattr(process, "ticlTracksterInterpretations"):
        return "ticlTracksterInterpretations"
    return "ticlTracksterLinks"


def customise(process):
    label = _finalTracksterLabel(process)
    process = addAdaptiveAssociator(process, tracksterCollections=(label,))

    process.abTracksterFeatures = cms.EDProducer(
        "TracksterFeatureFlatTableProducer",
        name=cms.string("TS"),
        tracksters=cms.InputTag(label),
        layerClusters=cms.InputTag("hgcalMergeLayerClusters"),
        association=cms.InputTag("tracksterToTruthBranch", label + "ToTruthBranch"),
        associationAdaptive=cms.InputTag("tracksterToTruthBranch", label + "ToTruthBranchAdaptive"),
        associationByHitsAdaptive=cms.InputTag(""),
        graph=cms.InputTag("truthLogicalGraphProducer"),
        minSharedEnergy=cms.double(0.5),
        minSharedByHits=cms.double(0.02),
    )

    # Sim side: branch-level completeness. shared_frac is the fraction of the truth
    # branch energy captured by its dominant reco trackster (the validator efficiency
    # quantity), and n_contrib counts how many reco tracksters the branch is split
    # across, which is what separates "less fragmentation" from "over-merging" when two
    # linkers disagree on the object count.
    process.branchSimTracksters = cms.EDProducer(
        "BranchSimTracksterProducer",
        src=cms.InputTag("truthLogicalGraphProducer"),
        hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
        layerClusters=cms.InputTag("hgcalMergeLayerClusters"),
    )
    process.abSimTable = cms.EDProducer(
        "BranchSimTracksterFlatTableProducer",
        name=cms.string("SB"),
        branches=cms.InputTag("branchSimTracksters"),
        level=cms.InputTag("branchSimTracksters", "level"),
        rootId=cms.InputTag("branchSimTracksters", "rootId"),
        pdgId=cms.InputTag("branchSimTracksters", "pdgId"),
        recoCollection=cms.InputTag(label),
        layerClusters=cms.InputTag("hgcalMergeLayerClusters"),
        hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
        graph=cms.InputTag("truthLogicalGraphProducer"),
        minSharedFraction=cms.double(0.5),
        minContribFraction=cms.double(0.1),
    )
    process.abFeatureTask = cms.Task(process.abTracksterFeatures, process.branchSimTracksters, process.abSimTable)

    if hasattr(process, "reconstruction_step"):
        process.reconstruction_step.associate(process.abFeatureTask)
    else:
        process.schedule.associate(process.abFeatureTask)

    process.abNanoOut = cms.OutputModule(
        "NanoAODOutputModule",
        fileName=cms.untracked.string("file:ab_nano.root"),
        outputCommands=cms.untracked.vstring("drop *", "keep nanoaodFlatTable_*_*_*"),
        compressionLevel=cms.untracked.int32(9),
    )
    process.abNanoStep = cms.EndPath(process.abNanoOut)
    process.schedule.append(process.abNanoStep)
    return process
