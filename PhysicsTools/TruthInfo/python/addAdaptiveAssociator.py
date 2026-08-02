# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

# RECO-step customise that associates reco tracksters to truth-graph branches with
# AllTracksterToTruthBranchAssociatorsProducer, including the adaptive-level match.
#
# The truth graph and the per-particle per-cell hit index are READ from the products
# built at DIGI (enableTruth is in the Run4 era, so they are already in the event);
# nothing is rebuilt here. The layer clusters and tracksters come from this RECO job.
#
# Per trackster collection the producer emits four association maps:
#   <label>ToTruthBranch          / TruthBranchTo<label>           fixed-level
#   <label>ToTruthBranchAdaptive  / TruthBranchTo<label>Adaptive   adaptive-level
# The adaptive maps hold, per trackster, the single graph level minimizing
# score + adaptiveReverseWeight * reverseScore, rejecting levels whose branch spread
# (contamination) exceeds adaptiveMaxReverseScore.

import FWCore.ParameterSet.Config as cms


def addAdaptiveAssociator(process,
                          tracksterCollections=("ticlTrackstersCLUE3DHigh",),
                          layerClusters="hgcalMergeLayerClusters",
                          adaptiveReverseWeight=1.0,
                          adaptiveMaxReverseScore=1.0,
                          label="tracksterToTruthBranch"):
    """Schedule the trackster-to-truth-branch associator at RECO and keep its products.

    tracksterCollections: labels (or "label:instance") of the trackster collections to
    associate; each yields its own pair of fixed-level and adaptive-level maps.
    adaptiveReverseWeight: how strongly branch spread counts against climbing a level.
    adaptiveMaxReverseScore: contamination ceiling above which a level is rejected.
    """
    def _tag(name):
        parts = name.split(":")
        return cms.InputTag(*parts)

    associator = cms.EDProducer(
        "AllTracksterToTruthBranchAssociatorsProducer",
        src=cms.InputTag("truthLogicalGraphProducer"),
        hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
        layerClusters=cms.InputTag(layerClusters),
        tracksterCollections=cms.VInputTag(*[_tag(t) for t in tracksterCollections]),
        rootsSrc=cms.InputTag(""),      # empty: use the calo-boundary root selection
        branchPdgIds=cms.vint32(),      # empty: keep every branch root
        adaptiveReverseWeight=cms.double(adaptiveReverseWeight),
        adaptiveMaxReverseScore=cms.double(adaptiveMaxReverseScore),
    )
    setattr(process, label, associator)

    pathName = label + "Path"
    setattr(process, pathName, cms.Path(getattr(process, label)))
    if process.schedule is not None:
        process.schedule.append(getattr(process, pathName))

    for out in process.outputModules_().values():
        out.outputCommands.append("keep *_" + label + "_*_*")

    return process
