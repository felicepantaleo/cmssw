"""Customises for the trackster-transformer PID training dump.

Two steps (same pattern as customiseTruthBranchTraining):
  - customiseReco : at RECO, DISABLE the trackster filtering (so low-energy fakes
    survive to be labeled), run the truth-graph + trackster-to-branch association
    chain, and persist the graph + association + tracksters + layer clusters.
  - customiseNano : at the NanoAOD step, add TracksterFeatureFlatTableProducer, which
    reads those persisted products and emits the per-trackster and per-LC feature
    tables (see PhysicsTools/TruthInfo/plugins/TracksterFeatureFlatTableProducer.cc).

The dumped collection is ticlTrackstersCLUE3DHigh (the complete, pre-linking set, the
natural per-shower unit); the association instance is the matching CLUE3DHigh one.
"""
import FWCore.ParameterSet.Config as cms


def _disable_trackster_filtering(process):
    # TracksterLinkingbySkeletons::isGoodTrackster gates tracksters by raw energy and
    # LC count; at the defaults (20 GeV, 15 LCs) it starves the low-energy fake sample.
    if hasattr(process, "ticlTracksterLinks") and hasattr(process.ticlTracksterLinks, "linkingPSet"):
        ps = process.ticlTracksterLinks.linkingPSet
        if hasattr(ps, "min_trackster_energy"):
            ps.min_trackster_energy = cms.double(0.0)
        if hasattr(ps, "min_num_lcs"):
            ps.min_num_lcs = cms.uint32(0)
    return process


def customiseReco(process):
    process = _disable_trackster_filtering(process)
    from PhysicsTools.TruthInfo.customiseTruthBranchTraining import customise as _training
    process = _training(process)
    return process


def _feature_table(process):
    process.tracksterFeatureTable = cms.EDProducer(
        "TracksterFeatureFlatTableProducer",
        name=cms.string("TICLTrackster"),
        tracksters=cms.InputTag("ticlTrackstersCLUE3DHigh"),
        layerClusters=cms.InputTag("hgcalMergeLayerClusters"),
        association=cms.InputTag("allTrackstersToTruthBranchAssociations",
                                 "ticlTrackstersCLUE3DHighToTruthBranch"),
        associationAdaptive=cms.InputTag("allTrackstersToTruthBranchAssociations",
                                         "ticlTrackstersCLUE3DHighToTruthBranchAdaptive"),
        graph=cms.InputTag("truthLogicalGraphProducer"),
        minSharedEnergy=cms.double(0.5),
    )
    return process.tracksterFeatureTable


# v6 downstream trackster stages to dump features+truth for (table name, module label).
# Each survives only when the v6 chain is applied (customiseApplyV6).
_V6_TRACKSTER_STAGES = [
    ("TICLTrackster", "ticlTrackstersCLUE3DHigh"),
    ("TICLTracksterLinks", "ticlTracksterLinks"),
    ("TICLTracksterSupercls", "ticlTracksterLinksSuperclusteringDNN"),
    ("TICLTracksterInterp", "ticlTracksterInterpretations"),
]


def _feature_table_for(process, name, collection):
    tbl = cms.EDProducer(
        "TracksterFeatureFlatTableProducer",
        name=cms.string(name),
        tracksters=cms.InputTag(collection),
        layerClusters=cms.InputTag("hgcalMergeLayerClusters"),
        association=cms.InputTag("allTrackstersToTruthBranchAssociations", collection + "ToTruthBranch"),
        associationAdaptive=cms.InputTag("allTrackstersToTruthBranchAssociations", collection + "ToTruthBranchAdaptive"),
        associationByHitsAdaptive=cms.InputTag("allTrackstersToTruthBranchByHitsAssociations",
                                               collection + "ToTruthBranchByHitsAdaptive"),
        graph=cms.InputTag("truthLogicalGraphProducer"),
        minSharedEnergy=cms.double(0.5),
        minSharedByHits=cms.double(0.02),  # coverage-matched to the default (composition E ~25x lower)
    )
    setattr(process, "tracksterFeatureTable" + name, tbl)  # no underscore in module labels
    return tbl


def customiseMixedAll(process, stages=None):
    """Pileup-aware dump of ALL v6 downstream trackster stages (CLUE3DHigh, links,
    superclustering, interpretations), each with its own feature table + truth
    association. Requires the v6 chain (customiseApplyV6) to have produced the
    collections. Single job (DIGI+RECO+features)."""
    stages = stages or _V6_TRACKSTER_STAGES
    stages = [(n, c) for (n, c) in stages if hasattr(process, c)]  # only produced collections
    labels = [c for _, c in stages]
    process = _disable_trackster_filtering(process)
    from PhysicsTools.TruthInfo.customiseTruthMixedReco import customise as _mixed
    process = _mixed(process)
    from PhysicsTools.TruthInfo.allTrackstersToTruthBranchAssociations_cfi import allTrackstersToTruthBranchAssociations
    from PhysicsTools.TruthInfo.allTrackstersToTruthBranchByHitsAssociations_cfi import allTrackstersToTruthBranchByHitsAssociations
    from PhysicsTools.TruthInfo.allLayerClustersToTruthBranchAssociations_cfi import allLayerClustersToTruthBranchAssociations
    from PhysicsTools.TruthInfo.branchSimTracksters_cfi import branchSimTracksters
    # Shared low-level hit maps that the by-hits (composition) associators walk against (no
    # per-object merge-join): the rechit index map, hit -> LC, and hit -> trackster. All are
    # standard modules, added only if the chain lacks them.
    from RecoLocalCalo.HGCalRecProducers.recHitMapProducer_cfi import recHitMapProducer
    from SimCalorimetry.HGCalAssociatorProducers.hitToLayerClusterAssociator_cfi import hitToLayerClusterAssociator
    from SimCalorimetry.HGCalAssociatorProducers.AllHitToTracksterAssociatorsProducer_cfi import AllHitToTracksterAssociatorsProducer
    if not hasattr(process, "recHitMapProducer"):
        process.recHitMapProducer = recHitMapProducer.clone()
    if not hasattr(process, "hitToLayerClusterAssociator"):
        process.hitToLayerClusterAssociator = hitToLayerClusterAssociator.clone()
    process.allHitToTracksterAssociations = AllHitToTracksterAssociatorsProducer.clone(
        tracksterCollections=labels,
    )
    # Default trackster<->branch association: the BranchHitAssociator (unchanged, the
    # deployed PID training labels).
    process.allTrackstersToTruthBranchAssociations = allTrackstersToTruthBranchAssociations.clone(
        tracksterCollections=labels,
    )
    process.branchSimTracksters = branchSimTracksters
    process.allTrackstersToTruthBranchAssociationsAllLevels = allTrackstersToTruthBranchAssociations.clone(
        tracksterCollections=labels,
        rootsSrc=("branchSimTracksters", "roots"),
    )
    # Parallel "by hits" (composition) association, dumped alongside for label comparison.
    process.allTrackstersToTruthBranchByHitsAssociations = allTrackstersToTruthBranchByHitsAssociations.clone(
        tracksterCollections=labels,
        hitToTracksterProducer="allHitToTracksterAssociations",
    )
    process.allLayerClustersToTruthBranchAssociations = allLayerClustersToTruthBranchAssociations.clone()
    process.truthMixedAssocPath = cms.Path(
        process.recHitMapProducer
        + process.hitToLayerClusterAssociator
        + process.allHitToTracksterAssociations
        + process.allTrackstersToTruthBranchAssociations
        + process.allTrackstersToTruthBranchByHitsAssociations
        + process.branchSimTracksters
        + process.allTrackstersToTruthBranchAssociationsAllLevels
        + process.allLayerClustersToTruthBranchAssociations
    )
    tables = [_feature_table_for(process, n, c) for (n, c) in stages]
    seq = tables[0]
    for t in tables[1:]:
        seq = seq + t
    # sim-side adaptive-branch table: one row per truth branch with its best reco match
    # from the reverse adaptive association. Supplies the efficiency/duplicate
    # denominators that the reco-side trackster tables lack (adaptive-branch validation).
    process.simBranchTableCLUE3D = cms.EDProducer(
        "BranchSimTracksterFlatTableProducer",
        name=cms.string("SimBranchCLUE3D"),
        branches=cms.InputTag("branchSimTracksters"),
        level=cms.InputTag("branchSimTracksters", "level"),
        rootId=cms.InputTag("branchSimTracksters", "rootId"),
        pdgId=cms.InputTag("branchSimTracksters", "pdgId"),
        recoCollection=cms.InputTag("ticlTrackstersCLUE3DHigh"),
        layerClusters=cms.InputTag("hgcalMergeLayerClusters"),
        hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
        graph=cms.InputTag("truthLogicalGraphProducer"),
        minSharedFraction=cms.double(0.5),
        minContribFraction=cms.double(0.1),
    )
    seq = seq + process.simBranchTableCLUE3D
    # per-LC truth table: consumes the LC-to-branch adaptive association, one row per
    # layer cluster with its single-shower truth class, match score/shared energy and
    # signal/pileup provenance (LC-granularity validation).
    process.layerClusterTruthTable = cms.EDProducer(
        "LayerClusterTruthFlatTableProducer",
        name=cms.string("LayerClusterTruth"),
        layerClusters=cms.InputTag("hgcalMergeLayerClusters"),
        association=cms.InputTag("allLayerClustersToTruthBranchAssociations",
                                 "hgcalMergeLayerClustersToTruthBranch"),
        graph=cms.InputTag("truthLogicalGraphProducer"),
    )
    seq = seq + process.layerClusterTruthTable
    process.tracksterFeatureTablesPath = cms.Path(seq)
    if process.schedule is not None:
        process.schedule.append(process.truthMixedAssocPath)
        process.schedule.append(process.tracksterFeatureTablesPath)
    for outName in ("FEVTDEBUGHLToutput", "RECOSIMoutput", "NANOAODoutput", "output"):
        if hasattr(process, outName):
            getattr(process, outName).outputCommands.append("keep nanoaodFlatTable_tracksterFeatureTable*_*_*")
            getattr(process, outName).outputCommands.append("keep nanoaodFlatTable_simBranchTable*_*_*")
            getattr(process, outName).outputCommands.append("keep nanoaodFlatTable_layerClusterTruthTable*_*_*")
    return process


def customiseNano(process):
    table = _feature_table(process)
    # NanoAODOutputModule writes every nanoaod::FlatTable product, so just make it run.
    for taskName in ("nanoTableTaskCommon", "nanoTableTaskFS", "nanoSequenceCommon"):
        if hasattr(process, taskName):
            getattr(process, taskName).add(table)
            return process
    process.tracksterFeatureTablePath = cms.Path(table)
    if process.schedule is not None:
        process.schedule.append(process.tracksterFeatureTablePath)
    return process


# Pileup-aware single-job features: the mixed (signal + pileup) truth graph built by
# the MixingModule accumulator, instead of the signal-only graph. Pairs with
# mixedTruthGraphCustomize.addTruthGraphAccumulator at DIGI (which produces TruthGraph
# 'mix' and keeps mix:mergedHGCHits). Use this at RECO for a PU sample so pileup
# tracksters get their real particle type and the is_primary flag is meaningful,
# instead of every pileup trackster falling to the fake class.
def customiseMixed(process):
    process = _disable_trackster_filtering(process)
    from PhysicsTools.TruthInfo.customiseTruthMixedReco import customise as _mixed
    process = _mixed(process)  # mixed (signal+pileup) graph + hit index (src=mix)
    # _mixed builds the mixed graph + hit index but NOT the trackster-to-branch
    # associator that _feature_table consumes; add it here, reading the mixed graph
    # and hit index (the same-named modules _mixed just repointed at 'mix').
    from PhysicsTools.TruthInfo.allTrackstersToTruthBranchAssociations_cfi import allTrackstersToTruthBranchAssociations
    from PhysicsTools.TruthInfo.branchSimTracksters_cfi import branchSimTracksters
    # The cfi default tracksterCollections includes the v6 'ticlTracksterInterpretations',
    # absent from a standard v5 reco; restrict to the produced CLUE3DHigh collection
    # (all the feature table consumes) so the associator does not throw ProductNotFound.
    process.allTrackstersToTruthBranchAssociations = allTrackstersToTruthBranchAssociations.clone(
        tracksterCollections=["ticlTrackstersCLUE3DHigh"],
    )
    process.branchSimTracksters = branchSimTracksters
    process.allTrackstersToTruthBranchAssociationsAllLevels = allTrackstersToTruthBranchAssociations.clone(
        tracksterCollections=["ticlTrackstersCLUE3DHigh"],
        rootsSrc=("branchSimTracksters", "roots"),
    )
    process.truthMixedAssocPath = cms.Path(
        process.allTrackstersToTruthBranchAssociations
        + process.branchSimTracksters
        + process.allTrackstersToTruthBranchAssociationsAllLevels
    )
    if process.schedule is not None:
        process.schedule.append(process.truthMixedAssocPath)
    table = _feature_table(process)
    process.tracksterFeatureTablePath = cms.Path(table)
    if process.schedule is not None:
        process.schedule.append(process.tracksterFeatureTablePath)
    for outName in ("FEVTDEBUGHLToutput", "RECOSIMoutput", "NANOAODoutput", "output"):
        if hasattr(process, outName):
            getattr(process, outName).outputCommands.append(
                "keep nanoaodFlatTable_tracksterFeatureTable_*_*")
    return process


# Single-job convenience: RECO + feature table in one process (no split, no persist).
# Use when running DIGI+RECO+features together; the tables land in the job output.
def customise(process):
    process = customiseReco(process)
    table = _feature_table(process)
    if hasattr(process, "truthBranchTrainingPath"):
        process.truthBranchTrainingPath += table
    else:
        process.tracksterFeatureTablePath = cms.Path(table)
        if process.schedule is not None:
            process.schedule.append(process.tracksterFeatureTablePath)
    for outName in ("FEVTDEBUGHLToutput", "RECOSIMoutput", "NANOAODoutput", "output"):
        if hasattr(process, outName):
            getattr(process, outName).outputCommands.append(
                "keep nanoaodFlatTable_tracksterFeatureTable_*_*")
    return process
