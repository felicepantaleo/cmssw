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
        graph=cms.InputTag("truthLogicalGraphProducer"),
        minSharedEnergy=cms.double(0.5),
        ambiguousFraction=cms.double(0.5),
    )
    return process.tracksterFeatureTable


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
