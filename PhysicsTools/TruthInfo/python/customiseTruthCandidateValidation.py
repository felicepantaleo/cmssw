# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

# cmsDriver customise for the TICLCandidate physics-performance scan. Appended to the
# RECO step (step3), it schedules the truth-graph producer chain and the dedicated
# two-channel BranchTICLCandidateValidator INLINE, where every sim product the truth
# producers need is still live, and writes only the candidate DQMIO (the bulky RECO
# PoolOutput is dropped - the scan needs the plots, not the events). The fired species
# is read from TICL_GUN_PDGID so the branch reference is the fired GEN primary (a clean
# antichain). Harvest candidate_dqm.root with test/harvestBranchCandidateDQM_cfg.py.

import os
import FWCore.ParameterSet.Config as cms


def customise(process):
    from PhysicsTools.TruthInfo.truthGraphValidation_cff import (
        truthGraphProducer,
        truthLogicalGraphProducer,
        detIdToRecHitMapProducer,
        truthLogicalGraphHitIndexProducer,
        branchTICLCandidateValidator,
    )

    process.truthGraphProducer = truthGraphProducer
    process.truthLogicalGraphProducer = truthLogicalGraphProducer
    process.detIdToRecHitMapProducer = detIdToRecHitMapProducer
    process.truthLogicalGraphHitIndexProducer = truthLogicalGraphHitIndexProducer

    pdgId = int(os.environ.get("TICL_GUN_PDGID", "0"))
    validator = branchTICLCandidateValidator.clone()
    if pdgId != 0:
        validator.interestingPdgIds = cms.vint32(pdgId, -pdgId)
        validator.onlyGenPrimaries = cms.bool(True)
    process.branchTICLCandidateValidator = validator

    # Superclustering check: compare the CLUE3D tracksters (the fragmented input) with
    # the superclustered tracksters (ticlTracksterLinksSuperclusteringDNN, the EM
    # supercluster output) against the same truth branch. Superclustering should merge
    # the EM fragments into one high-purity object -> lower duplicate rate. Meaningful
    # for the EM guns (photon, electron); harmless for the others.
    from PhysicsTools.TruthInfo.truthGraphValidation_cff import branchTracksterRecoValidator
    _tsCommon = dict(layerClusters=cms.InputTag("hgcalMergeLayerClusters"),
                     xMax=cms.double(2500.0), minX=cms.double(0.0),
                     minAbsEta=cms.double(1.5), maxAbsEta=cms.double(3.0))
    clue3d = branchTracksterRecoValidator.clone(
        recoCollection=cms.InputTag("ticlTrackstersCLUE3DHigh"),
        folder=cms.string("HGCAL/BranchValidator/TracksterCLUE3D"), **_tsCommon)
    supercls = branchTracksterRecoValidator.clone(
        recoCollection=cms.InputTag("ticlTracksterLinksSuperclusteringDNN"),
        folder=cms.string("HGCAL/BranchValidator/TracksterSupercls"), **_tsCommon)
    if pdgId != 0:
        for m in (clue3d, supercls):
            m.interestingPdgIds = cms.vint32(pdgId, -pdgId)
            m.onlyGenPrimaries = cms.bool(True)
    process.branchTracksterCLUE3D = clue3d
    process.branchTracksterSupercls = supercls

    process.load("DQMServices.Core.DQMStore_cfi")
    process.truthCandidatePath = cms.Path(
        process.truthGraphProducer
        + process.truthLogicalGraphProducer
        + process.detIdToRecHitMapProducer
        + process.truthLogicalGraphHitIndexProducer
        + process.branchTICLCandidateValidator
        + process.branchTracksterCLUE3D
        + process.branchTracksterSupercls
    )
    process.candidateDQMoutput = cms.OutputModule(
        "DQMRootOutputModule",
        fileName=cms.untracked.string(os.environ.get("TICL_CAND_DQM", "candidate_dqm.root")),
    )
    process.candidateDQMEnd = cms.EndPath(process.candidateDQMoutput)

    process.schedule.append(process.truthCandidatePath)
    process.schedule.append(process.candidateDQMEnd)

    # Drop the heavy RECO PoolOutputs: the scan keeps only the candidate DQM. Remove
    # every EndPath that hosts a PoolOutputModule from the schedule (and detach it),
    # leaving the DQM (MEtoEDMConverter) and the candidate output paths in place.
    dropped = []
    for name, outmod in process.outputModules_().items():
        if outmod.type_() == "PoolOutputModule":
            for ep_name, ep in process.endpaths_().items():
                if name in ep.moduleNames():
                    dropped.append(ep_name)
    for ep_name in dropped:
        ep = getattr(process, ep_name)
        if process.schedule is not None:
            try:
                process.schedule.remove(ep)
            except ValueError:
                pass
        delattr(process, ep_name)

    return process
