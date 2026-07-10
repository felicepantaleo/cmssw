"""RECO-side customise for the pileup truth chain: build the merged (signal+pileup)
logical graph from the MixingModule accumulator's raw TruthGraph (label mix),
resolve its SimHit associations against the mixed rechits, and run the
trackster-to-branch associators. Pairs with mixedTruthGraphCustomize.addTruthGraph
Accumulator at DIGI. Gives a pileup-aware truth for fake-rate measurement."""

import FWCore.ParameterSet.Config as cms


def customise(process):
    from PhysicsTools.TruthInfo.truthGraphValidation_cff import (
        truthLogicalGraphProducer,
        detIdToRecHitMapProducer,
        truthLogicalGraphHitIndexProducer,
    )
    # The merged raw TruthGraph comes from the mixing accumulator, not the
    # standalone truthGraphProducer.
    process.truthLogicalGraphProducer = truthLogicalGraphProducer.clone(
        src=cms.InputTag("mix"),
    )
    process.detIdToRecHitMapProducer = detIdToRecHitMapProducer
    # The hit index also reads the RAW merged graph (for trackId->node); point it at mix.
    process.truthLogicalGraphHitIndexProducer = truthLogicalGraphHitIndexProducer.clone(
        rawSrc=cms.InputTag('mix'),
        # Read the merged (signal+pileup) HGCal sim-hits from the accumulator, each
        # tagged with its sub-event EncodedEventId, instead of the signal-only
        # g4SimHits (which lack pileup at RECO).
        simHitCollections=cms.VInputTag(cms.InputTag('mix', 'mergedHGCHits')),
    )

    from PhysicsTools.TruthInfo.allTrackstersToTruthBranchAssociations_cfi import (
        allTrackstersToTruthBranchAssociations,
    )
    process.allTrackstersToTruthBranchAssociations = allTrackstersToTruthBranchAssociations

    process.truthMixedRecoPath = cms.Path(
        process.truthLogicalGraphProducer
        + process.detIdToRecHitMapProducer
        + process.truthLogicalGraphHitIndexProducer
        + process.allTrackstersToTruthBranchAssociations
    )
    process.schedule.append(process.truthMixedRecoPath)

    for out in process.outputModules_().values():
        out.outputCommands.extend([
            "keep *_truthLogicalGraphProducer_*_*",
            "keep *_truthLogicalGraphHitIndexProducer_*_*",
            "keep *_allTrackstersToTruthBranchAssociations_*_*",
        ])
    return process
