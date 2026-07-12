"""GEN-SIM customise for the lightweight truth library (pileup or signal). Builds
the logical truth graph and the node->SimHit associations from GEN+SIM (no rechits),
persists those two, and keeps the raw TruthGraph transient (not written to disk).
The SimHit associations carry an invalid recHitIndex to be resolved after mixing at
RECO, when the detId->rechit map is available."""

import FWCore.ParameterSet.Config as cms


def customise(process):
    from PhysicsTools.TruthInfo.truthGraphValidation_cff import (
        truthGraphProducer,
        truthLogicalGraphProducer,
    )
    from Validation.Configuration.truthPrevalidation_cff import truthLogicalGraphHitIndexProducer

    process.truthGraphProducer = truthGraphProducer
    process.truthLogicalGraphProducer = truthLogicalGraphProducer
    # No detIdToRecHitMapProducer at GEN-SIM: the hit index builds the SimHit side
    # only, with recHitIndex = invalid (resolved at RECO after mixing).
    process.truthLogicalGraphHitIndexProducer = truthLogicalGraphHitIndexProducer.clone(
        recHitMap=cms.InputTag(""),
    )
    process.truthLibraryPath = cms.Path(
        process.truthGraphProducer
        + process.truthLogicalGraphProducer
        + process.truthLogicalGraphHitIndexProducer
    )
    process.schedule.append(process.truthLibraryPath)

    for outName in ("FEVTDEBUGoutput", "FEVTDEBUGHLToutput", "output"):
        if hasattr(process, outName):
            oc = getattr(process, outName).outputCommands
            # Keep the logical graph and the SimHit associations.
            oc.extend([
                "keep *_truthLogicalGraphProducer_*_*",
                "keep *_truthLogicalGraphHitIndexProducer_*_*",
            ])
            # The raw TruthGraph is transient: never written to disk.
            oc.append("drop *_truthGraphProducer_*_*")
    return process
