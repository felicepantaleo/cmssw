# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

# Phase-A pileup customise: enable the SimTrack/SimVertex crossing frames and run
# TruthGraphMixedProducer in the DIGI step (the only place the transient
# CrossingFrame<SimTrack/SimVertex> products live), then keep the compact mixed
# raw TruthGraph in the output so downstream steps can read signal+pileup truth.

import FWCore.ParameterSet.Config as cms
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn


def addMixedTruthGraph(process):
    # makeCrossingFrame=True for SimTrack/SimVertex (transient, in-process only).
    process = setCrossingFrameOn(process)

    process.truthGraphMixedProducer = cms.EDProducer(
        "TruthGraphMixedProducer",
        simTracks=cms.InputTag("mix", "g4SimHits"),
        simVertices=cms.InputTag("mix", "g4SimHits"),
    )

    process.truthGraphMixedPath = cms.Path(process.truthGraphMixedProducer)
    if process.schedule is not None:
        process.schedule.append(process.truthGraphMixedPath)

    for out in process.outputModules_().values():
        out.outputCommands.append("keep *_truthGraphMixedProducer_*_*")

    return process


def addTruthGraphAccumulator(process,
                             pileupBunchCrossings=(0,),
                             collapsePileupGen=True):
    """Phase-B (B1): register TruthGraphAccumulator inside the MixingModule.

    The accumulator builds the mixed (signal + pileup) raw TruthGraph from the
    native per-sub-event SimTrack/SimVertex collections. By default only in-time
    pileup (bx 0) is included; pass pileupBunchCrossings to widen. The mixed graph
    is kept in the output as TruthGraph_mix__<process>.
    """
    process.mix.digitizers.truthGraph = cms.PSet(
        accumulatorType=cms.string("TruthGraphAccumulator"),
        simTracks=cms.InputTag("g4SimHits"),
        simVertices=cms.InputTag("g4SimHits"),
        genEventHepMC3=cms.InputTag("generatorSmeared"),
        genEventHepMC=cms.InputTag("generatorSmeared"),
        caloHits=cms.VInputTag(
            cms.InputTag("g4SimHits", "HGCHitsEE"),
            cms.InputTag("g4SimHits", "HGCHitsHEfront"),
            cms.InputTag("g4SimHits", "HGCHitsHEback"),
        ),
        # Barrel calorimeters, kept in separate products so the RECO consumer applies
        # the right sim-to-reco DetId relabelling per collection (ECAL barrel needs
        # none, HCAL uses HcalHitRelabeller).
        ecalHits=cms.VInputTag(cms.InputTag("g4SimHits", "EcalHitsEB")),
        hcalHits=cms.VInputTag(cms.InputTag("g4SimHits", "HcalHits")),
        # Tracking detectors, as PSimHit. WARNING: the tracker pileup PSimHits are by
        # far the largest sim-hit family at PU200 (millions of hits/event); keeping the
        # merged tracker collection dominates the event size. For a truth graph that only
        # needs the calorimeter footprint, leave trackerHits empty.
        trackerHits=cms.VInputTag(
            cms.InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"),
            cms.InputTag("g4SimHits", "TrackerHitsPixelBarrelHighTof"),
            cms.InputTag("g4SimHits", "TrackerHitsPixelEndcapLowTof"),
            cms.InputTag("g4SimHits", "TrackerHitsPixelEndcapHighTof"),
            cms.InputTag("g4SimHits", "TrackerHitsTIBLowTof"),
            cms.InputTag("g4SimHits", "TrackerHitsTIBHighTof"),
            cms.InputTag("g4SimHits", "TrackerHitsTIDLowTof"),
            cms.InputTag("g4SimHits", "TrackerHitsTIDHighTof"),
            cms.InputTag("g4SimHits", "TrackerHitsTOBLowTof"),
            cms.InputTag("g4SimHits", "TrackerHitsTOBHighTof"),
            cms.InputTag("g4SimHits", "TrackerHitsTECLowTof"),
            cms.InputTag("g4SimHits", "TrackerHitsTECHighTof"),
        ),
        muonHits=cms.VInputTag(
            cms.InputTag("g4SimHits", "MuonDTHits"),
            cms.InputTag("g4SimHits", "MuonCSCHits"),
            cms.InputTag("g4SimHits", "MuonRPCHits"),
            cms.InputTag("g4SimHits", "MuonGEMHits"),
            cms.InputTag("g4SimHits", "MuonME0Hits"),
        ),
        mtdHits=cms.VInputTag(
            cms.InputTag("g4SimHits", "FastTimerHitsBarrel"),
            cms.InputTag("g4SimHits", "FastTimerHitsEndcap"),
        ),
        pileupBunchCrossings=cms.vint32(*pileupBunchCrossings),
        collapsePileupGen=cms.bool(collapsePileupGen),
        collapseSignalGen=cms.bool(False),
    )

    for out in process.outputModules_().values():
        out.outputCommands.append("keep TruthGraph_mix_*_*")
        # mergedHGCHits is the union of signal + all kept pileup HGCal PCaloHits
        # (O(1e5-1e6) hits, tens of MB/event at PU200). It is the dominant event-size
        # term of this feature and must bridge a split DIGI->RECO job, since the pileup
        # hits are gone after mixing; the RECO customise does not re-keep it. Drop this
        # line for a single-job DIGI+RECO where the hit index is built in the same process.
        out.outputCommands.append("keep *_mix_mergedHGCHits_*")
        out.outputCommands.append("keep *_mix_mergedEcalHits_*")
        out.outputCommands.append("keep *_mix_mergedHcalHits_*")
        out.outputCommands.append("keep *_mix_mergedTrackerHits_*")
        out.outputCommands.append("keep *_mix_mergedMuonHits_*")
        out.outputCommands.append("keep *_mix_mergedMtdHits_*")

    return process
