# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# Customise of the pileup-aware truth that the DIGI step builds by default
# (truthGraphMixedDigi_cff, registered under enableTruth).

import FWCore.ParameterSet.Config as cms


def customiseTruthReduced(process):
    """Reduced variant: drop the tracker hits from the DIGI-built truth and keep only
    the Calo and Muon channels in the hit index. Apply it at the DIGI step for a
    production that does not need track-based candidate matching."""
    acc = process.mix.digitizers.truthGraph
    acc.trackerHits = cms.VInputTag()
    idx = process.truthLogicalGraphHitIndexProducer
    idx.subdetectors = cms.vstring("Calo", "Muon")
    # The pruning's detector scope stays equal to the index's, otherwise it prunes on
    # tracker hits that are no longer accumulated.
    process.truthLogicalGraphProducer.trackerSimHitCollections = cms.VInputTag()
    return process
