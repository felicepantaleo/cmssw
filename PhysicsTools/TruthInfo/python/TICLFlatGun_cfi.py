# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

# Single-particle flat-pt gun for the TICLCandidate physics-performance scan:
# one particle per endcap, flat pt in [2, 200] GeV, eta in [1.5, 3.1] (the HGCAL
# acceptance). The fired species is taken from the TICL_GUN_PDGID environment
# variable so one fragment drives every per-species job. AddAntiParticle fires the
# charge-conjugate into the opposite endcap, doubling the statistics and covering
# both z sides. Used as the GEN fragment of the enableTruth D120 no-PU workflow.

import os
import FWCore.ParameterSet.Config as cms

_pdgId = int(os.environ.get("TICL_GUN_PDGID", "22"))

generator = cms.EDProducer(
    "FlatRandomPtGunProducer",
    PGunParameters=cms.PSet(
        PartID=cms.vint32(_pdgId),
        MinPt=cms.double(2.0),
        MaxPt=cms.double(200.0),
        MinEta=cms.double(1.5),
        MaxEta=cms.double(3.1),
        MinPhi=cms.double(-3.14159265359),
        MaxPhi=cms.double(3.14159265359),
    ),
    AddAntiParticle=cms.bool(True),
    Verbosity=cms.untracked.int32(0),
    firstRun=cms.untracked.uint32(1),
    psethack=cms.string("single pdgId %d, flat pt 2-200, eta 1.5-3.1" % _pdgId),
)

ProductionFilterSequence = cms.Sequence(generator)
