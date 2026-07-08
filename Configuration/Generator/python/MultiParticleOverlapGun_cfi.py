# Dense multi-particle overlap gun for linking and candidate fake-rate studies:
# sixteen mixed particles (EM + charged and neutral hadrons) from the vertex,
# aimed so that ALL pairwise (eta,phi) distances AT THE CALORIMETER FACE are at
# most 0.1 (charged particles are helix-precompensated by the gun). Charged
# particles produce tracker tracks; energies concentrate around the historical
# candidate cuts (2-10 GeV), so opened track and trackster thresholds are
# testable against truth-labelled fakes at maximum shower confusion.
import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("CaloFaceClusteredGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(22, 11, -211, 211, 130, 22, -11, 211),
        NParticles = cms.int32(16),
        MaxDeltaR = cms.double(0.1),
        ZFace = cms.double(322.0),
        BField = cms.double(3.8),
        MinEta = cms.double(1.8),
        MaxEta = cms.double(2.5),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE = cms.double(2.0),
        # aligned with PartID: electrons need ~15 GeV to be aimable (brems),
        # charged hadrons ~4 GeV (loopers/scattering), neutrals unconstrained
        MinEPart = cms.vdouble(2.0, 15.0, 4.0, 4.0, 2.0, 2.0, 15.0, 4.0),
        MaxE = cms.double(60.0)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('sixteen mixed particles clustered at the HGCAL face'),
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
)
