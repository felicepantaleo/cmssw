# Multi-particle overlap gun for linking and candidate fake-rate studies: eight
# mixed particles (EM + charged and neutral hadrons) shot FROM THE VERTEX into a
# narrow eta/phi cone of one HGCAL endcap, so charged particles produce tracker
# tracks (a CloseBy gun at the calorimeter face does not) and the showers overlap
# statistically. Energies concentrate around the historical candidate cuts
# (2-10 GeV), so opened track and trackster thresholds are testable against
# truth-labelled fakes.
import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(22, 11, -211, 211, 130, 22, -11, 211),
        MinEta = cms.double(1.8),
        MaxEta = cms.double(2.4),
        MinPhi = cms.double(0.0),
        MaxPhi = cms.double(0.5),
        MinE = cms.double(2.0),
        MaxE = cms.double(60.0)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('overlapping multi-species particles from the vertex'),
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
)
