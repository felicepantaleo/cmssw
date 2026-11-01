import FWCore.ParameterSet.Config as cms


def customiseTICLEGammaEnergy(process):
    """The TICLv6 photon and electron candidates take the energy of their e/gamma supercluster
    (ticlEGammaSuperClusters:tracksterEnergy)."""
    process.ticlCandidate.eGammaEnergy = cms.InputTag("ticlEGammaSuperClusters", "tracksterEnergy")
    process.ticlCandidate.eGammaTracksters = cms.InputTag("ticlEGammaTracksters")
    return process
