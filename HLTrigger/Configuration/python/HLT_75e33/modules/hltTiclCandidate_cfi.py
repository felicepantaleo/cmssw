import FWCore.ParameterSet.Config as cms

hltTiclCandidate = cms.EDProducer("TICLCandidateProducer",
    buildTrackOnlyCandidates = cms.bool(False),
    delta_tk_gsf = cms.double(0.05),
    detector = cms.string('HGCAL'),
    gsf_tracks = cms.InputTag("electronGsfTracks"),
    interpretations = cms.InputTag("hltTiclTracksterInterpretations"),
    mightGet = cms.optional.untracked.vstring,
    propagator = cms.string('PropagatorWithMaterial'),
    timingQualityThreshold = cms.double(0.5),
    timingSoA = cms.InputTag("mtdSoA"),
    trackOnlyDeltaR = cms.double(0.1),
    trackOnlyNearbyEnergyFloor = cms.double(2),
    trackOnlyNearbyEnergyFraction = cms.double(0.2),
    tracks = cms.InputTag("hltGeneralTracks"),
    useGsfTracks = cms.bool(False),
    useMTDTiming = cms.bool(False),
    useTimingAverage = cms.bool(False)
)
