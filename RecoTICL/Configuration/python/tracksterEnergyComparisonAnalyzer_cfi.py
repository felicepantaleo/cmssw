import FWCore.ParameterSet.Config as cms


tracksterEnergyComparisonAnalyzer = cms.EDAnalyzer(
    "TracksterEnergyComparisonAnalyzer",
    reference=cms.InputTag(""),
    candidate=cms.InputTag(""),
    absoluteTolerance=cms.double(1.e-5),
    relativeTolerance=cms.double(1.e-4),
    maxMismatchesToPrint=cms.uint32(10),
    failOnMismatch=cms.bool(False),
    reportEveryEvent=cms.bool(True),
)

# Optional CLUE3DHigh validation; load this fragment and schedule explicitly.
ticlTracksterEnergyComparison = tracksterEnergyComparisonAnalyzer.clone(
    reference="ticlTrackstersToSoAProducer",
    candidate="ticlTracksterInferenceProducer",
    absoluteTolerance=1.e-5,
    relativeTolerance=1.e-4,
    maxMismatchesToPrint=3,
    failOnMismatch=False,
    reportEveryEvent=True,
)
