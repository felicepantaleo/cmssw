import FWCore.ParameterSet.Config as cms


tracksterPIDComparisonAnalyzer = cms.EDAnalyzer(
    "TracksterPIDComparisonAnalyzer",
    reference=cms.InputTag(""),
    candidate=cms.InputTag(""),
    absoluteTolerance=cms.double(1.e-5),
    relativeTolerance=cms.double(1.e-4),
    maxMismatchesToPrint=cms.uint32(10),
    failOnMismatch=cms.bool(False),
    reportEveryEvent=cms.bool(True),
)

# Optional CLUE3DHigh validation; schedule explicitly on an EndPath.
ticlTracksterPIDComparison = tracksterPIDComparisonAnalyzer.clone(
    reference="ticlTrackstersToSoAProducer",
    candidate="ticlTracksterInferenceProducer",
    # Start loose enough to accommodate the different inference runtimes.
    # Tighten these after inspecting the measured maximum differences.
    absoluteTolerance=1.e-5,
    relativeTolerance=1.e-4,
    maxMismatchesToPrint=3,
    failOnMismatch=False,
    reportEveryEvent=True,
)
