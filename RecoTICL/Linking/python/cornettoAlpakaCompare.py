import FWCore.ParameterSet.Config as cms


def compareCornettoBackends(process):
    """Run CPU Cornetto and device Cornetto over the SAME multi-collection input in
    the SAME job, and require the emitted components to be identical.

    Both paths concatenate the full tracksters_collections (CLUE3DHigh + Recovery)
    through edm::MultiSpan, so they link over one shared global index space. No
    single-collection restriction: the device SoA producer flattens the same
    MultiSpan the CPU plugin links over, so the comparison is on the real config.
    """
    collections = process.ticlTracksterLinks.tracksters_collections

    process.ticlTracksterLinks.linkingPSet = cms.PSet(
        type=cms.string("Cornetto"),
        algo_verbosity=cms.int32(0),
        etaWindow=cms.double(0.3),
        maxLongitudinalDistance=cms.double(30.0),
        transverseRadius0=cms.double(5.0),
        transverseSlope=cms.double(0.05),
        timeCompatibilityNSigma=cms.double(3.0),
        maxLongitudinalSlope=cms.double(0.3),
        longitudinalZRef=cms.double(320.0),
    )

    from RecoTICL.Linking.legacyTracksterToSoAProducer_cfi import legacyTracksterToSoAProducer
    from RecoTICL.Linking.tracksterLinksFromComponentsProducer_cfi import tracksterLinksFromComponentsProducer
    from RecoTICL.Linking.cornettoBackendComparator_cfi import cornettoBackendComparator
    from RecoTICL.Linking.tracksterLinkingByCornettoAlpakaProducer_cfi import (
        tracksterLinkingByCornettoAlpakaProducer,
    )

    process.tracksterSoACLUE3D = legacyTracksterToSoAProducer.clone(tracksters_collections=collections)
    process.ticlCornettoLinksAlpaka = tracksterLinkingByCornettoAlpakaProducer.clone(
        tracksterSoA=cms.InputTag("tracksterSoACLUE3D"),
        etaWindow=cms.double(0.3),
        maxLongitudinalDistance=cms.double(30.0),
        transverseRadius0=cms.double(5.0),
        transverseSlope=cms.double(0.05),
        timeCompatibilityNSigma=cms.double(3.0),
        maxLongitudinalSlope=cms.double(0.3),
        longitudinalZRef=cms.double(320.0),
    )
    process.ticlCornettoLinksHost = tracksterLinksFromComponentsProducer.clone(
        tracksters_collections=collections,
        components=cms.InputTag("ticlCornettoLinksAlpaka"),
    )
    process.cornettoCompare = cornettoBackendComparator.clone(
        reference=cms.InputTag("ticlTracksterLinks", "linkedTracksterIdToInputTracksterId"),
        test=cms.InputTag("ticlCornettoLinksHost", "linkedTracksterIdToInputTracksterId"),
    )

    process.cornettoComparePath = cms.Path(
        process.tracksterSoACLUE3D
        + process.ticlCornettoLinksAlpaka
        + process.ticlCornettoLinksHost
        + process.cornettoCompare
    )
    process.schedule.append(process.cornettoComparePath)
    return process
