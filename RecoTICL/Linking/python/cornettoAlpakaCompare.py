import FWCore.ParameterSet.Config as cms


def compareCornettoBackends(process):
    """Run CPU Cornetto and device Cornetto on the SAME single input collection in
    the SAME job, and require the emitted components to be identical.

    tracksters_collections is reduced to ticlTrackstersCLUE3DHigh alone so that both
    paths see exactly the same input: the CPU plugin links a MultiSpan of several
    collections, while the device path consumes one SoA. Comparing them on a
    multi-collection input would compare two different index spaces.
    """
    process.ticlTracksterLinks.tracksters_collections = cms.VInputTag("ticlTrackstersCLUE3DHigh")
    process.ticlTracksterLinks.linkingPSet = cms.PSet(
        type=cms.string("Cornetto"),
        algo_verbosity=cms.int32(0),
        etaWindow=cms.double(0.3),
        maxLongitudinalDistance=cms.double(60.0),
        transverseRadius0=cms.double(5.0),
        transverseSlope=cms.double(0.05),
        timeCompatibilityNSigma=cms.double(3.0),
    )

    from RecoTICL.Linking.legacyTracksterToSoAProducer_cfi import legacyTracksterToSoAProducer
    from RecoTICL.Linking.tracksterLinksFromComponentsProducer_cfi import tracksterLinksFromComponentsProducer
    from RecoTICL.Linking.cornettoBackendComparator_cfi import cornettoBackendComparator

    process.tracksterSoACLUE3D = legacyTracksterToSoAProducer.clone(
        tracksters=cms.InputTag("ticlTrackstersCLUE3DHigh")
    )

    from RecoTICL.Linking.tracksterLinkingByCornettoAlpakaProducer_cfi import (
        tracksterLinkingByCornettoAlpakaProducer,
    )

    process.ticlCornettoLinksAlpaka = tracksterLinkingByCornettoAlpakaProducer.clone(
        tracksterSoA=cms.InputTag("tracksterSoACLUE3D"),
        etaWindow=cms.double(0.3),
        maxLongitudinalDistance=cms.double(60.0),
        transverseRadius0=cms.double(5.0),
        transverseSlope=cms.double(0.05),
        timeCompatibilityNSigma=cms.double(3.0),
    )
    process.ticlCornettoLinksHost = tracksterLinksFromComponentsProducer.clone(
        tracksters=cms.InputTag("ticlTrackstersCLUE3DHigh"),
        components=cms.InputTag("ticlCornettoLinksAlpaka"),
    )
    process.cornettoCompare = cornettoBackendComparator.clone(
        reference=cms.InputTag("ticlTracksterLinks", "linkedTracksterIdToInputTracksterId"),
        test=cms.InputTag("ticlCornettoLinksHost"),
    )

    process.cornettoComparePath = cms.Path(
        process.tracksterSoACLUE3D
        + process.ticlCornettoLinksAlpaka
        + process.ticlCornettoLinksHost
        + process.cornettoCompare
    )
    process.schedule.append(process.cornettoComparePath)
    return process
