import FWCore.ParameterSet.Config as cms


def useCornetto(process):
    """Swap TICL trackster linking from Skeletons to Cornetto and expose the linked
    tracksters as a portable SoA via LegacyTracksterToSoAProducer.

    Only the linking plugin changes; every other TICL parameter is untouched, so a
    run with and without this customise differ by the linking algorithm alone.
    """
    process.ticlTracksterLinks.linkingPSet = cms.PSet(
        type=cms.string("Cornetto"),
        algo_verbosity=cms.int32(0),
        etaWindow=cms.double(0.3),
        maxLongitudinalDistance=cms.double(60.0),
        transverseRadius0=cms.double(5.0),
        transverseSlope=cms.double(0.05),
        timeCompatibilityNSigma=cms.double(3.0),
    )

    from RecoTICL.Linking.legacyTracksterToSoAProducer_cfi import (
        legacyTracksterToSoAProducer,
    )

    # The SoA the device-side Cornetto would consume: the concatenation of the same
    # collections the CPU linking step links over (CLUE3DHigh + Recovery), and one
    # on the linked output.
    process.tracksterSoACLUE3D = legacyTracksterToSoAProducer.clone(
        tracksters_collections=process.ticlTracksterLinks.tracksters_collections
    )
    process.tracksterSoALinks = legacyTracksterToSoAProducer.clone(
        tracksters_collections=cms.VInputTag("ticlTracksterLinks")
    )
    process.tracksterSoATask = cms.Task(
        process.tracksterSoACLUE3D, process.tracksterSoALinks
    )
    process.schedule.associate(process.tracksterSoATask)

    if hasattr(process, "FEVTDEBUGHLToutput"):
        process.FEVTDEBUGHLToutput.outputCommands.extend(
            [
                "keep *_tracksterSoACLUE3D_*_*",
                "keep *_tracksterSoALinks_*_*",
                "keep *_ticlTrackstersCLUE3DHigh_*_*",
                "keep *_ticlTracksterLinks_*_*",
            ]
        )

    return process
