import FWCore.ParameterSet.Config as cms

# Sources of Geant4 hits for the HGCAL portable digitizer, with the TOF delay of each section
hgcalSimHitListSources = cms.VPSet(
    cms.PSet(det=cms.int32(8), iscalib=cms.bool(False), hits=cms.InputTag("g4SimHits", "HGCHitsEE"), tof_delay=cms.double(-9)),
    cms.PSet(det=cms.int32(8), iscalib=cms.bool(True), hits=cms.InputTag("g4SimHits", "CalibrationHGCHitsEE"), tof_delay=cms.double(-9)),
    cms.PSet(det=cms.int32(9), iscalib=cms.bool(False), hits=cms.InputTag("g4SimHits", "HGCHitsHEfront"), tof_delay=cms.double(-11)),
    cms.PSet(det=cms.int32(9), iscalib=cms.bool(True), hits=cms.InputTag("g4SimHits", "CalibrationHGCHitsHEfront"), tof_delay=cms.double(-11)),
    cms.PSet(det=cms.int32(10), iscalib=cms.bool(False), hits=cms.InputTag("g4SimHits", "HGCHitsHEback"), tof_delay=cms.double(-14)),
)

# Entry of mix.digitizers: collects the hits of the signal and of the pileup into one list
hgcalSimHitListDigitizer = cms.PSet(
    accumulatorType=cms.string("HGCalSimHitListMixMod"),
    instanceLabel=cms.string("HGCalSimHitList"),
    produceExtra=cms.bool(False),
    indexingSource=cms.ESInputTag(),
    cellMappingSource=cms.ESInputTag(),
    g4HitsConfig=hgcalSimHitListSources,
)


def customiseHGCalSimHitListInMix(process):
    """Adds the hit list of the HGCAL portable digitizer to mix.digitizers. The legacy HGCAL digitizers stay.
    mix reads the pileup hits and shifts them in time only for the collections in mix.mixObjects.mixCH."""
    process.mix.digitizers.hgcalSimHitList = hgcalSimHitListDigitizer.clone()
    mixCH = process.mix.mixObjects.mixCH
    for source in hgcalSimHitListSources:
        if source.hits not in mixCH.input:
            mixCH.input.append(source.hits)
            mixCH.subdets.append(source.hits.productInstanceLabel)
    return process
