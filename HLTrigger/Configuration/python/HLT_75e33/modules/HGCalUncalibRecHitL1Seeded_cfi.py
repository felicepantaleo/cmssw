import FWCore.ParameterSet.Config as cms
from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHitProducer_cfi import HGCalUncalibRecHitProducer

HGCalUncalibRecHitL1Seeded = HGCalUncalibRecHitProducer.clone(
    HGCEEConfig = cms.PSet(
        adcNbits = cms.uint32(10),
        adcSaturation = cms.double(100),
        fCPerMIP = cms.vdouble(2.06, 3.43, 5.15),
        isSiFE = cms.bool(True),
        tdcNbits = cms.uint32(12),
        tdcOnset = cms.double(60),
        tdcSaturation = cms.double(10000),
        tofDelay = cms.double(-9),
        toaLSB_ns = cms.double(0.0244)
    ),
    HGCEEdigiCollection = cms.InputTag("hgcalDigisL1Seeded","EE"),
    HGCHEBConfig = cms.PSet(
        adcNbits = cms.uint32(10),
        adcSaturation = cms.double(68.75),
        fCPerMIP = cms.vdouble(1.0, 1.0, 1.0),
        isSiFE = cms.bool(True),
        tdcNbits = cms.uint32(12),
        tdcOnset = cms.double(55),
        tdcSaturation = cms.double(1000),
        tofDelay = cms.double(-14),
        toaLSB_ns = cms.double(0.0244)
    ),
    HGCHEBdigiCollection = cms.InputTag("hgcalDigisL1Seeded","HEback"),
    HGCHEFConfig = cms.PSet(
        adcNbits = cms.uint32(10),
        adcSaturation = cms.double(100),
        fCPerMIP = cms.vdouble(2.06, 3.43, 5.15),
        isSiFE = cms.bool(True),
        tdcNbits = cms.uint32(12),
        tdcOnset = cms.double(60),
        tdcSaturation = cms.double(10000),
        tofDelay = cms.double(-11),
        toaLSB_ns = cms.double(0.0244)
    ),
    HGCHEFdigiCollection = cms.InputTag("hgcalDigisL1Seeded","HEfront"),
    HGCHFNoseConfig = cms.PSet(
        adcNbits = cms.uint32(10),
        adcSaturation = cms.double(100),
        fCPerMIP = cms.vdouble(1.25, 2.57, 3.88),
        isSiFE = cms.bool(False),
        tdcNbits = cms.uint32(12),
        tdcOnset = cms.double(60),
        tdcSaturation = cms.double(10000),
        tofDelay = cms.double(-33),
        toaLSB_ns = cms.double(0.0244)
    )
)
