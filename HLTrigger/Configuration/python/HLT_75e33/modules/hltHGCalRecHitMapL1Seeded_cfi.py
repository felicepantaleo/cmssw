import FWCore.ParameterSet.Config as cms

hltHGCalRecHitMapL1Seeded = cms.EDProducer("RecHitMapProducer",
    hgcalOnly = cms.bool(True),
    hits = cms.VInputTag("hltHGCalRecHitL1Seeded:HGCEERecHits", "hltHGCalRecHitL1Seeded:HGCHEFRecHits", "hltHGCalRecHitL1Seeded:HGCHEBRecHits")
)
