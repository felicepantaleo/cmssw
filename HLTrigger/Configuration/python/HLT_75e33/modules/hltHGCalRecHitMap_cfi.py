import FWCore.ParameterSet.Config as cms

hltHGCalRecHitMap = cms.EDProducer("RecHitMapProducer",
    hgcalOnly = cms.bool(True),
    hits = cms.VInputTag("hltHGCalRecHit:HGCEERecHits", "hltHGCalRecHit:HGCHEFRecHits", "hltHGCalRecHit:HGCHEBRecHits")
)
