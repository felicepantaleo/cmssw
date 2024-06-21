import FWCore.ParameterSet.Config as cms
from Geometry.HGCalGeometry.TICLGeomESProducer_cfi import TICLGeomESProducer

# ECAL
ticlGeomESProducerECAL = TICLGeomESProducer.clone(
    label = "ECAL",
    detectors = ["ECAL"]
)

# HCAL
ticlGeomESProducerHCAL = TICLGeomESProducer.clone(
    label = "HCAL",
    detectors = ["HCAL"]
)

# HGCal
ticlGeomESProducerHGCal = TICLGeomESProducer.clone(
    label = "HGCal",
    detectors = ["HGCal"]
)

# HFNose
ticlGeomESProducerHFNose = TICLGeomESProducer.clone(
    label = "HFNose",
    detectors = ["HFNose"]
)