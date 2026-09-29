# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# DQM analyzers that compare truth::Branch to the legacy truth objects, and the
# cluster-to-TrackingParticle map they read. truthGraphDQMHarvester_cff holds the harvesting.
# globalValidation includes this file under enableTruth.

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

# The mixing accumulator chain builds the logical graph and the hit index at DIGI.
# They arrive at RECO through the input file. Do not import the signal-only build producers here:
# they would attach to the RECO process and shadow the DIGI-built products.

branchHGCalValidator = DQMEDAnalyzer(
    "BranchHGCalValidator",
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("mix"),  # merged raw graph, built at DIGI by the accumulator
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    caloParticles=cms.InputTag("mix", "MergedCaloTruth"),
    simClusters=cms.InputTag("mix", "MergedCaloTruth"),
    folder=cms.string("HGCAL/BranchValidator"),
    minPt=cms.double(1.0),
    maxEta=cms.double(3.0),
)

# Tracker validator. The reco track links a Branch to a TrackingParticle: the track matches
# Branches by shared tracker cells and TrackingParticles through ClusterTPAssociation.
# Phase-2 tracker: pixel and outer tracker (Phase2TrackerCluster1D), no strips.
from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import tpClusterProducer as _tpClusterProducer
truthTpClusterProducer = _tpClusterProducer.clone(
    pixelClusterSrc=cms.InputTag("siPixelClusters"),
    phase2OTClusterSrc=cms.InputTag("siPhase2Clusters"),
    pixelSimLinkSrc=cms.InputTag("simSiPixelDigis", "Pixel"),
    phase2OTSimLinkSrc=cms.InputTag("simSiPixelDigis", "Tracker"),
    trackingParticleSrc=cms.InputTag("mix", "MergedTrackTruth"),
    throwOnMissingCollections=cms.bool(False),
)

branchTrackingValidator = DQMEDAnalyzer(
    "BranchTrackingValidator",
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("mix"),  # merged raw graph, built at DIGI by the accumulator
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    tracks=cms.InputTag("generalTracks"),
    clusterTPMap=cms.InputTag("truthTpClusterProducer"),
    folder=cms.string("Tracking/BranchValidator"),
    minPt=cms.double(0.9),
    maxEta=cms.double(3.0),
)

# Content of the graph per interaction, to show whether it is built as intended.
truthGraphSummaryValidator = DQMEDAnalyzer(
    "TruthGraphSummaryValidator",
    src=cms.InputTag("truthLogicalGraphProducer"),
    folder=cms.string("TruthInfo/Graph"),
)

# The EDProducers go in the prevalidation Path, the analyzers in the validation EndPath.
truthGraphValidationProducers = cms.Sequence(truthTpClusterProducer)
truthGraphValidationAnalyzers = cms.Sequence(
    branchHGCalValidator
    + branchTrackingValidator
    + truthGraphSummaryValidator
)
