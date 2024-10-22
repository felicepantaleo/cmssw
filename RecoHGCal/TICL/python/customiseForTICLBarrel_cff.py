import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.recHitMapProducer_cfi import recHitMapProducer
from RecoParticleFlow.PFClusterProducer.barrelLayerClusters_cfi import barrelLayerClusters
from RecoHGCal.TICL.ticlDumper_cfi import ticlDumper as ticlDumper_

from SimCalorimetry.HGCalAssociatorProducers.barrelLCToCPAssociatorByEnergyScoreProducer_cfi import barrelLCToCPAssociatorByEnergyScoreProducer as barrelLCToCPAssociatorByEnergyScoreProducer
from SimCalorimetry.HGCalAssociatorProducers.barrelLCToSCAssociatorByEnergyScoreProducer_cfi import barrelLCToSCAssociatorByEnergyScoreProducer as barrelLCToSCAssociatorByEnergyScoreProducer
from SimCalorimetry.HGCalAssociatorProducers.LCToCPAssociation_cfi import barrelLayerClusterCaloParticleAssociation as barrelLayerClusterCaloParticleAssociationProducer
from SimCalorimetry.HGCalAssociatorProducers.LCToSCAssociation_cfi import barrelLayerClusterSimClusterAssociation as barrelLayerClusterSimClusterAssociationProducer

def customiseForTICLBarrel(process, pfComparison=False):
    
    process.recHitMapProducer.hgcalOnly = cms.bool(False)

    # Parameters for CLUE in the barrel
    process.barrelLayerClusters = barrelLayerClusters.clone(
        timeClname = cms.string('timeLayerCluster'),
        nHitsTime = cms.uint32(3),
        EBInput = cms.InputTag('particleFlowRecHitECAL'),
        HBInput = cms.InputTag('particleFlowRecHitHBHE'),
        ebplugin = cms.PSet(
            deltac = cms.double(1.8*0.0175),
            kappa = cms.double(3.5),
            maxLayerIndex = cms.int32(0),
            outlierDeltaFactor = cms.double(2.7*0.0175),
            fractionCutoff = cms.double(0),
            doSharing = cms.bool(False),
            type = cms.string('EBCLUE')
        ),
        hbplugin = cms.PSet(
            kappa = cms.double(0),
            maxLayerIndex = cms.int32(4),
            outlierDeltaFactor = cms.double(5*0.087),
            deltac = cms.double(3*0.087),
            fractionCutoff = cms.double(0),
            doSharing = cms.bool(False),
            type = cms.string('HBCLUE')
        ),
        timeResolutionCalc = cms.PSet(
            noiseTerm = cms.double(1.10889),
            constantTerm = cms.double(0.428192),
            corrTermLowE = cms.double(0.0510871),
            threshLowE = cms.double(0.5),
            constantTermLowE = cms.double(0),
            noiseTermLowE = cms.double(1.31883),
            threshHighE = cms.double(5)
        )
    )
    if not pfComparison:                                                                                           
        process.barrelLayerClustersTask = cms.Task(process.barrelLayerClusters)
    else:
        process.barrelLayerClustersTask = cms.Task(process.barrelLayerClusters, process.lcFromPFClusterProducer)

    process.ticlBarrel = cms.Path(process.barrelLayerClustersTask)
    

    ## associators
    process.barrelLCToCPAssociatorByEnergyScoreProducer = barrelLCToCPAssociatorByEnergyScoreProducer.clone()
    process.barrelLayerClusterCaloParticleAssociationProducer = barrelLayerClusterCaloParticleAssociationProducer.clone()

    process.barrelLCToSCAssociatorByEnergyScoreProducer = barrelLCToSCAssociatorByEnergyScoreProducer.clone()
    process.barrelLayerClusterSimClusterAssociationProducer = barrelLayerClusterSimClusterAssociationProducer.clone()

    process.ticlAssociators = cms.Path(process.recHitMapProducer
                                       +process.barrelLCToCPAssociatorByEnergyScoreProducer
                                       +process.barrelLCToSCAssociatorByEnergyScoreProducer
                                       +process.barrelLayerClusterCaloParticleAssociationProducer
                                       +process.barrelLayerClusterSimClusterAssociationProducer
    )

    #process.ticlDumper = ticlDumper_.clone()

    process.consumer = cms.EDAnalyzer("GenericConsumer",
        eventProducts = cms.untracked.vstring(['barrelLayerClusters',
                                               'barrelLayerClusterCaloParticleAssociationProducer',
                                               'barrelLayerClusterSimClusterAssociationProducer'])
    )


    process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput
                                                  #+process.ticlDumper
                                                  +process.consumer)

    process.schedule = cms.Schedule(process.ticlBarrel
                                    ,process.ticlAssociators
                                    ,process.FEVTDEBUGHLToutput_step)
    
    return process
