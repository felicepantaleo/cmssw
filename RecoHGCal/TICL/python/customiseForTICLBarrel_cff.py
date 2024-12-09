import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.recHitMapProducer_cfi import recHitMapProducer as _recHitMapProducer
from RecoParticleFlow.PFClusterProducer.barrelLayerClusters_cfi import barrelLayerClusters as _barrelLayerClusters
from RecoHGCal.TICL.lcFromPFClusterProducer_cfi import lcFromPFClusterProducer as _lcFromPFClusterProducer
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.ticlSeedingRegionProducer_cfi import ticlSeedingRegionProducer as _ticlSeedingRegionProducer
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer as _ticlLayerTileProducer
from RecoHGCal.TICL.simTrackstersProducer_cfi import simTrackstersProducer as _simTrackstersProducer
from RecoHGCal.TICL.ticlDumper_cfi import ticlDumper as _ticlDumper

from SimCalorimetry.HGCalAssociatorProducers.barrelLCToCPAssociatorByEnergyScoreProducer_cfi import barrelLCToCPAssociatorByEnergyScoreProducer as barrelLCToCPAssociatorByEnergyScoreProducer
from SimCalorimetry.HGCalAssociatorProducers.barrelLCToSCAssociatorByEnergyScoreProducer_cfi import barrelLCToSCAssociatorByEnergyScoreProducer as barrelLCToSCAssociatorByEnergyScoreProducer
from SimCalorimetry.HGCalAssociatorProducers.LCToCPAssociation_cfi import barrelLayerClusterCaloParticleAssociation as barrelLayerClusterCaloParticleAssociationProducer
from SimCalorimetry.HGCalAssociatorProducers.LCToSCAssociation_cfi import barrelLayerClusterSimClusterAssociation as barrelLayerClusterSimClusterAssociationProducer

from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import allBarrelTrackstersToSimTrackstersAssociationsByLCs as _allBarrelTrackstersToSimTrackstersAssociationsByLCs 
from SimCalorimetry.HGCalAssociatorProducers.LCToTSAssociator_cfi import allBarrelLayerClusterToTracksterAssociations as _allBarrelLayerClusterToTracksterAssociations
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociationByHits_cfi import allBarrelTrackstersToSimTrackstersAssociationsByHits as _allBarrelTrackstersToSimTrackstersAssociationsByHits
from SimCalorimetry.HGCalAssociatorProducers.barrelHitToSimClusterCaloParticleAssociator_cfi import barrelHitToSimClusterCaloParticleAssociator as _barrelHitToSimClusterCaloParticleAssociator
from SimCalorimetry.HGCalAssociatorProducers.SimClusterToCaloParticleAssociatorProducer_cfi import SimClusterToCaloParticleAssociatorProducer as _SimClusterToCaloParticleAssociator
from SimCalorimetry.HGCalAssociatorProducers.HitToTracksterAssociation_cfi import allHitToBarrelTracksterAssociations as _allHitToBarrelTracksterAssociations

def customiseForTICLBarrel_legacyPFClusters(process, pfComparison=False):
    
    process.recHitMapProducer.hgcalOnly = cms.bool(False)

    # Parameters for CLUE in the barrel
    process.barrelLayerClusters = _barrelLayerClusters.clone(
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
    
    process.lcFromPFClusterProducer = _lcFromPFClusterProducer.clone()
    process.barrelLayerClustersTask = cms.Task(process.barrelLayerClusters, process.lcFromPFClusterProducer)

    process.ticlBarrelTracksters = _trackstersProducer.clone(
        detector = "Barrel",
        layer_clusters = "barrelLayerClusters",
        time_layerclusters = "barrelLayerClusters:timeLayerCluster",
        filtered_mask = "barrelLayerClusters:InitialLayerClustersMask",
        original_mask = "barrelLayerClusters:InitialLayerClustersMask",
        seeding_regions = "ticlSeedingGlobal",
        itername = "CLUE3D",
        patternRecognitionBy = "CLUE3D",
        pluginPatternRecognitionByFastJet = dict(
            antikt_radius = 0.1,
            minNumLayerCluster = 0,
            algo_verbosity = 2
        ),
        pluginPatternRecognitionByCLUE3D = dict(
            criticalDensity = cms.vdouble(0.5, 0.5, 0.5),
            criticalSelfDensity = cms.vdouble(0., 0., 0.),
            criticalEtaPhiDistance = cms.vdouble(7*0.087, 7*0.087, 7*0.087, 7*0.087),
            densityEtaPhiDistanceSqr = cms.vdouble(0.37, 0.37, 0.37, 0.37),
            nearestHigherOnSameLayer = cms.bool(False),
            densityOnSameLayer = cms.bool(False),
            minNumLayerCluster = cms.vint32(1, 1, 1),
            useAbsoluteProjectiveScale = cms.bool(False),
            densitySiblingLayers = cms.vint32(4, 4, 4)
        )
    )
    process.ticlSeedingGlobal = _ticlSeedingRegionProducer.clone()
    process.ticlLayerTileProducer = _ticlLayerTileProducer.clone(
        detector = cms.string("Barrel")
    )

    process.ticlBarrelSimTracksters = _simTrackstersProducer.clone(
        layer_clusters = cms.InputTag("barrelLayerClusters"),
        filtered_mask = cms.InputTag("barrelLayerClusters:InitialLayerClustersMask"),
        time_layerclusters = cms.InputTag("barrelLayerClusters:timeLayerCluster"),
        layerClusterCaloParticleAssociator = cms.InputTag("barrelLayerClusterCaloParticleAssociationProducer"),
        layerClusterSimClusterAssociator = cms.InputTag("barrelLayerClusterSimClusterAssociationProducer"),
        detector = cms.string("Barrel"),
    )

    process.ticlBarrelTrackstersTask = cms.Task(process.ticlLayerTileProducer
                                                ,process.ticlSeedingGlobal
                                                ,process.ticlBarrelTracksters)

    process.ticlBarrel = cms.Path(process.barrelLayerClustersTask,process.ticlBarrelTrackstersTask)
    
    ticlBarrelTrackstersLabels = ["ticlBarrelTracksters"]
    ticlBarrelSimTrackstersLabels = ["ticlBarrelSimTrackster"]

    ## associators

    process.barrelLCToCPAssociatorByEnergyScoreProducer = barrelLCToCPAssociatorByEnergyScoreProducer.clone()
    process.barrelLayerClusterCaloParticleAssociationProducer = barrelLayerClusterCaloParticleAssociationProducer.clone()

    process.barrelLCToSCAssociatorByEnergyScoreProducer = barrelLCToSCAssociatorByEnergyScoreProducer.clone()
    process.barrelLayerClusterSimClusterAssociationProducer = barrelLayerClusterSimClusterAssociationProducer.clone()


    ## associators for pf
    process.barrelLCToCPAssociatorByEnergyScoreProducerPF = barrelLCToCPAssociatorByEnergyScoreProducer.clone()
    process.barrelLayerClusterCaloParticleAssociationProducerPF = barrelLayerClusterCaloParticleAssociationProducer.clone(
        label_lc = cms.InputTag("lcFromPFClusterProducer")
    )

    process.barrelLCToSCAssociatorByEnergyScoreProducerPF = barrelLCToCPAssociatorByEnergyScoreProducer.clone()
    process.barrelLayerClusterSimClusterAssociationProducerPF = barrelLayerClusterSimClusterAssociationProducer.clone(
        label_lc = cms.InputTag("lcFromPFClusterProducer")
    )

    process.allBarrelLayerClusterToTracksterAssociations = _allBarrelLayerClusterToTracksterAssociations.clone()
    process.SimClusterToCaloParticleAssociator = _SimClusterToCaloParticleAssociator.clone()
    process.barrelHitToSimClusterCaloParticleAssociator = _barrelHitToSimClusterCaloParticleAssociator.clone()

    # TS-STS associations by LCs
    process.allBarrelTrackstersToSimTrackstersAssociationsByLCs = _allBarrelTrackstersToSimTrackstersAssociationsByLCs.clone()

    # TS-STS associations by hits
    process.allHitToBarrelTracksterAssociations = _allHitToBarrelTracksterAssociations.clone()
    process.allBarrelTrackstersToSimTrackstersAssociationsByHits = _allBarrelTrackstersToSimTrackstersAssociationsByHits.clone(
        hitToCaloParticleMap = cms.InputTag('barrelHitToSimClusterCaloParticleAssociator', 'hitToCaloParticleMap'),
        hitToSimClusterMap = cms.InputTag('barrelHitToSimClusterCaloParticleAssociator', 'hitToSimClusterMap'),
        hitToTracksterMap = cms.string('allHitToBarrelTracksterAssociations')
    )
    
    process.ticlAssociators = cms.Path(process.recHitMapProducer
                                       +process.barrelLCToCPAssociatorByEnergyScoreProducer
                                       +process.barrelLCToSCAssociatorByEnergyScoreProducer
                                       +process.barrelLCToCPAssociatorByEnergyScoreProducerPF
                                       +process.barrelLCToSCAssociatorByEnergyScoreProducerPF
                                       +process.barrelLayerClusterCaloParticleAssociationProducer
                                       +process.barrelLayerClusterSimClusterAssociationProducer
                                       +process.barrelLayerClusterCaloParticleAssociationProducerPF
                                       +process.barrelLayerClusterSimClusterAssociationProducerPF
                                       +process.ticlBarrelSimTracksters
                                       +process.barrelHitToSimClusterCaloParticleAssociator
                                       +process.SimClusterToCaloParticleAssociator
                                       +process.allBarrelLayerClusterToTracksterAssociations
                                       +process.allHitToBarrelTracksterAssociations
                                       +process.allBarrelTrackstersToSimTrackstersAssociationsByLCs
                                       +process.allBarrelTrackstersToSimTrackstersAssociationsByHits
    )

    process.ticlDumper = _ticlDumper.clone(
        tracksterCollections = [*[cms.PSet(treeName=cms.string(label), inputTag=cms.InputTag(label)) for label in ticlBarrelTrackstersLabels],
            cms.PSet(
                treeName = cms.string("simtrackstersSC"),
                inputTag = cms.InputTag("ticlBarrelSimTracksters"),
                tracksterType = cms.string("SimTracksterSC")
            ),
            cms.PSet(
                treeName = cms.string("simtrackstersCP"),
                inputTag = cms.InputTag("ticlBarrelSimTracksters", "fromCPs"),
                tracksterType = cms.string("SimTracksterCP")
            ),
        ],
        associators = [
            cms.PSet(
                branchName = cms.string("tsToStsByLC"),
                suffix = cms.string("CP"),
                associatorRecoToSimInputTag = cms.InputTag("allBarrelTrackstersToSimTrackstersAssociationsByLCs", "ticlBarrelTrackstersToticlBarrelSimTrackstersfromCPs"),
                associatorSimToRecoInputTag = cms.InputTag("allBarrelTrackstersToSimTrackstersAssociationsByLCs", "ticlBarrelSimTrackstersfromCPsToticlBarrelTracksters")
            ),
            cms.PSet(
                branchName = cms.string("tsToStsByLC"),
                suffix = cms.string("SC"),
                associatorRecoToSimInputTag = cms.InputTag("allBarrelTrackstersToSimTrackstersAssociationsByLCs", "ticlBarrelTrackstersToticlBarrelSimTracksters"),
                associatorSimToRecoInputTag = cms.InputTag("allBarrelTrackstersToSimTrackstersAssociationsByLCs", "ticlBarrelSimTrackstersToticlBarrelTracksters")
            ),
             cms.PSet(                                                                                                                                                                
                 branchName = cms.string("tsToStsByHit"),
                 suffix = cms.string("CP"),
                 associatorRecoToSimInputTag = cms.InputTag("allBarrelTrackstersToSimTrackstersAssociationsByHits", "ticlBarrelTrackstersToticlBarrelSimTrackstersfromCPs"),
                 associatorSimToRecoInputTag = cms.InputTag("allBarrelTrackstersToSimTrackstersAssociationsByHits", "ticlBarrelSimTrackstersfromCPsToticlBarrelTracksters")
             ),
             cms.PSet(
                 branchName = cms.string("tsToStsByHit"),
                 suffix = cms.string("SC"),
                 associatorRecoToSimInputTag = cms.InputTag("allBarrelTrackstersToSimTrackstersAssociationsByHits", "ticlBarrelTrackstersToticlBarrelSimTracksters"),
                 associatorSimToRecoInputTag = cms.InputTag("allBarrelTrackstersToSimTrackstersAssociationsByHits", "ticlBarrelSimTrackstersToticlBarrelTracksters")
             )
        ],
        saveLCs = cms.bool(True),
        layerClusters = cms.InputTag("lcFromPFClusterProducer"),
        layer_clustersTime = cms.InputTag("barrelLayerClusters:timeLayerCluster"),
        lcRecoToSimAssociatorCP = cms.InputTag("barrelLayerClusterCaloParticleAssociationProducerPF"),
        lcSimToRecoAssociatorCP = cms.InputTag("barrelLayerClusterCaloParticleAssociationProducerPF"),
        lcRecoToSimAssociatorSC = cms.InputTag("barrelLayerClusterSimClusterAssociationProducerPF"),
        lcSimToRecoAssociatorSC = cms.InputTag("barrelLayerClusterSimClusterAssociationProducerPF"),
        saveTICLCandidate = cms.bool(False),
        saveSimTICLCandidate = cms.bool(False),
        saveTracks = cms.bool(False),
        saveSuperclustering = cms.bool(False),
        saveRecoSuperclusters = cms.bool(False),
        saveCaloParticles = cms.bool(True)
    )

    process.consumer = cms.EDAnalyzer("GenericConsumer",
        eventProducts = cms.untracked.vstring(['barrelLayerClusters',
                                               'barrelLayerClusterCaloParticleAssociationProducer',
                                               'barrelLayerClusterSimClusterAssociationProducer',
                                               'ticlBarrelTracksters',
                                               'ticlBarrelSimTracksters',
                                               'barrelSimTracksterAssociationPR',
                                               'barrelTracksterSimTracksterAssociationLinkingPR'])
    )


    process.TFileService = cms.Service("TFileService",
        fileName = cms.string("histo.root")
    )

    process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput
                                                  +process.ticlDumper
                                                  +process.consumer)

    process.schedule = cms.Schedule(process.ticlBarrel
                                    ,process.ticlAssociators
                                    ,process.FEVTDEBUGHLToutput_step)
    
    return process



def customiseForTICLBarrel(process, pfComparison=False):
    
    process.recHitMapProducer.hgcalOnly = cms.bool(False)

    # Parameters for CLUE in the barrel
    process.barrelLayerClusters = _barrelLayerClusters.clone(
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
    
    process.barrelLayerClustersTask = cms.Task(process.barrelLayerClusters)

    process.ticlBarrelTracksters = _trackstersProducer.clone(
        detector = "Barrel",
        layer_clusters = "barrelLayerClusters",
        time_layerclusters = "barrelLayerClusters:timeLayerCluster",
        filtered_mask = "barrelLayerClusters:InitialLayerClustersMask",
        original_mask = "barrelLayerClusters:InitialLayerClustersMask",
        seeding_regions = "ticlSeedingGlobal",
        itername = "CLUE3D",
        patternRecognitionBy = "CLUE3D",
        pluginPatternRecognitionByFastJet = dict(
            antikt_radius = 0.1,
            minNumLayerCluster = 0,
            algo_verbosity = 2
        ),
        pluginPatternRecognitionByCLUE3D = dict(
            criticalDensity = cms.vdouble(0.5, 0.5, 0.5),
            criticalSelfDensity = cms.vdouble(0., 0., 0.),
            criticalEtaPhiDistance = cms.vdouble(7*0.087, 7*0.087, 7*0.087, 7*0.087),
            densityEtaPhiDistanceSqr = cms.vdouble(0.37, 0.37, 0.37, 0.37),
            nearestHigherOnSameLayer = cms.bool(False),
            densityOnSameLayer = cms.bool(False),
            minNumLayerCluster = cms.vint32(1, 1, 1),
            useAbsoluteProjectiveScale = cms.bool(False),
            densitySiblingLayers = cms.vint32(4, 4, 4)
        )
    )
    process.ticlSeedingGlobal = _ticlSeedingRegionProducer.clone()
    process.ticlLayerTileProducer = _ticlLayerTileProducer.clone(
        detector = cms.string("Barrel")
    )

    process.ticlBarrelSimTracksters = _simTrackstersProducer.clone(
        layer_clusters = cms.InputTag("barrelLayerClusters"),
        filtered_mask = cms.InputTag("barrelLayerClusters:InitialLayerClustersMask"),
        time_layerclusters = cms.InputTag("barrelLayerClusters:timeLayerCluster"),
        layerClusterCaloParticleAssociator = cms.InputTag("barrelLayerClusterCaloParticleAssociationProducer"),
        layerClusterSimClusterAssociator = cms.InputTag("barrelLayerClusterSimClusterAssociationProducer"),
        detector = cms.string("Barrel"),
    )

    process.ticlBarrelTrackstersTask = cms.Task(process.ticlLayerTileProducer
                                                ,process.ticlSeedingGlobal
                                                ,process.ticlBarrelTracksters)

    process.ticlBarrel = cms.Path(process.barrelLayerClustersTask,process.ticlBarrelTrackstersTask)
    
    ticlBarrelTrackstersLabels = ["ticlBarrelTracksters"]
    ticlBarrelSimTrackstersLabels = ["ticlBarrelSimTrackster"]

    ## associators

    process.barrelLCToCPAssociatorByEnergyScoreProducer = barrelLCToCPAssociatorByEnergyScoreProducer.clone()
    process.barrelLayerClusterCaloParticleAssociationProducer = barrelLayerClusterCaloParticleAssociationProducer.clone()

    process.barrelLCToSCAssociatorByEnergyScoreProducer = barrelLCToSCAssociatorByEnergyScoreProducer.clone()
    process.barrelLayerClusterSimClusterAssociationProducer = barrelLayerClusterSimClusterAssociationProducer.clone()


    process.allBarrelLayerClusterToTracksterAssociations = _allBarrelLayerClusterToTracksterAssociations.clone()
    process.SimClusterToCaloParticleAssociator = _SimClusterToCaloParticleAssociator.clone()
    process.barrelHitToSimClusterCaloParticleAssociator = _barrelHitToSimClusterCaloParticleAssociator.clone()

    # TS-STS associations by LCs
    process.allBarrelTrackstersToSimTrackstersAssociationsByLCs = _allBarrelTrackstersToSimTrackstersAssociationsByLCs.clone()

    # TS-STS associations by hits
    process.allHitToBarrelTracksterAssociations = _allHitToBarrelTracksterAssociations.clone()
    process.allBarrelTrackstersToSimTrackstersAssociationsByHits = _allBarrelTrackstersToSimTrackstersAssociationsByHits.clone(
        hitToCaloParticleMap = cms.InputTag('barrelHitToSimClusterCaloParticleAssociator', 'hitToCaloParticleMap'),
        hitToSimClusterMap = cms.InputTag('barrelHitToSimClusterCaloParticleAssociator', 'hitToSimClusterMap'),
        hitToTracksterMap = cms.string('allHitToBarrelTracksterAssociations')
    )
    
    process.ticlAssociators = cms.Path(process.recHitMapProducer
                                       +process.barrelLCToCPAssociatorByEnergyScoreProducer
                                       +process.barrelLCToSCAssociatorByEnergyScoreProducer
                                       +process.barrelLayerClusterCaloParticleAssociationProducer
                                       +process.barrelLayerClusterSimClusterAssociationProducer
                                       +process.ticlBarrelSimTracksters
                                       +process.barrelHitToSimClusterCaloParticleAssociator
                                       +process.SimClusterToCaloParticleAssociator
                                       +process.allBarrelLayerClusterToTracksterAssociations
                                       +process.allHitToBarrelTracksterAssociations
                                       +process.allBarrelTrackstersToSimTrackstersAssociationsByLCs
                                       +process.allBarrelTrackstersToSimTrackstersAssociationsByHits
    )

    process.ticlDumper = _ticlDumper.clone(
        tracksterCollections = [*[cms.PSet(treeName=cms.string(label), inputTag=cms.InputTag(label)) for label in ticlBarrelTrackstersLabels],
            cms.PSet(
                treeName = cms.string("simtrackstersSC"),
                inputTag = cms.InputTag("ticlBarrelSimTracksters"),
                tracksterType = cms.string("SimTracksterSC")
            ),
            cms.PSet(
                treeName = cms.string("simtrackstersCP"),
                inputTag = cms.InputTag("ticlBarrelSimTracksters", "fromCPs"),
                tracksterType = cms.string("SimTracksterCP")
            ),
        ],
        associators = [
            cms.PSet(
                branchName = cms.string("tsToStsByLC"),
                suffix = cms.string("CP"),
                associatorRecoToSimInputTag = cms.InputTag("allBarrelTrackstersToSimTrackstersAssociationsByLCs", "ticlBarrelTrackstersToticlBarrelSimTrackstersfromCPs"),
                associatorSimToRecoInputTag = cms.InputTag("allBarrelTrackstersToSimTrackstersAssociationsByLCs", "ticlBarrelSimTrackstersfromCPsToticlBarrelTracksters")
            ),
            cms.PSet(
                branchName = cms.string("tsToStsByLC"),
                suffix = cms.string("SC"),
                associatorRecoToSimInputTag = cms.InputTag("allBarrelTrackstersToSimTrackstersAssociationsByLCs", "ticlBarrelTrackstersToticlBarrelSimTracksters"),
                associatorSimToRecoInputTag = cms.InputTag("allBarrelTrackstersToSimTrackstersAssociationsByLCs", "ticlBarrelSimTrackstersToticlBarrelTracksters")
            ),
             cms.PSet(                                                                                                                                                                
                 branchName = cms.string("tsToStsByHit"),
                 suffix = cms.string("CP"),
                 associatorRecoToSimInputTag = cms.InputTag("allBarrelTrackstersToSimTrackstersAssociationsByHits", "ticlBarrelTrackstersToticlBarrelSimTrackstersfromCPs"),
                 associatorSimToRecoInputTag = cms.InputTag("allBarrelTrackstersToSimTrackstersAssociationsByHits", "ticlBarrelSimTrackstersfromCPsToticlBarrelTracksters")
             ),
             cms.PSet(
                 branchName = cms.string("tsToStsByHit"),
                 suffix = cms.string("SC"),
                 associatorRecoToSimInputTag = cms.InputTag("allBarrelTrackstersToSimTrackstersAssociationsByHits", "ticlBarrelTrackstersToticlBarrelSimTracksters"),
                 associatorSimToRecoInputTag = cms.InputTag("allBarrelTrackstersToSimTrackstersAssociationsByHits", "ticlBarrelSimTrackstersToticlBarrelTracksters")
             )
        ],
        saveLCs = cms.bool(True),
        layerClusters = cms.InputTag("barrelLayerClusters"),
        layer_clustersTime = cms.InputTag("barrelLayerClusters:timeLayerCluster"),
        lcRecoToSimAssociatorCP = cms.InputTag("barrelLayerClusterCaloParticleAssociationProducer"),
        lcSimToRecoAssociatorCP = cms.InputTag("barrelLayerClusterCaloParticleAssociationProducer"),
        lcRecoToSimAssociatorSC = cms.InputTag("barrelLayerClusterSimClusterAssociationProducer"),
        lcSimToRecoAssociatorSC = cms.InputTag("barrelLayerClusterSimClusterAssociationProducer"),
        saveTICLCandidate = cms.bool(False),
        saveSimTICLCandidate = cms.bool(False),
        saveTracks = cms.bool(False),
        saveSuperclustering = cms.bool(False),
        saveRecoSuperclusters = cms.bool(False),
        saveCaloParticles = cms.bool(True)
    )

    process.consumer = cms.EDAnalyzer("GenericConsumer",
        eventProducts = cms.untracked.vstring(['barrelLayerClusters',
                                               'barrelLayerClusterCaloParticleAssociationProducer',
                                               'barrelLayerClusterSimClusterAssociationProducer',
                                               'ticlBarrelTracksters',
                                               'ticlBarrelSimTracksters',
                                               'barrelSimTracksterAssociationPR',
                                               'barrelTracksterSimTracksterAssociationLinkingPR'])
    )


    process.TFileService = cms.Service("TFileService",
        fileName = cms.string("histo.root")
    )

    process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput
                                                  +process.ticlDumper
                                                  +process.consumer)

    process.schedule = cms.Schedule(process.ticlBarrel
                                    ,process.ticlAssociators
                                    ,process.FEVTDEBUGHLToutput_step)
    
    return process
