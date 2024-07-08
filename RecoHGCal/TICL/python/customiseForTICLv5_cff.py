import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff import hgcalLayerClustersEE, hgcalLayerClustersHSi, hgcalLayerClustersHSci
from RecoLocalCalo.HGCalRecProducers.hgcalMergeLayerClusters_cfi import hgcalMergeLayerClusters
from RecoTracker.IterativeTracking.iterativeTk_cff import trackdnn_source
from RecoLocalCalo.HGCalRecProducers.hgcalRecHitMapProducer_cfi import hgcalRecHitMapProducer

from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer

from RecoHGCal.TICL.CLUE3DEM_cff import *
from RecoHGCal.TICL.CLUE3DHAD_cff import *
from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer as _pfTICLProducer

from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoHGCal.TICL.tracksterSelectionTf_cfi import *

from RecoHGCal.TICL.tracksterLinksProducer_cfi import tracksterLinksProducer as _tracksterLinksProducer
from RecoHGCal.TICL.ticlCandidateProducer_cfi import ticlCandidateProducer as _ticlCandidateProducer
from RecoHGCal.Configuration.RecoHGCal_EventContent_cff import customiseForTICLv5EventContent
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels, ticlIterLabelsMerge
from RecoHGCal.TICL.ticlDumper_cfi import ticlDumper
from RecoHGCal.TICL.mergedTrackstersProducer_cfi import mergedTrackstersProducer as _mergedTrackstersProducer
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import tracksterSimTracksterFromCPsAssociationPR as _tracksterSimTracksterFromCPsAssociationPR
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import tracksterSimTracksterAssociationPR  as _tracksterSimTracksterAssociationPR
from Validation.HGCalValidation.HGCalValidator_cff import hgcalValidator
from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import HGCalUncalibRecHit
from RecoHGCal.TICL.SimTracksters_cff import ticlSimTracksters, ticlSimTrackstersTask

from RecoHGCal.TICL.FastJetStep_cff import ticlTrackstersFastJet
from RecoHGCal.TICL.EMStep_cff import ticlTrackstersEM, ticlTrackstersHFNoseEM
from RecoHGCal.TICL.TrkStep_cff import ticlTrackstersTrk, ticlTrackstersHFNoseTrk
from RecoHGCal.TICL.MIPStep_cff import ticlTrackstersMIP, ticlTrackstersHFNoseMIP
from RecoHGCal.TICL.HADStep_cff import ticlTrackstersHAD, ticlTrackstersHFNoseHAD
from RecoHGCal.TICL.CLUE3DEM_cff import ticlTrackstersCLUE3DEM
from RecoHGCal.TICL.CLUE3DHAD_cff import ticlTrackstersCLUE3DHAD
from RecoHGCal.TICL.CLUE3DHighStep_cff import ticlTrackstersCLUE3DHigh
from RecoHGCal.TICL.TrkEMStep_cff import ticlTrackstersTrkEM, filteredLayerClustersHFNoseTrkEM

from RecoHGCal.TICL.mtdSoAProducer_cfi import mtdSoAProducer as _mtdSoAProducer

def customiseTICLv5FromReco(process, enableDumper = False):
    # TensorFlow ESSource

    process.TFESSource = cms.Task(process.trackdnn_source)

    process.hgcalLayerClustersTask = cms.Task(process.hgcalLayerClustersEE,
                                              process.hgcalLayerClustersHSi,
                                              process.hgcalLayerClustersHSci,
                                              process.hgcalMergeLayerClusters)

    # Reconstruction

    process.ticlSimTracksters.computeLocalTime = cms.bool(True)

    process.ticlTrackstersCLUE3DHigh.pluginPatternRecognitionByCLUE3D.computeLocalTime = cms.bool(True)

    '''for future CLUE3D separate iterations
    process.ticlTrackstersCLUE3DHAD.pluginPatternRecognitionByCLUE3D.computeLocalTime = cms.bool(True)
    process.ticlTrackstersCLUE3DEM.pluginPatternRecognitionByCLUE3D.computeLocalTime = cms.bool(True)
    '''

    process.ticlLayerTileTask = cms.Task(ticlLayerTileProducer)

    process.ticlIterationsTask = cms.Task(
        process.ticlTrackstersCLUE3DHigh,
    )

    process.mtdSoA = _mtdSoAProducer.clone()
    process.mtdSoATask = cms.Task(process.mtdSoA)

    process.ticlTracksterLinks = _tracksterLinksProducer.clone()
    process.ticlTracksterLinks = _tracksterLinksProducer.clone(
            tracksters_collections = cms.VInputTag(
              'ticlTrackstersCLUE3DHigh'
            ),
    )

    process.ticlCandidate = _ticlCandidateProducer.clone()
    process.ticlCandidateTask = cms.Task(process.ticlCandidate)

    process.tracksterSimTracksterFromCPsAssociationPRHigh = _tracksterSimTracksterFromCPsAssociationPR.clone(
        label_tst = cms.InputTag("ticlTrackstersCLUE3DHigh")
        )
    process.tracksterSimTracksterAssociationPRHigh = _tracksterSimTracksterAssociationPR.clone(
        label_tst = cms.InputTag("ticlTrackstersCLUE3DHigh")
        )


    process.iterTICLTask = cms.Path(process.hgcalLayerClustersTask,
                            process.TFESSource,
                            process.ticlLayerTileTask,
                            process.mtdSoATask,
                            process.ticlIterationsTask,
                            process.ticlTracksterLinksTask,
                            process.ticlCandidateTask)

    process.particleFlowClusterHGCal.initialClusteringStep.tracksterSrc = "ticlCandidate"
    process.globalrecoTask.remove(process.ticlTrackstersMerge)

    process.tracksterSimTracksterAssociationLinking.label_tst = cms.InputTag("ticlCandidate")
    process.tracksterSimTracksterAssociationPR.label_tst = cms.InputTag("ticlCandidate")

    process.tracksterSimTracksterAssociationLinkingPU.label_tst = cms.InputTag("ticlCandidate")
    process.tracksterSimTracksterAssociationPRPU.label_tst = cms.InputTag("ticlCandidate")
    process.mergeTICLTask = cms.Task()
    process.pfTICL = _pfTICLProducer.clone(
      ticlCandidateSrc = cms.InputTag('ticlCandidate'),
      isTICLv5 = cms.bool(True)
    )
    process.hgcalAssociators = cms.Task(process.hgcalRecHitMapProducer, process.lcAssocByEnergyScoreProducer, process.layerClusterCaloParticleAssociationProducer,
                            process.scAssocByEnergyScoreProducer, process.layerClusterSimClusterAssociationProducer,
                            # FP 07/2024 new associators:
                            layerClusterToCLUE3DTracksterAssociation, layerClusterToTracksterMergeAssociation,
                            layerClusterToSimTracksterAssociation,
                            hitToTrackstersAssociationLinking, hitToTrackstersAssociationPR,
                            hitToSimTracksterAssociation, hitToSimTracksterFromCPsAssociation,
                            tracksterSimTracksterAssociationByHitsLinking, tracksterSimTracksterAssociationByHitsPR,
                            tracksterSimTracksterFromCPsAssociationLinking, tracksterSimTracksterAssociationLinking, tracksterSimTracksterFromCPsAssociationPR, tracksterSimTracksterAssociationPR,
                            hitToSimClusterCaloParticleAssociator
                            )

    if(enableDumper):
        process.ticlDumper = ticlDumper.clone(
            saveLCs=True,
            saveCLUE3DTracksters=True,
            saveTrackstersMerged=True,
            saveSimTrackstersSC=True,
            saveSimTrackstersCP=True,
            saveTICLCandidate=True,
            saveSimTICLCandidate=True,
            saveTracks=True,
            saveAssociations=True,
            trackstersclue3d = cms.InputTag('ticlTrackstersCLUE3DHigh'),
            ticlcandidates = cms.InputTag("ticlCandidate"),
            trackstersmerged = cms.InputTag("ticlCandidate"),
            trackstersInCand = cms.InputTag("ticlCandidate")
        )
        process.TFileService = cms.Service("TFileService",
                                           fileName=cms.string("histo.root")
                                           )

    process.FEVTDEBUGHLToutput_step = cms.EndPath(process.ticlDumper)

    process.TICL_Validation = cms.Path(process.ticlSimTrackstersTask, process.hgcalAssociators)

# Schedule definition
    process.schedule = cms.Schedule(process.iterTICLTask,
                                    process.TICL_Validation,
                                    process.FEVTDEBUGHLToutput_step)
    process = customiseForTICLv5EventContent(process)

    return process
