import FWCore.ParameterSet.Config as cms

from RecoTICL.LayerClustersProducers.ticlLayerTileProducer_cfi import ticlLayerTileProducer

from RecoTICL.TrackstersProducers.CLUE3DEM_cff import *
from RecoTICL.TrackstersProducers.CLUE3DHAD_cff import *
from RecoTICL.CandidatesProducers.pfTICLProducerV5_cfi import pfTICLProducerV5 as _pfTICLProducerV5

from RecoTICL.LayerClustersProducers.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoTICL.TrackstersProducers.tracksterSelectionTf_cfi import *

from RecoTICL.LinkingProducers.tracksterLinksProducer_cfi import tracksterLinksProducer as _tracksterLinksProducer
from RecoTICL.CandidatesProducers.ticlCandidateProducer_cfi import ticlCandidateProducer as _ticlCandidateProducer
from RecoTICL.Configuration.RecoTICL_EventContent_cff import customiseForTICLv5EventContent
from RecoTICL.CandidatesProducers.iterativeTICL_cff import ticlIterLabels, ticlIterLabelsMerge
from RecoTICL.CandidatesProducers.ticlDumper_cfi import ticlDumper
from RecoTICL.LinkingProducers.mergedTrackstersProducer_cfi import mergedTrackstersProducer as _mergedTrackstersProducer
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import tracksterSimTracksterAssociationLinkingbyCLUE3D as _tracksterSimTracksterAssociationLinkingbyCLUE3D
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import tracksterSimTracksterAssociationPRbyCLUE3D  as _tracksterSimTracksterAssociationPRbyCLUE3D
from Validation.HGCalValidation.HGCalValidator_cff import hgcalValidator
from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import HGCalUncalibRecHit
from RecoTICL.TrackstersProducers.SimTracksters_cff import ticlSimTracksters

from RecoTICL.TrackstersProducers.FastJetStep_cff import ticlTrackstersFastJet
from RecoTICL.TrackstersProducers.EMStep_cff import ticlTrackstersEM, ticlTrackstersHFNoseEM
from RecoTICL.TrackstersProducers.TrkStep_cff import ticlTrackstersTrk, ticlTrackstersHFNoseTrk
from RecoTICL.TrackstersProducers.MIPStep_cff import ticlTrackstersMIP, ticlTrackstersHFNoseMIP
from RecoTICL.TrackstersProducers.HADStep_cff import ticlTrackstersHAD, ticlTrackstersHFNoseHAD
from RecoTICL.TrackstersProducers.CLUE3DEM_cff import ticlTrackstersCLUE3DEM
from RecoTICL.TrackstersProducers.CLUE3DHAD_cff import ticlTrackstersCLUE3DHAD
from RecoTICL.TrackstersProducers.CLUE3DHighStep_cff import ticlTrackstersCLUE3DHigh
from RecoTICL.TrackstersProducers.TrkEMStep_cff import ticlTrackstersTrkEM, filteredLayerClustersHFNoseTrkEM

from RecoTICL.CandidatesProducers.mtdSoAProducer_cfi import mtdSoAProducer as _mtdSoAProducer

def customiseForTICLv5(process, enableDumper = False):

    process.HGCalUncalibRecHit.computeLocalTime = cms.bool(True)
    process.ticlSimTracksters.computeLocalTime = cms.bool(True)

    process.ticlTrackstersFastJet.pluginPatternRecognitionByFastJet.computeLocalTime = cms.bool(True)

    process.ticlTrackstersEM.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)
    process.ticlTrackstersHFNoseEM.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)

    process.ticlTrackstersTrk.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)
    process.ticlTrackstersHFNoseTrk.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)

    process.ticlTrackstersMIP.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)
    process.ticlTrackstersHFNoseMIP.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)

    process.ticlTrackstersHAD.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)
    process.ticlTrackstersHFNoseHAD.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)

    process.ticlTrackstersCLUE3DHAD.pluginPatternRecognitionByCLUE3D.computeLocalTime = cms.bool(True)
    process.ticlTrackstersCLUE3DEM.pluginPatternRecognitionByCLUE3D.computeLocalTime = cms.bool(True)
    process.ticlTrackstersCLUE3DHigh.pluginPatternRecognitionByCLUE3D.computeLocalTime = cms.bool(True)

    process.ticlTrackstersTrkEM.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)
    process.ticlTrackstersHFNoseTrkEM.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)

    process.ticlLayerTileTask = cms.Task(ticlLayerTileProducer)

    process.ticlIterationsTask = cms.Task(
        ticlCLUE3DEMStepTask,
        ticlCLUE3DHADStepTask,
    )

    process.mtdSoA = _mtdSoAProducer.clone()
    process.mtdSoATask = cms.Task(process.mtdSoA)

    process.ticlTracksterLinks = _tracksterLinksProducer.clone()
    process.ticlTracksterLinksTask = cms.Task(process.ticlTracksterLinks)

    process.ticlCandidate = _ticlCandidateProducer.clone()
    process.ticlCandidateTask = cms.Task(process.ticlCandidate)

    process.tracksterSimTracksterAssociationLinkingbyCLUE3DEM = _tracksterSimTracksterAssociationLinkingbyCLUE3D.clone(
        label_tst = cms.InputTag("ticlTrackstersCLUE3DEM")
        )
    process.tracksterSimTracksterAssociationPRbyCLUE3DEM = _tracksterSimTracksterAssociationPRbyCLUE3D.clone(
        label_tst = cms.InputTag("ticlTrackstersCLUE3DEM")
        )
    process.tracksterSimTracksterAssociationLinkingbyCLUE3DHAD = _tracksterSimTracksterAssociationLinkingbyCLUE3D.clone(
        label_tst = cms.InputTag("ticlTrackstersCLUE3DHAD")
        )
    process.tracksterSimTracksterAssociationPRbyCLUE3DHAD = _tracksterSimTracksterAssociationPRbyCLUE3D.clone(
        label_tst = cms.InputTag("ticlTrackstersCLUE3DHAD")
        )

    process.mergedTrackstersProducer = _mergedTrackstersProducer.clone()

    process.tracksterSimTracksterAssociationLinkingbyCLUE3D = _tracksterSimTracksterAssociationLinkingbyCLUE3D.clone(
        label_tst = cms.InputTag("mergedTrackstersProducer")
        )
    process.tracksterSimTracksterAssociationPRbyCLUE3D = _tracksterSimTracksterAssociationPRbyCLUE3D.clone(
        label_tst = cms.InputTag("mergedTrackstersProducer")
        )
    process.iterTICLTask = cms.Task(process.ticlLayerTileTask,
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
    process.pfTICL = _pfTICLProducerV5.clone()
    process.hgcalAssociators = cms.Task(process.mergedTrackstersProducer, process.lcAssocByEnergyScoreProducer, process.layerClusterCaloParticleAssociationProducer,
                            process.scAssocByEnergyScoreProducer, process.layerClusterSimClusterAssociationProducer,
                            process.lcSimTSAssocByEnergyScoreProducer, process.layerClusterSimTracksterAssociationProducer,
                            process.simTsAssocByEnergyScoreProducer,  process.simTracksterHitLCAssociatorByEnergyScoreProducer,
                            process.tracksterSimTracksterAssociationLinking, process.tracksterSimTracksterAssociationPR,
                            process.tracksterSimTracksterAssociationLinkingbyCLUE3D, process.tracksterSimTracksterAssociationPRbyCLUE3D,
                            process.tracksterSimTracksterAssociationLinkingbyCLUE3DEM, process.tracksterSimTracksterAssociationPRbyCLUE3DEM,
                            process.tracksterSimTracksterAssociationLinkingbyCLUE3DHAD, process.tracksterSimTracksterAssociationPRbyCLUE3DHAD,
                            process.tracksterSimTracksterAssociationLinkingPU, process.tracksterSimTracksterAssociationPRPU
                            )


    labelTst = ["ticlTrackstersCLUE3DEM", "ticlTrackstersCLUE3DHAD", "ticlTracksterLinks"]
    labelTst.extend([cms.InputTag("ticlSimTracksters", "fromCPs"), cms.InputTag("ticlSimTracksters")])
    lcInputMask  = ["ticlTrackstersCLUE3DEM", "ticlTrackstersCLUE3DHAD", "ticlTracksterLinks"]
    lcInputMask.extend([cms.InputTag("ticlSimTracksters", "fromCPs"), cms.InputTag("ticlSimTracksters")])
    process.hgcalValidator = hgcalValidator.clone(
        label_tst = cms.VInputTag(labelTst),
        LayerClustersInputMask = cms.VInputTag(lcInputMask),
        ticlTrackstersMerge = cms.InputTag("ticlCandidate"),
    )

    process.hgcalValidatorSequence = cms.Sequence(process.hgcalValidator)
    process.hgcalValidation = cms.Sequence(process.hgcalSimHitValidationEE+process.hgcalSimHitValidationHEF+process.hgcalSimHitValidationHEB+process.hgcalDigiValidationEE+process.hgcalDigiValidationHEF+process.hgcalDigiValidationHEB+process.hgcalRecHitValidationEE+process.hgcalRecHitValidationHEF+process.hgcalRecHitValidationHEB+process.hgcalHitValidationSequence+process.hgcalValidatorSequence+process.hgcalTiclPFValidation+process.hgcalPFJetValidation)
    process.globalValidationHGCal = cms.Sequence(process.hgcalValidation)
    process.validation_step9 = cms.EndPath(process.globalValidationHGCal)
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
            trackstersclue3d = cms.InputTag('mergedTrackstersProducer'),
            ticlcandidates = cms.InputTag("ticlCandidate"),
            trackstersmerged = cms.InputTag("ticlCandidate"),
            trackstersInCand = cms.InputTag("ticlCandidate")
        )
        process.TFileService = cms.Service("TFileService",
                                           fileName=cms.string("histo.root")
                                           )
        process.FEVTDEBUGHLToutput_step = cms.EndPath(
            process.FEVTDEBUGHLToutput + process.ticlDumper)


    process = customiseForTICLv5EventContent(process)

    return process
