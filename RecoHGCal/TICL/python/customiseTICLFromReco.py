# Reconstruction
from RecoHGCal.TICL.iterativeTICL_cff import *
from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff import hgcalLayerClusters
# Validation
from Validation.HGCalValidation.HGCalValidator_cfi import *
from RecoLocalCalo.HGCalRecProducers.hgcalRecHitMapProducer_cfi import hgcalRecHitMapProducer

# Load DNN ESSource
from RecoTracker.IterativeTracking.iterativeTk_cff import trackdnn_source

# Automatic addition of the customisation function from RecoHGCal.Configuration.RecoHGCal_EventContent_cff
from RecoHGCal.Configuration.RecoHGCal_EventContent_cff import customiseHGCalOnlyEventContent



def customiseTICLFromReco(process):

# TensorFlow ESSource
    process.TFileService = cms.Service("TFileService",
                                       # fileName = cms.string('/afs/cern.ch/work/w/wredjeb/public/EnergyRegression/TFInterface/Rebase/CMSSW_12_4_0_pre2/src/38693.0_CloseByParticleGun+2026D86+CloseByParticle_Photon_ERZRanges_GenSimHLBeamSpotHGCALCloseBy+DigiTrigger+RecoGlobal+HARVESTGlobal/ticlTracksterCLUE3DKaonsAllProb.root')
					fileName = cms.string('/afs/cern.ch/work/w/wredjeb/public/EnergyRegression/TFInterface/Rebase/CMSSW_12_4_0_pre2/src/38694.203_CloseByPGun_CE_E_Front_300um+2026D86_ticl_v4+CE_E_Front_300um_GenSimHLBeamSpotHGCALCloseBy+DigiTrigger+RecoGlobal+HARVESTGlobal/ticlTrackstersCLUE3DElectronAllProb.root')
                                    )
    process.TFESSource = cms.Task(process.trackdnn_source)

    process.TICL = cms.Path(process.hgcalLayerClusters,
                            process.TFESSource,
                            process.ticlLayerTileTask,
                            process.ticlIterationsTask,
                            process.ticlTracksterMergeTask)

    process.trackstersNtuplerCLUE3D = cms.EDAnalyzer('TracksterNtupler',
        tracksters = cms.InputTag('ticlTrackstersCLUE3DHigh'),
        caloParticles = cms.InputTag('mix', 'MergedCaloTruth'),
        layerClusters=cms.InputTag('hgcalLayerClusters'),
        outfilePath = cms.untracked.string('/afs/cern.ch/work/w/wredjeb/public/EnergyRegression/TFInterface/Rebase/CMSSW_12_4_0_pre2/src/38693.0_CloseByParticleGun+2026D86+CloseByParticle_Photon_ERZRanges_GenSimHLBeamSpotHGCALCloseBy+DigiTrigger+RecoGlobal+HARVESTGlobal/ticlTracksterCLUE3D_newRegression.root')
    )


    process.TFileService = cms.Service("TFileService",
            fileName = cms.string("ticlTracksterCLUE3D_newRegressionFS.root")
    )

# Validation
    process.TICL_ValidationProducers = cms.Task(process.hgcalRecHitMapProducer,
                                                process.lcAssocByEnergyScoreProducer,
                                                process.layerClusterCaloParticleAssociationProducer,
                                                process.scAssocByEnergyScoreProducer,
                                                process.layerClusterSimClusterAssociationProducer,
                                               )
    process.TICL_Validator = cms.Task(process.hgcalValidator)
    process.TICL_Validation = cms.Path(process.TICL_ValidationProducers,
                                       process.TICL_Validator
                                      )
# Path and EndPath definitions
    process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput + process.trackstersNtuplerCLUE3D)
    process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
    process.schedule = cms.Schedule(process.TICL,
                                    # process.TICL_Validation,
                                    process.FEVTDEBUGHLToutput_step,
                                    # process.DQMoutput_step,
                                    )
#call to customisation function customiseHGCalOnlyEventContent imported from RecoHGCal.Configuration.RecoHGCal_EventContent_cff
    process = customiseHGCalOnlyEventContent(process)

    return process
