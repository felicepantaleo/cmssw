import FWCore.ParameterSet.Config as cms

enableGPU = True
from Configuration.ProcessModifiers.gpu_cff import gpu

from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit
from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import HGCAL_noise_fC, HGCAL_chargeCollectionEfficiencies

process = cms.Process("TESTgpu", gpu) if enableGPU else cms.Process("TESTnongpu")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.Geometry.GeometryExtended2026D46Reco_cff')
process.load('HeterogeneousCore.CUDAServices.CUDAService_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi')
process.load('SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 10 ))

fNames = ['file:/afs/cern.ch/user/b/bfontana/CMSSW_11_0_0_pre11_Patatrack/src/UserCode/Samples/20495.0_CloseByParticleGun_CE_E_Front_200um+CE_E_Front_200um_2026D41_GenSimHLBeamSpotFull+DigiFullTrigger_2026D41+RecoFullGlobal_2026D41+HARVESTFullGlobal_2026D41/step3.root']
keep = 'keep *'
drop = 'drop CSCDetIdCSCALCTPreTriggerDigiMuonDigiCollection_simCscTriggerPrimitiveDigis__HLT'
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(fNames),
                            inputCommands = cms.untracked.vstring([keep, drop]),
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"))

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( False )) #add option for edmStreams
process.HeterogeneousHGCalEERecHitProducer = cms.EDProducer('HeterogeneousHGCalEERecHitProducer',
                                                            HGCEEUncalibRecHitsTok = cms.InputTag('HGCalUncalibRecHit', 'HGCEEUncalibRecHits'),
                                                            nhitsmax       = cms.uint32(40000),
                                                            HGCEE_keV2DIGI = HGCalRecHit.__dict__['HGCEE_keV2DIGI'],
                                                            minValSiPar    = HGCalRecHit.__dict__['minValSiPar'],
                                                            maxValSiPar    = HGCalRecHit.__dict__['maxValSiPar'],
                                                            constSiPar     = HGCalRecHit.__dict__['constSiPar'],
                                                            noiseSiPar     = HGCalRecHit.__dict__['noiseSiPar'],
                                                            HGCEE_fCPerMIP = HGCalRecHit.__dict__['HGCEE_fCPerMIP'],
                                                            HGCEE_isSiFE   = HGCalRecHit.__dict__['HGCEE_isSiFE'],
                                                            HGCEE_noise_fC = HGCalRecHit.__dict__['HGCEE_noise_fC'],
                                                            HGCEE_cce      = HGCalRecHit.__dict__['HGCEE_cce'],
                                                            rangeMatch     = HGCalRecHit.__dict__['rangeMatch'],
                                                            rangeMask      = HGCalRecHit.__dict__['rangeMask'],
                                                            rcorr          = HGCalRecHit.__dict__['thicknessCorrection'],
                                                            weights        = HGCalRecHit.__dict__['layerWeights']
)
process.HeterogeneousHGCalHEFRecHitProducer = cms.EDProducer('HeterogeneousHGCalHEFRecHitProducer',
                                                             HGCHEFUncalibRecHitsTok = cms.InputTag('HGCalUncalibRecHit', 'HGCHEFUncalibRecHits'),
                                                             nhitsmax        = cms.uint32(6000),
                                                             HGCHEF_keV2DIGI  = HGCalRecHit.__dict__['HGCHEF_keV2DIGI'],
                                                             minValSiPar     = HGCalRecHit.__dict__['minValSiPar'],
                                                             maxValSiPar     = HGCalRecHit.__dict__['maxValSiPar'],
                                                             constSiPar      = HGCalRecHit.__dict__['constSiPar'],
                                                             noiseSiPar      = HGCalRecHit.__dict__['noiseSiPar'],
                                                             HGCHEF_fCPerMIP = HGCalRecHit.__dict__['HGCHEF_fCPerMIP'],
                                                             HGCHEF_isSiFE   = HGCalRecHit.__dict__['HGCHEF_isSiFE'],
                                                             HGCHEF_noise_fC = HGCalRecHit.__dict__['HGCHEF_noise_fC'],
                                                             HGCHEF_cce      = HGCalRecHit.__dict__['HGCHEF_cce'],
                                                             rangeMatch      = HGCalRecHit.__dict__['rangeMatch'],
                                                             rangeMask       = HGCalRecHit.__dict__['rangeMask'],
                                                             rcorr           = HGCalRecHit.__dict__['thicknessCorrection'],
                                                             weights         = HGCalRecHit.__dict__['layerWeights'],
                                                             offset          = cms.uint32(22)
                                                         )
process.HeterogeneousHGCalHEBRecHitProducer = cms.EDProducer('HeterogeneousHGCalHEBRecHitProducer',
                                                             HGCHEBUncalibRecHitsTok = cms.InputTag('HGCalUncalibRecHit', 'HGCHEBUncalibRecHits'),
                                                             nhitsmax         = cms.uint32(1000),
                                                             HGCHEB_keV2DIGI  = HGCalRecHit.__dict__['HGCHEB_keV2DIGI'],
                                                             HGCHEB_noise_MIP = HGCalRecHit.__dict__['HGCHEB_noise_MIP'],
                                                             minValSiPar      = HGCalRecHit.__dict__['minValSiPar'],
                                                             maxValSiPar      = HGCalRecHit.__dict__['maxValSiPar'],
                                                             constSiPar       = HGCalRecHit.__dict__['constSiPar'],
                                                             noiseSiPar       = HGCalRecHit.__dict__['noiseSiPar'],
                                                             HGCHEB_isSiFE    = HGCalRecHit.__dict__['HGCHEB_isSiFE'],
                                                             rangeMatch       = HGCalRecHit.__dict__['rangeMatch'],
                                                             rangeMask        = HGCalRecHit.__dict__['rangeMask'],
                                                             weights          = HGCalRecHit.__dict__['layerWeights'],
                                                             offset           = cms.uint32(22)
                                                         )

fNameOut = 'out'
#convert this to a task!!!!!
process.task = cms.Task( process.HeterogeneousHGCalEERecHitProducer, process.HeterogeneousHGCalHEFRecHitProducer, process.HeterogeneousHGCalHEBRecHitProducer )
#process.task = cms.Task( process.HeterogeneousHGCalHEFRecHitProducer )
process.path = cms.Path( process.task )

process.TFileService = cms.Service("TFileService", 
                                   fileName = cms.string("histo.root"),
                                   closeFileFast = cms.untracked.bool(True) #safe as long as the file doesn't contain multiple references to the same object, for example a histogram and a TCanvas containing that histogram.
)

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string(fNameOut+".root"))
process.outpath = cms.EndPath(process.out)
