import FWCore.ParameterSet.Config as cms

TICLParticleFlowBlock = cms.EDProducer(
    "TICLPFBlockProducer",
    # verbosity
    verbose = cms.untracked.bool(False),
    # Debug flag
    debug = cms.untracked.bool(False),

    #define what we are importing into particle flow
    #from the various subdetectors
    # importers are executed in the order they are defined here!!!
    #order matters for some modules (it is pointed out where this is important)
    # you can find a list of all available importers in:
    #  plugins/importers
    elementImporters = cms.VPSet(
        cms.PSet( importerName = cms.string("TICLGSFTrackImporter"),
                  source = cms.InputTag("pfTrackElec"),
                  gsfsAreSecondary = cms.bool(False),
                  superClustersArePF = cms.bool(True) ),
        cms.PSet( importerName = cms.string("TICLConvBremTrackImporter"),
                  source = cms.InputTag("pfTrackElec"),
                  vetoEndcap = cms.bool(False)),
        cms.PSet( importerName = cms.string("TICLSuperClusterImporter"),
                  source_eb = cms.InputTag("particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel"),
                  source_ee = cms.InputTag("particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower"),
                  maximumHoverE = cms.double(0.5),
                  minSuperClusterPt = cms.double(10.0),
                  minPTforBypass = cms.double(100.0),
                  hbheRecHitsTag = cms.InputTag('hbhereco'),
                  maxSeverityHB = cms.int32(9),
                  maxSeverityHE = cms.int32(9),
                  usePFThresholdsFromDB = cms.bool(False),
                  superClustersArePF = cms.bool(True) ),
        cms.PSet( importerName = cms.string("TICLConversionTrackImporter"),
                  source = cms.InputTag("pfConversions"),
                  vetoEndcap = cms.bool(False)),
        # V0's not actually used in particle flow block building so far
        #cms.PSet( importerName = cms.string("TICLV0TrackImporter"),
        #          source = cms.InputTag("pfV0"),
        #          vetoEndcap = cms.bool(False)),
        #NuclearInteraction's also come in Loose and VeryLoose varieties
        cms.PSet( importerName = cms.string("TICLNuclearInteractionTrackImporter"),
                  source = cms.InputTag("pfDisplacedTrackerVertex"),
                  vetoEndcap = cms.bool(False)),
        #for best timing GeneralTracksImporter should come after
        # all secondary track importers
        cms.PSet( importerName = cms.string("TICLGeneralTracksImporter"),
                  source = cms.InputTag("pfTrack"),
                  vetoEndcap = cms.bool(False),
                  muonSrc = cms.InputTag("muons1stStep"),
		  trackQuality = cms.string("highPurity"),
                  cleanBadConvertedBrems = cms.bool(True),
                  useIterativeTracking = cms.bool(True),
                  DPtOverPtCuts_byTrackAlgo = cms.vdouble(10.0,10.0,10.0,
                                                           10.0,10.0,5.0),
                  NHitCuts_byTrackAlgo = cms.vuint32(3,3,3,3,3,3),
                  muonMaxDPtOPt = cms.double(1)
                  ),
        # secondary GSF tracks are also turned off
        #cms.PSet( importerName = cms.string("TICLGSFTrackImporter"),
        #          source = cms.InputTag("pfTrackElec:Secondary"),
        #          gsfsAreSecondary = cms.bool(True),
        #          superClustersArePF = cms.bool(True) ),
        # to properly set SC based links you need to run ECAL importer
        # after you've imported all SCs to the block
        cms.PSet( importerName = cms.string("TICLECALClusterImporter"),
                  source = cms.InputTag("particleFlowClusterECAL"),
                  BCtoPFCMap = cms.InputTag('particleFlowSuperClusterECAL:PFClusterAssociationEBEE') ),
        cms.PSet( importerName = cms.string("TICLGenericClusterImporter"),
                  source = cms.InputTag("particleFlowClusterHCAL") ),
        cms.PSet( importerName = cms.string("TICLGenericClusterImporter"),
                  source = cms.InputTag("particleFlowBadHcalPseudoCluster") ),
        cms.PSet( importerName = cms.string("TICLGenericClusterImporter"),
                  source = cms.InputTag("particleFlowClusterHO") ),
        cms.PSet( importerName = cms.string("TICLGenericClusterImporter"),
                  source = cms.InputTag("particleFlowClusterHF") ),
        cms.PSet( importerName = cms.string("TICLGenericClusterImporter"),
                  source = cms.InputTag("particleFlowClusterPS") ),
        ),

    #linking definitions
    # you can find a list of all available linkers in:
    #  plugins/linkers
    # see : plugins/kdtrees for available KDTree Types
    # to enable a KDTree for a linking pair, write a KDTree linker
    # and set useKDTree = True in the linker PSet
    #order does not matter here since we are defining a lookup table
    linkDefinitions = cms.VPSet(
        cms.PSet( linkerName = cms.string("TICLPreshowerAndECALLinker"),
                  linkType   = cms.string("PS1:ECAL"),
                  useKDTree  = cms.bool(True) ),
        cms.PSet( linkerName = cms.string("TICLPreshowerAndECALLinker"),
                  linkType   = cms.string("PS2:ECAL"),
                  useKDTree  = cms.bool(True) ),
        cms.PSet( linkerName = cms.string("TICLTrackAndECALLinker"),
                  linkType   = cms.string("TRACK:ECAL"),
                  useKDTree  = cms.bool(True) ),
        cms.PSet( linkerName = cms.string("TICLTrackAndHCALLinker"),
                  linkType   = cms.string("TRACK:HCAL"),
                  useKDTree  = cms.bool(True),
                  trajectoryLayerEntrance = cms.string("HCALEntrance"),
                  trajectoryLayerExit = cms.string("HCALExit"),
                  nMaxHcalLinksPerTrack = cms.int32(1) # the max hcal links per track (negative values: no restriction)
        ),
        cms.PSet( linkerName = cms.string("TICLTrackAndHOLinker"),
                  linkType   = cms.string("TRACK:HO"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("TICLECALAndHCALLinker"),
                  linkType   = cms.string("ECAL:HCAL"),
                  minAbsEtaEcal = cms.double(2.5),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("TICLHCALAndHOLinker"),
                  linkType   = cms.string("HCAL:HO"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("TICLHFEMAndHFHADLinker"),
                  linkType   = cms.string("HFEM:HFHAD"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("TICLTrackAndTrackLinker"),
                  linkType   = cms.string("TRACK:TRACK"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("TICLECALAndECALLinker"),
                  linkType   = cms.string("ECAL:ECAL"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("TICLGSFAndECALLinker"),
                  linkType   = cms.string("GSF:ECAL"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("TICLTrackAndGSFLinker"),
                  linkType   = cms.string("TRACK:GSF"),
                  useKDTree  = cms.bool(False),
                  useConvertedBrems = cms.bool(True) ),
        cms.PSet( linkerName = cms.string("TICLGSFAndBREMLinker"),
                  linkType   = cms.string("GSF:BREM"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("TICLGSFAndGSFLinker"),
                  linkType   = cms.string("GSF:GSF"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("TICLECALAndBREMLinker"),
                  linkType   = cms.string("ECAL:BREM"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("TICLGSFAndHCALLinker"),
                  linkType   = cms.string("GSF:HCAL"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("TICLHCALAndBREMLinker"),
                  linkType   = cms.string("HCAL:BREM"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("TICLSCAndECALLinker"),
                  linkType   = cms.string("SC:ECAL"),
                  useKDTree  = cms.bool(False),
                  SuperClusterMatchByRef = cms.bool(True) )
        )
)

print("puppa")

for imp in TICLParticleFlowBlock.elementImporters:
  if imp.importerName.value() == "SuperClusterImporter":
    _scImporter = imp

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(_scImporter,
                                minSuperClusterPt = 1.0,
                                minPTforBypass = 0.0)

#
# kill pfTICL tracks
def _findIndicesByModule(name):
   ret = []
   for i, pset in enumerate(TICLParticleFlowBlock.elementImporters):
        if pset.importerName.value() == name:
            ret.append(i)
   return ret

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
_insertTrackImportersWithVeto = {}
_trackImporters = ['GeneralTracksImporter','ConvBremTrackImporter',
                   'ConversionTrackImporter','NuclearInteractionTrackImporter']
for importer in _trackImporters:
  for idx in _findIndicesByModule(importer):
    _insertTrackImportersWithVeto[idx] = dict(
      vetoEndcap = True,
      vetoMode = cms.uint32(2), # pfTICL candidate list
      vetoSrc = cms.InputTag("pfTICL")
    )
phase2_hgcal.toModify(
    TICLParticleFlowBlock,
    elementImporters = _insertTrackImportersWithVeto
)

#
# append track-HF linkers
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
_addTrackHFLinks = TICLParticleFlowBlock.linkDefinitions.copy()
_addTrackHFLinks.append(
  cms.PSet( linkerName = cms.string("TICLTrackAndHCALLinker"),
            linkType   = cms.string("TRACK:HFEM"),
            useKDTree  = cms.bool(True),
            trajectoryLayerEntrance = cms.string("VFcalEntrance"),
            trajectoryLayerExit = cms.string(""),
            nMaxHcalLinksPerTrack = cms.int32(-1) # Keep all track-HFEM links
          )
)
_addTrackHFLinks.append(
  cms.PSet( linkerName = cms.string("TICLTrackAndHCALLinker"),
            linkType   = cms.string("TRACK:HFHAD"),
            useKDTree  = cms.bool(True),
            trajectoryLayerEntrance = cms.string("VFcalEntrance"),
            trajectoryLayerExit = cms.string(""),
            nMaxHcalLinksPerTrack = cms.int32(-1) # Keep all track-HFHAD links for now
          )
)
phase2_tracker.toModify(
    TICLParticleFlowBlock,
    linkDefinitions = _addTrackHFLinks
)

#
# for precision timing
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
_addTiming = TICLParticleFlowBlock.elementImporters.copy()
_addTiming.append( cms.PSet( importerName = cms.string("TICLTrackTimingImporter"),
                             timeValueMap = cms.InputTag("trackTimeValueMapProducer:generalTracksConfigurableFlatResolutionModel"),
                             timeErrorMap = cms.InputTag("trackTimeValueMapProducer:generalTracksConfigurableFlatResolutionModelResolution"),
                             timeValueMapGsf = cms.InputTag("gsfTrackTimeValueMapProducer:electronGsfTracksConfigurableFlatResolutionModel"),
                             timeErrorMapGsf = cms.InputTag("gsfTrackTimeValueMapProducer:electronGsfTracksConfigurableFlatResolutionModelResolution")
                             )
                   )

from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
_addTimingLayer = TICLParticleFlowBlock.elementImporters.copy()
_addTimingLayer.append( cms.PSet( importerName = cms.string("TICLTrackTimingImporter"),
                             timeValueMap = cms.InputTag("tofPID:t0"),
                             timeErrorMap = cms.InputTag("tofPID:sigmat0"),
                             timeQualityMap = cms.InputTag("mtdTrackQualityMVA:mtdQualMVA"),
                             timeQualityThreshold = cms.double(0.5),
                             #this will cause no time to be set for gsf tracks
                             #(since this is not available for the fullsim/reconstruction yet)
                             #*TODO* update when gsf times are available
                             timeValueMapGsf = cms.InputTag("tofPID:t0"),
                             timeErrorMapGsf = cms.InputTag("tofPID:sigmat0"),
                             timeQualityMapGsf = cms.InputTag("mtdTrackQualityMVA:mtdQualMVA"),
                             )
                   )

phase2_timing.toModify(
    TICLParticleFlowBlock,
    elementImporters = _addTiming
)

phase2_timing_layer.toModify(
    TICLParticleFlowBlock,
    elementImporters = _addTimingLayer
)

#--- Use DB conditions for cuts&seeds for Run3 and phase2
from Configuration.Eras.Modifier_hcalPfCutsFromDB_cff import hcalPfCutsFromDB
hcalPfCutsFromDB.toModify( _scImporter,
                           usePFThresholdsFromDB = True)
