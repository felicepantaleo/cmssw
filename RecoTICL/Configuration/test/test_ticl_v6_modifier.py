#!/usr/bin/env python3
"""The ticl_v6 process modifier gives the modules of the pyTICL preset v6, and the preset validates."""

import sys

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C26I13M9_cff import Phase2C26I13M9
from Configuration.ProcessModifiers.ticl_v6_cff import ticl_v6

from RecoTICL.Configuration import presets

LABELS = ("ticlTracksterLinks", "ticlTracksterLinksSuperclusteringDNN", "ticlTracksterInterpretations", "ticlCandidate",
          "pfTICL")


def main():
    failures = []
    process = cms.Process("TEST", Phase2C26I13M9, ticl_v6)
    process.load("RecoTICL.Configuration.iterativeTICL_cff")
    reference = presets.v6().validate().modules
    for label in LABELS:
        if getattr(process, label).dumpPython() != reference[label].dumpPython():
            failures.append("%s differs from presets.v6()" % label)
    if "ticlTracksterInterpretations" not in process.iterTICLTask.moduleNames():
        failures.append("ticlTracksterInterpretations is not in iterTICLTask")

    # The consumers of the final tracksters read them from the interpretation stage.
    process.load("RecoTICL.Configuration.ticlEventContent_cff")
    for block in ("TICL_RECO", "TICL_FEVT", "TICL_FEVTHLT"):
        if "keep *_ticlTracksterInterpretations_*_*" not in getattr(process, block).outputCommands:
            failures.append("%s does not keep ticlTracksterInterpretations" % block)
    # MiniAOD keeps the candidates and the hypotheses; AOD keeps no product of the two modules.
    process.load("Configuration.EventContent.EventContent_cff")
    for block in ("MINIAODEventContent", "MINIAODSIMEventContent"):
        for keep in ("keep TICLCandidates_ticlCandidate__*",
                     "keep ticlInterpretationHypotheses_ticlTracksterInterpretations_hypotheses_*"):
            if keep not in getattr(process, block).outputCommands:
                failures.append("%s does not have '%s'" % (block, keep))
    for command in process.AODSIMEventContent.outputCommands:
        if "ticlCandidate" in command or "ticlTracksterInterpretations" in command:
            failures.append("AODSIMEventContent has '%s'" % command)
    # The electrons and photons of AOD and RECO point to the TICL e/gamma superclusters.
    for block in ("AODSIMEventContent", "RECOSIMEventContent"):
        for keep in ("keep recoSuperClusters_ticlEGammaSuperClusters__*",
                     "keep recoCaloClusters_ticlEGammaSuperClusters_clusters_*"):
            if keep not in getattr(process, block).outputCommands:
                failures.append("%s does not have '%s'" % (block, keep))
    process.load("Validation.HGCalValidation.HGCalValidator_cff")
    process.load("Validation.HGCalValidation.HLTHGCalValidator_cff")
    process.load("DPGAnalysis.HGCalNanoAOD.hgcalTICLCandidates_cfi")
    process.load("RecoEcal.EgammaClusterProducers.particleFlowSuperClusteringSequence_cff")
    process.load("RecoLocalCalo.Configuration.hgcalLocalReco_cff")
    process.load("RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeeds_cff")
    process.load("RecoParticleFlow.PFTracking.mergedElectronSeeds_cfi")
    process.load("RecoEgamma.EgammaPhotonProducers.photonCore_cfi")
    expected = {
        "hgcalValidator.mergedTracksters": "ticlTracksterInterpretations",
        "hltHgcalValidator.mergedTracksters": "hltTiclCandidate",
        "ticlCandidateExtraTable.tracksters": "ticlTracksterInterpretations",
        "ticlEGammaTracksters.interpretations": "ticlTracksterInterpretations",
        "ticlEGammaSuperClusters.eGammaTracksters": "ticlEGammaTracksters",
        "ticlElectronSuperClusters.superClusters": "ticlEGammaSuperClusters",
        "ticlElectronSuperClusters.eGammaTracksters": "ticlEGammaTracksters",
        "ecalDrivenElectronSeeds.endcapSuperClusters": "ticlElectronSuperClusters",
        "ticlElectronSeeds.seeds": "ecalDrivenElectronSeeds",
        "ticlElectronSeeds.superClusters": "ticlEGammaSuperClusters",
        "electronMergedSeeds.EcalBasedSeeds": "ticlElectronSeeds",
        "photonCoreHGC.scIslandEndcapProducer": "ticlEGammaSuperClusters",
    }
    # The HGCAL e/gamma superclusters are the TICL e/gamma tracksters: no HGCAL PF clusters, no PF superclusters.
    for label in ("ticlEGammaTracksters", "ticlEGammaSuperClusters", "ticlElectronSuperClusters", "ticlElectronSeeds"):
        if label not in process.iterTICLTask.moduleNames():
            failures.append("%s is not in iterTICLTask" % label)
    if "particleFlowClusterHGCal" in process.hgcalLocalRecoTask.moduleNames():
        failures.append("particleFlowClusterHGCal is in hgcalLocalRecoTask")
    if "particleFlowSuperClusterHGCal" in process.particleFlowSuperClusteringTask.moduleNames():
        failures.append("particleFlowSuperClusterHGCal is in particleFlowSuperClusteringTask")
    # The e/gamma DQM merges the TICL e/gamma superclusters.
    process.load("DQMOffline.EGamma.electronAnalyzerSequence_cff")
    for label in ("mergedSuperClusters", "mergedSuperClustersHGC"):
        sources = [tag.getModuleLabel() for tag in getattr(process, label).src]
        if "ticlEGammaSuperClusters" not in sources or "particleFlowSuperClusterHGCal" in sources:
            failures.append("%s reads %s" % (label, sources))
    for path, label in expected.items():
        obj = process
        for part in path.split("."):
            obj = getattr(obj, part)
        if obj.getModuleLabel() != label:
            failures.append("%s is %s, expected %s" % (path, obj.getModuleLabel(), label))
    if hasattr(process.ticlCandidateExtraTable, "linkedTracksters"):
        failures.append("ticlCandidateExtraTable still reads linkedTracksters")
    for failure in failures:
        print("FAIL:", failure)
    print("ticl_v6 modifier: %s" % ("FAILED" if failures else "OK"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
