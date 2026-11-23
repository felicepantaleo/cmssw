#!/usr/bin/env python3
"""With ticl_v6, the Phase-2 HLT menus run TICLv6: on every path, each TICL product that a module reads comes from a
module earlier on the same path, and no TICLv5-only module stays. The unseeded e/gamma candidates read the TICLv6
e/gamma superclusters. Without ticl_v6, the menus run TICLv5. In both cases, only a path with the HGCAL PF clusters
runs the HGCAL PF rechits."""

import subprocess
import sys

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.SequenceTypes import ModuleNodeVisitor

MENUS = (("HLT_NGTScouting_cff", ("alpaka", "ngtScouting")), ("HLT_75e33_timing_cff", ()), ("HLT_75e33_cff", ()))
V5_ONLY = ("hltFilteredLayerClustersCLUE3DHigh", "hltTiclSeedingGlobal", "hltTiclTrackstersCLUE3DHigh",
           "hltFilteredLayerClustersRecovery")
TICL_PREFIXES = ("hltTicl", "hltFilteredLayerClusters", "hltPfTICL", "hltHgcalSoAMergeLayerClusters", "hltHGCalRecHitMap")
HGCAL_PF_RECHITS = ("hltParticleFlowRecHitHGC", "hltParticleFlowRecHitHGCL1Seeded")
# Inputs that a module reads only when a switch of the module is on.
SWITCHES = {"timingSoA": "useMTDTiming", "gsf_tracks": "useGsfTracks"}


def input_labels(pset):
    for name in pset.parameterNames_():
        value = getattr(pset, name)
        if name in SWITCHES and hasattr(pset, SWITCHES[name]) and not getattr(pset, SWITCHES[name]).value():
            continue
        if name == "layer_clusters_hfnose_tiles" or name == "layer_clusters_barrel_tiles":
            continue
        if isinstance(value, cms.InputTag):
            yield value.getModuleLabel()
        elif isinstance(value, cms.VInputTag):
            for tag in value:
                yield tag.getModuleLabel() if isinstance(tag, cms.InputTag) else str(tag).split(":")[0]
        elif isinstance(value, cms.PSet):
            yield from input_labels(value)
        elif isinstance(value, cms.VPSet):
            for item in value:
                yield from input_labels(item)


def check(menu, names, v6):
    from Configuration.Eras.Era_Phase2C26I13M9_cff import Phase2C26I13M9
    names = tuple(names) + (("ticl_v6",) if v6 else ())
    modifiers = [getattr(__import__("Configuration.ProcessModifiers.%s_cff" % n, fromlist=[n]), n) for n in names]
    process = cms.Process("HLTX", Phase2C26I13M9, *modifiers)
    process.load("HLTrigger.Configuration." + menu)
    failures = []
    candidate = "TICLCandidateArbitrationProducer" if v6 else "TICLCandidateProducer"
    if process.hltTiclCandidate.type_() != candidate:
        failures.append("hltTiclCandidate is a %s" % process.hltTiclCandidate.type_())
    if v6:
        failures += ["%s is in the process" % label for label in V5_ONLY if hasattr(process, label)]
        if hasattr(process, "hltTiclEGammaSuperClusterProducerUnseeded"):
            failures.append("hltTiclEGammaSuperClusterProducerUnseeded is in the process")
        if (hasattr(process, "hltEgammaCandidatesUnseeded") and
                process.hltEgammaCandidatesUnseeded.scIslandEndcapProducer.getModuleLabel() !=
                "hltTiclEGammaSuperClustersUnseeded"):
            failures.append("hltEgammaCandidatesUnseeded does not read hltTiclEGammaSuperClustersUnseeded")
    for name, path in process.paths_().items():
        modules = []
        path.visit(ModuleNodeVisitor(modules))
        seen = set()
        for module in modules:
            for used in input_labels(module):
                if used.startswith(TICL_PREFIXES) and used not in seen:
                    failures.append("%s: %s reads %s, which no earlier module makes" % (name, module.label_(), used))
            seen.add(module.label_())
        if "hltParticleFlowClusterHGCal" not in seen:
            failures += ["%s runs %s" % (name, label) for label in HGCAL_PF_RECHITS if label in seen]
    for failure in failures:
        print("FAIL [%s, %s]: %s" % (menu, "ticl_v6" if v6 else "default", failure))
    return 1 if failures else 0


def main():
    if len(sys.argv) > 1:
        menu, names = MENUS[int(sys.argv[1])]
        return check(menu, names, sys.argv[2] == "v6")
    # One process for each menu and modifier set: a modifier acts when a configuration file is first imported.
    status = 0
    for index in range(len(MENUS)):
        for leg in ("v5", "v6"):
            status |= subprocess.call([sys.executable, __file__, str(index), leg])
    print("TICL in the HLT menus: %s" % ("FAILED" if status else "OK"))
    return status


if __name__ == "__main__":
    sys.exit(main())
