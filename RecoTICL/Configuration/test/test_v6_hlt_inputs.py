#!/usr/bin/env python3
"""No producer of the v6 configuration on the HLT target reads an offline input: tracks, muons, GSF tracks, MTD
timing, layer clusters and tracksters all come from the HLT target."""

import sys

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration import presets

OFFLINE_INPUTS = {"generalTracks", "muons1stStep", "electronGsfTracks", "mtdSoA", "hgcalMergeLayerClusters"}


def input_labels(pset, prefix=""):
    for name in pset.parameterNames_():
        value = getattr(pset, name)
        if isinstance(value, cms.InputTag):
            yield prefix + name, value.getModuleLabel()
        elif isinstance(value, cms.VInputTag):
            for tag in value:
                yield prefix + name, (tag.getModuleLabel() if isinstance(tag, cms.InputTag) else str(tag).split(":")[0])
        elif isinstance(value, cms.PSet):
            yield from input_labels(value, prefix + name + ".")


def main():
    process = cms.Process("TEST")
    presets.v6(target="HLT").assemble().add_to_process(process)
    failures = []
    for label, module in sorted(process.producers_().items()):
        # Inputs that a switch turns off are not read; the HFNose tiles are read only by an HFNose iteration.
        off = {name for name, switch in (("timingSoA", "useMTDTiming"), ("gsf_tracks", "useGsfTracks"))
               if hasattr(module, switch) and not getattr(module, switch).value()}
        if hasattr(module, "detector") and module.detector.value() != "HFNose":
            off.add("layer_clusters_hfnose_tiles")
        for name, used in input_labels(module):
            if name in off:
                continue
            if used in OFFLINE_INPUTS or (used.startswith("ticl") and not used.startswith("hltTicl")):
                failures.append("%s.%s reads the offline input %s" % (label, name, used))
    for failure in failures:
        print("FAIL:", failure)
    print("v6 HLT inputs: %s" % ("FAILED" if failures else "OK"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
