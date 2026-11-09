#!/usr/bin/env python3
"""reconstruction_hgcalOnly reads only the HGCAL digis and the products of its own modules: by default, with ticl_v6
and with ticl_v6 on the device. No module reads tracks, vertices, MTD or muons."""

import subprocess
import sys

import FWCore.ParameterSet.Config as cms

MODIFIER_SETS = ((), ("ticl_v6",), ("ticl_v6", "ticl_v6_heterogeneous"))
# the products of RAW2DIGI
EXTERNAL = {"hgcalDigis", "hfnoseDigis"}
# Inputs that a module reads only under a condition on its own parameters.
CONDITIONAL = {"layer_clusters_hfnose_tiles": lambda m: m.detector.value() == "HFNose",
               "layer_HFNose_clusters": lambda m: m.detector.value() == "HFNose"}


def input_labels(pset):
    for name in pset.parameterNames_():
        if name in CONDITIONAL and not CONDITIONAL[name](pset):
            continue
        value = getattr(pset, name)
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


def check(names):
    from Configuration.Eras.Era_Phase2C26I13M9_cff import Phase2C26I13M9
    modifiers = [getattr(__import__("Configuration.ProcessModifiers.%s_cff" % n, fromlist=[n]), n) for n in names]
    process = cms.Process("TEST", Phase2C26I13M9, *modifiers)
    process.load("Configuration.StandardSequences.Reconstruction_cff")
    labels = set(process.reconstruction_hgcalOnly.moduleNames())
    failures = []
    for label in sorted(labels):
        module = getattr(process, label)
        if not isinstance(module, (cms.EDProducer, cms.EDFilter)):
            continue
        for used in input_labels(module):
            if used and used not in labels and used not in EXTERNAL:
                failures.append("%s reads %s" % (label, used))
    for failure in failures:
        print("FAIL [%s]: %s" % (", ".join(names), failure))
    return 1 if failures else 0


def main():
    if len(sys.argv) > 1:
        return check([n for n in sys.argv[1].split(",") if n])
    # One process per interpreter: a second cms.Process with other modifiers fails in the same session.
    rc = 0
    for names in MODIFIER_SETS:
        rc |= subprocess.call([sys.executable, __file__, ",".join(names)])
    print("reconstruction_hgcalOnly inputs: %s" % ("FAILED" if rc else "OK"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
