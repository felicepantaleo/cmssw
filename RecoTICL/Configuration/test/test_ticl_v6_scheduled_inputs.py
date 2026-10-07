#!/usr/bin/env python3
"""With ticl_v6, alone and with each superclustering modifier, no module of iterTICLTask reads a module that is in
no task."""

import subprocess
import sys

import FWCore.ParameterSet.Config as cms

MODIFIER_SETS = (
    ("ticl_v6",),
    ("ticl_v6", "ticl_superclustering_mustache_ticl"),
)


# Inputs that a module reads only under a condition on its own parameters.
CONDITIONAL = {"layer_clusters_hfnose_tiles": lambda m: m.detector.value() == "HFNose"}


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
    process.load("RecoTICL.Configuration.iterativeTICL_cff")
    scheduled = set(process.iterTICLTask.moduleNames())
    # A module of another task (HFNose, barrel) can run; a module of no task cannot.
    defined = set(process.producers_()) - {m for task in process.tasks_().values() for m in task.moduleNames()}
    failures = []
    for label in sorted(scheduled):
        module = getattr(process, label, None)
        if module is None:
            continue
        for used in input_labels(module):
            if used in defined:
                failures.append("%s reads %s, which is in no task" % (label, used))
    for failure in failures:
        print("FAIL [%s]: %s" % (", ".join(names), failure))
    return 1 if failures else 0


def main():
    if len(sys.argv) > 1:
        return check(sys.argv[1].split(","))
    # One process per interpreter: a second cms.Process with other modifiers fails in the same session.
    rc = 0
    for names in MODIFIER_SETS:
        rc |= subprocess.call([sys.executable, __file__, ",".join(names)])
    print("ticl_v6 scheduled inputs: %s" % ("FAILED" if rc else "OK"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
