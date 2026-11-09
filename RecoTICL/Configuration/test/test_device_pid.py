#!/usr/bin/env python3
"""TICLConfig.device_pid schedules any model on the CLUEstering tracksters, makeTracksterInference makes every family,
and the invalid requests are rejected."""

import os
import sys

from RecoTICL.Configuration.model import TICLConfig
from RecoTICL.Configuration.presets import v6
from RecoTICL.Configuration.tracksterInference_cff import makeTracksterInference

# model -> arguments of device_pid, and whether the module reads the layer clusters
MODELS = {
    "ShowerShape": (dict(), False),
}


def found(path):
    return any(os.path.isfile(os.path.join(area, path)) for area in os.environ["CMSSW_SEARCH_PATH"].split(":"))


def pid_module(target, model, overrides):
    config = TICLConfig("test", target=target).iteration("CLUEstering").preset().device_pid(model=model, **overrides)
    return config, config.assemble().modules


def main():
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    for model, (overrides, reads_clusters) in MODELS.items():
        _, modules = pid_module("offline", model, overrides)
        pid = modules["ticlSoATrackstersPID"]
        check(pid.type_() == "Trackster%sInferenceProducer@alpaka" % model, "%s: module %s" % (model, pid.type_()))
        check(pid.tracksters.value() == "ticlSoATracksters", "%s: reads %s" % (model, pid.tracksters.value()))
        check(modules["ticlTrackstersCLUEstering"].src.value() == "ticlSoATrackstersPID", model + ": conversion input")
        check(found(pid.pidModel.value()), "%s: no file %s" % (model, pid.pidModel.value()))
        if overrides.get("doRegression"):
            check(found(pid.regressionModel.value()), "%s: no file %s" % (model, pid.regressionModel.value()))
        check(hasattr(pid, "layerClusters") == reads_clusters, model + ": layer clusters")
        check(hasattr(pid, "alpaka"), model + ": not cloned from its cfi")

        # the layer clusters of the target
        _, modules = pid_module("HLT", model, overrides)
        pid = modules["hltTiclSoATrackstersPID"]
        check(pid.tracksters.value() == "hltTiclSoATracksters", "%s: HLT reads %s" % (model, pid.tracksters.value()))
        if reads_clusters:
            check(pid.layerClusters.value() == "hltHgcalSoAMergeLayerClusters", model + ": HLT layer clusters")

    for arguments in (dict(model="BDT"), dict(model="CNN", doRegression=True),
                      dict(model="ShowerShape", doRegression=True)):
        try:
            makeTracksterInference(**arguments)
            failures.append("accepted %s" % arguments)
        except ValueError:
            pass

    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        return 1
    print("OK: device_pid and makeTracksterInference for %s" % ", ".join(MODELS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
