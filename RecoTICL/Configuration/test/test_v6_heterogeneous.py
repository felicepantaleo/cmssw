#!/usr/bin/env python3
"""TICLv6 on the device is v6 with the Recovery iteration and the Cornetto linking as portable modules: the same
persisted products under the same labels, the same Cornetto parameters, and the ticl_v6_heterogeneous modifier gives
the modules of the preset. pyTICL refuses a device chain that cannot give the result of the host chain."""

import sys
import types

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C26I13M9_cff import Phase2C26I13M9
from Configuration.ProcessModifiers.ticl_v6_cff import ticl_v6
from Configuration.ProcessModifiers.ticl_v6_heterogeneous_cff import ticl_v6_heterogeneous

from RecoTICL.Configuration import presets
from RecoTICL.Configuration.backend import ALPAKA_BACKEND, apply_backends
from RecoTICL.Configuration.catalog import GPU
from RecoTICL.Configuration.model import PyTICLError

# the portable modules and the conversions that the device chain adds, and the host modules it removes
EXPECTED_ADDED = {
    "ticlTrackstersCLUEsteringRefs": "TracksterCollectionRefsProducer@alpaka",
    "ticlLayerClusterMaskCLUEsteringSoA": "LayerClusterMaskSoAProducer@alpaka",
    "filteredLayerClustersRecoverySoA": "FilteredLayerClustersSoAProducer@alpaka",
    "ticlTrackstersRecoverySoA": "HGCalRecoveryTrackstersSoAProducer@alpaka",
    "ticlTrackstersCLUEsteringPidSoA": "TracksterPidSoAProducer@alpaka",
    "ticlTracksterLinksInputRefs": "TracksterCollectionRefsProducer@alpaka",
    "ticlTracksterLinksSoA": "TracksterLinksSoAProducer@alpaka",
}
EXPECTED_REMOVED = {"ticlSeedingGlobal", "filteredLayerClustersRecovery"}
# label -> (host producer, producer with the device chain)
EXPECTED_REPLACED = {"ticlTrackstersRecovery": ("TrackstersProducer", "HGCalTrackstersFromSoAProducer"),
                     "ticlTracksterLinks": ("TracksterLinksProducer", "TracksterLinksFromSoAProducer")}
# the modules of the modifier that the test compares with the preset
LABELS = tuple(EXPECTED_ADDED) + tuple(EXPECTED_REPLACED)


class _Modules:
    """The assembled modules as attributes, as a process has them. One cms.Process per interpreter: the modifier
    process below must be the only one."""

    def __init__(self, cfg):
        self.modules = cfg.assemble().modules

    def __getattr__(self, label):
        return self.modules[label]


def refusals():
    """The configurations that pyTICL refuses, each with the reason."""
    def links_override(**kw):
        def change(cfg):
            cfg.links_spec.overrides.update(kw)
        return change

    def linking_pset(**kw):
        def change(cfg):
            pset = cfg.links_spec.overrides["linkingPSet"].clone(**kw)
            cfg.links_spec.overrides["linkingPSet"] = pset
        return change

    def pid_cut(cfg):
        cfg._by_name["CLUEstering"].trackster_extra["cutHadProb"] = cms.double(0.5)

    def recovery_on_gpu(cfg):
        cfg._by_name["Recovery"].backend = GPU

    def five_collections(cfg):
        cfg.links_spec.collections = ["CLUEstering", "Recovery"] * 2 + ["CLUEstering"]

    return {
        "regression and PID of the linked tracksters": links_override(regressionAndPid=cms.bool(True)),
        "inference of the linked tracksters": links_override(inferenceAlgo=cms.string("TracksterInferenceByPFN")),
        "second pass": linking_pset(mergeFragments=cms.bool(True)),
        "PID cut of the host tracksters": pid_cut,
        "Recovery on GPU after CLUEstering on CPU": recovery_on_gpu,
        "more than 4 collections": five_collections,
    }


def check_refusals(failures):
    for reason, change in refusals().items():
        cfg = presets.v6(heterogeneous=True)
        change(cfg)
        try:
            cfg.validate()
            failures.append("pyTICL accepts a device chain with: %s" % reason)
        except PyTICLError:
            pass


def check_backends(failures):
    # CLUEstering on GPU puts the whole device chain on GPU
    cfg = presets.v6(heterogeneous=True)
    cfg._by_name["CLUEstering"].backend = GPU
    assembled = cfg.validate()
    # a stand-in for the process: a second cms.Process with other modifiers fails in the same interpreter
    apply_backends(assembled, types.SimpleNamespace())
    for label in EXPECTED_ADDED:
        backend = assembled.modules[label].alpaka.backend.value()
        if backend != ALPAKA_BACKEND[GPU]:
            failures.append("%s runs on %s with CLUEstering on GPU" % (label, backend))


def check_hlt_labels(failures):
    labels = presets.v6(target="HLT", heterogeneous=True).assemble().modules
    for label in labels:
        if label.startswith("ticl") or label.startswith("filteredLayerClusters"):
            failures.append("the HLT device chain has the offline label %s" % label)


def main():
    failures = []
    check_refusals(failures)
    check_backends(failures)
    check_hlt_labels(failures)
    host, device = presets.v6(), presets.v6(heterogeneous=True)
    device.validate()
    h, d = _Modules(host), _Modules(device)

    h_labels, d_labels = set(h.modules), set(d.modules)
    if d_labels - h_labels != set(EXPECTED_ADDED) or h_labels - d_labels != EXPECTED_REMOVED:
        failures.append("only on the device %s, only on the host %s"
                        % (sorted(d_labels - h_labels), sorted(h_labels - d_labels)))
    for label, kind in EXPECTED_ADDED.items():
        if label in d_labels and d.modules[label].type_() != kind:
            failures.append("%s is a %s, expected %s" % (label, d.modules[label].type_(), kind))
    for label, (kind_h, kind_d) in EXPECTED_REPLACED.items():
        got = (h.modules[label].type_(), d.modules[label].type_())
        if got != (kind_h, kind_d):
            failures.append("%s producers are %s, expected %s" % (label, got, (kind_h, kind_d)))
    # every other module of the chain is the same
    for label in sorted((h_labels & d_labels) - set(EXPECTED_REPLACED)):
        if h.modules[label].dumpPython() != d.modules[label].dumpPython():
            failures.append("%s differs between the host and the device chain" % label)
    # the same persisted products
    if host.event_content() != device.event_content():
        failures.append("the event content differs: %s vs %s" % (host.event_content(), device.event_content()))

    # the device linking has the Cornetto parameters of the host linking, as double
    pset = h.ticlTracksterLinks.linkingPSet
    for name in pset.parameterNames_():
        if name in ("type", "algo_verbosity"):
            continue
        want = getattr(pset, name).value()
        got = getattr(d.ticlTracksterLinksSoA, name, None)
        if got is None or got.value() != want:
            failures.append("ticlTracksterLinksSoA.%s is %r, the host linking has %r" % (name, got, want))
    # the device linking reads the two collections in the order of the host linking, with the PID of CLUEstering
    def labels(tags):
        return [cms.InputTag(t).getModuleLabel() if isinstance(t, str) else t.getModuleLabel() for t in tags]
    if labels(d.ticlTracksterLinksSoA.tracksters) != ["ticlSoATracksters", "ticlTrackstersRecoverySoA"]:
        failures.append("ticlTracksterLinksSoA.tracksters is %s" % d.ticlTracksterLinksSoA.tracksters)
    if labels(d.ticlTracksterLinksSoA.pid) != ["ticlTrackstersCLUEsteringPidSoA", ""]:
        failures.append("ticlTracksterLinksSoA.pid is %s" % d.ticlTracksterLinksSoA.pid)
    # the Recovery on the device has the filter and the energy threshold of the host Recovery
    if d.filteredLayerClustersRecoverySoA.algo_number.value() != h.filteredLayerClustersRecovery.algo_number.value():
        failures.append("filteredLayerClustersRecoverySoA.algo_number differs from the host filter")
    if (d.filteredLayerClustersRecoverySoA.min_cluster_size.value()
            != h.filteredLayerClustersRecovery.min_cluster_size.value()):
        failures.append("filteredLayerClustersRecoverySoA.min_cluster_size differs from the host filter")
    if (d.ticlTrackstersRecoverySoA.minEnergy.value()
            != h.ticlTrackstersRecovery.pluginPatternRecognitionByRecovery.minEnergy.value()):
        failures.append("ticlTrackstersRecoverySoA.minEnergy differs from the host Recovery")

    # the modifier gives the modules of the preset
    process = cms.Process("TEST", Phase2C26I13M9, ticl_v6, ticl_v6_heterogeneous)
    process.load("RecoTICL.Configuration.iterativeTICL_cff")
    reference = d.modules
    for label in LABELS:
        if getattr(process, label).dumpPython() != reference[label].dumpPython():
            failures.append("%s of the modifier differs from presets.v6(heterogeneous=True)" % label)
    scheduled = set(process.iterTICLTask.moduleNames())
    for label in LABELS:
        if label not in scheduled:
            failures.append("%s is not in iterTICLTask" % label)
    for label in EXPECTED_REMOVED:
        if label in scheduled:
            failures.append("%s is in iterTICLTask" % label)

    for failure in failures:
        print("FAIL: " + failure)
    print("v6 heterogeneous: %s" % ("FAILED" if failures else "OK"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
