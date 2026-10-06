#!/usr/bin/env python3
# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""TICLv6 is v5 plus the changes declared here: the added module, the replaced module, the parameters that v6
changes in the shared modules, and the module defaults of the v6-only modules. An undeclared change fails the test.
"""

import sys

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration import presets

# module -> subtrees that differ as a whole: a new plugin type replaces the whole parameter set.
EXPECTED_WHOLESALE = {
    "ticlTracksterLinks": ("linkingPSet",),
}

# module -> {parameter path: (v5 value, v6 value)}. The path is dotted for nested PSets.
EXPECTED_DELTA = {
    "ticlTracksterLinks": {
        "linkingPSet.type": ("Skeletons", "Cornetto"),
    },
    "ticlTracksterLinksSuperclusteringDNN": {
        "linkingPSet.PIDThreshold": (0.8, 0.1),
        "linkingPSet.emissionPIDThreshold": (0.0, 0.3),
        # absent in v5, so it carries the plugin default (photon, electron) there
        "linkingPSet.tracksterPIDCategoriesToFilter": (None, [0, 1, 3]),
    },
    # The v6 candidates carry the muon kinematics; pfTICL copies their muon decisions.
    "pfTICL": {
        "muonsFromCandidates": (False, True),
    },
}

# The interpretation stage exists only in v6.
EXPECTED_ADDED = {"ticlTracksterInterpretations": "TICLInterpretationProducer"}
# label -> (v5 producer, v6 producer)
EXPECTED_REPLACED = {"ticlCandidate": ("TICLCandidateProducer", "TICLCandidateArbitrationProducer")}

# The v6-only modules run with their module defaults, except these parameters (subtrees end with a dot).
EXPECTED_FROM_DEFAULT = {
    "ticlTracksterInterpretations": ("tracksters_collections", "egamma_tracksters_collections",
                                     "pluginInferenceAlgoTracksterInferenceByPFN."),
    "ticlCandidate": (),
}
CFI = {
    "TICLInterpretationProducer": ("RecoTICL.Interpretation.ticlInterpretationProducer_cfi",
                                   "ticlInterpretationProducer"),
    "TICLCandidateArbitrationProducer": ("RecoTICL.Interpretation.ticlCandidateArbitrationProducer_cfi",
                                         "ticlCandidateArbitrationProducer"),
}


def _assemble(cfg):
    p = cms.Process("TEST")
    cfg.assemble().add_to_process(p)
    return p


def _get(module, path):
    obj = module
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj.value()


def _flatten(pset, prefix=""):
    """Every leaf parameter of a module as {dotted path: repr}, nested PSets included."""
    out = {}
    for name in pset.parameterNames_():
        p = getattr(pset, name)
        key = prefix + name
        if isinstance(p, cms.PSet):
            out.update(_flatten(p, key + "."))
        else:
            out[key] = p.dumpPython()
    return out


def _changed_paths(a, b):
    """Dotted paths that differ between two modules, including added and removed ones."""
    fa, fb = _flatten(a), _flatten(b)
    return {k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}


def main():
    v5, v6 = _assemble(presets.v5()), _assemble(presets.v6())
    ok = True

    v5_labels = set(v5.iterTICLTask.moduleNames())
    v6_labels = set(v6.iterTICLTask.moduleNames())
    if v5_labels - v6_labels or v6_labels - v5_labels != set(EXPECTED_ADDED):
        print("FAIL: only in v5 %s, only in v6 %s, expected only in v6 %s"
              % (sorted(v5_labels - v6_labels), sorted(v6_labels - v5_labels), sorted(EXPECTED_ADDED)))
        ok = False
    for label, kind in EXPECTED_ADDED.items():
        if label in v6_labels and getattr(v6, label).type_() != kind:
            print("FAIL: %s is a %s, expected %s" % (label, getattr(v6, label).type_(), kind))
            ok = False
    for label, allowed in EXPECTED_FROM_DEFAULT.items():
        module = getattr(v6, label)
        cfi_module, cfi_symbol = CFI[module.type_()]
        default = getattr(__import__(cfi_module, fromlist=[cfi_symbol]), cfi_symbol)
        moved = {p for p in _changed_paths(default, module)
                 if not any(p == a or (a.endswith(".") and p.startswith(a)) for a in allowed)}
        if moved:
            print("FAIL: %s differs from its module defaults in %s" % (label, sorted(moved)))
            ok = False
    for label, (type5, type6) in EXPECTED_REPLACED.items():
        got = (getattr(v5, label).type_(), getattr(v6, label).type_())
        if got != (type5, type6):
            print("FAIL: %s producers are %s, expected %s" % (label, got, (type5, type6)))
            ok = False

    shared = (v5_labels & v6_labels) - set(EXPECTED_REPLACED)
    differing = {l for l in shared if getattr(v5, l).dumpPython() != getattr(v6, l).dumpPython()}
    if differing != set(EXPECTED_DELTA):
        print("FAIL: v6 touches %s, expected %s"
              % (sorted(differing), sorted(EXPECTED_DELTA)))
        ok = False

    for label, params in EXPECTED_DELTA.items():
        if label not in v5_labels & v6_labels:
            continue
        for path, (want5, want6) in params.items():
            got5, got6 = _get(getattr(v5, label), path), _get(getattr(v6, label), path)
            if (got5, got6) != (want5, want6):
                print("FAIL: %s.%s is (v5=%r, v6=%r), expected (v5=%r, v6=%r)"
                      % (label, path, got5, got6, want5, want6))
                ok = False
        # Every changed parameter is declared in EXPECTED_DELTA.
        actual = _changed_paths(getattr(v5, label), getattr(v6, label))
        for sub in EXPECTED_WHOLESALE.get(label, ()):
            actual = {p for p in actual if not p.startswith(sub + ".") or p in params}
        unlisted = actual - set(params)
        vanished = set(params) - actual
        if unlisted:
            print("FAIL: %s also differs in %s, which is not declared in EXPECTED_DELTA"
                  % (label, sorted(unlisted)))
            ok = False
        if vanished:
            print("FAIL: %s no longer differs in %s, delete it from EXPECTED_DELTA"
                  % (label, sorted(vanished)))
            ok = False

    if ok:
        n = sum(len(p) for p in EXPECTED_DELTA.values())
        print("OK: v6 is v5 plus %s, with %s replaced and exactly %d parameter changes across %d modules"
              % (sorted(EXPECTED_ADDED), sorted(EXPECTED_REPLACED), n, len(EXPECTED_DELTA)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
