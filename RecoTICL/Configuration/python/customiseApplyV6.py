"""cmsDriver customise that switches a standard reco onto the pyTICL v6 chain.

It does three things, and only these, so that TICLv5 cfi defaults are left untouched
and v5 stays bit-identical when the customise is not applied:

1. Applies the pyTICL v6 preset (the two-stage chain: ticlTracksterInterpretations
   produces the final tracksters, TICLCandidateArbitrationProducer the candidates).
2. Repoints the final-trackster consumers from ticlCandidate (the v5 producer of the
   final tracksters) to ticlTracksterInterpretations (the v6 producer). The walk is
   recursive so nested PSets are covered (e.g. particleFlowClusterHGCal's
   initialClusteringStep.tracksterSrc).
3. Re-associates the swapped iterTICLTask to reconstruction_step, because
   add_to_process replaces process.iterTICLTask while the reconstruction path already
   captured the old task by reference, so the new interpretation module would not be
   scheduled otherwise.

Usage:
    cmsDriver.py ... --customise RecoTICL/Configuration/customiseApplyV6.customiseApplyV6
"""

import FWCore.ParameterSet.Config as cms
from RecoTICL.Configuration import presets

_INTERP = cms.InputTag("ticlTracksterInterpretations")
# parameter names that carry the FINAL trackster collection (not candidate refs)
_TRACKSTER_PARAMS = ("tracksterSrc", "ticlTrackstersMerge", "tracksters", "Tracksters")


def _to_interp(v):
    if isinstance(v, cms.InputTag) and v.getModuleLabel() == "ticlCandidate":
        return _INTERP
    return v


def _repoint(pset):
    for name in list(pset.parameterNames_()):
        p = getattr(pset, name)
        if isinstance(p, cms.PSet):
            _repoint(p)
        elif isinstance(p, cms.VPSet):
            for sub in p:
                _repoint(sub)
        elif name in _TRACKSTER_PARAMS:
            if isinstance(p, cms.InputTag):
                setattr(pset, name, _to_interp(p))
            elif isinstance(p, cms.VInputTag):
                setattr(pset, name, cms.VInputTag([_to_interp(t) for t in p]))


def customiseApplyV6(process):
    presets.v6().assemble().add_to_process(process)
    for collection in (process.producers_(), process.filters_(), process.analyzers_()):
        for module in collection.values():
            _repoint(module)
    if hasattr(process, "reconstruction_step"):
        process.reconstruction_step.associate(process.iterTICLTask)
    return process
