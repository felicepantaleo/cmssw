"""cmsDriver customise that switches a standard reco onto the pyTICL v6 chain.

It does four things, and only these, so that TICLv5 cfi defaults are left untouched
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
4. Retargets HGCalValidator and its trackster-to-simTrackster associators onto the
   collections this chain actually produces, when a VALIDATION step is present.

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


def _retargetValidation(process, cfg):
    """Point HGCalValidator and its associators at the collections this chain produces.

    The mainline label list names ``ticlCandidate`` as a trackster collection. In the
    two-stage chain that label holds ``vector<TICLCandidate>`` and the final tracksters
    come from ``ticlTracksterInterpretations``, so the mainline list books a full set of
    monitor elements that can never be filled and nothing throws.

    Parameters are copied onto the EXISTING module objects rather than replacing them.
    A Task holds its modules by reference, so assigning a new object to the process
    attribute leaves the Task pointing at the old one and the module stops being
    scheduled, which silently empties every folder rather than just the wrong ones.
    """
    if not hasattr(process, "hgcalValidator"):
        return
    modules, _, labels = cfg.build_validation()

    # The trackster-to-simTrackster associators read per-label products from the
    # layer-cluster and hit associators, so those have to learn the same list or the
    # ByLCs producer throws on the first label it was not told about.
    upstream = cms.VInputTag(*[cms.InputTag(l) for l in labels]
                             + [cms.InputTag("ticlSimTracksters"),
                                cms.InputTag("ticlSimTracksters", "fromCPs")])
    for name in ("allLayerClusterToTracksterAssociations", "allHitToTracksterAssociations"):
        if hasattr(process, name):
            getattr(process, name).tracksterCollections = upstream

    for name, module in modules.items():
        if not hasattr(process, name):
            continue
        target = getattr(process, name)
        for pname in module.parameterNames_():
            setattr(target, pname, getattr(module, pname))


def customiseApplyV6(process):
    cfg = presets.v6()
    cfg.assemble().add_to_process(process)
    for collection in (process.producers_(), process.filters_(), process.analyzers_()):
        for module in collection.values():
            _repoint(module)
    if hasattr(process, "reconstruction_step"):
        process.reconstruction_step.associate(process.iterTICLTask)
    _retargetValidation(process, cfg)
    return process
