# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""The single place that says WHICH TICL collections exist.

Everything downstream keys off this list: the hit-to-trackster and trackster-to-simTrackster
associators, the truth-branch association labels and their DQM validators and harvesters,
the TICL dumper, and the HGCal nano tables. When the list and the configured chain
disagree the failure is silent, because a consumer of a collection that is not produced
either throws a ProductNotFound the framework is often told to continue past, or simply
books an empty folder that looks like a finished measurement. So the list must be derived
from the chain, not maintained beside it.

``tracksterLabels`` does that derivation from an assembled pyTICL configuration, which is
the only object that knows what was actually scheduled. ``ticlIterLabelsPSet`` keeps the
legacy value for the mainline ``RecoHGCal/TICL`` chain, which is what importers get today;
a configuration that departs from it (the v6 two-stage chain, an era that adds an
iteration) must retarget its consumers with the derived list rather than assume this one.
"""

import FWCore.ParameterSet.Config as cms

# Module types that emit a vector<ticl::Trackster>, plus the candidate assembler whose
# collection is tracked alongside them.
_TRACKSTER_TYPES = ("TrackstersProducer", "TracksterLinksProducer")
_CANDIDATE_TYPES = ("TICLCandidateProducer", "TICLCandidateArbitrationProducer")


def tracksterLabels(assembled, config=None, includeCandidate=True):
    """Trackster collection labels an assembled pyTICL configuration actually produces.

    An iteration marked ``persist = False`` is intermediate (Recovery is linked, never
    kept), so it is excluded, matching the Event Content. Order follows the assembly, so
    the pattern-recognition collections come before the linked ones.
    """
    transient = set()
    if config is not None:
        from RecoTICL.Configuration.assembler import trackster_label

        transient = {trackster_label(it.name) for it in config.iterations if not it.persist}

    labels = []
    for label, module in assembled.modules.items():
        kind = module.type_()
        if kind in _TRACKSTER_TYPES and label not in transient:
            labels.append(label)
        elif includeCandidate and kind in _CANDIDATE_TYPES:
            labels.append(label)
    return labels


def tracksterLabelsInProcess(process, includeCandidate=True):
    """Trackster collections actually scheduled in a process, in scheduling order.

    The counterpart of tracksterLabels for consumers that run after the chain is built
    (a RECO-step customise, a validation sequence): it asks the process what exists
    rather than trusting a list written elsewhere, so it is correct for the mainline
    chain, for anything pyTICL assembles, and for a configuration nobody anticipated.
    """
    scheduled = _scheduledModuleLabels(process)
    labels = []
    for label, module in process.producers_().items():
        if label not in scheduled:
            continue  # defined by a loaded cff but not run: consuming it books empty folders
        kind = module.type_()
        if kind in _TRACKSTER_TYPES:
            labels.append(label)
        elif includeCandidate and kind in _CANDIDATE_TYPES:
            labels.append(label)
    return labels


def _scheduledModuleLabels(process):
    """Labels of the modules the process actually runs.

    process.producers_() is every module a loaded cff DEFINED, which for a RECO job is
    every TICL iteration that exists anywhere, not the handful this chain schedules. A
    consumer keyed on the defined set asks for products nobody produces, which surfaces
    as ProductNotFound or, when the job is told to continue past that, as an empty folder
    that reads like a finished measurement.
    """
    from FWCore.ParameterSet.SequenceTypes import ModuleNodeVisitor

    modules = []
    schedule = getattr(process, "schedule", None)
    paths = list(schedule) if schedule is not None else list(process.paths_().values()) + list(
        process.endpaths_().values())
    for path in paths:
        try:
            path.visit(ModuleNodeVisitor(modules))
        except Exception:
            continue
    return {m.label_() for m in modules if hasattr(m, "label_")}


# The mainline RecoHGCal/TICL chain, which is what every current importer expects. Kept
# as a literal because that chain is built by cff files rather than assembled, so there
# is nothing to derive it from; use tracksterLabels for anything pyTICL assembles.
ticlIterLabels = [
    "ticlTrackstersCLUE3DHigh",
    "ticlTracksterLinks",
    "ticlCandidate",
    "ticlTracksterLinksSuperclusteringDNN",
]

ticlIterLabelsPSet = cms.PSet(labels=cms.vstring(*ticlIterLabels))
