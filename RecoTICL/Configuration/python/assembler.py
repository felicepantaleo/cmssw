# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Assemble a :class:`~RecoTICL.Configuration.model.TICLConfig` into ``cms`` objects.

Each node is built by cloning the *real* ``_cfi`` default (so values match the
baseline) and applying (a) the algorithm overrides recorded in the model and
(b) the *plumbing* ``InputTag``s computed here from the iteration graph and the
selected :class:`~RecoTICL.Configuration.target.Target` (offline or HLT).  The result
is a set of labelled modules plus the grouping (``cms.Task`` offline /
``cms.Sequence`` HLT).
"""

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration.catalog import CATALOG
from RecoTICL.Configuration.model import PyTICLError
from RecoTICL.Configuration.tracksterInference_cff import makeTracksterInference


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _cfi_default(key):
    spec = CATALOG[key]
    mod = __import__(spec.cfi_module, fromlist=[spec.cfi_symbol])
    return getattr(mod, spec.cfi_symbol)


# offline default label helpers (used by the validator and by callers that
# pre-date targets); the target's own helpers are preferred inside assembly.
def trackster_label(name):
    return "ticlTracksters" + name


def filter_label(name):
    return "filteredLayerClusters" + name


def step_task_name(name):
    return "ticl" + name + "StepTask"


def iteration_labels(it, target):
    """The labels of the modules of iteration ``it`` whose backend the iteration selects: the cluster filter and the
    tracksters, or for CLUEstering its device modules, with the merge on the device when its step schedules it, or
    for an iteration on the device its mask, filter and tracksters on the device."""
    if it.pattern_type == "CLUEstering":
        merge = [target.merged_lc_soa] if target.merged_lc_soa_cfi is not None else []
        pid = [target.soa_pid_tracksters_label] if it.device_pid is not None else []
        return merge + [target.cluestering_label, target.soa_tracksters_label] + pid
    if it.on_device:
        return [target.refs_label(it.masks_from), target.mask_soa_label(it.masks_from),
                target.filter_soa_label(it.name), target.soa_trackster_label(it.name)]
    return [target.filter_label(it.name), target.trackster_label(it.name)]


# ticl::cornetto::kMaxCollections of the linking on the device
MAX_DEVICE_LINKED_COLLECTIONS = 4


def device_trackster_label(it, target):
    """The label of the tracksters of iteration ``it`` on the device, or None when the iteration has none. For a
    CLUEstering iteration with a PID on the device, the tracksters with that PID: the same tracksters in the same order,
    so that the conversion, the superclustering, the layer cluster mask and the linking on the device read one
    collection, and the linking reads the PID from it."""
    if it.pattern_type == "CLUEstering":
        return target.soa_pid_tracksters_label if it.device_pid is not None else target.soa_tracksters_label
    if it.on_device:
        return target.soa_trackster_label(it.name)
    return None


def check_device_source(m, it, target):
    """Raises when the host tracksters of iteration ``it`` are not the device tracksters: a PID cut of the host
    conversion erases tracksters that the device consumers still read."""
    host = m.get(target.trackster_label(it.name))
    if host is None or not hasattr(host, "doPidCut") or not host.doPidCut.value():
        return
    if host.cutHadProb.value() <= 1.:
        raise PyTICLError("iteration %r: the device modules read its device tracksters, which are before the PID cut "
                          "of %s (cutHadProb %g); the cut must keep every trackster (cutHadProb > 1)"
                          % (it.name, target.trackster_label(it.name), host.cutHadProb.value()))


def device_links_labels(cfg, target):
    """The labels of the portable modules of a linking on the device: the PID SoAs, the ordered list and the
    linking."""
    if cfg.links_spec is None or not cfg.links_spec.on_device:
        return []
    pid = [target.pid_soa_label(n) for n in cfg.links_spec.collections
           if cfg._by_name[n].trackster_extra.get("inferenceAlgo", cms.string("")).value()]
    return pid + [target.links_refs_label, target.links_soa_label]


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #

class Assembled:
    """The product of assembling a config: labelled modules + named groups."""

    def __init__(self, config):
        self.config = config
        self.target = config.target
        self.modules = {}        # label -> cms module   (insertion order preserved)
        self.tasks = {}          # name  -> cms.Task / cms.Sequence
        self.task_children = {}  # name  -> [child identifier, ...]
        self.top = None          # the top group (iterTICLTask / HLTIterTICLSequence)

    def uses_onnx(self):
        """True if any module runs an ONNX Runtime model, and requires the ONNXService."""
        return any(_uses_onnx(mod) for mod in self.modules.values())

    def add_to_process(self, process):
        for label, mod in self.modules.items():
            setattr(process, label, mod)
        for name, group in self.tasks.items():
            setattr(process, name, group)
        if self.uses_onnx():
            process.load("PhysicsTools.ONNXRuntime.ONNXService_cfi")
        return process


def _uses_onnx(mod):
    """True if the module runs an ONNX Runtime model (see TICLONNXGlobalCache.h, GNNInterpretationAlgo.cc and
    EGammaSuperclusterProducer.cc)."""
    def value(pset, name, default):
        return getattr(pset, name).value() if hasattr(pset, name) else default

    if mod.type_() == "EGammaSuperclusterProducer":
        return bool(value(mod, "enableRegression", False))
    if hasattr(mod, "linkingPSet") and value(mod.linkingPSet, "onnxModelPath", ""):
        return True
    if hasattr(mod, "interpretationDescPSet") and value(mod.interpretationDescPSet, "type", "") == "GNNLink":
        return True
    algo = value(mod, "inferenceAlgo", "")
    if algo and hasattr(mod, "pluginInferenceAlgo" + algo):
        pset = getattr(mod, "pluginInferenceAlgo" + algo)
        return any(value(pset, key, "") for key in ("onnxModelPath", "onnxPIDModelPath", "onnxEnergyModelPath"))
    return False


# --------------------------------------------------------------------------- #
# per-node builders (target-aware)
# --------------------------------------------------------------------------- #

def _build_seeding(seeding_type):
    base = _cfi_default("TICLSeedingRegionProducer")
    return base.clone(seedingPSet=base.seedingPSet.clone(type=seeding_type))


def _build_layer_tile(target):
    ov = {}
    if target.lc_tag() is not None:
        ov["layer_clusters"] = target.lc_tag()
    return _cfi_default("TICLLayerTileProducer").clone(**ov)


def _build_barrel_tile(target):
    ov = dict(detector="Barrel")
    if target.lc_tag() is not None:
        ov["layer_clusters"] = target.lc_tag()
    return _cfi_default("TICLLayerTileProducer").clone(**ov)


def _build_filter(it, prev_trackster, target):
    base = _cfi_default("FilteredLayerClustersProducer")
    ov = dict(clusterFilter=it.filter_type, iteration_label=it.name)
    ov.update(it.filter_params)
    if target.lc_tag() is not None:
        ov["LayerClusters"] = target.lc_tag()
    if prev_trackster:
        ov["LayerClustersInputMask"] = cms.InputTag(prev_trackster)
    elif target.initial_mask_tag() is not None:
        ov["LayerClustersInputMask"] = target.initial_mask_tag()
    return base.clone(**ov)


def _build_trackster(it, flabel, seed_label, prev_trackster, target):
    base = _cfi_default("TrackstersProducer")
    ov = dict(
        filtered_mask=cms.InputTag(flabel, it.name),
        seeding_regions=cms.InputTag(seed_label),
        itername=it.name,
        patternRecognitionBy=it.pattern_type,
    )
    ov["pluginPatternRecognitionBy" + it.pattern_type] = dict(**it.pattern_params)
    if prev_trackster:
        ov["original_mask"] = cms.InputTag(prev_trackster)
    if target.lc_tag() is not None:  # HLT-style: remap the shared inputs
        ov["layer_clusters"] = target.lc_tag()
        ov["time_layerclusters"] = target.lc_time_tag()
        ov["layer_clusters_tiles"] = cms.InputTag(target.layer_tile_label)
        if target.barrel_tile_tag is not None:
            ov["layer_clusters_barrel_tiles"] = target.barrel_tile_tag
        if not prev_trackster:
            ov["original_mask"] = target.initial_mask_tag()
    if it.detector == "Barrel":   # barrel iteration: detector + barrel tile
        ov["detector"] = "Barrel"
        ov["layer_clusters_barrel_tiles"] = cms.InputTag(
            target.barrel_tile_label, "ticlLayerTilesBarrel")
    ov.update(it.trackster_extra)
    return base.clone(**ov)


def _build_cluestering(it, target):
    """The modules of a CLUEstering iteration as [(label, module)]: CLUEstering on the layer clusters merged on the
    device, the tracksters built on the device, their PID on the device when the iteration has one, and
    their conversion to ticl::Trackster with the mask of their layer clusters."""
    if target.merged_lc is not None and target.merged_lc_soa is None:
        raise PyTICLError("iteration %r: target %r has no layer clusters merged on the device for CLUEstering"
                          % (it.name, target.name))
    lc_soa = {} if target.merged_lc_soa is None else dict(hgcalLayerClustersSoA=cms.InputTag(target.merged_lc_soa))
    clustering = _cfi_default("HGCalCLUEsteringTrackstersProducer@alpaka").clone(**lc_soa, **it.pattern_params)
    soa = _cfi_default("HGCalSoATrackstersProducer@alpaka").clone(
        hgcalLayerClustersTrackstersSoA=cms.InputTag(target.cluestering_label), **lc_soa)
    modules = [(target.cluestering_label, clustering), (target.soa_tracksters_label, soa)]
    if it.device_pid is not None:
        merged = {} if target.merged_lc_soa is None else dict(layerClusters=target.merged_lc_soa)
        pid = makeTracksterInference(tracksters=target.soa_tracksters_label, **merged, **it.device_pid)
        modules.append((target.soa_pid_tracksters_label, pid))
    ov = dict(src=cms.InputTag(device_trackster_label(it, target)))
    if target.initial_mask_tag() is not None:
        ov["originalMask"] = target.initial_mask_tag()
    if target.lc_tag() is not None: 
        ov["layer_clusters"] = target.lc_tag()
    ov.update(it.trackster_extra)
    modules.append((target.trackster_label(it.name), _cfi_default("HGCalTrackstersFromSoAProducer").clone(**ov)))
    return modules


def _merged_lc_soa(target, param):
    """The merged layer clusters on the device for a portable module, or nothing to keep the cfi default."""
    return {} if target.merged_lc_soa is None else {param: cms.InputTag(target.merged_lc_soa)}


def _build_refs(soa_labels):
    """The ordered list of the device trackster collections ``soa_labels``."""
    return _cfi_default("TracksterCollectionRefsProducer@alpaka").clone(tracksters=cms.VInputTag(*soa_labels))


def _build_mask_soa(soa_label, refs_label, target):
    """The layer cluster mask on the device after the iteration whose device tracksters are ``soa_label``: 0 for
    their layer clusters, 1 for the others."""
    return _cfi_default("LayerClusterMaskSoAProducer@alpaka").clone(
        tracksters=cms.VInputTag(soa_label), trackstersRefs=cms.InputTag(refs_label),
        **_merged_lc_soa(target, "layerClusters"))


def _build_filter_soa(it, mask_label, target):
    """The cluster filter of iteration ``it`` on the device, from the mask on the device after the iteration it masks
    from."""
    if it.filter_type != "ClusterFilterByAlgoAndSize":
        raise PyTICLError("iteration %r: the device has the ClusterFilterByAlgoAndSize filter only, not %r"
                          % (it.name, it.filter_type))
    return _cfi_default("FilteredLayerClustersSoAProducer@alpaka").clone(
        LayerClustersInputMask=cms.InputTag(mask_label), **_merged_lc_soa(target, "LayerClusters"),
        **it.filter_params)


def _build_recovery_soa(it, filter_soa_label, target):
    """The Recovery of iteration ``it`` on the device. ``algo_verbosity`` has no device counterpart."""
    params = {k: v for k, v in it.pattern_params.items() if k != "algo_verbosity"}
    return _cfi_default("HGCalRecoveryTrackstersSoAProducer@alpaka").clone(
        filtered_mask=cms.InputTag(filter_soa_label), **_merged_lc_soa(target, "layer_clusters"), **params)


def _build_tracksters_from_soa(it, soa_label, prev_trackster, target):
    """The conversion of the device tracksters of iteration ``it`` to ticl::Trackster, with the mask of their layer
    clusters, as TrackstersProducer puts them."""
    ov = dict(src=cms.InputTag(soa_label))
    if prev_trackster:
        ov["originalMask"] = cms.InputTag(prev_trackster)
    elif target.initial_mask_tag() is not None:
        ov["originalMask"] = target.initial_mask_tag()
    if target.lc_tag() is not None:
        ov["layer_clusters"] = target.lc_tag()
    ov.update(it.trackster_extra)
    return _cfi_default("HGCalTrackstersFromSoAProducer").clone(**ov)


# The settings of TracksterLinksProducer that change the result and that the linking on the device does not have
_HOST_ONLY_LINKING_SETTINGS = ("regressionAndPid", "inferenceAlgo", "detector")


def _device_linking_parameters(overrides):
    """The parameters of the device Cornetto linking from the ``linkingPSet`` of the host linking: the same names
    and values, the floats as double. ``type`` and ``algo_verbosity`` have no device counterpart. The parameters of
    the second pass have none either, so the second pass must be off. The host linking must not run the regression
    and the PID, and must be on HGCAL."""
    overrides = overrides or {}
    pset = overrides.get("linkingPSet")
    if pset is None or getattr(pset, "type").value() != "Cornetto":
        raise PyTICLError("the linking on the device needs a linkingPSet of type Cornetto")
    host = _cfi_default("TracksterLinksProducer")
    for name in _HOST_ONLY_LINKING_SETTINGS:
        value = overrides.get(name, getattr(host, name))
        value = value.value() if hasattr(value, "value") else value
        if value != getattr(host, name).value():
            raise PyTICLError("the linking on the device has no %s: it must keep %r, got %r"
                              % (name, getattr(host, name).value(), value))
    device = set(_cfi_default("TracksterLinksSoAProducer@alpaka").parameterNames_())
    for name in ("mergeFragments", "mergeDropped"):
        if hasattr(pset, name) and getattr(pset, name).value():
            raise PyTICLError("the linking on the device has no second pass: linkingPSet.%s must be False" % name)
    params = {}
    for name in pset.parameterNames_():
        if name in ("type", "algo_verbosity"):
            continue
        if name not in device:
            if name.startswith("merge"):
                continue  # a parameter of the second pass, which is off
            raise PyTICLError("linkingPSet.%s has no device counterpart" % name)
        value = getattr(pset, name)
        if isinstance(value, (cms.float, cms.double)):
            params[name] = cms.double(value.value())
        elif isinstance(value, (cms.vint32, cms.bool)):
            params[name] = value
        else:
            raise PyTICLError("linkingPSet.%s: a %s has no device counterpart" % (name, type(value).__name__))
    return params


def _build_links_soa(soa_labels, pid_tags, overrides, target):
    """The Cornetto linking on the device of the device trackster collections ``soa_labels``, each with its PID SoA in
    ``pid_tags`` (an empty tag for the PID of the trackster SoA)."""
    return _cfi_default("TracksterLinksSoAProducer@alpaka").clone(
        tracksters=cms.VInputTag(*soa_labels), trackstersRefs=cms.InputTag(target.links_refs_label),
        pid=cms.VInputTag(*pid_tags), **_merged_lc_soa(target, "layer_clusters"),
        **_device_linking_parameters(overrides))


def _build_links_from_soa(target):
    """The products of TracksterLinksProducer from the linking on the device."""
    ov = dict(src=cms.InputTag(target.links_soa_label), links=cms.InputTag(target.links_soa_label, "links"))
    if target.lc_tag() is not None:
        ov["layer_clusters"] = target.lc_tag()
        ov["original_masks"] = cms.VInputTag(target.initial_mask_str())
    return _cfi_default("TracksterLinksFromSoAProducer").clone(**ov)


def _build_links(collection_labels, overrides, target):
    base = _cfi_default("TracksterLinksProducer")
    ov = dict(tracksters_collections=cms.VInputTag(*collection_labels))
    if target.lc_tag() is not None:
        ov["layer_clusters"] = target.lc_tag()
        ov["layer_clustersTime"] = target.lc_time_tag()
        ov["original_masks"] = cms.VInputTag(target.initial_mask_str())
    ov.update(overrides or {})
    return base.clone(**ov)


def _build_supercluster_dnn(source_label, overrides, target):
    base = _cfi_default("TracksterLinksProducer")
    ov = dict(tracksters_collections=[cms.InputTag(source_label)])
    if target.lc_tag() is not None:
        ov["layer_clusters"] = target.lc_tag()
        ov["layer_clustersTime"] = target.lc_time_tag()
        ov["original_masks"] = cms.VInputTag(target.initial_mask_str())
    ov.update(overrides or {})
    return base.clone(**ov)


# The cuts that both the device preselection of the pairs and their association on the host apply.
_SHARED_SUPERCLUSTERING_CUTS = ("seedPtThreshold", "filterByTracksterPID", "tracksterPIDCategoriesToFilter")


def _build_supercluster_dnn_heterogeneous(source_label, source, sc, target):
    """The superclustering DNN on the device: the pairs of the device tracksters of CLUEstering iteration ``source``
    and their DNN scores, and the linking module that associates them on the host, from
    superclusteringDNNHeterogeneous_cff."""
    from RecoTICL.Superclustering.superclusteringDNNHeterogeneous_cff import ticlTracksterLinksSuperclusteringDNN
    producer = _cfi_default("SuperclusteringDNNProducer@alpaka").clone(
        tracksters=cms.InputTag(device_trackster_label(source, target)), **sc.producer)
    ov = dict(tracksters_collections=[cms.InputTag(source_label)])
    if target.lc_tag() is not None:
        ov["layer_clusters"] = target.lc_tag()
        ov["layer_clustersTime"] = target.lc_time_tag()
        ov["original_masks"] = cms.VInputTag(target.initial_mask_str())
    ov.update(sc.overrides or {})
    linking = ticlTracksterLinksSuperclusteringDNN.clone(**ov)
    linking.linkingPSet.pairs = cms.InputTag(target.supercluster_producer_label)
    linking.linkingPSet.scores = cms.InputTag(target.supercluster_producer_label, "scores")
    # A cut the linking PSet omits takes its plugin default, which is the default of the device module.
    default = _cfi_default("SuperclusteringDNNProducer@alpaka")
    for name in _SHARED_SUPERCLUSTERING_CUTS:
        host = getattr(linking.linkingPSet, name, getattr(default, name)).value()
        device = getattr(producer, name).value()
        if host != device:
            raise PyTICLError("heterogeneous superclustering: %s is %r in the device module and %r in the linking"
                              % (name, device, host))
    return producer, linking


def _external_inputs(target, tracks="tracks", muons="muons"):
    """The inputs from outside TICL of a candidate-stage module: tracks, muons and MTD timing."""
    ov = {tracks: cms.InputTag(target.tracks_label), muons: cms.InputTag(target.muons_label)}
    if target.timing_label is None:
        ov.update(useMTDTiming=cms.bool(False))
    else:
        ov.update(useMTDTiming=cms.bool(True), timingSoA=cms.InputTag(target.timing_label))
    return ov


def _build_interpretations(overrides, links_label, sc_label, claim_labels, target):
    """The interpretation stage (``TICLInterpretationProducer``): the final tracksters and the per-track assignment
    maps, from the linked tracksters and the superclusters. The layer clusters of the ``claim_labels`` collections
    that are in no linked trackster join the track claim: the linking inputs, so that a group the linking does not
    emit stays available to the tracks."""
    base = _cfi_default("TICLInterpretationProducer")
    ov = dict(
        tracksters_collections=cms.VInputTag(links_label),
        egamma_tracksters_collections=cms.VInputTag(cms.InputTag(sc_label)),
        claimUnlinkedTracksters=cms.VInputTag(*claim_labels),
        **_external_inputs(target),
    )
    if target.lc_tag() is not None:  # HLT-style: remap shared inputs
        ov["layer_clusters"] = target.lc_tag()
        ov["layer_clustersTime"] = target.lc_time_tag()
    ov.update(overrides or {})
    return base.clone(**ov)


def _build_single_stage_candidate(overrides, links_label, target):
    """The single-stage candidate producer (v5): the final tracksters and the TICLCandidates from the linked
    tracksters."""
    base = _cfi_default("TICLCandidateProducer")
    links = cms.VInputTag(links_label)
    ov = dict(egamma_tracksters_collections=links,
              egamma_tracksterlinks_collections=links,
              general_tracksters_collections=links,
              general_tracksterlinks_collections=links,
              **_external_inputs(target))
    if target.timing_label is None:
        ov["useTimingAverage"] = cms.bool(False)
    if target.lc_tag() is not None:  # HLT-style: remap shared inputs
        ov["layer_clusters"] = target.lc_tag()
        ov["layer_clustersTime"] = target.lc_time_tag()
        ov["original_masks"] = cms.VInputTag(target.initial_mask_str())
    ov.update(overrides or {})
    return base.clone(**ov)


def _build_candidate(overrides, target):
    """The candidate assembly after the interpretation stage: consumes its final tracksters
    and assignment maps (plus the GSF tracks) and builds the TICLCandidates."""
    base = _cfi_default("TICLCandidateArbitrationProducer")
    ov = dict(interpretations=cms.InputTag(target.interpretations_label), **_external_inputs(target, muons="muonSrc"))
    if target.timing_label is None:
        ov["useTimingAverage"] = cms.bool(False)
    if target.gsf_tracks_label is None:
        ov["useGsfTracks"] = cms.bool(False)
    else:
        ov["gsf_tracks"] = cms.InputTag(target.gsf_tracks_label)
    if target.lc_tag() is not None:  # HLT-style: remap shared inputs
        ov["layerClusters"] = target.lc_tag()
        ov["candidateTracksters"] = base.candidateTracksters.clone(layerClustersTime=target.lc_time_tag())
    ov.update(overrides or {})
    return base.clone(**ov)


def _build_mtd():
    return _cfi_default("MTDSoAProducer").clone()


def _build_pf(overrides, target):
    base = _cfi_default("PFTICLProducer")
    ov = dict(ticlCandidateSrc=cms.InputTag(target.candidate_label), muonSrc=cms.InputTag(target.muons_label))
    ov.update(overrides or {})
    pf = base.clone(**ov)
    if pf.muonsFromCandidates.value():
        # The candidates carry the muon decisions: pfTICL reads no muons and runs no PFMuonAlgo.
        del pf.muonSrc
        del pf.pfMuonAlgoParameters
    return pf


# --------------------------------------------------------------------------- #
# top-level assembly
# --------------------------------------------------------------------------- #

def assemble(cfg):
    res = Assembled(cfg)
    target = cfg.target
    m = res.modules
    t = res.tasks

    def group(*items):
        if target.grouping == "Sequence":
            combined = items[0]
            for nxt in items[1:]:
                combined = combined + nxt
            return cms.Sequence(combined)
        return cms.Task(*items)

    def mkgroup(name, *children):
        objs = []
        for ch in children:
            if ch in m:
                objs.append(m[ch])
            elif ch in t:
                objs.append(t[ch])
            else:
                raise PyTICLError("group %r references unknown child %r" % (name, ch))
        t[name] = group(*objs)
        res.task_children[name] = list(children)
        return t[name]

    def ensure_seeding(stype):
        label = target.seeding_label(stype)
        if label is None:
            raise PyTICLError("target %r has no seeding module for seeding type %r"
                              % (target.name, stype))
        if label not in m:
            m[label] = _build_seeding(stype)
        return label

    # shared infrastructure: layer tile(s)
    tile_children = []
    if cfg.include_layer_tile:
        m[target.layer_tile_label] = _build_layer_tile(target)
        tile_children.append(target.layer_tile_label)
    if any(it.detector == "Barrel" for it in cfg.iterations):
        m[target.barrel_tile_label] = _build_barrel_tile(target)
        tile_children.append(target.barrel_tile_label)
    if tile_children:
        mkgroup(target.gname("layer_tile"), *tile_children)

    # iterations
    step_groups = []
    for it in cfg.iterations:
        if it.pattern_type == "CLUEstering":
            # CLUEstering reads every layer cluster, without an input mask: it must be the first iteration. Its
            # device modules have fixed labels, so there is one CLUEstering iteration.
            if it is not cfg.iterations[0] or it.masks_from:
                raise PyTICLError("iteration %r: CLUEstering must be the first iteration, without masks_from"
                                  % it.name)
            labels = []
            for label, module in _build_cluestering(it, target):
                m[label] = module
                labels.append(label)
            if target.merged_lc_soa_cfi is not None:
                # the step schedules the merge on the device, which only CLUEstering reads
                cfi = __import__(target.merged_lc_soa_cfi, fromlist=[target.merged_lc_soa])
                m[target.merged_lc_soa] = getattr(cfi, target.merged_lc_soa)
                labels.insert(0, target.merged_lc_soa)
            gname = target.gname("step", it.name)
            mkgroup(gname, *labels)
            step_groups.append(gname)
            continue
        if it.device_pid is not None:
            raise PyTICLError("iteration %r: the PID on the device needs the device tracksters of CLUEstering" % it.name)
        if it.on_device:
            # The device has the Recovery pattern recognition, on the mask after the iteration it masks from. The
            # mask on the device and the ordered list of that iteration are made here, since only a device consumer
            # reads them.
            if it.pattern_type != "Recovery":
                raise PyTICLError("iteration %r: the device has the Recovery pattern recognition only, not %r"
                                  % (it.name, it.pattern_type))
            prev_it = cfg._by_name.get(it.masks_from) if it.masks_from else None
            prev_soa = device_trackster_label(prev_it, target) if prev_it is not None else None
            if prev_soa is None:
                raise PyTICLError("iteration %r on the device must mask from an iteration with tracksters on the "
                                  "device" % it.name)
            check_device_source(m, prev_it, target)
            refs_label = target.refs_label(it.masks_from)
            mask_label = target.mask_soa_label(it.masks_from)
            if refs_label not in m:
                m[refs_label] = _build_refs([prev_soa])
            if mask_label not in m:
                m[mask_label] = _build_mask_soa(prev_soa, refs_label, target)
            flabel = target.filter_soa_label(it.name)
            slabel = target.soa_trackster_label(it.name)
            tlabel = target.trackster_label(it.name)
            m[flabel] = _build_filter_soa(it, mask_label, target)
            m[slabel] = _build_recovery_soa(it, flabel, target)
            m[tlabel] = _build_tracksters_from_soa(it, slabel, target.trackster_label(it.masks_from), target)
            gname = target.gname("step", it.name)
            mkgroup(gname, refs_label, mask_label, flabel, slabel, tlabel)
            step_groups.append(gname)
            continue
        if it.seeding_type is None:
            raise PyTICLError("iteration %r has no seeding region" % it.name)
        if it.filter_type is None:
            raise PyTICLError("iteration %r has no cluster filter" % it.name)
        if it.pattern_type is None:
            raise PyTICLError("iteration %r has no pattern recognition" % it.name)
        seed_label = ensure_seeding(it.seeding_type)
        flabel = target.filter_label(it.name)
        tlabel = target.trackster_label(it.name)
        prev = None
        if it.masks_from:
            if it.masks_from not in cfg._by_name:
                raise PyTICLError("iteration %r masks_from unknown iteration %r"
                                  % (it.name, it.masks_from))
            prev = target.trackster_label(it.masks_from)
        m[flabel] = _build_filter(it, prev, target)
        m[tlabel] = _build_trackster(it, flabel, seed_label, prev, target)
        gname = target.gname("step", it.name)
        mkgroup(gname, seed_label, flabel, tlabel)
        step_groups.append(gname)

    if step_groups:
        mkgroup(target.gname("iterations"), *step_groups)

    # links + superclustering
    links_children = []
    if cfg.links_spec and cfg.links_spec.on_device:
        # The linking on the device reads the device tracksters of the collections, with the PID of the host
        # tracksters of the collections that have one, and the result is converted under the usual label.
        soa_labels, pid_tags = [], []
        if len(cfg.links_spec.collections) > MAX_DEVICE_LINKED_COLLECTIONS:
            raise PyTICLError("the linking on the device links at most %d collections, got %d"
                              % (MAX_DEVICE_LINKED_COLLECTIONS, len(cfg.links_spec.collections)))
        for n in cfg.links_spec.collections:
            it = cfg._by_name.get(n)
            soa = device_trackster_label(it, target) if it is not None else None
            if soa is None:
                raise PyTICLError("the linking on the device needs the tracksters of iteration %r on the device" % n)
            check_device_source(m, it, target)
            soa_labels.append(soa)
            if it.trackster_extra.get("inferenceAlgo", cms.string("")).value():
                pid_label = target.pid_soa_label(n)
                m[pid_label] = _cfi_default("TracksterPidSoAProducer@alpaka").clone(
                    src=cms.InputTag(target.trackster_label(n)), reference=cms.InputTag(soa))
                links_children.append(pid_label)
                pid_tags.append(cms.InputTag(pid_label))
            else:
                pid_tags.append(cms.InputTag(""))
        m[target.links_refs_label] = _build_refs(soa_labels)
        m[target.links_soa_label] = _build_links_soa(soa_labels, pid_tags, cfg.links_spec.overrides, target)
        m[target.links_label] = _build_links_from_soa(target)
        links_children += [target.links_refs_label, target.links_soa_label]
    elif cfg.links_spec:
        labels = [target.trackster_label(n) for n in cfg.links_spec.collections]
        m[target.links_label] = _build_links(labels, cfg.links_spec.overrides, target)

    if cfg.superclustering_spec:
        sc = cfg.superclustering_spec
        src = target.trackster_label(sc.source)
        sc_labels = [target.supercluster_dnn_label]
        if sc.kind == "DNNHeterogeneous":
            # the pairs index the device tracksters of a CLUEstering iteration
            source = cfg._by_name.get(sc.source)
            if source is None or source.pattern_type != "CLUEstering":
                raise PyTICLError("heterogeneous superclustering: the source %r is not a CLUEstering iteration"
                                  % sc.source)
            producer, linking = _build_supercluster_dnn_heterogeneous(src, source, sc, target)
            m[target.supercluster_producer_label] = producer
            m[target.supercluster_dnn_label] = linking
            sc_labels = [target.supercluster_producer_label] + sc_labels
        else:
            m[target.supercluster_dnn_label] = _build_supercluster_dnn(src, sc.overrides, target)
        mkgroup(target.gname("superclustering"), *sc_labels)


    if target.links_label in m:
        links_children.append(target.links_label)
        if target.gname("superclustering") in t:
            links_children.append(target.gname("superclustering"))
        mkgroup(target.gname("links"), *links_children)

    # mergeTICL
    merge = [target.gname(r) for r in ("layer_tile", "iterations", "links")
             if target.gname(r) in t]
    mkgroup(target.gname("merge"), *merge)

    # candidate / mtd / pf
    if cfg.include_mtd:
        m[target.mtd_label] = _build_mtd()
        mkgroup(target.gname("mtd"), target.mtd_label)
    if cfg.include_candidate and cfg.interpretations_spec is not None:
        # Two stages: the interpretations give the final tracksters and the assignment maps, the candidate
        # producer builds the candidates.
        sc_label = (target.supercluster_dnn_label
                    if cfg.superclustering_spec is not None else target.links_label)
        claim_labels = [target.trackster_label(n) for n in cfg.links_spec.collections]
        m[target.interpretations_label] = _build_interpretations(
            cfg.interpretations_spec, target.links_label, sc_label, claim_labels, target)
        m[target.candidate_label] = _build_candidate(cfg.candidate_spec, target)
        mkgroup(target.gname("candidate"), target.interpretations_label, target.candidate_label)
    elif cfg.include_candidate:
        m[target.candidate_label] = _build_single_stage_candidate(cfg.candidate_spec, target.links_label, target)
        mkgroup(target.gname("candidate"), target.candidate_label)
    if cfg.include_pf:
        m[target.pf_label] = _build_pf(cfg.pf_spec, target)
        mkgroup(target.gname("pf"), target.pf_label)

    # top group
    top = [target.gname("merge")] + [target.gname(r) for r in ("mtd", "candidate", "pf")
                                     if target.gname(r) in t]
    mkgroup(target.gname("top"), *top)
    res.top = t[target.gname("top")]
    return res
