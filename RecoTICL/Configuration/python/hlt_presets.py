# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Phase-2 HLT presets for the TICL chain (HLTIterTICLSequence).

The HLT iterations (CLUE3DHigh, Recovery) share the offline algorithm presets;
only the labels and inputs differ (handled by the HLT :class:`Target`).  The
singletons (links / candidate / pf) are HLT-tuned and defined here.  pyTICL
reproduces ``HLTIterTICLSequence`` byte-for-byte (see test_reproduce_hlt.py).
"""

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration.model import TICLConfig


def hlt_links_defaults():
    """Overrides for hltTiclTracksterLinks (HLT Skeletons linking, empty ONNX)."""
    return dict(
        regressionAndPid=cms.bool(False),
        inferenceAlgo=cms.string(""),
        linkingPSet=cms.PSet(
            algo_verbosity=cms.int32(0),
            cylinder_radius_sqr=cms.vfloat(9, 15),
            cylinder_radius_sqr_split=cms.float(9),
            deltaRxy=cms.float(4),
            dot_prod_th=cms.float(0.97),
            lower_boundary=cms.vfloat(20, 10),
            lower_distance_projective_sqr=cms.vfloat(4, 60),
            lower_distance_projective_sqr_closest_points=cms.vfloat(10, 50),
            max_z_distance_closest_points=cms.vfloat(35, 35),
            min_distance_z=cms.vfloat(35, 35),
            min_num_lcs=cms.uint32(15),
            min_trackster_energy=cms.float(20),
            onnxModelPath=cms.string(""),
            pca_quality_th=cms.float(0.85),
            proj_distance_split=cms.float(5),
            track_time_quality_threshold=cms.float(0.5),
            type=cms.string("Skeletons"),
            upper_boundary=cms.vfloat(150, 100),
            upper_distance_projective_sqr=cms.vfloat(4, 60),
            upper_distance_projective_sqr_closest_points=cms.vfloat(5, 30),
        ),
        pluginInferenceAlgoTracksterInferenceByDNN=cms.PSet(
            algo_verbosity=cms.int32(0),
            doPID=cms.int32(1),
            doRegression=cms.int32(1),
            eid_min_cluster_energy=cms.float(1),
            eid_n_clusters=cms.int32(10),
            eid_n_layers=cms.int32(50),
            inputNames=cms.vstring("input"),
            onnxEnergyModelPath=cms.string(""),
            onnxPIDModelPath=cms.string(""),
            output_en=cms.vstring("enreg_output"),
            output_id=cms.vstring("pid_output"),
            type=cms.string("TracksterInferenceByDNN"),
        ),
        pluginInferenceAlgoTracksterInferenceByPFN=cms.PSet(
            algo_verbosity=cms.int32(0),
            doPID=cms.int32(1),
            doRegression=cms.int32(1),
            eid_min_cluster_energy=cms.float(1),
            eid_n_clusters=cms.int32(10),
            eid_n_layers=cms.int32(50),
            inputNames=cms.vstring("input", "input_tr_features"),
            miniBatchSize=cms.untracked.int32(64),
            onnxEnergyModelPath=cms.string(""),
            onnxPIDModelPath=cms.string(""),
            output_en=cms.vstring("enreg_output"),
            output_id=cms.vstring("pid_output"),
            type=cms.string("TracksterInferenceByPFN"),
        ),
    )


def hlt_clue3dhigh_inference():
    """HLT CLUE3DHigh pattern-recognition DNN/PFN PSets (differ from offline:
    no eid_min_cluster_energy in DNN, miniBatchSize added in PFN)."""
    return dict(
        pluginInferenceAlgoTracksterInferenceByDNN=cms.PSet(
            algo_verbosity=cms.int32(0),
            doPID=cms.int32(1),
            doRegression=cms.int32(0),
            eid_n_clusters=cms.int32(10),
            eid_n_layers=cms.int32(50),
            inputNames=cms.vstring("input"),
            onnxEnergyModelPath=cms.string(""),
            onnxPIDModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/DNN/patternrecognition/id_v0.onnx"),
            output_en=cms.vstring("enreg_output"),
            output_id=cms.vstring("pid_output"),
            type=cms.string("TracksterInferenceByDNN"),
        ),
        pluginInferenceAlgoTracksterInferenceByPFN=cms.PSet(
            algo_verbosity=cms.int32(0),
            doPID=cms.int32(1),
            doRegression=cms.int32(0),
            eid_n_clusters=cms.int32(10),
            eid_n_layers=cms.int32(50),
            inputNames=cms.vstring("input", "input_tr_features"),
            miniBatchSize=cms.untracked.int32(64),
            onnxEnergyModelPath=cms.string(""),
            onnxPIDModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/patternrecognition/id_v0.onnx"),
            output_en=cms.vstring("enreg_output"),
            output_id=cms.vstring("pid_output"),
            type=cms.string("TracksterInferenceByPFN"),
        ),
    )


def hlt_candidate_defaults():
    """Overrides for hltTiclCandidate (PFN inference). The HLT target sets the track, muon and timing inputs."""
    return dict(
        inferenceAlgo=cms.string("TracksterInferenceByPFN"),
        regressionAndPid=cms.bool(True),
        pluginInferenceAlgoTracksterInferenceByPFN=cms.PSet(
            algo_verbosity=cms.int32(0),
            doPID=cms.int32(1),
            doRegression=cms.int32(1),
            eid_min_cluster_energy=cms.float(2.5),
            eid_n_clusters=cms.int32(10),
            eid_n_layers=cms.int32(50),
            inputNames=cms.vstring("input", "input_tr_features"),
            onnxEnergyModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/linking/energy_v1.onnx"),
            onnxPIDModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/CNN/linking/id_v0.onnx"),
            output_en=cms.vstring("enreg_output"),
            output_id=cms.vstring("pid_output"),
            type=cms.string("TracksterInferenceByPFN"),
        ),
    )


def v5_hlt(name="v5_hlt"):
    """Return a :class:`TICLConfig` reproducing HLTIterTICLSequence (9 modules)."""
    cfg = (TICLConfig(name, target="HLT")
           .iteration("CLUE3DHigh").preset().trackster_params(**hlt_clue3dhigh_inference())
           .iteration("Recovery").preset().masks_from("CLUE3DHigh")
           .links(["CLUE3DHigh", "Recovery"], **hlt_links_defaults())
           .candidate(**hlt_candidate_defaults())
           .pf())
    cfg.include_mtd = False   # the HLT iterTICL sequence has no mtdSoA stage
    return cfg


def v6_hlt(name="v6_hlt"):
    """Return the :class:`TICLConfig` of TICLv6 on the HLT target, with the device chain."""
    from RecoTICL.Configuration.presets import v6
    return v6(name, target="HLT", heterogeneous=True)


# The TICLv5 modules and sequences of the HLT menu that TICLv6 does not use.
_V5_ONLY_MODULES = ("hltFilteredLayerClustersCLUE3DHigh", "hltTiclSeedingGlobal", "hltTiclTrackstersCLUE3DHigh",
                    "hltFilteredLayerClustersRecovery")
_V5_ONLY_SEQUENCES = ("HLTTiclTrackstersCLUE3DHighStepSequence", "HLTTiclTrackstersRecoverySequence")


def _replace_input_label(pset, old, new):
    for name in pset.parameterNames_():
        value = getattr(pset, name)
        if isinstance(value, cms.InputTag):
            if value.getModuleLabel() == old:
                value.setModuleLabel(new)
        elif isinstance(value, cms.VInputTag):
            for i, tag in enumerate(value):
                if isinstance(tag, cms.InputTag):
                    if tag.getModuleLabel() == old:
                        tag.setModuleLabel(new)
                elif tag.split(":")[0] == old:
                    value[i] = ":".join([new] + tag.split(":")[1:])
        elif isinstance(value, cms.PSet):
            _replace_input_label(value, old, new)
        elif isinstance(value, cms.VPSet):
            for item in value:
                _replace_input_label(item, old, new)


def use_v6(process):
    """Make the Phase-2 HLT menu in ``process`` run TICLv6 (:func:`v6_hlt`) in the place of TICLv5 (:func:`v5_hlt`).
    The unseeded e/gamma superclusters read the CLUEstering tracksters and the TICLv6 superclustering DNN. The
    L1-seeded e/gamma chain does not change."""
    assembled = v6_hlt().assemble()
    target = assembled.target
    old_sequences = dict(process.sequences_())
    old_first = process.hltTiclTrackstersCLUE3DHigh
    for label, module in assembled.modules.items():
        setattr(process, label, module)
    for name, sequence in assembled.tasks.items():
        setattr(process, name, sequence)
    # A sequence of the menu that runs the first iteration or the superclustering alone runs the TICLv6 sequence.
    superclustering = getattr(process, target.supercluster_dnn_label)
    containers = [s for n, s in old_sequences.items() if n not in assembled.tasks and n not in _V5_ONLY_SEQUENCES]
    containers += list(process.paths_().values())
    for container in containers:
        container.replace(old_first, process.HLTTiclTrackstersCLUEsteringStepSequence)
        container.replace(superclustering, process.HLTTiclSuperclusteringSequence)
    for name in _V5_ONLY_SEQUENCES + _V5_ONLY_MODULES:
        if hasattr(process, name):
            delattr(process, name)
    # A reader of the first iteration reads CLUEstering. A reader of the final tracksters, other than the module of
    # the PF candidates, reads the interpretation stage.
    for modules in (process.producers_(), process.filters_(), process.analyzers_()):
        for label, module in modules.items():
            _replace_input_label(module, "hltTiclTrackstersCLUE3DHigh", target.trackster_label("CLUEstering"))
            if label != target.pf_label:
                _replace_input_label(module, target.candidate_label, target.interpretations_label)
    return process
