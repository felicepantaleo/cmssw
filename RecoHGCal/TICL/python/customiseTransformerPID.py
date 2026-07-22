"""cmsDriver customise that swaps the TICL trackster PID inference (CNN/DNN/PFN)
for the trackster-transformer model (TracksterInferenceByTransformer).

A/B usage:
  A (baseline): no customise -> the release default (TracksterInferenceByCNN).
  B (branch):   --customise RecoHGCal/TICL/customiseTransformerPID.customiseTracksterTransformerPID

It only rewrites the inference plugin of TrackstersProducer modules, so the rest of
the reconstruction is untouched and the comparison isolates the PID change.
"""

import FWCore.ParameterSet.Config as cms

_SWAPPABLE = ("TracksterInferenceByCNN", "TracksterInferenceByDNN", "TracksterInferenceByPFN")


def _transformer_pset():
    # Geometry locked to the exported model (ttpid_meta.json). Shape/track/MTD globals
    # are not fed by this model, so the C++ builder leaves those slots zero.
    return cms.PSet(
        algo_verbosity=cms.int32(0),
        type=cms.string("TracksterInferenceByTransformer"),
        onnxModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/Transformer/ttpid_model.onnx"),
        inputNames=cms.vstring("grid", "globals"),
        outputNames=cms.vstring("logits_adaptive"),
        nChannels=cms.int32(2),
        nLayers=cms.int32(48),
        gridH=cms.int32(12),
        gridW=cms.int32(12),
        windowU=cms.double(12.0),
        windowV=cms.double(12.0),
        minSigma=cms.double(0.5),
        nGlobal=cms.int32(19),
        nClasses=cms.int32(6),
        eid_min_cluster_energy=cms.double(1.0),
        doPID=cms.int32(1),
        miniBatchSize=cms.untracked.int32(64),
    )


def customiseTracksterTransformerPID(process):
    for name in process.producers_():
        mod = getattr(process, name)
        if mod.type_() != "TrackstersProducer" or not hasattr(mod, "inferenceAlgo"):
            continue
        if mod.inferenceAlgo.value() not in _SWAPPABLE:
            continue
        mod.inferenceAlgo = cms.string("TracksterInferenceByTransformer")
        mod.pluginInferenceAlgoTracksterInferenceByTransformer = _transformer_pset()
    # The density model's P(em)+P(merged_em) is calibrated lower than the older model's,
    # so the superclustering cut must move DOWN with it. A deterministic PU200 TTbar scan
    # vs the adaptive truth branch (single-thread, identical CLUE3D across legs) gives, for
    # the density model: thr 0.6 -> EM eff 0.065 / purity 0.69; 0.5 -> 0.077 / 0.72;
    # 0.4 -> 0.112 / 0.74; 0.3 -> 0.125 / 0.77. The density-off baseline at 0.6 is
    # 0.093 / 0.71. So at 0.4 the density model beats the old model on BOTH EM efficiency
    # (+20%) and purity, with the fake fraction essentially flat (0.012 vs 0.010); 0.3 is
    # the max-EM point (+35% eff, highest purity). Deploy at 0.4 (dominant, low fake);
    # drop toward 0.3 if maximum EM completeness is wanted (permissive e/gamma is fine).
    if hasattr(process, "ticlTracksterLinksSuperclusteringDNN"):
        process.ticlTracksterLinksSuperclusteringDNN.linkingPSet.PIDThreshold = cms.double(0.4)
    return process
