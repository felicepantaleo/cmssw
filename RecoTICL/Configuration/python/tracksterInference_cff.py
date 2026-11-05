"""Factory for the device trackster inference producers (TracksterInferenceProducer@alpaka)."""
import importlib
import math

import FWCore.ParameterSet.Config as cms

# The models that read the layer clusters, and all the models
LAYER_CLUSTER_MODELS = ("CNN", "DNN", "PFN")
MODELS = LAYER_CLUSTER_MODELS + ("ShowerShape",)

_SHOWER_SHAPE_PID = "RecoTICL/Inference/data/showerShapePID/mixPID_hi_v1.pt"


def makeTracksterInference(*, model="CNN", doPID=True, doRegression=False, batchSize=64,
                           tracksters="ticlSoATracksters", layerClusters="hgcalSoAMergeLayerClusters",
                           pidModel=None, regressionModel=None, energyThreshold=1.0,
                           convertToFP16=False, warmupIterations=3):
    if model not in MODELS:
        raise ValueError("model must be one of " + ", ".join(MODELS))
    if not isinstance(doPID, bool) or not isinstance(doRegression, bool):
        raise TypeError("doPID and doRegression must be bool")
    if model in ("CNN", "ShowerShape") and doRegression:
        raise ValueError(model + " supports PID only; energy regression requires DNN or PFN")
    if type(batchSize) is not int or not 0 < batchSize <= 2147483647:
        raise ValueError("batchSize must be a positive int fitting the SoA size type")
    if type(warmupIterations) is not int or not 0 <= warmupIterations <= 0xFFFFFFFF:
        raise ValueError("warmupIterations must fit uint32")
    if not math.isfinite(energyThreshold):
        raise ValueError("energyThreshold must be finite")
    if not isinstance(layerClusters, (str, cms.InputTag)):
        raise TypeError("layerClusters must be the merged CaloClusterSoA InputTag")
    if pidModel is None:
        pidModel = (_SHOWER_SHAPE_PID if model == "ShowerShape" else
                    f"RecoTICL/Inference/data/trackster_pid_{model.lower()}.pt") if doPID else ""
    if regressionModel is None:
        regressionModel = f"RecoTICL/Inference/data/trackster_energy_{model.lower()}.pt" if doRegression else ""
    if doPID and not pidModel:
        raise ValueError("An enabled PID task requires a nonempty pidModel path")
    if doRegression and not regressionModel:
        raise ValueError("An enabled regression task requires a nonempty regressionModel path")
    name = "trackster" + model + "InferenceProducer"
    default = getattr(importlib.import_module("RecoTICL.Inference." + name + "_cfi"), name)
    inputs = dict(layerClusters=layerClusters) if model in LAYER_CLUSTER_MODELS else {}
    return default.clone(tracksters=tracksters, pidModel=pidModel, regressionModel=regressionModel, doPID=doPID,
                         doRegression=doRegression, batchSize=batchSize, convertToFP16=convertToFP16,
                         warmupIterations=warmupIterations, energyThreshold=energyThreshold, **inputs)
