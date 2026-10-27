"""Factory for paired Alpaka feature-extraction and inference producers."""
import math

import FWCore.ParameterSet.Config as cms



def makeTracksterInference(*, model="CNN", doPID=True, doRegression=False, batchSize=64,
                           tracksters="ticlSoATracksters", layerClusters="hgcalSoAMergeLayerClusters",
                           pidModel=None, regressionModel=None, minClusterEnergy=1.0,
                           convertToFP16=False, warmupIterations=3):
    if model not in ("CNN", "DNN", "PFN"):
        raise ValueError("model must be CNN, DNN or PFN")
    if not isinstance(doPID, bool) or not isinstance(doRegression, bool):
        raise TypeError("doPID and doRegression must be bool")
    if model == "CNN" and doRegression:
        raise ValueError("CNN supports PID only; energy regression requires DNN or PFN")
    if type(batchSize) is not int or not 0 < batchSize <= 2147483647:
        raise ValueError("batchSize must be a positive int fitting the SoA size type")
    if type(warmupIterations) is not int or not 0 <= warmupIterations <= 0xFFFFFFFF:
        raise ValueError("warmupIterations must fit uint32")
    if not math.isfinite(minClusterEnergy):
        raise ValueError("minClusterEnergy must be finite")
    if not isinstance(layerClusters, (str, cms.InputTag)):
        raise TypeError("layerClusters must be the merged CaloClusterSoA InputTag")
    if pidModel is None:
        pidModel = "RecoTICL/Inference/data/trackster_pid_" + model.lower() + ".pt"
    if regressionModel is None:
        regressionModel = "RecoTICL/Inference/data/trackster_energy_" + model.lower() + ".pt"
    if doPID and not pidModel:
        raise ValueError("An enabled PID task requires a nonempty pidModel path")
    if doRegression and not regressionModel:
        raise ValueError("An enabled regression task requires a nonempty regressionModel path")
    feature_plugin = "Trackster" + model + "FeaturesProducer"
    features = cms.EDProducer(
        feature_plugin + "@alpaka",
        detector=cms.string("HGCAL"),
        tracksters=cms.InputTag(tracksters) if isinstance(tracksters, str) else tracksters,
        layerClusters=cms.InputTag(layerClusters) if isinstance(layerClusters, str) else layerClusters,
        minClusterEnergy=cms.double(minClusterEnergy),
    )
    producer = cms.EDProducer(
        "Trackster" + model + "InferenceProducer@alpaka",
        tracksters=features.tracksters, features=cms.InputTag(""),
        pidModel=cms.string(pidModel), regressionModel=cms.string(regressionModel),
        doPID=cms.bool(doPID), doRegression=cms.bool(doRegression),
        batchSize=cms.uint32(batchSize), convertToFP16=cms.bool(convertToFP16),
        warmupIterations=cms.uint32(warmupIterations), minClusterEnergy=cms.double(minClusterEnergy),
    )
    return features, producer
