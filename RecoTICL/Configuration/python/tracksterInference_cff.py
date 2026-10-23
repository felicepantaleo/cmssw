"""Interchangeable portable inference. Build once, then install with a unique prefix.

    inference = makeTracksterInference(model="PFN", doPID=True, doRegression=True)
    inference.install(process, "ticlTracksterInference")
    process.reconstruction_step.associate(inference.task)
    # inference.output is cms.InputTag("ticlTracksterInference")

Call the factory again for another pipeline; it creates independent modules.
Only install() assigns labels and wires the feature InputTag.
"""
import math

import FWCore.ParameterSet.Config as cms


class TracksterInferenceConfiguration:
    def __init__(self, features, producer):
        self.features = features
        self.producer = producer
        self.task = None
        self._prefix = None

    def install(self, process, prefix="ticlTracksterInference"):
        if self._prefix is not None:
            raise ValueError("This inference configuration is already installed; call the factory for another pipeline")
        if not isinstance(prefix, str) or not prefix.isidentifier():
            raise ValueError("prefix must be a Python identifier")
        names = (prefix, prefix + "Features", prefix + "Task")
        if any(hasattr(process, name) for name in names):
            raise ValueError("Inference module/task label already exists: " + prefix)
        features = self.features.clone()
        producer = self.producer.clone(features=cms.InputTag(prefix + "Features", "features"))
        setattr(process, prefix + "Features", features)
        setattr(process, prefix, producer)
        self.features = getattr(process, prefix + "Features")
        self.producer = getattr(process, prefix)
        self.task = cms.Task(self.features, self.producer)
        setattr(process, prefix + "Task", self.task)
        self._prefix = prefix
        return self

    @property
    def output(self):
        if self._prefix is None:
            raise ValueError("Call install(process, prefix) before requesting the output InputTag")
        return cms.InputTag(self._prefix)


def makeTracksterInference(*, model="CNN", doPID=True, doRegression=False, batchSize=64,
                           tracksters="ticlTrackstersToSoAProducer", layerClusters=None,
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
    if layerClusters is None:
        layerClusters = ["hgcalSoALayerClustersEE", "hgcalSoALayerClustersHSi", "hgcalSoALayerClustersHSci"]
    if len(layerClusters) != 3:
        raise ValueError("layerClusters must contain three collections in Trackster vertex-index order")
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
        layerClusters=cms.VInputTag(*layerClusters),
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
    return TracksterInferenceConfiguration(features, producer)
