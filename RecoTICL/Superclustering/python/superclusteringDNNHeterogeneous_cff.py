"""Superclustering DNN with PyTorch heterogeneous inference.

  ticlSuperclusteringDNNPairFeatures     (@alpaka) SoA tracksters -> (seed, candidate) pairs and their DNN features
  ticlSuperclusteringDNNInference        (@alpaka) pair features  -> pair scores
  ticlTracksterLinksSuperclusteringDNNHeterogeneous  (host) pair scores -> superclusters, same products as
                                                     ticlTracksterLinksSuperclusteringDNN

To use it in place of the ONNX version:
    ticlEGammaSuperClusterProducer.ticlSuperClusters = "ticlTracksterLinksSuperclusteringDNNHeterogeneous"
    ticlSuperclusteringTask -> ticlSuperclusteringDNNHeterogeneousTask
"""
import FWCore.ParameterSet.Config as cms

from RecoTICL.Superclustering.superclusteringDNNPairFeaturesProducer_cfi import superclusteringDNNPairFeaturesProducer as _superclusteringDNNPairFeaturesProducer
from RecoTICL.Superclustering.superclusteringDNNInferenceProducer_cfi import superclusteringDNNInferenceProducer as _superclusteringDNNInferenceProducer
from RecoTICL.Superclustering.superclustersFromDNNScoresProducer_cfi import superclustersFromDNNScoresProducer as _superclustersFromDNNScoresProducer
from RecoTICL.Superclustering.ticlEGammaSuperClusterProducer_cfi import ticlEGammaSuperClusterProducer

ticlSuperclusteringDNNPairFeatures = _superclusteringDNNPairFeaturesProducer.clone(
    tracksters = "ticlSoATracksters",
    dnnInputsVersion = "v3",
)

ticlSuperclusteringDNNInference = _superclusteringDNNInferenceProducer.clone(
    features = ("ticlSuperclusteringDNNPairFeatures", "features"),
    modelPath = "RecoTICL/Superclustering/data/supercls_v3.pt",
    dnnInputsVersion = ticlSuperclusteringDNNPairFeatures.dnnInputsVersion,
)

ticlTracksterLinksSuperclusteringDNNHeterogeneous = _superclustersFromDNNScoresProducer.clone(
    tracksters = "ticlTrackstersCLUEstering",
    pairs = "ticlSuperclusteringDNNPairFeatures",
    scores = "ticlSuperclusteringDNNInference",
    nnWorkingPoint = 0.57247,
)

ticlSuperclusteringDNNHeterogeneousTask = cms.Task(
    ticlSuperclusteringDNNPairFeatures,
    ticlSuperclusteringDNNInference,
    ticlTracksterLinksSuperclusteringDNNHeterogeneous,
)
