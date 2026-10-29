"""Superclustering DNN with heterogeneous PyTorch inference (alpaka): the TICLv6 superclustering.

    ticlSuperclusteringDNNPairFeatures    (@alpaka)  device tracksters -> (seed, candidate) pairs and their features
    ticlSuperclusteringDNNInference       (@alpaka)  pair features     -> one DNN score per pair (PyTorch)
    ticlTracksterLinksSuperclusteringDNN  (host)     pair scores       -> superclusters, with the
                                                     "SuperClusteringDNNHeterogeneous" linking plugin

The linking module has the label and the products of the legacy one (superclustering_cff), so the e/gamma supercluster
producer and the interpretation read it unchanged. The pairs index the device tracksters, so the linking reads the
legacy tracksters converted from them, in the same order: ticlSoATracksters -> ticlTrackstersCLUEstering.

The cuts that both the device preselection and the host association apply (seedPtThreshold, filterByTracksterPID,
tracksterPIDCategoriesToFilter) are in both modules and must agree; pyTICL (presets.v6) sets them together.
The alpaka modules run on the backend that the process picks (a GPU when there is one).
"""
import FWCore.ParameterSet.Config as cms

from RecoTICL.Linking.tracksterLinksProducer_cfi import tracksterLinksProducer as _tracksterLinksProducer
from RecoTICL.Superclustering.superclusteringDNNPairFeaturesProducer_cfi import superclusteringDNNPairFeaturesProducer as _superclusteringDNNPairFeaturesProducer
from RecoTICL.Superclustering.superclusteringDNNInferenceProducer_cfi import superclusteringDNNInferenceProducer as _superclusteringDNNInferenceProducer
from RecoTICL.Superclustering.ticlEGammaSuperClusterProducer_cfi import ticlEGammaSuperClusterProducer as _ticlEGammaSuperClusterProducer

ticlSuperclusteringDNNPairFeatures = _superclusteringDNNPairFeaturesProducer.clone(
    tracksters = "ticlSoATracksters",
)

ticlSuperclusteringDNNInference = _superclusteringDNNInferenceProducer.clone(
    features = ("ticlSuperclusteringDNNPairFeatures", "features"),
    modelPath = "RecoTICL/Superclustering/data/supercls_v3.pt",  # supercls_v3.onnx exported to TorchScript
)

ticlTracksterLinksSuperclusteringDNN = _tracksterLinksProducer.clone(
    linkingPSet = cms.PSet(
        type = cms.string("SuperClusteringDNNHeterogeneous"),
        algo_verbosity = cms.int32(0),
        pairs = cms.InputTag("ticlSuperclusteringDNNPairFeatures"),
        scores = cms.InputTag("ticlSuperclusteringDNNInference"),
        nnWorkingPoint = cms.float(0.57247),
    ),
    tracksters_collections = [cms.InputTag("ticlTrackstersCLUEstering")],
)

ticlEGammaSuperClusterProducer = _ticlEGammaSuperClusterProducer.clone(
    ticlTrackstersEM = "ticlTrackstersCLUEstering",
)

ticlSuperclusteringTask = cms.Task(ticlSuperclusteringDNNPairFeatures,
                                   ticlSuperclusteringDNNInference,
                                   ticlTracksterLinksSuperclusteringDNN,
                                   ticlEGammaSuperClusterProducer)
