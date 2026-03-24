/**
Trackster particle identification (electromagnetic vs hadronic) using a simple transformer architecture

Uses layer cluster features (position, energy) and trackster features (eta, phi, first and last layer, energy in CEE)

Author : Theo Cuisset (LLR)
Network design & training : Shamik Ghosh, Alessandra Cappati (LLR)
*/
#include "RecoHGCal/TICL/interface/TracksterInferenceByTransformer.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace ticl {

  TracksterInferenceByTransformer::TracksterInferenceByTransformer(const edm::ParameterSet& conf,
                                                                   TICLONNXGlobalCache const* cache)
      : TracksterInferenceAlgoBase(conf, cache),
        inputNames_(conf.getParameter<std::vector<std::string>>("inputNames")),
        output_id_(conf.getParameter<std::vector<std::string>>("output_id")),
        eidMinClusterEnergy_(conf.getParameter<double>("eid_min_cluster_energy")),
        eidNClusters_(conf.getParameter<int>("eid_n_clusters")),
        doPID_(conf.getParameter<int>("doPID")),
        miniBatchSize_(conf.getUntrackedParameter<int>("miniBatchSize", 256)) {
    const std::string pidModel = conf.getParameter<std::string>("onnxPIDModelPath");

    if (cache_ != nullptr) {
      if (!pidModel.empty()) {
        onnxPIDSession_ = cache_->getByModelPathString(pidModel);
      }
    }

    enabled_ = ((doPID_ != 0 && onnxPIDSession_ != nullptr));

    ortScratch_.inputs.resize(3);
    ortScratch_.input_shapes.resize(3);
  }

  void TracksterInferenceByTransformer::runInference(const std::vector<reco::CaloCluster>& layerClusters,
                                                     std::vector<Trackster>& tracksters,
                                                     const hgcal::RecHitTools& rhtools) const {
    if (!enabled_ || tracksters.empty()) {
      return;
    }

    // ---- select tracksters (same physics logic), reset outputs once
    std::vector<int> indices;
    indices.reserve(tracksters.size());

    for (int i = 0; i < static_cast<int>(tracksters.size()); ++i) {
      float sumClusterEnergy = 0.f;

      // Note: keep the same semantics you had (skip barrel clusters, sum endcap energy)
      for (const unsigned int& v : tracksters[i].vertices()) {
        if (rhtools.isBarrel(layerClusters[v].seed())) {
          continue;
        }
        sumClusterEnergy += static_cast<float>(layerClusters[v].energy());
        if (sumClusterEnergy >= eidMinClusterEnergy_) {
          tracksters[i].setRegressedEnergy(0.f);
          tracksters[i].zeroProbabilities();
          indices.push_back(i);
          break;
        }
      }
    }

    const int total = static_cast<int>(indices.size());
    if (total == 0) {
      return;
    }

    const int mb = std::max(1, miniBatchSize_);

    // Reuse buffers across events
    ortScratch_.clearPerEvent();

    std::vector<int> clusterIndices;

    // input tensor
    std::vector<float> layerClusterFeatures;
    std::vector<uint8_t> layerClusterMask;  // Here we use uint8_t to avoid std::vector<bool> specialization
    std::vector<float> tracksterFeatures;

    for (int start = 0; start < total; start += mb) {
      const int nTrackstersInBatch = std::min(mb, total - start);  ///< nTracksters in batch

      // shape: layer cluster features = [trackster, maxLayerClusterCount, nFeats]
      ortScratch_.input_shapes[0] = {nTrackstersInBatch, eidNClusters_, eidNFeatures_};
      // shape: layer cluster mask = [trackster, maxLayerClusterCount]
      ortScratch_.input_shapes[1] = {nTrackstersInBatch, eidNClusters_};
      // shape: trackster features : [trackster, nTracksterFeatures]
      ortScratch_.input_shapes[2] = {nTrackstersInBatch, eidNTracksterFeatures_};

      const size_t nFloats = static_cast<size_t>(nTrackstersInBatch) * eidNClusters_ * eidNFeatures_;
      layerClusterFeatures.assign(nFloats, 0.f);  // sparse fill -> must zero

      layerClusterMask.assign(static_cast<size_t>(nTrackstersInBatch) * eidNClusters_, 1);
      tracksterFeatures.assign(static_cast<size_t>(nTrackstersInBatch) * eidNTracksterFeatures_, 0.f);

      // ---- build sparse tensor for this minibatch
      // loop on trackster
      for (int tracksterIdxInBatch = 0; tracksterIdxInBatch < nTrackstersInBatch; ++tracksterIdxInBatch) {
        const int tsIdx = indices[start + tracksterIdxInBatch];
        Trackster const& ts = tracksters[tsIdx];

        const int vtxCount = static_cast<int>(ts.vertices().size());
        clusterIndices.resize(vtxCount);
        std::iota(clusterIndices.begin(), clusterIndices.end(), 0);

        // the layer cluster mask must be True (=1) for LCs that are masked (ie not in use because trackster has < eidNClusters_ LCs), and False for LCs in use (=0)
        std::fill_n(
            layerClusterMask.begin() + tracksterIdxInBatch * eidNClusters_, std::min(vtxCount, eidNClusters_), 0);

        // std::sort(clusterIndices.begin(), clusterIndices.end(), [&layerClusters, &ts](int a, int b) {
        //   return layerClusters[ts.vertices(a)].energy() > layerClusters[ts.vertices(b)].energy();
        // });

        unsigned int minLayer = std::numeric_limits<unsigned int>::max();
        unsigned int maxLayer = 0;

        int layerClusterCount = 0;
        for (int k : clusterIndices) {
          const unsigned int v = ts.vertices(k);
          auto const& cl = layerClusters[v];

          const unsigned int layer = rhtools.getLayerWithOffset(cl.hitsAndFractions()[0].first);
          minLayer = std::min(layer, minLayer);
          maxLayer = std::max(layer, maxLayer);

          if (layerClusterCount >= eidNClusters_)
            break;

          // trackster x LCs x features
          const size_t base = static_cast<size_t>(tracksterIdxInBatch) * (eidNClusters_ * eidNFeatures_) +
                              static_cast<size_t>(layerClusterCount) * eidNFeatures_;

          layerClusterFeatures[base + 0] = static_cast<float>(cl.x());
          layerClusterFeatures[base + 1] = static_cast<float>(cl.y());
          layerClusterFeatures[base + 2] = static_cast<float>(layer);
          layerClusterFeatures[base + 3] =
              static_cast<float>(cl.energy() / static_cast<float>(ts.vertex_multiplicity(k)));

          ++layerClusterCount;
        }

        tracksterFeatures[static_cast<size_t>(tracksterIdxInBatch) * eidNTracksterFeatures_ + 0] =
            std::abs(ts.barycenter().eta());
        tracksterFeatures[static_cast<size_t>(tracksterIdxInBatch) * eidNTracksterFeatures_ + 1] =
            ts.barycenter().phi();
        tracksterFeatures[static_cast<size_t>(tracksterIdxInBatch) * eidNTracksterFeatures_ + 2] = ts.raw_em_energy();
        tracksterFeatures[static_cast<size_t>(tracksterIdxInBatch) * eidNTracksterFeatures_ + 3] = minLayer;
        tracksterFeatures[static_cast<size_t>(tracksterIdxInBatch) * eidNTracksterFeatures_ + 4] = maxLayer;
      }

      // ---- PID
      ortScratch_.outputs.clear();

      onnxPIDSession_->runIntoTemplated(
          std::make_tuple(
              cms::Ort::ONNXRuntime::InputTensorConfig{
                  inputNames_[0], layerClusterFeatures, ortScratch_.input_shapes[0]},
              // important to use InputTensorConfigBool to avoid std::vector<bool> but to actually create an ONNX bool tensor
              cms::Ort::ONNXRuntime::InputTensorConfigBool{
                  inputNames_[1], layerClusterMask, ortScratch_.input_shapes[1]},
              cms::Ort::ONNXRuntime::InputTensorConfig{inputNames_[2], tracksterFeatures, ortScratch_.input_shapes[2]}),
          output_id_,
          ortScratch_.outputs,
          {},
          nTrackstersInBatch);

      if (!ortScratch_.outputs.empty() && !output_id_.empty()) {
        for (int bi = 0; bi < nTrackstersInBatch; ++bi) {
          auto& ts = tracksters[indices[start + bi]];
          float output_proba_had = ortScratch_.outputs[0][bi * 2];
          float output_proba_em = ortScratch_.outputs[0][bi * 2 + 1];
          ts.setIdProbability(Trackster::ParticleType::charged_hadron, output_proba_had * 0.5f);
          ts.setIdProbability(Trackster::ParticleType::neutral_hadron, output_proba_had * 0.5f);
          ts.setIdProbability(Trackster::ParticleType::photon, output_proba_em * 0.5f);
          ts.setIdProbability(Trackster::ParticleType::electron, output_proba_em * 0.5f);
        }
      }
    }
  }

  void TracksterInferenceByTransformer::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    TracksterInferenceAlgoBase::fillPSetDescription(iDesc);

    iDesc.add<std::string>("onnxPIDModelPath", "")
        ->setComment("Path to ONNX PID model. If empty, PID inference is skipped.");

    iDesc.add<std::vector<std::string>>("inputNames",
                                        {"layerCluster_features", "layerCluster_mask", "trackster_features"});
    iDesc.add<std::vector<std::string>>("output_id", {"pid_output"});

    iDesc.add<double>("eid_min_cluster_energy", 1.0);
    iDesc.add<int>("eid_n_clusters", 150)
        ->setComment("Maximum number of layer clusters of the model architecture (must match the model)");
    iDesc.add<int>("doPID", 1);

    iDesc.addUntracked<int>("miniBatchSize", 256)
        ->setComment("Mini-batch size for inference to limit peak memory usage.");
  }

}  // namespace ticl
