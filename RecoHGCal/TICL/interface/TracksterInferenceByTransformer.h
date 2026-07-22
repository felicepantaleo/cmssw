#ifndef RecoHGCal_TICL_TracksterInferenceByTransformer_h
#define RecoHGCal_TICL_TracksterInferenceByTransformer_h

#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "RecoHGCal/TICL/interface/TICLONNXGlobalCache.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoBase.h"

#include <string>
#include <vector>

namespace hgcal {
  class RecHitTools;
}

namespace ticl {

  // Trackster-transformer PID. Builds, per trackster, a barycenter-frame smeared
  // grid [n_channels, n_layers, grid_h, grid_w] plus a global-feature vector, feeds
  // both to a two-input ONNX model, and writes the softmaxed 6-class adaptive output
  // into id_probabilities_ via the em/mip/hadronic/fake -> ParticleType mapping.
  // Feature construction mirrors the training builder (barycenter (u,v) frame,
  // energy-conserving Gaussian splat at a fixed sigma); the grid geometry is a
  // parameter so it stays locked to the exported model.
  class TracksterInferenceByTransformer : public TracksterInferenceAlgoBase {
  public:
    TracksterInferenceByTransformer(const edm::ParameterSet& conf, TICLONNXGlobalCache const* cache);

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

    void runInference(const std::vector<reco::CaloCluster>& layerClusters,
                      std::vector<Trackster>& tracksters,
                      const hgcal::RecHitTools& rhtools) const override;

  private:
    // Fill grid (size n_channels*n_layers*grid_h*grid_w) and globals (size n_global)
    // for one trackster, both zero-initialised by the caller.
    void buildFeatures(const Trackster& ts,
                       const std::vector<reco::CaloCluster>& layerClusters,
                       const hgcal::RecHitTools& rhtools,
                       float* grid,
                       float* globals) const;

    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;

    // Grid geometry, locked to the exported model.
    int nChannels_;
    int nLayers_;
    int gridH_;
    int gridW_;
    double windowU_;
    double windowV_;
    double minSigma_;
    int nGlobal_;
    int nClasses_;

    double eidMinClusterEnergy_;
    int doPID_;
    int miniBatchSize_;

    cms::Ort::ONNXRuntime const* onnxSession_ = nullptr;
    bool enabled_ = false;
  };

}  // namespace ticl

#endif
