#ifndef RecoTICL_Inference_TracksterInferenceByShowerShape_h
#define RecoTICL_Inference_TracksterInferenceByShowerShape_h

// EM identification of tracksters from their shower shape: one ONNX model on 16 trackster quantities gives the EM score
// s, on the scale of the CNN identification that it replaces (the same hadron mis-identification at every threshold).
// The trackster gets P(photon) = s and P(charged hadron) = 1 - s.

#include <array>
#include <string>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "RecoTICL/Inference/interface/TICLONNXGlobalCache.h"
#include "RecoTICL/Inference/interface/TracksterInferenceAlgoBase.h"

namespace ticlgeom {
  class Tools;
}

namespace ticl {

  class TracksterInferenceByShowerShape : public TracksterInferenceAlgoBase {
  public:
    static constexpr std::size_t kNFeatures = 16;

    TracksterInferenceByShowerShape(const edm::ParameterSet& conf, TICLONNXGlobalCache const* cache);

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

    void runInference(const std::vector<reco::CaloCluster>& layerClusters,
                      std::vector<Trackster>& tracksters,
                      const ticlgeom::Tools& rhtools) const override;

    // log(1 + raw energy), CE-E energy over raw energy, |eta| and |z| of the barycenter, log(1 + number of layer
    // clusters), the logs of the three PCA eigenvalues, the first eigenvalue over their sum, the three sigmas, the three
    // PCA sigmas, and |eta of the main axis - eta of the barycenter|.
    static std::array<float, kNFeatures> features(const Trackster& trackster);

  private:
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    float minEnergy_;
    int doPID_;
    cms::Ort::ONNXRuntime const* onnxSession_ = nullptr;
  };

}  // namespace ticl

#endif
