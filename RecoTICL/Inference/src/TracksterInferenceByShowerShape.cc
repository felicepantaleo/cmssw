#include "RecoTICL/Inference/interface/TracksterInferenceByShowerShape.h"
#include "RecoTICL/Inference/interface/TracksterInferenceAlgoFactory.h"

#include <algorithm>
#include <cmath>

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace ticl {

  TracksterInferenceByShowerShape::TracksterInferenceByShowerShape(const edm::ParameterSet& conf,
                                                                   TICLONNXGlobalCache const* cache)
      : TracksterInferenceAlgoBase(conf, cache),
        inputNames_(conf.getParameter<std::vector<std::string>>("inputNames")),
        outputNames_(conf.getParameter<std::vector<std::string>>("outputNames")),
        minEnergy_(conf.getParameter<double>("minEnergy")),
        doPID_(conf.getParameter<int>("doPID")) {
    const std::string modelPath = conf.getParameter<std::string>("onnxModelPath");
    if (cache_ != nullptr && !modelPath.empty())
      onnxSession_ = cache_->getByModelPathString(modelPath);
  }

  std::array<float, TracksterInferenceByShowerShape::kNFeatures> TracksterInferenceByShowerShape::features(
      const Trackster& trackster) {
    const float energy = trackster.raw_energy();
    const auto& barycenter = trackster.barycenter();
    const float eta = barycenter.eta();
    std::array<float, 3> eigenvalues;
    for (int k = 0; k < 3; ++k)
      eigenvalues[k] = std::max(trackster.eigenvalues()[k], 1e-6f);
    const auto axis = trackster.eigenvectors(0);
    const float axisEta = std::asinh(axis.z() / std::max(std::hypot(axis.x(), axis.y()), 1e-6f));
    return {std::log1p(energy),
            trackster.raw_em_energy() / std::max(energy, 1e-6f),
            std::abs(eta),
            std::abs(barycenter.z()),
            std::log1p(static_cast<float>(trackster.vertices().size())),
            std::log(eigenvalues[0]),
            std::log(eigenvalues[1]),
            std::log(eigenvalues[2]),
            eigenvalues[0] / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]),
            trackster.sigmas()[0],
            trackster.sigmas()[1],
            trackster.sigmas()[2],
            trackster.sigmasPCA()[0],
            trackster.sigmasPCA()[1],
            trackster.sigmasPCA()[2],
            std::abs(axisEta - eta)};
  }

  void TracksterInferenceByShowerShape::runInference(const std::vector<reco::CaloCluster>& layerClusters,
                                                     std::vector<Trackster>& tracksters,
                                                     const ticlgeom::Tools& rhtools) const {
    if (doPID_ == 0 || onnxSession_ == nullptr || tracksters.empty())
      return;
    std::vector<int> indices;
    cms::Ort::FloatArrays input(1);
    for (int i = 0; i < static_cast<int>(tracksters.size()); ++i) {
      if (tracksters[i].raw_energy() < minEnergy_)
        continue;
      const auto x = features(tracksters[i]);
      input[0].insert(input[0].end(), x.begin(), x.end());
      indices.push_back(i);
    }
    if (indices.empty())
      return;
    const auto n = static_cast<int64_t>(indices.size());
    const auto output = onnxSession_->run(inputNames_, input, {{n, static_cast<int64_t>(kNFeatures)}}, outputNames_, n);
    if (output.empty() || output[0].size() != indices.size())
      throw cms::Exception("LogicError") << "TracksterInferenceByShowerShape: the model gives "
                                         << (output.empty() ? 0 : output[0].size()) << " scores for " << indices.size()
                                         << " tracksters";
    for (size_t k = 0; k < indices.size(); ++k) {
      auto& trackster = tracksters[indices[k]];
      const float s = std::clamp(output[0][k], 0.f, 1.f);
      trackster.zeroProbabilities();
      trackster.setIdProbability(Trackster::ParticleType::photon, s);
      trackster.setIdProbability(Trackster::ParticleType::charged_hadron, 1.f - s);
    }
  }

  void TracksterInferenceByShowerShape::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    TracksterInferenceAlgoBase::fillPSetDescription(iDesc);
    iDesc.add<std::string>("onnxModelPath", "RecoTICL/Inference/data/showerShapePID/shapePID_v1.onnx")
        ->setComment("ONNX model: input [N, 16] trackster quantities, output [N] EM score.");
    iDesc.add<std::vector<std::string>>("inputNames", {"input"});
    iDesc.add<std::vector<std::string>>("outputNames", {"em"});
    iDesc.add<double>("minEnergy", 1.0)->setComment("Min raw energy [GeV] of a trackster to get an identification.");
    iDesc.add<int>("doPID", 1);
  }

}  // namespace ticl
