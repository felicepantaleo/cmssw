#include "RecoTICL/Inference/interface/TracksterInferenceByShowerShape.h"
#include "RecoTICL/Inference/interface/TracksterInferenceAlgoFactory.h"
#include "RecoTICL/Inference/interface/TracksterLayerClusterProfile.h"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <utility>

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace ticl {

  namespace {
    constexpr std::size_t kNTypes =
        std::tuple_size_v<std::remove_cvref_t<decltype(std::declval<const Trackster&>().id_probabilities())>>;
  }  // namespace

  TracksterInferenceByShowerShape::TracksterInferenceByShowerShape(const edm::ParameterSet& conf,
                                                                   TICLONNXGlobalCache const* cache)
      : TracksterInferenceAlgoBase(conf, cache),
        mixture_(conf.getParameter<std::string>("mode") == "mixture"),
        inputNames_(conf.getParameter<std::vector<std::string>>("inputNames")),
        outputNames_(conf.getParameter<std::vector<std::string>>("outputNames")),
        minEnergy_(conf.getParameter<double>("minEnergy")),
        doPID_(conf.getParameter<int>("doPID")) {
    const std::string mode = conf.getParameter<std::string>("mode");
    if (mode != "em" && mode != "mixture")
      throw cms::Exception("Configuration") << "TracksterInferenceByShowerShape: unknown mode '" << mode << "'";
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
      if (mixture_) {
        const auto profile = layerClusterProfile(tracksters[i], layerClusters);
        input[0].insert(input[0].end(), profile.begin(), profile.end());
      }
      indices.push_back(i);
    }
    if (indices.empty())
      return;
    const auto n = static_cast<int64_t>(indices.size());
    const auto nFeatures = static_cast<int64_t>(mixture_ ? kNMixtureFeatures : kNFeatures);
    const std::size_t nOut = mixture_ ? kNTypes : 1;
    const auto output = onnxSession_->run(inputNames_, input, {{n, nFeatures}}, outputNames_, n);
    if (output.empty() || output[0].size() != nOut * indices.size())
      throw cms::Exception("LogicError") << "TracksterInferenceByShowerShape: the model gives "
                                         << (output.empty() ? 0 : output[0].size()) << " values for " << indices.size()
                                         << " tracksters, expected " << nOut << " per trackster";
    for (size_t k = 0; k < indices.size(); ++k) {
      auto& trackster = tracksters[indices[k]];
      trackster.zeroProbabilities();
      if (mixture_) {
        for (std::size_t c = 0; c < nOut; ++c)
          trackster.setIdProbability(static_cast<Trackster::ParticleType>(c),
                                     std::clamp(output[0][nOut * k + c], 0.f, 1.f));
      } else {
        const float s = std::clamp(output[0][k], 0.f, 1.f);
        trackster.setIdProbability(Trackster::ParticleType::photon, s);
        trackster.setIdProbability(Trackster::ParticleType::charged_hadron, 1.f - s);
      }
    }
  }

  void TracksterInferenceByShowerShape::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    TracksterInferenceAlgoBase::fillPSetDescription(iDesc);
    iDesc.add<std::string>("mode", "em")
        ->setComment(
            "em: input [N, 16], output [N] EM score; mixture: input [N, 22] (with the layer-cluster profile), "
            "output [N, 8] probabilities in ParticleType order.");
    iDesc.add<std::string>("onnxModelPath", "RecoTICL/Inference/data/showerShapePID/shapePID_v1.onnx")
        ->setComment("ONNX model of the mode.");
    iDesc.add<std::vector<std::string>>("inputNames", {"input"});
    iDesc.add<std::vector<std::string>>("outputNames", {"em"});
    iDesc.add<double>("minEnergy", 1.0)->setComment("Min raw energy [GeV] of a trackster to get an identification.");
    iDesc.add<int>("doPID", 1);
  }

}  // namespace ticl
