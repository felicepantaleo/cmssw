#ifndef RecoTICL_Inference_TracksterInferenceSoA_h
#define RecoTICL_Inference_TracksterInferenceSoA_h

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

// The inputs and outputs of the trackster models on the device, one row per selected trackster. The columns of each
// model tensor are consecutive, in the order of the tensor.
namespace ticl {

  inline constexpr int kMaxLayers = 50;
  inline constexpr int kMaxClusters = 10;

  // The most energetic layer clusters of each layer, by decreasing energy
  using TracksterFeaturesImage = Eigen::Matrix<float, kMaxLayers, kMaxClusters, Eigen::ColMajor>;

  // CNN and DNN: [batch, 3, layers, clusters]
  GENERATE_SOA_LAYOUT(TracksterImageInputsLayout,
                      SOA_EIGEN_COLUMN(TracksterFeaturesImage, energy),
                      SOA_EIGEN_COLUMN(TracksterFeaturesImage, absEta),
                      SOA_EIGEN_COLUMN(TracksterFeaturesImage, phi))

  // PFN: the image [batch, 7, layers, clusters] and the trackster quantities [batch, 7]
  GENERATE_SOA_LAYOUT(TracksterPFNInputsLayout,
                      SOA_EIGEN_COLUMN(TracksterFeaturesImage, energy),
                      SOA_EIGEN_COLUMN(TracksterFeaturesImage, absEta),
                      SOA_EIGEN_COLUMN(TracksterFeaturesImage, phi),
                      SOA_EIGEN_COLUMN(TracksterFeaturesImage, x),
                      SOA_EIGEN_COLUMN(TracksterFeaturesImage, y),
                      SOA_EIGEN_COLUMN(TracksterFeaturesImage, absZ),
                      SOA_EIGEN_COLUMN(TracksterFeaturesImage, cells),
                      SOA_COLUMN(float, rawEnergy),
                      SOA_COLUMN(float, rawEmEnergy),
                      SOA_COLUMN(float, barycenterX),
                      SOA_COLUMN(float, barycenterY),
                      SOA_COLUMN(float, barycenterAbsZ),
                      SOA_COLUMN(float, barycenterAbsEta),
                      SOA_COLUMN(float, barycenterPhi))

  // Shower-shape PID in mixture mode, [batch, 22]: the 16 quantities of TracksterInferenceByShowerShape::features, then
  // the energy profile of the trackster (ticl::layerClusterProfile)
  GENERATE_SOA_LAYOUT(TracksterShowerShapeInputsLayout,
                      SOA_COLUMN(float, logRawEnergy),
                      SOA_COLUMN(float, emFraction),
                      SOA_COLUMN(float, barycenterAbsEta),
                      SOA_COLUMN(float, barycenterAbsZ),
                      SOA_COLUMN(float, logNLayerClusters),
                      SOA_COLUMN(float, logEigenvalue0),
                      SOA_COLUMN(float, logEigenvalue1),
                      SOA_COLUMN(float, logEigenvalue2),
                      SOA_COLUMN(float, eigenvalue0Fraction),
                      SOA_COLUMN(float, sigma0),
                      SOA_COLUMN(float, sigma1),
                      SOA_COLUMN(float, sigma2),
                      SOA_COLUMN(float, sigmaPCA0),
                      SOA_COLUMN(float, sigmaPCA1),
                      SOA_COLUMN(float, sigmaPCA2),
                      SOA_COLUMN(float, axisDeltaAbsEta),
                      SOA_COLUMN(float, maxEnergyFraction),
                      SOA_COLUMN(float, core2EnergyFraction),
                      SOA_COLUMN(float, core5EnergyFraction),
                      SOA_COLUMN(float, rmsRadius),
                      SOA_COLUMN(float, minAbsZ),
                      SOA_COLUMN(float, lengthZ))

  // The outputs: the 8 probabilities in ticl::Trackster::ParticleType order [batch, 8], and the energy [batch, 1]
  GENERATE_SOA_LAYOUT(TracksterPIDScoresLayout,
                      SOA_COLUMN(float, probability0),
                      SOA_COLUMN(float, probability1),
                      SOA_COLUMN(float, probability2),
                      SOA_COLUMN(float, probability3),
                      SOA_COLUMN(float, probability4),
                      SOA_COLUMN(float, probability5),
                      SOA_COLUMN(float, probability6),
                      SOA_COLUMN(float, probability7))

  GENERATE_SOA_LAYOUT(TracksterEnergyLayout, SOA_COLUMN(float, energy))

}  // namespace ticl

#endif
