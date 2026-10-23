#ifndef RecoHGCal_TICL_TracksterInferenceSoA_h
#define RecoHGCal_TICL_TracksterInferenceSoA_h

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace ticl {

  inline constexpr int kMaxLayers = 50;
  inline constexpr int kMaxClusters = 10;

  using TracksterFeaturesImage = Eigen::Matrix<float, kMaxLayers, kMaxClusters, Eigen::ColMajor>;

  GENERATE_SOA_LAYOUT(TracksterInferenceFeaturesLayout,
                      SOA_EIGEN_COLUMN(TracksterFeaturesImage, energy),
                      SOA_EIGEN_COLUMN(TracksterFeaturesImage, absEta),
                      SOA_EIGEN_COLUMN(TracksterFeaturesImage, phi),
                      SOA_COLUMN(uint32_t, selectedTracksterIndex))

  GENERATE_SOA_LAYOUT(TracksterInferencePIDScoresLayout,
                      SOA_COLUMN(float, id_probabilities0),
                      SOA_COLUMN(float, id_probabilities1),
                      SOA_COLUMN(float, id_probabilities2),
                      SOA_COLUMN(float, id_probabilities3),
                      SOA_COLUMN(float, id_probabilities4),
                      SOA_COLUMN(float, id_probabilities5),
                      SOA_COLUMN(float, id_probabilities6),
                      SOA_COLUMN(float, id_probabilities7))

  GENERATE_SOA_LAYOUT(TracksterPFNFeaturesLayout,
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
                      SOA_COLUMN(float, barycenterPhi),
                      SOA_COLUMN(uint32_t, selectedTracksterIndex))

  GENERATE_SOA_LAYOUT(TracksterInferenceEnergyLayout, SOA_COLUMN(float, energy))
  using TracksterInferenceSoA = TracksterInferenceFeaturesLayout<>;

  using TracksterCNNFeaturesSoA = TracksterInferenceSoA;
  using TracksterDNNFeaturesSoA = TracksterInferenceSoA;
  using TracksterPFNFeaturesSoA = TracksterPFNFeaturesLayout<>;

  using TracksterInferencePIDScoresSoA = TracksterInferencePIDScoresLayout<>;
  using TracksterInferenceEnergySoA = TracksterInferenceEnergyLayout<>;
}  // namespace ticl

#endif
