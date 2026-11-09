#ifndef DataFormats_HGCalReco_interface_TracksterSoA_h
#define DataFormats_HGCalReco_interface_TracksterSoA_h

#include <cstdint>
#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/TICL/interface/AssociationMap.h"

namespace ticl {

  using TracksterIdProbabilities = Eigen::Matrix<float, 8, 1>;

  GENERATE_SOA_LAYOUT(TracksterSoAEnergy,
                      SOA_COLUMN(float, rawEnergy),
                      SOA_COLUMN(float, rawEmEnergy),
                      SOA_COLUMN(float, regressedEnergy))

  GENERATE_SOA_LAYOUT(TracksterSoABarycenter, SOA_COLUMN(float, x), SOA_COLUMN(float, y), SOA_COLUMN(float, z))

  GENERATE_SOA_LAYOUT(TracksterSoATiming, SOA_COLUMN(float, time), SOA_COLUMN(float, timeError))

  // PCA: eigenvalues in decreasing order, eigenvectors.col(i) the ith principal axis
  GENERATE_SOA_LAYOUT(TracksterSoAPCA,
                      SOA_EIGEN_COLUMN(Eigen::Vector3f, eigenvalues),
                      SOA_EIGEN_COLUMN(Eigen::Matrix3f, eigenvectors),
                      SOA_EIGEN_COLUMN(Eigen::Vector3f, sigmas),
                      SOA_EIGEN_COLUMN(Eigen::Vector3f, sigmasPCA))

  GENERATE_SOA_LAYOUT(TracksterSoAId, SOA_EIGEN_COLUMN(TracksterIdProbabilities, idProbabilities))

  // The radii [cm] of the core energies of the profile, shared with ticl::layerClusterProfile on the host
  inline constexpr float kProfileCore2Radius = 2.f;
  inline constexpr float kProfileCore5Radius = 5.f;

  // The energy profile of the layer clusters around the first principal axis (the barycenter direction without one),
  // as ticl::layerClusterProfile: the fractions of the energy in the most energetic layer cluster and within 2 and
  // 5 cm of the axis, the energy-weighted rms distance from the axis [cm], and the |z| range [cm]. Without energy:
  // -1, -1, -1, -1, -1, 0.
  GENERATE_SOA_LAYOUT(TracksterSoAProfile,
                      SOA_COLUMN(float, maxEnergyFraction),
                      SOA_COLUMN(float, core2EnergyFraction),
                      SOA_COLUMN(float, core5EnergyFraction),
                      SOA_COLUMN(float, rmsRadius),
                      SOA_COLUMN(float, minAbsZ),
                      SOA_COLUMN(float, lengthZ))

  // The layer clusters of each trackster, as indices in the merged layer clusters.
  template <CMS_SOA_BYTE_SIZE_TYPE ALIGNMENT, bool ALIGNMENT_ENFORCEMENT>
  using TracksterSoAVertices = AssociationMapLayout<uint32_t, uint32_t>::Layout<ALIGNMENT, ALIGNMENT_ENFORCEMENT>;

  GENERATE_SOA_BLOCKS(TracksterSoALayout,
                      SOA_BLOCK(energy, TracksterSoAEnergy),
                      SOA_BLOCK(barycenter, TracksterSoABarycenter),
                      SOA_BLOCK(timing, TracksterSoATiming),
                      SOA_BLOCK(pca, TracksterSoAPCA),
                      SOA_BLOCK(id, TracksterSoAId),
                      SOA_BLOCK(profile, TracksterSoAProfile),
                      SOA_BLOCK(vertices, TracksterSoAVertices))

  using TracksterSoA = TracksterSoALayout<>;
  using TracksterSoAView = TracksterSoA::View;
  using TracksterSoAConstView = TracksterSoA::ConstView;

}  // namespace ticl

#endif  // DataFormats_HGCalReco_interface_TracksterSoA_h
