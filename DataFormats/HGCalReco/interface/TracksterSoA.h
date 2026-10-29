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

  // The layer clusters of each trackster, as indices in the merged layer clusters. 
  template <CMS_SOA_BYTE_SIZE_TYPE ALIGNMENT, bool ALIGNMENT_ENFORCEMENT>
  using TracksterSoAVertices =
      typename AssociationMapLayout<uint32_t, uint32_t>::template Layout<ALIGNMENT, ALIGNMENT_ENFORCEMENT>;

  GENERATE_SOA_BLOCKS(TracksterSoALayout,
                      SOA_BLOCK(energy, TracksterSoAEnergy),
                      SOA_BLOCK(barycenter, TracksterSoABarycenter),
                      SOA_BLOCK(timing, TracksterSoATiming),
                      SOA_BLOCK(pca, TracksterSoAPCA),
                      SOA_BLOCK(id, TracksterSoAId),
                      SOA_BLOCK(vertices, TracksterSoAVertices))

  using TracksterSoA = TracksterSoALayout<>;
  using TracksterSoAView = TracksterSoA::View;
  using TracksterSoAConstView = TracksterSoA::ConstView;

}  // namespace ticl

#endif  // DataFormats_HGCalReco_interface_TracksterSoA_h
