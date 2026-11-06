// Eigen must precede the SoA headers to enable the Eigen columns of the TracksterSoA
#include <Eigen/Core>

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "HGCalRecoveryTrackstersSoAAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  namespace {

    constexpr uint32_t kBlockSize = 256;

    // The selection of PatternRecognitionbyRecovery, with the same comparisons: the energy compares to minEnergy in
    // double precision
    struct KernelSelect {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    double minEnergy,
                                    reco::CaloClusterDeviceCollection::ConstView layerClusters,
                                    ticl::LayerClusterMaskDeviceCollection::ConstView mask,
                                    uint32_t* selected) const {
        for (auto i : uniform_elements(acc, mask.metadata().size())) {
          const bool skip =
              mask[i].mask() == 0.f || static_cast<double>(layerClusters.energy()[i].energy()) < minEnergy;
          selected[i] = skip ? 0u : 1u;
        }
      }
    };

    struct KernelFill {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    float zLimitEm,
                                    reco::CaloClusterDeviceCollection::ConstView layerClusters,
                                    uint32_t const* selected,
                                    uint32_t const* positions,
                                    ticl::TracksterDeviceCollection::View tracksters) const {
        auto vertices = tracksters.vertices();
        for (auto i : uniform_elements(acc, layerClusters.position().metadata().size())) {
          if (selected[i] == 0u) {
            continue;
          }
          const uint32_t t = positions[i] - 1;
          vertices.offsets()[t + 1].keys_offsets() = t + 1;
          vertices.content()[t].values() = i;

          const float energy = layerClusters.energy()[i].energy();
          const float z = layerClusters.position()[i].z();
          tracksters.energy()[t].rawEnergy() = energy;
          tracksters.energy()[t].rawEmEnergy() = (z <= zLimitEm && z >= -zLimitEm) ? energy : 0.f;
          tracksters.energy()[t].regressedEnergy() = 0.f;
          tracksters.barycenter()[t].x() = layerClusters.position()[i].x();
          tracksters.barycenter()[t].y() = layerClusters.position()[i].y();
          tracksters.barycenter()[t].z() = z;
          tracksters.timing()[t].time() = layerClusters.timing()[i].time();
          tracksters.timing()[t].timeError() = layerClusters.timing()[i].timeError();
          tracksters.pca()[t].eigenvalues().setZero();
          tracksters.pca()[t].eigenvectors().setZero();
          tracksters.pca()[t].sigmas().setZero();
          tracksters.pca()[t].sigmasPCA().setZero();
          tracksters.id()[t].idProbabilities().setZero();
        }
      }
    };

  }  // namespace

  void HGCalRecoveryTrackstersSoAAlgoWrapper::selectLayerClusters(
      Queue& queue,
      double minEnergy,
      reco::CaloClusterDeviceCollection::ConstView layerClusters,
      ticl::LayerClusterMaskDeviceCollection::ConstView mask,
      uint32_t* selected,
      uint32_t* positions) const {
    const uint32_t n = mask.metadata().size();
    if (n == 0) {
      return;
    }
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(n, kBlockSize), kBlockSize),
                        KernelSelect{},
                        minEnergy,
                        layerClusters,
                        mask,
                        selected);
    iterativePrefixScan<Acc1D>(queue, selected, positions, n);
  }

  void HGCalRecoveryTrackstersSoAAlgoWrapper::fillTracksters(Queue& queue,
                                                             float zLimitEm,
                                                             reco::CaloClusterDeviceCollection::ConstView layerClusters,
                                                             uint32_t const* selected,
                                                             uint32_t const* positions,
                                                             ticl::TracksterDeviceCollection::View tracksters) const {
    // the vertices of the first trackster start at offset 0, also when there is no trackster
    auto firstOffset = make_device_view(queue, tracksters.vertices().offsets().keys_offsets().data(), 1);
    alpaka::memset(queue, firstOffset, 0x0);
    const uint32_t n = layerClusters.position().metadata().size();
    if (n == 0 || tracksters.energy().metadata().size() == 0) {
      return;
    }
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(n, kBlockSize), kBlockSize),
                        KernelFill{},
                        zLimitEm,
                        layerClusters,
                        selected,
                        positions,
                        tracksters);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
