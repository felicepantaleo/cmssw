// Eigen must precede the SoA headers to enable the Eigen columns of the TracksterSoA
#include <Eigen/Core>

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "LayerClusterMaskAlgos.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using namespace cms::alpakatools;

  namespace {

    constexpr uint32_t kBlockSize = 256;

    // ticlgeom::Tools::isSilicon on the raw DetId: CE-E, CE-H silicon or HFNose.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isSilicon(uint32_t rawId) {
      const uint32_t det = (rawId >> DetId::kDetOffset) & DetId::kDetMask;
      const uint32_t subdet = (rawId >> DetId::kSubdetOffset) & DetId::kSubdetMask;
      return det == DetId::HGCalEE || det == DetId::HGCalHSi ||
             (det == DetId::Forward && subdet == static_cast<uint32_t>(ForwardSubdetector::HFNose));
    }

    struct KernelFillOnes {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc, LayerClusterMaskDeviceCollection::View mask) const {
        for (auto i : uniform_elements(acc, mask.metadata().size())) {
          mask[i].mask() = 1.f;
        }
      }
    };

    struct KernelCopy {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    LayerClusterMaskDeviceCollection::ConstView input,
                                    LayerClusterMaskDeviceCollection::View output) const {
        for (auto i : uniform_elements(acc, output.metadata().size())) {
          output[i].mask() = input[i].mask();
        }
      }
    };

    // Per vertex of the tracksters: its layer cluster gets 0
    struct KernelMaskVertices {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    TracksterDeviceCollection::ConstView tracksters,
                                    LayerClusterMaskDeviceCollection::View mask) const {
        const auto vertices = tracksters.vertices();
        const uint32_t nTracksters = tracksters.energy().metadata().size();
        const uint32_t nVertices = vertices.offsets()[nTracksters].keys_offsets();
        const uint32_t nLayerClusters = mask.metadata().size();
        for (auto v : uniform_elements(acc, nVertices)) {
          const uint32_t layerCluster = vertices.content()[v].values();
          if (layerCluster < nLayerClusters) {
            mask[layerCluster].mask() = 0.f;
          }
        }
      }
    };

    struct KernelFilterByAlgoAndSize {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    AlgoAndSizeFilter filter,
                                    reco::CaloClusterDeviceCollection::ConstView layerClusters,
                                    LayerClusterMaskDeviceCollection::ConstView input,
                                    LayerClusterMaskDeviceCollection::View output) const {
        for (auto i : uniform_elements(acc, output.metadata().size())) {
          const int algo = static_cast<int>(layerClusters.indexes()[i].algoID());
          bool algoAccepted = false;
          for (unsigned int a = 0; a < filter.nAlgos; ++a) {
            algoAccepted |= (filter.algos[a] == algo);
          }
          const int cells = layerClusters.position()[i].cells();
          const bool rejected = !algoAccepted || cells > filter.maxSize ||
                                (cells < filter.minSize && isSilicon(layerClusters.indexes()[i].seedID().rawId()));
          output[i].mask() = rejected ? 0.f : input[i].mask();
        }
      }
    };

  }  // namespace

  void fillMaskWithOnes(Queue& queue, LayerClusterMaskDeviceCollection::View mask) {
    const uint32_t n = mask.metadata().size();
    if (n == 0) {
      return;
    }
    alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(divide_up_by(n, kBlockSize), kBlockSize), KernelFillOnes{}, mask);
  }

  void copyMask(Queue& queue,
                LayerClusterMaskDeviceCollection::ConstView input,
                LayerClusterMaskDeviceCollection::View output) {
    const uint32_t n = output.metadata().size();
    if (n == 0) {
      return;
    }
    alpaka::exec<Acc1D>(
        queue, make_workdiv<Acc1D>(divide_up_by(n, kBlockSize), kBlockSize), KernelCopy{}, input, output);
  }

  void maskLayerClustersOfTracksters(Queue& queue,
                                     TracksterDeviceCollection::ConstView tracksters,
                                     LayerClusterMaskDeviceCollection::View mask) {
    // The number of vertices is on the device. The kernel loops over them with the stride of the grid.
    const uint32_t n = mask.metadata().size();
    if (n == 0 || tracksters.energy().metadata().size() == 0) {
      return;
    }
    alpaka::exec<Acc1D>(
        queue, make_workdiv<Acc1D>(divide_up_by(n, kBlockSize), kBlockSize), KernelMaskVertices{}, tracksters, mask);
  }

  void filterByAlgoAndSize(Queue& queue,
                           AlgoAndSizeFilter const& filter,
                           reco::CaloClusterDeviceCollection::ConstView layerClusters,
                           LayerClusterMaskDeviceCollection::ConstView input,
                           LayerClusterMaskDeviceCollection::View output) {
    const uint32_t n = output.metadata().size();
    if (n == 0) {
      return;
    }
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(n, kBlockSize), kBlockSize),
                        KernelFilterByAlgoAndSize{},
                        filter,
                        layerClusters,
                        input,
                        output);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl
