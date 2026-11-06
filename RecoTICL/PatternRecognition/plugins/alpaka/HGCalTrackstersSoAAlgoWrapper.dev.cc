// Eigen must precede the SoA headers to enable the Eigen columns of the TracksterSoA
#include <Eigen/Core>

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTICL/Common/interface/alpaka/TracksterPropertiesKernel.h"

#include "HGCalTrackstersSoAAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  namespace {

    constexpr int kInvalidTrackster = -1;

    // Per layer cluster -> per trackster: the number of layer clusters
    struct KernelTracksterSizes {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    HGCalSoALayerClustersExtraDeviceCollection::ConstView layerClustersClustering,
                                    uint32_t* tracksterSizes) const {
        for (auto i : uniform_elements(acc, layerClustersClustering.metadata().size())) {
          const int trackster = layerClustersClustering[i].tracksterIndex();
          if (trackster != kInvalidTrackster) {
            alpaka::atomicAdd(acc, &tracksterSizes[trackster], 1u);
          }
        }
      }
    };

    // Per layer cluster: its index in the layer clusters of its trackster, which end at the offset of the next
    // trackster. The sizes count down the layer clusters still to place, so that they fill the trackster from its start.
    struct KernelTracksterVertices {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    HGCalSoALayerClustersExtraDeviceCollection::ConstView layerClustersClustering,
                                    uint32_t* tracksterSizes,
                                    ticl::TracksterDeviceCollection::View tracksters) const {
        auto vertices = tracksters.vertices();
        for (auto i : uniform_elements(acc, layerClustersClustering.metadata().size())) {
          const int trackster = layerClustersClustering[i].tracksterIndex();
          if (trackster == kInvalidTrackster) {
            continue;
          }
          const uint32_t vertex =
              vertices.offsets()[trackster + 1].keys_offsets() - alpaka::atomicSub(acc, &tracksterSizes[trackster], 1u);
          vertices.content()[vertex].values() = i;
        }
      }
    };

  }  // namespace

  void HGCalTrackstersSoAAlgoWrapper::run(Queue& queue,
                                          float zLimitEm,
                                          reco::CaloClusterDeviceCollection::ConstView layerClusters,
                                          HGCalSoALayerClustersExtraDeviceCollection::ConstView layerClustersClustering,
                                          ticl::TracksterDeviceCollection::View tracksters) const {
    // the layer clusters of the first trackster start at 0, also when there is no trackster
    uint32_t* offsets = tracksters.vertices().offsets().keys_offsets().data();
    auto firstOffset = make_device_view(queue, offsets, 1);
    alpaka::memset(queue, firstOffset, 0x0);
    const uint32_t nTracksters = tracksters.energy().metadata().size();
    if (nTracksters == 0) {
      return;
    }
    const uint32_t nLayerClusters = layerClustersClustering.metadata().size();

    auto tracksterSizes = make_device_buffer<uint32_t[]>(queue, nTracksters);
    alpaka::memset(queue, tracksterSizes, 0x0);

    constexpr uint32_t blockSize = 256;
    const auto workDivLayerClusters = make_workdiv<Acc1D>(divide_up_by(nLayerClusters, blockSize), blockSize);
    alpaka::exec<Acc1D>(
        queue, workDivLayerClusters, KernelTracksterSizes{}, layerClustersClustering, tracksterSizes.data());
    // inclusive scan of the sizes, so offsets[t + 1] is where the layer clusters of trackster t end. The scan leaves
    // the sizes unchanged.
    iterativePrefixScan<Acc1D>(queue, tracksterSizes.data(), offsets + 1, nTracksters);
    alpaka::exec<Acc1D>(queue,
                        workDivLayerClusters,
                        KernelTracksterVertices{},
                        layerClustersClustering,
                        tracksterSizes.data(),
                        tracksters);

    // the PCA with the cleaning window, as the pattern recognition does
    ticl::properties::fillTracksterProperties<true>(queue, zLimitEm, layerClusters, tracksters);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
