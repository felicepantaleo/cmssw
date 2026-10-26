#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "HGCalMergeLayerClustersSoAAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  namespace {

    // Per layer cluster: copies all its columns to row offset + i of the merged layer clusters
    struct KernelMergeLayerClusters {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    reco::CaloClusterDeviceCollection::ConstView layerClusters,
                                    uint32_t offset,
                                    reco::CaloClusterDeviceCollection::View mergedLayerClusters) const {
        for (auto i : uniform_elements(acc, layerClusters.position().metadata().size())) {
          mergedLayerClusters.position()[offset + i] = layerClusters.position()[i];
          mergedLayerClusters.energy()[offset + i] = layerClusters.energy()[i];
          mergedLayerClusters.indexes()[offset + i] = layerClusters.indexes()[i];
          mergedLayerClusters.timing()[offset + i] = layerClusters.timing()[i];
        }
      }
    };

  }  // namespace

  void HGCalMergeLayerClustersSoAAlgoWrapper::run(Queue& queue,
                                                  reco::CaloClusterDeviceCollection::ConstView layerClusters,
                                                  uint32_t offset,
                                                  reco::CaloClusterDeviceCollection::View mergedLayerClusters) const {
    const uint32_t nLayerClusters = layerClusters.position().metadata().size();
    if (nLayerClusters == 0) {
      return;
    }

    constexpr uint32_t blockSize = 256;
    const auto workDivLayerClusters = make_workdiv<Acc1D>(divide_up_by(nLayerClusters, blockSize), blockSize);
    alpaka::exec<Acc1D>(
        queue, workDivLayerClusters, KernelMergeLayerClusters{}, layerClusters, offset, mergedLayerClusters);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
