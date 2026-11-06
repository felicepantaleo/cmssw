#include <cmath>
#include <cstdint>
#include <span>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "HGCalCLUEsteringAlgoWrapper.h"

#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/core/Clusterer.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  namespace {

    // Per seed: its rank in rechit order is the index of its layer cluster. seedEnd is the inclusive scan of isSeed.
    struct KernelRankSeeds {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const int32_t* isSeed,
                                    const int32_t* seedEnd,
                                    HGCalSoARecHitsExtraDeviceCollection::ConstView recHitsClustering,
                                    int32_t* rankedIndex,
                                    uint32_t nRecHits) const {
        for (auto i : uniform_elements(acc, nRecHits)) {
          if (isSeed[i]) {
            rankedIndex[recHitsClustering[i].clusterIndex()] = seedEnd[i] - 1;
          }
        }
      }
    };

    // Per rechit: its layer cluster as the rank of the seed, its seed flag, and the number of layer clusters
    struct KernelMarkSeeds {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const int32_t* isSeed,
                                    const int32_t* rankedIndex,
                                    HGCalSoARecHitsExtraDeviceCollection::View recHitsClustering,
                                    uint32_t nRecHits) const {
        for (auto i : uniform_elements(acc, nRecHits)) {
          const int cluster = recHitsClustering[i].clusterIndex();
          if (cluster >= 0) {
            recHitsClustering[i].clusterIndex() = rankedIndex[cluster];
          }
          recHitsClustering[i].isSeed() = isSeed[i];
          if (isSeed[i]) {
            alpaka::atomicAdd(acc, &recHitsClustering.numberOfClustersScalar(), 1u);
          }
        }
      }
    };

  }  // namespace

  void HGCalCLUEsteringAlgoWrapper::run(Queue& queue,
                                        uint32_t nRecHits,
                                        float dc,
                                        float kappa,
                                        float outlierDeltaFactor,
                                        bool isScintillator,
                                        std::span<const uint32_t> batchItemSizes,
                                        HGCalSoARecHitsDeviceCollection::ConstView recHits,
                                        HGCalSoARecHitsExtraDeviceCollection::View recHitsClustering) const {
    auto nClusters = make_device_view<unsigned int>(queue, recHitsClustering.numberOfClustersScalar());
    alpaka::memset(queue, nClusters, 0x0);
    if (nRecHits == 0) {
      return;
    }

    clue::ConstPointsDevice<2, float> d_points(queue,
                                               static_cast<int32_t>(nRecHits),
                                               recHits.dim1().data(),
                                               recHits.dim2().data(),
                                               recHits.energy().data(),
                                               recHitsClustering.clusterIndex().data());
    d_points.set_density_uncertainty(recHits.sigmaNoise());
    d_points.set_tags(recHits.detid());

    clue::Clusterer<2> algo(queue, dc, kappa, dc * outlierDeltaFactor);
    if (isScintillator) {
      // Scintillator (BH) cells cluster in (eta, phi)
      constexpr float kTwoPi = 2.f * static_cast<float>(M_PI);
      algo.setWrappedCoordinates(0, 1);
      algo.make_clusters(queue,
                         d_points,
                         batchItemSizes,
                         clue::PeriodicEuclideanMetric<2, float>{0.f, kTwoPi},
                         clue::FlatKernel<float>{0.5f});
    } else {
      algo.make_clusters(queue, d_points, batchItemSizes);
    }

    // The layer clusters are numbered by their seeds in rechit order, so the result does not depend on the order in
    // which CLUEstering finds the seeds. d_points releases is_seed only after the queued kernels that read it.
    const int32_t* isSeed = d_points.view().is_seed().data();
    auto seedEnd = make_device_buffer<int32_t[]>(queue, nRecHits);
    iterativePrefixScan<Acc1D>(queue, isSeed, seedEnd.data(), nRecHits);
    // there are at most as many layer clusters as rechits
    auto rankedIndex = make_device_buffer<int32_t[]>(queue, nRecHits);
    constexpr uint32_t blockSize = 256;
    const auto workDivRecHits = make_workdiv<Acc1D>(divide_up_by(nRecHits, blockSize), blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDivRecHits,
                        KernelRankSeeds{},
                        isSeed,
                        seedEnd.data(),
                        recHitsClustering,
                        rankedIndex.data(),
                        nRecHits);
    alpaka::exec<Acc1D>(
        queue, workDivRecHits, KernelMarkSeeds{}, isSeed, rankedIndex.data(), recHitsClustering, nRecHits);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
