#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalSimHitListDevice.h"
#include "SimCalorimetry/HGCalSimProducers/interface/alpaka/HGCalSimHitListAlgorithms.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  using hgcaldigi::HGCalRealisticSimHitDevice;
  using hgcaldigi::HGCalSimHitListDevice;

  namespace {

    // Counts the hits of each cell.
    struct HGCalSimHitListKernel_count {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    HGCalSimHitListDevice::ConstView hits,
                                    uint32_t* __restrict__ counts) const {
        for (auto i : uniform_elements(acc, hits.metadata().size())) {
          alpaka::atomicAdd(acc, &counts[hits[i].denseIdx()], 1u, alpaka::hierarchy::Blocks{});
        }
      }
    };

    // Writes the index of each hit in the range of its cell. The order inside a range is arbitrary;
    // on a sequential backend it is the input order. At the end, all counts are zero.
    struct HGCalSimHitListKernel_fill {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    HGCalSimHitListDevice::ConstView hits,
                                    uint32_t const* __restrict__ ends,
                                    uint32_t* __restrict__ counts,
                                    uint32_t* __restrict__ order) const {
        for (auto i : uniform_elements(acc, hits.metadata().size())) {
          auto const cell = hits[i].denseIdx();
          auto const remaining = alpaka::atomicSub(acc, &counts[cell], 1u, alpaka::hierarchy::Blocks{});
          order[ends[cell] - remaining] = i;
        }
      }
    };

    // Sums the hits of each cell in input order, so the sums are equal to a sequential loop on the host.
    struct HGCalSimHitListKernel_sum {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    HGCalSimHitListDevice::ConstView hits,
                                    uint32_t const* __restrict__ ends,
                                    uint32_t* __restrict__ order,
                                    HGCalRealisticSimHitDevice::View cells) const {
        for (auto cell : uniform_elements(acc, cells.metadata().size())) {
          auto const begin = cell == 0 ? 0u : ends[cell - 1];
          auto const end = ends[cell];

          for (auto i = begin + 1; i < end; ++i) {
            auto const hit = order[i];
            auto j = i;
            for (; j > begin && order[j - 1] > hit; --j) {
              order[j] = order[j - 1];
            }
            order[j] = hit;
          }

          auto row = cells[cell];
          row.sumQ().setZero();
          row.avgTOA().setZero();
          for (auto i = begin; i < end; ++i) {
            auto const hit = hits[order[i]];
            row.sumQ()(hit.timeBin(), 0) += hit.energy();
            row.avgTOA()(hit.timeBin(), 0) += hit.energyXtof();
          }
        }
      }
    };

  }  // namespace

  HGCalRealisticSimHitDevice HGCalSimHitListAlgorithms::accumulate(Queue& queue,
                                                                   ::hgcaldigi::HGCalSimHitListHost const& hostHits,
                                                                   int32_t nCells) const {
    HGCalRealisticSimHitDevice cells(queue, nCells);
    if (nCells == 0) {
      return cells;
    }

    auto const nHits = hostHits.const_view().metadata().size();
    HGCalSimHitListDevice hits(queue, nHits);
    alpaka::memcpy(queue, hits.buffer(), hostHits.const_buffer());

    auto counts = make_device_buffer<uint32_t[]>(queue, nCells);
    auto ends = make_device_buffer<uint32_t[]>(queue, nCells);
    auto order = make_device_buffer<uint32_t[]>(queue, nHits);
    alpaka::memset(queue, counts, 0);

    uint32_t const items = nThreads_;
    if (nHits > 0) {
      auto const grid = make_workdiv<Acc1D>(divide_up_by(nHits, items), items);
      alpaka::exec<Acc1D>(queue, grid, HGCalSimHitListKernel_count{}, hits.const_view(), counts.data());
    }

    // ends[c] is the number of hits in the cells 0 to c
    prefixScan<Acc1D>(queue, counts.data(), ends.data(), static_cast<uint32_t>(nCells));

    if (nHits > 0) {
      auto const grid = make_workdiv<Acc1D>(divide_up_by(nHits, items), items);
      alpaka::exec<Acc1D>(
          queue, grid, HGCalSimHitListKernel_fill{}, hits.const_view(), ends.data(), counts.data(), order.data());
    }

    auto const grid = make_workdiv<Acc1D>(divide_up_by(nCells, items), items);
    alpaka::exec<Acc1D>(
        queue, grid, HGCalSimHitListKernel_sum{}, hits.const_view(), ends.data(), order.data(), cells.view());

    // the scratch buffers are freed in queue order by the caching allocator
    return cells;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
