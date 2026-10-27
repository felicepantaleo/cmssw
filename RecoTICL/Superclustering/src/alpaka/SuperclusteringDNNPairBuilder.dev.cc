#include <cstdint>
#include <utility>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNNPairBuilder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {

    // TODO: preselection of seeds and candidates (energy, pT, explained variance ratio, PID)
    struct PreselectTrackstersKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, ::ticl::TracksterSoA::ConstView tracksters) const {}
    };

    // TODO: number of compatible seeds (higher pT, in the eta-phi window) per candidate
    struct CountPairsKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, ::ticl::TracksterSoA::ConstView tracksters) const {}
    };

    // TODO: write the seed and candidate indices and the features of each pair, the device version of
    // SuperclusteringDNNInputV*::computeInto
    struct FillPairsKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::ticl::TracksterSoA::ConstView tracksters,
                                    ::ticl::SuperclusteringDNNPairsSoA::View pairs,
                                    ::ticl::SuperclusteringDNNPairFeaturesSoA::View features) const {}
    };

  }  // namespace

  SuperclusteringDNNPairBuilder::SuperclusteringDNNPairBuilder(SuperclusteringDNNPairSelection selection)
      : selection_{std::move(selection)} {}

  void SuperclusteringDNNPairBuilder::prepare(Queue& queue, ::ticl::TracksterSoA::ConstView tracksters) {
    pairCountHost_ = cms::alpakatools::make_host_buffer<uint32_t>(queue);
    *(*pairCountHost_) = 0u;
    // TODO: enqueue PreselectTrackstersKernel and CountPairsKernel, a prefix sum of the counts into
    // candidatePairOffsetsDevice_, and the copy of the total to pairCountHost_
  }

  uint32_t SuperclusteringDNNPairBuilder::pairCount() const { return *(*pairCountHost_); }

  void SuperclusteringDNNPairBuilder::fill(Queue& queue,
                                           ::ticl::TracksterSoA::ConstView tracksters,
                                           ticl::SuperclusteringDNNPairsDevice& pairs,
                                           ticl::SuperclusteringDNNPairFeaturesDevice& features) {
    if (pairCount() == 0u) {
      return;
    }
    // TODO: enqueue FillPairsKernel
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
