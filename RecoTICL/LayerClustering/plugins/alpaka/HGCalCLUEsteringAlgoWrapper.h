#ifndef RecoTICL_LayerClustering_plugins_alpaka_HGCalCLUEsteringAlgoWrapper_h
#define RecoTICL_LayerClustering_plugins_alpaka_HGCalCLUEsteringAlgoWrapper_h

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>

#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Clusters the rechits of one HGCAL component with CLUEstering. Each event uses its own CLUEstering buffers.
  class HGCalCLUEsteringAlgoWrapper {
  public:
    // The scintillator rechits cluster in (eta, phi), with a periodic phi in [0, 2 pi)
    HGCalCLUEsteringAlgoWrapper(float dc, float kappa, float outlierDeltaFactor, bool isScintillator);

    // The ranges of the two coordinates for the tiles. Without them, CLUEstering computes them from the rechits of
    // every event. The tiles set the order in which the densities are summed, so the ranges can change the rounding.
    void setExtremes(std::pair<float, float> dim1, std::pair<float, float> dim2);

    // Clusters the rechits of each layer (one batch per layer, of batchItemSizes rechits): writes the layer cluster
    // and the seed flag of each rechit, and the number of layer clusters. The layer clusters are numbered in the order
    // of their seeds in the rechits.
    void run(Queue& queue,
             uint32_t nRecHits,
             std::span<const uint32_t> batchItemSizes,
             HGCalSoARecHitsDeviceCollection::ConstView recHits,
             HGCalSoARecHitsExtraDeviceCollection::View recHitsClustering);

  private:
    float dc_;
    float kappa_;
    float outlierDistance_;
    std::optional<std::array<std::pair<float, float>, 2>> extremes_;
    bool isScintillator_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_LayerClustering_plugins_alpaka_HGCalCLUEsteringAlgoWrapper_h
