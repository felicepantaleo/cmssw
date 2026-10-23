
#include <cmath>
#include <cstdint>
#include <tuple>

#include <alpaka/alpaka.hpp>
#include <xtd/math.h>

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTICL/Inference/interface/TracksterInferenceSoA.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterFeatureExtractors.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {

    ALPAKA_FN_HOST_ACC float absoluteEta(float x, float y, float z) {
      const auto rho = xtd::sqrt(x * x + y * y);
      const auto absZ = xtd::abs(z);

      return xtd::asinh(absZ / rho);
    }

    // Same seed-ID predicate as ticlgeom::Tools::isBarrel().
    ALPAKA_FN_HOST_ACC bool isBarrel(DetId id) {
      return (id.det() == DetId::Ecal && id.subdetId() == 1) ||
             (id.det() == DetId::Hcal && (id.subdetId() == 1 || id.subdetId() == 3));
    }

    // ---------------------------------------------------------------------------
    // Trackster selection
    // ---------------------------------------------------------------------------

    template <TracksterModelFamily Family>
    struct SelectTrackstersKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::ticl::TracksterSoA::ConstView inputTracksters,
                                    LayerClusterEnergyMultiView clusterEnergies,
                                    LayerClusterIndexMultiView clusterIndexes,
                                    float minClusterEnergy,
                                    uint32_t* selectionMask,
                                    uint32_t nTracksters) const {
        const auto associations = inputTracksters.vertices();

        for (auto tracksterIndex : cms::alpakatools::uniform_elements(acc, nTracksters)) {
          const auto vertices = associations[tracksterIndex];
          bool selected = false;

          // CNN: select Tracksters above the energy threshold, including barrel clusters.
          if constexpr (Family == TracksterModelFamily::CNN) {
            auto clusterEnergy = 0.f;

            for (auto k = 0u; k < vertices.size(); ++k) {
              clusterEnergy += clusterEnergies[vertices[k]].energy();

              if (clusterEnergy >= minClusterEnergy) {
                selected = true;
                break;
              }
            }

            // DNN: apply the energy threshold using only non-barrel clusters.
          } else if constexpr (Family == TracksterModelFamily::DNN) {
            auto clusterEnergy = 0.f;

            for (auto k = 0u; k < vertices.size(); ++k) {
              if (isBarrel(clusterIndexes[vertices[k]].seedID()))
                continue;

              clusterEnergy += clusterEnergies[vertices[k]].energy();

              if (clusterEnergy >= minClusterEnergy) {
                selected = true;
                break;
              }
            }

            // PFN: reject Tracksters containing any barrel cluster.
          } else if constexpr (Family == TracksterModelFamily::PFN) {
            selected = true;

            for (auto k = 0u; k < vertices.size(); ++k) {
              if (isBarrel(clusterIndexes[vertices[k]].seedID())) {
                selected = false;
                break;
              }
            }
          }

          selectionMask[tracksterIndex] = selected ? 1u : 0u;
        }
      }
    };

    // ---------------------------------------------------------------------------
    // Feature extraction
    // ---------------------------------------------------------------------------

    template <TracksterModelFamily Family, typename FeatureView>
    struct FillInputFeaturesKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::ticl::TracksterSoA::ConstView inputTracksters,
                                    LayerClusterEnergyMultiView clusterEnergies,
                                    LayerClusterIndexMultiView clusterIndexes,
                                    LayerClusterPositionMultiView clusterPositions,
                                    uint32_t layersPerEndcap,
                                    uint32_t const* selectionMask,
                                    uint32_t const* selectionOffsets,
                                    uint32_t* selectedTracksters,
                                    FeatureView features,
                                    uint32_t nTracksters) const {
        const auto verticesAssociations = inputTracksters.vertices();
        const auto multiplicityAssociations = inputTracksters.multiplicity();

        constexpr auto maxClusters = ::ticl::kMaxClusters;
        constexpr auto nLayers = ::ticl::kMaxLayers;

        // One thread processes one Trackster.
        for (auto tracksterIndex : cms::alpakatools::uniform_elements(acc, nTracksters)) {
          if (selectionMask[tracksterIndex] == 0u)
            continue;

          const auto vertices = verticesAssociations[tracksterIndex];
          const auto multiplicities = multiplicityAssociations[tracksterIndex];

          // Map the original Trackster index to its compact output row.
          const auto row = selectionOffsets[tracksterIndex] - 1u;
          selectedTracksters[row] = tracksterIndex;
          auto feature = features[row];

          // PFN additionally uses global Trackster properties.
          if constexpr (Family == TracksterModelFamily::PFN) {
            const auto ts = inputTracksters.tracksters()[tracksterIndex];

            const auto x = ts.barycenterX();
            const auto y = ts.barycenterY();
            const auto z = xtd::abs(ts.barycenterZ());

            feature.rawEnergy() = ts.raw_energy();
            feature.rawEmEnergy() = ts.raw_em_energy();
            feature.barycenterX() = x;
            feature.barycenterY() = y;
            feature.barycenterAbsZ() = z;
            feature.barycenterAbsEta() = absoluteEta(x, y, z);
            feature.barycenterPhi() = xtd::atan2(y, x);
          }

          // Keep at most maxClusters vertices per layer, sorted by decreasing
          // LayerCluster energy. Equal-energy ordering is not required.
          uint32_t selected[nLayers][maxClusters];
          uint8_t counts[nLayers] = {};

          for (auto k = 0u; k < vertices.size(); ++k) {
            const auto vertexIndex = vertices[k];

            if constexpr (Family == TracksterModelFamily::DNN) {
              if (isBarrel(clusterIndexes[vertexIndex].seedID()))
                continue;
            }

            const auto position = clusterPositions[vertexIndex];
            const auto layer = static_cast<uint32_t>(position.layer()) % layersPerEndcap;

            if (layer >= nLayers)
              continue;

            const auto energy = clusterEnergies[vertexIndex].energy();
            auto& count = counts[layer];

            // Find the position of this vertex among the retained top-K.
            auto slot = 0;
            while (slot < count && clusterEnergies[vertices[selected[layer][slot]]].energy() >= energy)
              ++slot;

            if (slot >= maxClusters)
              continue;

            // Insert the vertex, discarding the lowest-energy one if full.
            const auto end = count < maxClusters ? count : maxClusters - 1;
            for (auto j = end; j > slot; --j)
              selected[layer][j] = selected[layer][j - 1];

            selected[layer][slot] = k;

            if (count < maxClusters)
              ++count;
          }

          // Fill the model input features from the retained LayerClusters.
          for (auto layer = 0u; layer < nLayers; ++layer) {
            for (auto slot = 0u; slot < counts[layer]; ++slot) {
              const auto k = selected[layer][slot];
              const auto vertexIndex = vertices[k];
              const auto multiplicity = multiplicities[k];

              const auto position = clusterPositions[vertexIndex];
              const auto x = position.x();
              const auto y = position.y();
              const auto z = position.z();

              // Sort by original LayerCluster energy, but store the energy
              // corrected for the Trackster vertex multiplicity.
              feature.energy()(layer, slot) = clusterEnergies[vertexIndex].energy() / multiplicity;
              feature.absEta()(layer, slot) = absoluteEta(x, y, z);
              feature.phi()(layer, slot) = xtd::atan2(y, x);

              if constexpr (Family == TracksterModelFamily::PFN) {
                feature.x()(layer, slot) = x;
                feature.y()(layer, slot) = y;
                feature.absZ()(layer, slot) = xtd::abs(z);
                feature.cells()(layer, slot) = static_cast<float>(position.cells());
              }
            }
          }
        }
      }
    };

  }  // namespace

  // ---------------------------------------------------------------------------
  // Trackster feature extractor
  // ---------------------------------------------------------------------------

  template <TracksterModelFamily Family>
  TracksterFeatureExtractor<Family>::TracksterFeatureExtractor(edm::ParameterSet const& config)
      : minClusterEnergy_{static_cast<float>(config.getParameter<double>("minClusterEnergy"))} {
    if (!std::isfinite(minClusterEnergy_))
      throw cms::Exception("Configuration") << "minClusterEnergy must be finite";
  }

  template <TracksterModelFamily Family>
  void TracksterFeatureExtractor<Family>::fillDescription(edm::ParameterSetDescription& description) {
    description.add<double>("minClusterEnergy", 1.0);
  }

  template <TracksterModelFamily Family>
  void TracksterFeatureExtractor<Family>::select(Queue& queue,
                                                 ::ticl::TracksterSoA::ConstView inputTracksters,
                                                 LayerClusterEnergyMultiView clusterEnergies,
                                                 LayerClusterIndexMultiView clusterIndexes,
                                                 uint32_t* selectionMask,
                                                 uint32_t* selectionOffsets,
                                                 uint32_t nTracksters) const {
    if (nTracksters == 0u)
      return;

    constexpr auto threadsPerBlock = 256u;
    const auto blocks = cms::alpakatools::divide_up_by(nTracksters, threadsPerBlock);
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SelectTrackstersKernel<Family>{},
                        inputTracksters,
                        clusterEnergies,
                        clusterIndexes,
                        minClusterEnergy_,
                        selectionMask,
                        nTracksters);

    auto blockCounter = cms::alpakatools::make_device_buffer<int32_t>(queue);
    alpaka::memset(queue, blockCounter, 0);

    cms::alpakatools::checkSharedMemoryPrefixScan<Acc1D>(nTracksters, blocks, alpaka::getDev(queue));

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        cms::alpakatools::multiBlockPrefixScan<uint32_t>{},
                        selectionMask,
                        selectionOffsets,
                        nTracksters,
                        static_cast<int32_t>(blocks),
                        blockCounter.data(),
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));
  }

  template <TracksterModelFamily Family>
  void TracksterFeatureExtractor<Family>::fill(Queue& queue,
                                               ::ticl::TracksterSoA::ConstView inputTracksters,
                                               LayerClusterEnergyMultiView clusterEnergies,
                                               LayerClusterIndexMultiView clusterIndexes,
                                               LayerClusterPositionMultiView clusterPositions,
                                               uint32_t layersPerEndcap,
                                               uint32_t const* selectionMask,
                                               uint32_t const* selectionOffsets,
                                               FeaturesDevice& features,
                                               uint32_t nTracksters) const {
    if (nTracksters == 0u)
      return;

    auto featureRecords = features.view().records();

    constexpr auto elementsPerBlock = 256u;
    const auto blocks = cms::alpakatools::divide_up_by(nTracksters, elementsPerBlock);
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, elementsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        FillInputFeaturesKernel<Family, typename FeaturesDevice::View>{},
                        inputTracksters,
                        clusterEnergies,
                        clusterIndexes,
                        clusterPositions,
                        layersPerEndcap,
                        selectionMask,
                        selectionOffsets,
                        std::get<0>(featureRecords.selectedTracksterIndex()).data(),
                        features.view(),
                        nTracksters);
  }

  template struct TracksterFeatureExtractor<TracksterModelFamily::CNN>;
  template struct TracksterFeatureExtractor<TracksterModelFamily::DNN>;
  template struct TracksterFeatureExtractor<TracksterModelFamily::PFN>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
