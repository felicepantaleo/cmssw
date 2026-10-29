#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include <alpaka/alpaka.hpp>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "HGCalCLUEsteringTrackstersAlgoWrapper.h"

#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/core/Clusterer.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  // The points are the layer clusters that are clustered, in the order of the layer clusters. Per point: the inputs of
  // CLUEstering and its output (the cluster index). The coordinates are dim1 = x / |z| / sigmaRef,
  // dim2 = y / |z| / sigmaRef and dim3 = layer / layerScale, with sigmaRef the smallest sigmaT; sigmaT is the one of
  // the HGCAL component of the point, relative to sigmaRef. Outside of the anonymous namespace, as nvcc warns about the
  // unused members of a SoA layout with internal linkage.
  GENERATE_SOA_LAYOUT(HGCalCLUEsteringTrackstersPointsLayout,
                      SOA_COLUMN(float, dim1),
                      SOA_COLUMN(float, dim2),
                      SOA_COLUMN(float, dim3),
                      SOA_COLUMN(float, weight),
                      SOA_COLUMN(float, rhocScale),
                      SOA_COLUMN(uint32_t, tag),
                      SOA_COLUMN(float, sigmaT),
                      SOA_COLUMN(float, searchSigma),
                      SOA_COLUMN(int, clusterIndex))

  namespace {

    constexpr uint32_t kComponents = HGCalCLUEsteringTrackstersAlgoWrapper::kComponents;
    constexpr int kInvalidCluster = -1;
    constexpr int kInvalidTrackster = -1;

    using PointsDeviceCollection = PortableCollection<HGCalCLUEsteringTrackstersPointsLayout<>>;

    // The HGCAL component of a layer cluster, from the DetId of its seed
    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t component(const DetId seed) { return seed.det() - DetId::HGCalEE; }

    // The clustering distance: max(transverse distance / mean sigmaT of the pair, layer distance). Its unit ball is a
    // disc times an interval along the layer axis, and the pairs of the coarser HGCAL components see their transverse
    // distance shrunk.
    struct CylinderMetric {
      using value_type = float;

      const float* sigmaT;

      template <typename TPoints>
      ALPAKA_FN_HOST_ACC float operator()(const TPoints& points, std::size_t i, std::size_t j) const {
        const auto pi = points[static_cast<int>(i)];
        const auto pj = points[static_cast<int>(j)];
        const float d1 = pi[0] - pj[0];
        const float d2 = pi[1] - pj[1];
        const float transverse = std::sqrt(d1 * d1 + d2 * d2) * 2.f / (sigmaT[i] + sigmaT[j]);
        return std::max(transverse, std::abs(pi[2] - pj[2]));
      }
    };

    // The ranges of layers and of transverse radii (in clustering coordinates) of the points of each HGCAL component on
    // each endcap, in slot 2 * component + endcap. The radii are stored as the bits of the non-negative floats, which
    // are ordered as their values, so that integer atomics can reduce them.
    constexpr uint32_t kSlots = 2 * kComponents;
    struct ComponentExtents {
      int32_t minLayer[kSlots];
      int32_t maxLayer[kSlots];
      uint32_t minRadius[kSlots];
      uint32_t maxRadius[kSlots];
    };

    // The extents before any point is added: empty ranges
    constexpr ComponentExtents emptyExtents() {
      ComponentExtents extents{};
      for (uint32_t slot = 0; slot < kSlots; ++slot) {
        extents.minLayer[slot] = std::numeric_limits<int32_t>::max();
        extents.maxLayer[slot] = std::numeric_limits<int32_t>::min();
        extents.minRadius[slot] = std::numeric_limits<uint32_t>::max();
        extents.maxRadius[slot] = 0;
      }
      return extents;
    }

    // Per layer cluster: whether it is a point, as in the CLUE3D layer cluster filter: a silicon layer cluster needs
    // minSiliconCells cells. Also resets its trackster, and the extents of the HGCAL components.
    struct KernelSelectPoints {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    uint32_t minSiliconCells,
                                    reco::CaloClusterDeviceCollection::ConstView layerClusters,
                                    HGCalSoALayerClustersExtraDeviceCollection::View layerClustersClustering,
                                    uint32_t* isPoint,
                                    ComponentExtents* extents) const {
        if (once_per_grid(acc)) {
          *extents = emptyExtents();
        }
        for (auto i : uniform_elements(acc, layerClusters.position().metadata().size())) {
          const bool isScintillator = layerClusters.indexes()[i].seedID().det() == DetId::HGCalHSc;
          isPoint[i] = isScintillator or layerClusters.position()[i].cells() >= static_cast<int>(minSiliconCells);
          layerClustersClustering[i].tracksterIndex() = kInvalidTrackster;
        }
      }
    };

    // Per layer cluster -> per point: the clustering coordinates, weight, seed threshold scale, tag (the seed DetId, to
    // break the ties deterministically) and sigmaT of the point, and the extents of the HGCAL components.
    // The extents are reduced within each block first, so that few atomics reach global memory.
    struct KernelFillPoints {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    float invSigmaRef,
                                    float invLayerScale,
                                    float endcapGap,
                                    float rhocEtaExponent,
                                    float rhocPivotRadius,
                                    std::array<float, kComponents> relativeSigmaT,
                                    reco::CaloClusterDeviceCollection::ConstView layerClusters,
                                    const uint32_t* isPoint,
                                    const uint32_t* pointEnd,
                                    PointsDeviceCollection::View points,
                                    ComponentExtents* extents) const {
        constexpr auto block = alpaka::hierarchy::Threads{};
        auto& blockExtents = alpaka::declareSharedVar<ComponentExtents, __COUNTER__>(acc);
        if (once_per_block(acc)) {
          blockExtents = emptyExtents();
        }
        alpaka::syncBlockThreads(acc);

        for (auto i : uniform_elements(acc, layerClusters.position().metadata().size())) {
          if (not isPoint[i]) {
            continue;
          }
          const uint32_t p = pointEnd[i] - 1;
          const auto position = layerClusters.position()[i];
          const DetId seed = layerClusters.indexes()[i].seedID();
          const uint32_t c = component(seed);
          const uint32_t endcap = position.z() > 0.f;

          // the gnomonic projection, and the layer: the endcaps are kept further apart than any clustering distance
          const float invAbsZ = 1.f / std::abs(position.z());
          const float u = position.x() * invAbsZ;
          const float v = position.y() * invAbsZ;
          const float dim1 = u * invSigmaRef;
          const float dim2 = v * invSigmaRef;
          points[p].dim1() = dim1;
          points[p].dim2() = dim2;
          points[p].dim3() = position.layer() * invLayerScale + endcap * endcapGap;
          points[p].weight() = layerClusters.energy()[i].energy();
          // the seed threshold grows with eta: r = 0 is outside of HGCAL
          const float r = std::sqrt(u * u + v * v);
          points[p].rhocScale() = std::pow(rhocPivotRadius / r, rhocEtaExponent);
          points[p].tag() = seed.rawId();
          points[p].sigmaT() = relativeSigmaT[c];

          // the radius in clustering coordinates, computed as KernelSearchWindow computes it
          const uint32_t slot = 2 * c + endcap;
          const uint32_t radius = std::bit_cast<uint32_t>(std::sqrt(dim1 * dim1 + dim2 * dim2));
          alpaka::atomicMin(acc, &blockExtents.minLayer[slot], position.layer(), block);
          alpaka::atomicMax(acc, &blockExtents.maxLayer[slot], position.layer(), block);
          alpaka::atomicMin(acc, &blockExtents.minRadius[slot], radius, block);
          alpaka::atomicMax(acc, &blockExtents.maxRadius[slot], radius, block);
        }
        alpaka::syncBlockThreads(acc);

        if (once_per_block(acc)) {
          for (uint32_t slot = 0; slot < kSlots; ++slot) {
            alpaka::atomicMin(acc, &extents->minLayer[slot], blockExtents.minLayer[slot]);
            alpaka::atomicMax(acc, &extents->maxLayer[slot], blockExtents.maxLayer[slot]);
            alpaka::atomicMin(acc, &extents->minRadius[slot], blockExtents.minRadius[slot]);
            alpaka::atomicMax(acc, &extents->maxRadius[slot], blockExtents.maxRadius[slot]);
          }
        }
      }
    };

    // Per point: the sigma of its transverse search window. A neighbour j is within the distance R of point i only if
    // their transverse distance, so the difference of their radii too, is at most R * (sigmaT_i + sigmaT_j) / 2: the
    // window of point i must cover the coarsest HGCAL component within reach along the layer axis and in radius, on its
    // endcap. CLUEstering searches within R * searchSigma * sqrt(2) along each transverse coordinate.
    struct KernelSearchWindow {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    float maxDistance,
                                    int32_t layerReach,
                                    std::array<float, kComponents> relativeSigmaT,
                                    reco::CaloClusterDeviceCollection::ConstView layerClusters,
                                    const uint32_t* isPoint,
                                    const uint32_t* pointEnd,
                                    const ComponentExtents* extents,
                                    PointsDeviceCollection::View points) const {
        // 1 / sqrt(2) rounded up, so that the window is never narrower than needed
        constexpr float invSqrt2 = 0.70710681f;
        for (auto i : uniform_elements(acc, layerClusters.position().metadata().size())) {
          if (not isPoint[i]) {
            continue;
          }
          const uint32_t p = pointEnd[i] - 1;
          const int32_t layer = layerClusters.position()[i].layer();
          const uint32_t endcap = layerClusters.position()[i].z() > 0.f;
          const float radius = std::sqrt(points[p].dim1() * points[p].dim1() + points[p].dim2() * points[p].dim2());
          const float sigmaT = points[p].sigmaT();
          float coarsestSigmaT = sigmaT;
          for (uint32_t c = 0; c < kComponents; ++c) {
            const uint32_t slot = 2 * c + endcap;
            // the transverse reach of the pairs with component c
            const float reach = maxDistance * 0.5f * (sigmaT + relativeSigmaT[c]);
            const bool withinReach = extents->minLayer[slot] <= layer + layerReach and
                                     extents->maxLayer[slot] >= layer - layerReach and
                                     std::bit_cast<float>(extents->minRadius[slot]) <= radius + reach and
                                     std::bit_cast<float>(extents->maxRadius[slot]) >= radius - reach;
            if (withinReach) {
              coarsestSigmaT = std::max(coarsestSigmaT, relativeSigmaT[c]);
            }
          }
          points[p].searchSigma() = 0.5f * (sigmaT + coarsestSigmaT) * invSqrt2;
        }
      }
    };

    // Per point -> per cluster: the number of points
    struct KernelCountClusterSizes {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    uint32_t nPoints,
                                    PointsDeviceCollection::ConstView points,
                                    uint32_t* clusterSizes) const {
        for (auto p : uniform_elements(acc, nPoints)) {
          const int cluster = points[p].clusterIndex();
          if (cluster != kInvalidCluster) {
            alpaka::atomicAdd(acc, &clusterSizes[cluster], 1u);
          }
        }
      }
    };

    // Per cluster: whether it is a trackster, having at least minLayerClusters (>= 1) layer clusters, and the number
    // of layer clusters in the tracksters
    struct KernelSelectTracksters {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    uint32_t nClusters,
                                    uint32_t minLayerClusters,
                                    const uint32_t* clusterSizes,
                                    uint32_t* isTrackster,
                                    HGCalSoALayerClustersExtraDeviceCollection::View layerClustersClustering) const {
        for (auto c : uniform_elements(acc, nClusters)) {
          const bool selected = clusterSizes[c] >= minLayerClusters;
          isTrackster[c] = selected;
          if (selected) {
            alpaka::atomicAdd(acc, &layerClustersClustering.numberOfVerticesScalar(), clusterSizes[c]);
          }
        }
      }
    };

    // Per layer cluster: its trackster, if it is a point and its cluster is a trackster. tracksterEnd[c] - 1 is the
    // trackster of cluster c, so the tracksters keep the order of the clusters.
    struct KernelWriteLayerClustersClustering {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    uint32_t nPoints,
                                    const uint32_t* isPoint,
                                    const uint32_t* pointEnd,
                                    PointsDeviceCollection::ConstView points,
                                    const uint32_t* isTrackster,
                                    const uint32_t* tracksterEnd,
                                    HGCalSoALayerClustersExtraDeviceCollection::View layerClustersClustering) const {
        if (once_per_grid(acc)) {
          layerClustersClustering.numberOfTrackstersScalar() = tracksterEnd[nPoints - 1];
        }
        for (auto i : uniform_elements(acc, layerClustersClustering.metadata().size())) {
          if (not isPoint[i]) {
            continue;
          }
          const int cluster = points[pointEnd[i] - 1].clusterIndex();
          if (cluster == kInvalidCluster or not isTrackster[cluster]) {
            continue;
          }
          layerClustersClustering[i].tracksterIndex() = tracksterEnd[cluster] - 1;
        }
      }
    };

  }  // namespace

  void HGCalCLUEsteringTrackstersAlgoWrapper::run(
      Queue& queue,
      const Parameters& parameters,
      reco::CaloClusterDeviceCollection::ConstView layerClusters,
      HGCalSoALayerClustersExtraDeviceCollection::View layerClustersClustering) const {

    auto nTracksters = make_device_view<unsigned int>(queue, layerClustersClustering.numberOfTrackstersScalar());
    alpaka::memset(queue, nTracksters, 0x0);
    auto nVertices = make_device_view<unsigned int>(queue, layerClustersClustering.numberOfVerticesScalar());
    alpaka::memset(queue, nVertices, 0x0);
    const uint32_t nLayerClusters = layerClusters.position().metadata().size();
    if (nLayerClusters == 0) {
      return;
    }

    // The coordinates are in units of the smallest sigmaT, and the sigmaT of the HGCAL components relative to it
    const float sigmaRef = std::ranges::min(parameters.sigmaT);
    std::array<float, kComponents> relativeSigmaT;
    for (uint32_t c = 0; c < kComponents; ++c) {
      relativeSigmaT[c] = parameters.sigmaT[c] / sigmaRef;
    }

    // Two points interact only within maxDistance: the gap between the endcaps is larger
    const float maxDistance = std::max(parameters.deltac, parameters.outlierDistance);
    const float endcapGap = 2.f * maxDistance;
    const int32_t layerReach = static_cast<int32_t>(std::ceil(maxDistance * parameters.layerScale));

    // Point pointEnd[i] - 1 is layer cluster i, if isPoint[i]. The number of points is not known on the host yet, so
    // the points are sized for all the layer clusters.
    auto isPoint = make_device_buffer<uint32_t[]>(queue, nLayerClusters);
    auto pointEnd = make_device_buffer<uint32_t[]>(queue, nLayerClusters);
    auto hostNPoints = make_host_buffer<uint32_t[]>(queue, 1);
    PointsDeviceCollection points(queue, nLayerClusters);
    auto extents = make_device_buffer<ComponentExtents>(queue);

    constexpr uint32_t blockSize = 256;
    const auto workDivLayerClusters = make_workdiv<Acc1D>(divide_up_by(nLayerClusters, blockSize), blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDivLayerClusters,
                        KernelSelectPoints{},
                        parameters.minSiliconCells,
                        layerClusters,
                        layerClustersClustering,
                        isPoint.data(),
                        extents.data());
    // inclusive scan of the point flags
    iterativePrefixScan<Acc1D>(isPoint.data(), pointEnd.data(), nLayerClusters, queue);
    alpaka::memcpy(queue, hostNPoints, make_device_view(queue, pointEnd.data() + nLayerClusters - 1, 1));
    alpaka::exec<Acc1D>(queue,
                        workDivLayerClusters,
                        KernelFillPoints{},
                        1.f / sigmaRef,
                        1.f / parameters.layerScale,
                        endcapGap,
                        parameters.rhocEtaExponent,
                        parameters.rhocPivotRadius,
                        relativeSigmaT,
                        layerClusters,
                        isPoint.data(),
                        pointEnd.data(),
                        points.view(),
                        extents.data());
    alpaka::exec<Acc1D>(queue,
                        workDivLayerClusters,
                        KernelSearchWindow{},
                        maxDistance,
                        layerReach,
                        relativeSigmaT,
                        layerClusters,
                        isPoint.data(),
                        pointEnd.data(),
                        extents.data(),
                        points.view());

    // CLUEstering needs the number of points on the host
    alpaka::wait(queue);
    const uint32_t nPoints = hostNPoints[0];
    if (nPoints == 0) {
      return;
    }

    auto pointsView = points.view();
    clue::PointsDevice<3, float> clusteringPoints(queue,
                                                  static_cast<int32_t>(nPoints),
                                                  pointsView.dim1().data(),
                                                  pointsView.dim2().data(),
                                                  pointsView.dim3().data(),
                                                  pointsView.weight().data(),
                                                  pointsView.clusterIndex().data());
    clusteringPoints.set_tags(std::span<const uint32_t>(pointsView.tag().data(), nPoints));
    clusteringPoints.set_density_uncertainty(std::span<const float>(pointsView.rhocScale().data(), nPoints));
    // the per point sigma of the transverse coordinates sets the search window
    clusteringPoints.set_sigma(0, std::span<const float>(pointsView.searchSigma().data(), nPoints));
    clusteringPoints.set_sigma(1, std::span<const float>(pointsView.searchSigma().data(), nPoints));

    clue::Clusterer<3> algo(
        queue, parameters.deltac, parameters.rhoc, parameters.outlierDistance, parameters.seedingDistance);
    algo.make_clusters(queue, clusteringPoints, CylinderMetric{pointsView.sigmaT().data()});

    // There are at most as many clusters as points: the sizes of the unused slots stay 0, below any minLayerClusters
    auto clusterSizes = make_device_buffer<uint32_t[]>(queue, nPoints);
    alpaka::memset(queue, clusterSizes, 0x0);
    auto isTrackster = make_device_buffer<uint32_t[]>(queue, nPoints);
    auto tracksterEnd = make_device_buffer<uint32_t[]>(queue, nPoints);

    const auto workDivPoints = make_workdiv<Acc1D>(divide_up_by(nPoints, blockSize), blockSize);
    alpaka::exec<Acc1D>(
        queue, workDivPoints, KernelCountClusterSizes{}, nPoints, points.const_view(), clusterSizes.data());
    alpaka::exec<Acc1D>(queue,
                        workDivPoints,
                        KernelSelectTracksters{},
                        nPoints,
                        std::max(parameters.minLayerClusters, 1u),
                        clusterSizes.data(),
                        isTrackster.data(),
                        layerClustersClustering);
    // inclusive scan of the trackster flags
    iterativePrefixScan<Acc1D>(isTrackster.data(), tracksterEnd.data(), nPoints, queue);
    alpaka::exec<Acc1D>(queue,
                        workDivLayerClusters,
                        KernelWriteLayerClustersClustering{},
                        nPoints,
                        isPoint.data(),
                        pointEnd.data(),
                        points.const_view(),
                        isTrackster.data(),
                        tracksterEnd.data(),
                        layerClustersClustering);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
