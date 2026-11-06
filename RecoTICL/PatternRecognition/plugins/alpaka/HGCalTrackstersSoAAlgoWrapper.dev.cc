// Eigen must precede the SoA headers to enable the Eigen columns of the TracksterSoA
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "HGCalTrackstersSoAAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  namespace {

    constexpr int kInvalidTrackster = -1;

    // The PCA uses the most energetic layer cluster of each layer within kCleaningLayers layers of the most energetic
    // layer cluster of the trackster
    constexpr int32_t kCleaningLayers = 10;
    constexpr int32_t kCleaningWindow = 2 * kCleaningLayers + 1;
    // A layer cluster contributes to the time of its trackster if it is within this squared distance [cm^2] of the
    // line through the origin and the barycenter
    constexpr float kTimeMaxDistance2 = 9.f;
    constexpr float kSpeedOfLight = 29.9792458f;  // cm/ns
    // The time of a trackster without layer clusters with a time, and its error [ns]
    constexpr float kNoTime = -99.f;
    constexpr float kNoTimeError = -1.f;

    // The most energetic layer cluster of a set is found with one atomicMax on this key. The bits of a non-negative
    // float, read as an integer, are ordered as its value is. For equal energies, ~layerCluster makes the lowest
    // layer cluster index win. This assumes energy is always positive
    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint64_t energyKey(const float energy, const uint32_t layerCluster) {
      return (static_cast<uint64_t>(std::bit_cast<uint32_t>(energy)) << 32) | ~layerCluster;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t layerClusterOfEnergyKey(const uint64_t key) {
      return ~static_cast<uint32_t>(key);
    }

    // The squared distance of a point from the line through the origin along direction. As in the legacy code, the
    // squares are summed in double precision (std::pow).
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float distance2FromLine(const float point[3], const float direction[3]) {
      float dot = 0.f;
      float direction2 = 0.f;
      for (int k = 0; k < 3; ++k) {
        dot += point[k] * direction[k];
        direction2 += direction[k] * direction[k];
      }
      const float projection = direction2 != 0.f ? dot / direction2 : 0.f;
      float distance2 = 0.f;
      for (int k = 0; k < 3; ++k) {
        const float d = point[k] - projection * direction[k];
        distance2 = static_cast<float>(distance2 + static_cast<double>(d) * d);
      }
      return distance2;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float dot3(const float a[3], const float b[3]) {
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    // The sums over the layer clusters of the trackster processed by a block, in shared memory
    struct TracksterSums {
      // all the layer clusters
      float energy;
      float emEnergy;
      float position[3];  // weighted by energy
      uint64_t maxEnergyKey;
      uint32_t minAbsZ;  // the bits of the non-negative floats, ordered as their values
      uint32_t maxAbsZ;
      float time;  // weighted by 1 / timeError^2
      float timeWeight;
      // the layer clusters used by the PCA: the most energetic one of each layer of the cleaning window
      uint64_t layerMaxEnergyKey[kCleaningWindow];
      uint32_t cleanedCount;
      float cleanedEnergy;
      float cleanedPosition[3];  // weighted by energy
      // weighted by energy / cleanedEnergy
      float weight2;
      float covariance[6];  // xx, yx, yy, zx, zy, zz
      float sigmasPCA2[3];  // along the principal axes, in increasing eigenvalue order
      // the eigen-decomposition of the covariance matrix, in increasing eigenvalue order, the eigenvectors column-major
      float eigenvalues[3];
      float eigenvectors[9];
      // the energy within 2 and 5 cm of the profile axis, and weighted by the squared distance from it
      float core2Energy;
      float core5Energy;
      float radius2Energy;
    };

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

    // Per trackster, one block: its energy, barycenter, time, principal components and energy profile from its layer
    // clusters, as ticl::assignPCAtoTracksters and ticl::layerClusterProfile do. The threads of the block share the
    // layer clusters of the trackster and sum over them in shared memory; after each sum, every thread computes what
    // depends on it.
    struct KernelTracksterProperties {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    float zLimitEm,
                                    reco::CaloClusterDeviceCollection::ConstView layerClusters,
                                    ticl::TracksterDeviceCollection::View tracksters) const {
        constexpr auto block = alpaka::hierarchy::Threads{};
        auto& s = alpaka::declareSharedVar<TracksterSums, __COUNTER__>(acc);
        auto vertices = tracksters.vertices();

        for (auto t : independent_groups(acc, tracksters.energy().metadata().size())) {
          const uint32_t first = vertices.offsets()[t].keys_offsets();
          const uint32_t last = vertices.offsets()[t + 1].keys_offsets();

          if (once_per_block(acc)) {
            s = TracksterSums{};
            s.minAbsZ = std::numeric_limits<uint32_t>::max();
          }
          alpaka::syncBlockThreads(acc);

          // the energy, the electromagnetic energy, the weighted position and the most energetic layer cluster
          for (auto vertex : independent_group_elements(acc, first, last)) {
            const uint32_t i = vertices.content()[vertex].values();
            const auto position = layerClusters.position()[i];
            const float energy = layerClusters.energy()[i].energy();
            alpaka::atomicAdd(acc, &s.energy, energy, block);
            if (std::abs(position.z()) <= zLimitEm) {
              alpaka::atomicAdd(acc, &s.emEnergy, energy, block);
            }
            alpaka::atomicAdd(acc, &s.position[0], energy * position.x(), block);
            alpaka::atomicAdd(acc, &s.position[1], energy * position.y(), block);
            alpaka::atomicAdd(acc, &s.position[2], energy * position.z(), block);
            alpaka::atomicMax(acc, &s.maxEnergyKey, energyKey(energy, i), block);
            const auto absZ = std::bit_cast<uint32_t>(std::abs(position.z()));
            alpaka::atomicMin(acc, &s.minAbsZ, absZ, block);
            alpaka::atomicMax(acc, &s.maxAbsZ, absZ, block);
          }
          alpaka::syncBlockThreads(acc);

          const float invEnergy = 1.f / s.energy;
          const float barycenter[3] = {s.position[0] * invEnergy, s.position[1] * invEnergy, s.position[2] * invEnergy};
          const int32_t maxEnergyLayer = layerClusters.position()[layerClusterOfEnergyKey(s.maxEnergyKey)].layer();
          // the slot of a layer in the cleaning window, or -1 outside of it
          auto cleaningSlot = [&](int32_t layer) {
            const int32_t slot = layer - maxEnergyLayer + kCleaningLayers;
            return (slot >= 0 and slot < kCleaningWindow) ? slot : -1;
          };

          // the most energetic layer cluster of each layer of the cleaning window, and the time: the weighted mean of
          // the times of the layer clusters near the axis, corrected for the path to the barycenter
          for (auto vertex : independent_group_elements(acc, first, last)) {
            const uint32_t i = vertices.content()[vertex].values();
            const auto position = layerClusters.position()[i];
            const int32_t slot = cleaningSlot(position.layer());
            if (slot >= 0) {
              alpaka::atomicMax(
                  acc, &s.layerMaxEnergyKey[slot], energyKey(layerClusters.energy()[i].energy(), i), block);
            }

            const auto timing = layerClusters.timing()[i];
            const float point[3] = {position.x(), position.y(), position.z()};
            if (timing.timeError() > 0.f and distance2FromLine(point, barycenter) < kTimeMaxDistance2) {
              const float scale = barycenter[2] * (1.f / position.z()) - 1.f;
              const float deltaZ = barycenter[2] - position.z();
              const float deltaT = 1.f / kSpeedOfLight *
                                   std::sqrt((scale * position.x()) * (scale * position.x()) +
                                             (scale * position.y()) * (scale * position.y()) + deltaZ * deltaZ);
              const float time =
                  std::abs(barycenter[2]) < std::abs(position.z()) ? timing.time() - deltaT : timing.time() + deltaT;
              // in double precision, as in the legacy code (std::pow)
              const float weight =
                  static_cast<float>(1. / (static_cast<double>(timing.timeError()) * timing.timeError()));
              alpaka::atomicAdd(acc, &s.time, time * weight, block);
              alpaka::atomicAdd(acc, &s.timeWeight, weight, block);
            }
          }
          alpaka::syncBlockThreads(acc);

          // the PCA uses the most energetic layer cluster of each layer of the cleaning window
          auto isCleaned = [&](uint32_t i) {
            const int32_t slot = cleaningSlot(layerClusters.position()[i].layer());
            return slot >= 0 and layerClusterOfEnergyKey(s.layerMaxEnergyKey[slot]) == i;
          };

          // The PCA needs at least 3 layer clusters, and at least 2 in the cleaned set with a spread to normalise:
          // 1 - weight2 is 0 when one layer cluster carries all the cleaned energy. Both conditions are the same for
          // all the threads of the block.
          const bool hasPCA = last - first > 2;
          bool validPCA = false;
          if (hasPCA) {
            for (auto vertex : independent_group_elements(acc, first, last)) {
              const uint32_t i = vertices.content()[vertex].values();
              if (isCleaned(i)) {
                const auto position = layerClusters.position()[i];
                const float energy = layerClusters.energy()[i].energy();
                alpaka::atomicAdd(acc, &s.cleanedCount, 1u, block);
                alpaka::atomicAdd(acc, &s.cleanedEnergy, energy, block);
                alpaka::atomicAdd(acc, &s.cleanedPosition[0], energy * position.x(), block);
                alpaka::atomicAdd(acc, &s.cleanedPosition[1], energy * position.y(), block);
                alpaka::atomicAdd(acc, &s.cleanedPosition[2], energy * position.z(), block);
              }
            }
            alpaka::syncBlockThreads(acc);

            // inverted in double precision, as in the legacy code
            const float invCleanedEnergy = static_cast<float>(1. / s.cleanedEnergy);
            const Eigen::Vector3f cleanedBarycenter =
                Eigen::Map<const Eigen::Vector3f>(s.cleanedPosition) * invCleanedEnergy;
            auto deviation = [&](uint32_t i) -> Eigen::Vector3f {
              const auto position = layerClusters.position()[i];
              return Eigen::Vector3f(position.x(), position.y(), position.z()) - cleanedBarycenter;
            };

            // the covariance matrix
            for (auto vertex : independent_group_elements(acc, first, last)) {
              const uint32_t i = vertices.content()[vertex].values();
              if (isCleaned(i)) {
                const float weight = layerClusters.energy()[i].energy() * invCleanedEnergy;
                const Eigen::Vector3f d = deviation(i);
                alpaka::atomicAdd(acc, &s.weight2, weight * weight, block);
                for (int a = 0, k = 0; a < 3; ++a) {
                  for (int b = 0; b <= a; ++b, ++k) {
                    alpaka::atomicAdd(acc, &s.covariance[k], weight * d(a) * d(b), block);
                  }
                }
              }
            }
            alpaka::syncBlockThreads(acc);

            validPCA = s.cleanedCount >= 2 and s.weight2 < 1.f;
            if (validPCA) {
              if (once_per_block(acc)) {
                const float* c = s.covariance;
                Eigen::Matrix3f covariance;
                covariance << c[0], c[1], c[3], c[1], c[2], c[4], c[3], c[4], c[5];
                covariance *= 1.f / (1.f - s.weight2);
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
                if (solver.info() == Eigen::Success) {
                  Eigen::Map<Eigen::Vector3f>(s.eigenvalues) = solver.eigenvalues();
                  Eigen::Map<Eigen::Matrix3f>(s.eigenvectors) = solver.eigenvectors();
                }
              }
              alpaka::syncBlockThreads(acc);

              // The spreads along the principal axes, with eigenvectors^T: the small eigenvalues carry the rounding
              // error of the largest one, and this sum does not.
              const Eigen::Map<const Eigen::Matrix3f> eigenvectors(s.eigenvectors);
              for (auto vertex : independent_group_elements(acc, first, last)) {
                const uint32_t i = vertices.content()[vertex].values();
                if (isCleaned(i)) {
                  const float weight = layerClusters.energy()[i].energy() * invCleanedEnergy;
                  const Eigen::Vector3f projected = eigenvectors.transpose() * deviation(i);
                  for (int k = 0; k < 3; ++k) {
                    alpaka::atomicAdd(acc, &s.sigmasPCA2[k], weight * projected(k) * projected(k), block);
                  }
                }
              }
              alpaka::syncBlockThreads(acc);
            }
          }

          // The energy profile around the first principal axis, or the barycenter direction without one, as
          // ticl::layerClusterProfile. Without a PCA the eigenvectors stay 0.
          float axis[3] = {s.eigenvectors[6], s.eigenvectors[7], s.eigenvectors[8]};
          float axis2 = dot3(axis, axis);
          if (not(axis2 > 0.f) or not std::isfinite(axis2)) {
            for (int k = 0; k < 3; ++k) {
              axis[k] = barycenter[k];
            }
            axis2 = dot3(axis, axis);
          }
          if (axis2 > 0.f) {
            const float norm = std::sqrt(axis2);
            for (int k = 0; k < 3; ++k) {
              axis[k] /= norm;
            }
          }
          for (auto vertex : independent_group_elements(acc, first, last)) {
            const uint32_t i = vertices.content()[vertex].values();
            const auto position = layerClusters.position()[i];
            const float energy = layerClusters.energy()[i].energy();
            const float d[3] = {
                position.x() - barycenter[0], position.y() - barycenter[1], position.z() - barycenter[2]};
            const float along = dot3(d, axis);
            const float r = std::sqrt(std::max(0.f, dot3(d, d) - along * along));
            if (r < ::ticl::kProfileCore2Radius) {
              alpaka::atomicAdd(acc, &s.core2Energy, energy, block);
            }
            if (r < ::ticl::kProfileCore5Radius) {
              alpaka::atomicAdd(acc, &s.core5Energy, energy, block);
            }
            alpaka::atomicAdd(acc, &s.radius2Energy, energy * r * r, block);
          }
          alpaka::syncBlockThreads(acc);

          if (once_per_block(acc)) {
            auto energy = tracksters.energy()[t];
            energy.rawEnergy() = s.energy;
            energy.rawEmEnergy() = s.emEnergy;
            energy.regressedEnergy() = 0.f;

            auto position = tracksters.barycenter()[t];
            position.x() = barycenter[0];
            position.y() = barycenter[1];
            position.z() = barycenter[2];

            auto timing = tracksters.timing()[t];
            timing.time() = s.timeWeight > 0.f ? s.time / s.timeWeight : kNoTime;
            timing.timeError() = s.timeWeight > 0.f ? 1.f / std::sqrt(s.timeWeight) : kNoTimeError;

            auto pca = tracksters.pca()[t];
            pca.eigenvalues().setZero();
            pca.eigenvectors().setZero();
            pca.sigmas().setZero();
            pca.sigmasPCA().setZero();
            if (validPCA) {
              const Eigen::Map<const Eigen::Matrix3f> eigenvectors(s.eigenvectors);
              const float normalisation = 1.f - s.weight2;
              // The spreads along x, y, z are the diagonal of the covariance matrix, each layer cluster with its own
              // weight.
              const float diagonal[3] = {s.covariance[0], s.covariance[2], s.covariance[5]};
              // in decreasing eigenvalue order
              for (int k = 0; k < 3; ++k) {
                pca.eigenvalues()(k) = s.eigenvalues[2 - k];
                pca.eigenvectors().col(k) = eigenvectors.col(2 - k);
                pca.sigmas()(k) = std::sqrt(diagonal[k] / normalisation);
                pca.sigmasPCA()(k) = std::sqrt(s.sigmasPCA2[2 - k] / normalisation);
              }
              // the principal axis points away from the interaction point
              if (pca.eigenvectors()(2, 0) * barycenter[2] < 0.f) {
                pca.eigenvectors().col(0) *= -1.f;
              }
            }

            tracksters.id()[t].idProbabilities().setZero();

            auto profile = tracksters.profile()[t];
            if (s.energy > 0.f) {
              // a non-finite value is 0
              const auto finite = [](float value) { return std::isfinite(value) ? value : 0.f; };
              const float maxEnergy = std::bit_cast<float>(static_cast<uint32_t>(s.maxEnergyKey >> 32));
              const float minAbsZ = std::bit_cast<float>(s.minAbsZ);
              profile.maxEnergyFraction() = finite(maxEnergy / s.energy);
              profile.core2EnergyFraction() = finite(s.core2Energy / s.energy);
              profile.core5EnergyFraction() = finite(s.core5Energy / s.energy);
              profile.rmsRadius() = finite(std::sqrt(s.radius2Energy / s.energy));
              profile.minAbsZ() = finite(minAbsZ);
              profile.lengthZ() = finite(std::bit_cast<float>(s.maxAbsZ) - minAbsZ);
            } else {
              profile.maxEnergyFraction() = -1.f;
              profile.core2EnergyFraction() = -1.f;
              profile.core5EnergyFraction() = -1.f;
              profile.rmsRadius() = -1.f;
              profile.minAbsZ() = -1.f;
              profile.lengthZ() = 0.f;
            }
          }
          // the shared sums are reset for the next trackster of the block
          alpaka::syncBlockThreads(acc);
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

    // One warp per trackster: most tracksters have a few layer clusters, the largest a few hundred
    const auto threadsPerTrackster = static_cast<uint32_t>(alpaka::getPreferredWarpSize(alpaka::getDev(queue)));
    const auto workDivTracksters = make_workdiv<Acc1D>(nTracksters, threadsPerTrackster);
    alpaka::exec<Acc1D>(queue, workDivTracksters, KernelTracksterProperties{}, zLimitEm, layerClusters, tracksters);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
