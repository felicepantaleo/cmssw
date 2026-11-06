// Eigen must precede the SoA headers to enable the Eigen columns of the TracksterSoA
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTICL/Common/interface/alpaka/TracksterPropertiesKernel.h"

#include "CornettoAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl::cornetto {

  using namespace cms::alpakatools;

  namespace {

    constexpr uint32_t kBlockSize = 256;
    constexpr float kInvSpeedOfLight = 1.f / 29.9792458f;  // ns/cm
    constexpr int kNumParticleTypes = static_cast<int>(::ticl::Trackster::ParticleType::unknown) + 1;
    constexpr int kUnknown = static_cast<int>(::ticl::Trackster::ParticleType::unknown);
    constexpr uint32_t kNoBin = std::numeric_limits<uint32_t>::max();

    // One ticl::TICLLayerTile per endcap, index 1 for z > 0
    using Tile = ::ticl::TileConstants;
    constexpr uint32_t kBins = 2 * Tile::nBins;

    // The quantities of the pair test, one entry per trackster
    struct Features {
      float* x;
      float* y;
      float* z;
      float* energy;
      float* eta;
      float* phi;
      float* rad;  // |barycenter|
      float* em;   // EM probability, negative without a PID
      float* time;
      float* timeError;
      uint32_t* valid;  // 1 for a finite barycenter and energy
    };

    // The operations of the host code of DataFormats/Math and ROOT::Math::Cartesian3D<float>, in the same order. The
    // compilers of the host and of a GPU can fuse a multiply and an add, and the math functions of a GPU (atan2, log,
    // asin) can differ from the host ones in the last bit, so a pair at the limit of a cut can differ on a GPU.

    // reco::reducePhiRange<float>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float reducePhiRange(float x) {
      constexpr float o2pi = 1. / (2. * M_PI);
      if (std::abs(x) <= static_cast<float>(M_PI))
        return x;
      const float n = std::round(x * o2pi);
      return x - n * static_cast<float>(2. * M_PI);
    }

    // ROOT::Math::Cartesian3D<float>::Eta()
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float etaOf(float x, float y, float z) {
      constexpr float kEtaMax = 22756.0f;  // ROOT::Math::etaMax<float>()
      // the limit of the Taylor expansion of ROOT, pow(epsilon, -0.25) in float
      constexpr float kBigZScaled = 53.8174f;
      const float rho = std::sqrt(x * x + y * y);
      if (rho > 0.f) {
        const float zScaled = z / rho;
        if (std::abs(zScaled) < kBigZScaled) {
          return static_cast<float>(std::log(zScaled + std::sqrt(zScaled * zScaled + 1.0)));
        }
        return z > 0.f ? static_cast<float>(std::log(2.0 * zScaled + 0.5 / zScaled))
                       : static_cast<float>(-std::log(-2.0 * zScaled));
      }
      if (z == 0.f)
        return 0.f;
      return z > 0.f ? z + kEtaMax : z - kEtaMax;
    }

    // TICLLayerTileT<TileConstants>::etaBin and phiBin
    ALPAKA_FN_ACC ALPAKA_FN_INLINE int etaBin(float eta) {
      constexpr float etaRange = Tile::maxEta - Tile::minEta;
      const float r = Tile::nEtaBins / etaRange;
      int bin = (std::abs(eta) - Tile::minEta) * r;
      return std::clamp(bin, 0, Tile::nEtaBins - 1);
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE int phiBin(float phi) {
      const float normPhi = reducePhiRange(phi);
      const float r = Tile::nPhiBins * M_1_PI * 0.5f;
      const int bin = (normPhi + M_PI) * r;
      // phi = pi gives nPhiBins: the device takes the last phi bin, the host tile bin 0 of the next eta bin
      return std::min(bin, Tile::nPhiBins - 1);
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t binOf(float eta, float phi, float z) {
      return (z > 0.f ? Tile::nBins : 0) + phiBin(phi) + etaBin(eta) * Tile::nPhiBins;
    }

    // ticl::cornetto::ranksAbove: a ranks above b when it has more energy, or the same energy and a smaller index
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool ranksAbove(const Features& f, uint32_t a, uint32_t b) {
      return f.energy[a] > f.energy[b] || (f.energy[a] == f.energy[b] && a < b);
    }

    // The EM probability of ticl::cornetto::emProbability: the probabilities of the categories over the sum of all,
    // -1 without a PID (the sum is not above the unknown probability). The categories are summed in the order of the
    // parameters.
    template <typename TProbabilities>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float emProbability(const TProbabilities& prob, const Parameters& p) {
      float sum = 0.f;
      for (int k = 0; k < kNumParticleTypes; ++k)
        sum += prob(k);
      if (!(sum > prob(kUnknown)))
        return -1.f;
      float em = 0.f;
      for (uint32_t k = 0; k < p.nEmCategories; ++k)
        em += prob(p.emCategories[k]);
      return em / sum;
    }

    // ticl::cornetto::separationInCone: the separation s of other along the anchor axis when other is inside the
    // cone of anchor, else false
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool separationInCone(
        const Cone& cone, const Features& f, uint32_t a, uint32_t b, float& s) {
      const float anchorRad = f.rad[a];
      if (!(anchorRad > 0.f))
        return false;
      // as ROOT::Math::DisplacementVector3D::operator/: a scale by the inverse
      const float invRad = 1.f / anchorRad;
      const float axis[3] = {f.x[a] * invRad, f.y[a] * invRad, f.z[a] * invRad};
      const float d[3] = {f.x[b] - f.x[a], f.y[b] - f.y[a], f.z[b] - f.z[a]};
      s = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
      if (!(s >= -cone.maxBackward && s <= cone.maxLongitudinal))
        return false;
      const float rT = cone.radius0 + cone.slope * std::abs(s);
      const float mag2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      return std::max(0.f, mag2 - s * s) <= rT * rT;
    }

    struct Window {
      float dEta;
      float dPhi;
    };

    // ticl::cornetto::searchWindow
    ALPAKA_FN_ACC ALPAKA_FN_INLINE Window searchWindow(const Cone& cone, float otherAbsEta, float otherRad, float cap) {
      constexpr float kRadiusMargin = 0.1f;
      constexpr float kRelativeMargin = 1.01f;
      constexpr float kAbsoluteMargin = 1.e-5f;
      constexpr float kMinPolarAngle = 1.e-4f;
      const float maxRadius = cone.radius0 + cone.slope * std::max(cone.maxBackward, cone.maxLongitudinal);
      const float alpha = std::asin(std::min(1.f, (maxRadius + kRadiusMargin) / otherRad));
      const float theta = 2.f * std::atan(std::exp(-otherAbsEta));
      const float sinLow = std::sin(std::max(theta - alpha, kMinPolarAngle));
      const float etaBound = kRelativeMargin * alpha / sinLow;
      const float phiBound = kRelativeMargin * 2.f *
                             std::asin(std::min(1.f, std::sin(0.5f * alpha) / std::sqrt(std::sin(theta) * sinLow)));
      return {std::min(cap, etaBound) + kAbsoluteMargin, std::min(cap, phiBound) + kAbsoluteMargin};
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool timeCompatible(
        const Parameters& p, float tAnchor, float errAnchor, float tOther, float errOther, float path) {
      if (!p.useTimeGate || !(errAnchor > 0.f) || !(errOther > 0.f))
        return true;
      const float dt = tOther - tAnchor - path * kInvSpeedOfLight;
      return dt * dt <= p.timeNSigma2 * (errAnchor * errAnchor + errOther * errOther + p.timeFloor2);
    }

    // The pair test of ticl::TracksterLinkingbyCornetto::compatibleIn with anchor a and other b
    ALPAKA_FN_ACC bool compatible(const Parameters& p, const Features& f, uint32_t a, uint32_t b) {
      if (!(f.z[a] * f.z[b] > 0.f))
        return false;  // same endcap only
      if (!(std::abs(f.eta[b] - f.eta[a]) <= p.etaWindow) ||
          !(std::abs(reducePhiRange(f.phi[b] - f.phi[a])) <= p.etaWindow))
        return false;
      const bool emAnchor = p.typeVetoProbability > 0.f && f.em[a] > p.typeVetoProbability;
      float s = 0.f;
      if (!separationInCone(emAnchor ? p.emCone : p.cone, f, a, b, s))
        return false;
      if (!timeCompatible(p, f.time[a], f.timeError[a], f.time[b], f.timeError[b], s))
        return false;
      if (p.typeVetoProbability > 0.f && f.em[a] >= 0.f && f.em[b] >= 0.f) {
        const float hi = p.typeVetoProbability;
        const float lo = 1.f - p.typeVetoProbability;
        if ((f.em[a] > hi && f.em[b] < lo) || (f.em[a] < lo && f.em[b] > hi))
          return false;
      }
      return true;
    }

    // ticl::TracksterLinkingbyCornetto::emits
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool emits(const Parameters& p, float energy, float pt, float em) {
      const bool isEM = p.typeVetoProbability > 0.f && em > p.typeVetoProbability;
      if (!(energy >= (isEM ? p.minEmittedEnergyEM : p.minEmittedEnergy)))
        return false;
      return pt >= (isEM ? p.minEmittedPtEM : p.minEmittedPt) ||
             (p.minEmittedEnergyAnyPt > 0.f && energy >= p.minEmittedEnergyAnyPt);
    }

    // Per trackster: its features, and the tile bin of a valid one (counted in binCounts)
    struct KernelFeatures {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const Parameters p,
                                    const Inputs inputs,
                                    Features f,
                                    uint32_t* bin,
                                    uint32_t* binCounts) const {
        for (auto i : uniform_elements(acc, inputs.size())) {
          const uint32_t c = inputs.collectionOf(i);
          const uint32_t local = i - inputs.offsets[c];
          const auto& tracksters = inputs.tracksters[c];
          const auto barycenter = tracksters.barycenter()[local];
          const float x = barycenter.x();
          const float y = barycenter.y();
          const float z = barycenter.z();
          const float energy = tracksters.energy()[local].rawEnergy();
          f.x[i] = x;
          f.y[i] = y;
          f.z[i] = z;
          f.energy[i] = energy;
          f.eta[i] = etaOf(x, y, z);
          f.phi[i] = std::atan2(y, x);
          f.rad[i] = std::sqrt(x * x + y * y + z * z);
          f.time[i] = tracksters.timing()[local].time();
          f.timeError[i] = tracksters.timing()[local].timeError();
          f.em[i] = inputs.hasPid[c] ? emProbability(inputs.pid[c][local].idProbabilities(), p)
                                     : emProbability(tracksters.id()[local].idProbabilities(), p);
          const bool valid = std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && std::isfinite(energy);
          f.valid[i] = valid ? 1u : 0u;
          if (valid) {
            const uint32_t b = binOf(f.eta[i], f.phi[i], z);
            bin[i] = b;
            alpaka::atomicAdd(acc, &binCounts[b], 1u);
          } else {
            bin[i] = kNoBin;
          }
        }
      }
    };

    // Per trackster: its slot in the content of its bin, in no particular order
    struct KernelFillBins {
      ALPAKA_FN_ACC void operator()(
          const Acc1D& acc, uint32_t n, const uint32_t* bin, uint32_t* binCursor, uint32_t* binContent) const {
        for (auto i : uniform_elements(acc, n)) {
          if (bin[i] != kNoBin) {
            const uint32_t slot = alpaka::atomicAdd(acc, &binCursor[bin[i]], 1u);
            binContent[slot] = i;
          }
        }
      }
    };

    // Per trackster: its parent, the highest-ranked compatible trackster above it in the bins of the search window
    // (ticl::TracksterLinkingbyCornetto::findParents). A trackster without a parent is its own parent.
    struct KernelParents {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const Parameters p,
                                    uint32_t n,
                                    const Features f,
                                    const uint32_t* binOffsets,
                                    const uint32_t* binContent,
                                    uint32_t* parents) const {
        for (auto i : uniform_elements(acc, n)) {
          uint32_t best = i;
          if (f.valid[i] != 0u && f.rad[i] > 0.f) {
            const float absEta = std::abs(f.eta[i]);
            const Window window = searchWindow(p.cone, absEta, f.rad[i], p.etaWindow);
            // local copies: a static member of the host tile constants has no definition in device code
            constexpr float kMinEta = Tile::minEta;
            constexpr float kMaxEta = Tile::maxEta;
            const float etaMin = std::min(std::max(absEta - window.dEta, kMinEta), kMaxEta);
            const float etaMax = std::min(std::max(absEta + window.dEta, kMinEta), kMaxEta);
            // TICLLayerTileT::searchBoxEtaPhi for one endcap
            const int etaBinMin = etaBin(etaMin);
            const int etaBinMax = etaBin(etaMax);
            const int phiBinMin = phiBin(f.phi[i] - window.dPhi);
            int phiBinMax = phiBin(f.phi[i] + window.dPhi);
            if (phiBinMax < phiBinMin) {
              phiBinMax += Tile::nPhiBins;
            }
            const uint32_t endcap = f.z[i] > 0.f ? Tile::nBins : 0;
            for (int e = etaBinMin; e <= etaBinMax; ++e) {
              for (int ph = phiBinMin; ph <= phiBinMax; ++ph) {
                const uint32_t b = endcap + (ph % Tile::nPhiBins) + e * Tile::nPhiBins;
                for (uint32_t slot = binOffsets[b]; slot < binOffsets[b + 1]; ++slot) {
                  const uint32_t j = binContent[slot];
                  if (ranksAbove(f, j, i) && (best == i || ranksAbove(f, j, best)) && compatible(p, f, j, i)) {
                    best = j;
                  }
                }
              }
            }
          }
          parents[i] = best;
        }
      }
    };

    // Per trackster: its root, by following its parents, and its group label: the root when the pair (root,
    // trackster) is compatible, else itself (ticl::cornetto::roots and groups). The label is counted and the
    // smallest member of each label is kept.
    struct KernelLabels {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const Parameters p,
                                    uint32_t n,
                                    const Features f,
                                    const uint32_t* parents,
                                    uint32_t* label,
                                    uint32_t* labelCount,
                                    uint32_t* labelMinMember) const {
        for (auto i : uniform_elements(acc, n)) {
          uint32_t root = i;
          while (parents[root] != root) {
            root = parents[root];
          }
          const uint32_t l = (root == i || compatible(p, f, root, i)) ? root : i;
          label[i] = l;
          alpaka::atomicAdd(acc, &labelCount[l], 1u);
          alpaka::atomicMin(acc, &labelMinMember[l], i);
        }
      }
    };

    // Per trackster: 1 when it is the smallest member of its group
    struct KernelGroupHeads {
      ALPAKA_FN_ACC void operator()(
          const Acc1D& acc, uint32_t n, const uint32_t* label, const uint32_t* labelMinMember, uint32_t* head) const {
        for (auto i : uniform_elements(acc, n)) {
          head[i] = labelMinMember[label[i]] == i ? 1u : 0u;
        }
      }
    };

    // Per group head: the group index of its label, in the order of the heads, and the size of the group
    struct KernelGroupsOfLabels {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    uint32_t n,
                                    const uint32_t* label,
                                    const uint32_t* head,
                                    const uint32_t* headEnd,
                                    const uint32_t* labelCount,
                                    uint32_t* groupOfLabel,
                                    uint32_t* groupSize) const {
        for (auto i : uniform_elements(acc, n)) {
          if (head[i] != 0u) {
            const uint32_t g = headEnd[i] - 1;
            groupOfLabel[label[i]] = g;
            groupSize[g] = labelCount[label[i]];
          }
        }
      }
    };

    // Per trackster: a slot in its group, in no particular order
    struct KernelFillGroups {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    uint32_t n,
                                    const uint32_t* label,
                                    const uint32_t* groupOfLabel,
                                    const uint32_t* groupSize,
                                    const uint32_t* groupEnd,
                                    uint32_t* groupCursor,
                                    uint32_t* unsorted) const {
        for (auto i : uniform_elements(acc, n)) {
          const uint32_t g = groupOfLabel[label[i]];
          const uint32_t slot = groupEnd[g] - groupSize[g] + alpaka::atomicAdd(acc, &groupCursor[g], 1u);
          unsorted[slot] = i;
        }
      }
    };

    // Per trackster: its slot in its group in ascending order, which is the number of members of the group below it
    struct KernelSortGroups {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    uint32_t n,
                                    const uint32_t* label,
                                    const uint32_t* groupOfLabel,
                                    const uint32_t* groupSize,
                                    const uint32_t* groupEnd,
                                    const uint32_t* unsorted,
                                    uint32_t* members) const {
        for (auto i : uniform_elements(acc, n)) {
          const uint32_t g = groupOfLabel[label[i]];
          const uint32_t first = groupEnd[g] - groupSize[g];
          uint32_t rank = 0;
          for (uint32_t slot = first; slot < groupEnd[g]; ++slot) {
            if (unsorted[slot] < i) {
              ++rank;
            }
          }
          members[first + rank] = i;
        }
      }
    };

    // Per group, one thread: the features of the group from its valid members (ticl::cornetto::groupFeatures) and
    // the emission rule. The members are summed in ascending order, as the host does. For an emitted group, its
    // number of layer clusters and of members; 0 for the others and after the last group.
    struct KernelEmission {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const Parameters p,
                                    const Inputs inputs,
                                    uint32_t n,
                                    const Features f,
                                    const uint32_t* headEnd,
                                    const uint32_t* groupSize,
                                    const uint32_t* groupEnd,
                                    const uint32_t* members,
                                    uint32_t* emitted,
                                    uint32_t* emittedVertices,
                                    uint32_t* emittedMembers) const {
        const uint32_t nGroups = headEnd[n - 1];
        for (auto g : uniform_elements(acc, nGroups)) {
          const uint32_t first = groupEnd[g] - groupSize[g];
          const uint32_t last = groupEnd[g];
          float energy = 0.f, emWeight = 0.f, em = 0.f;
          float wx = 0.f, wy = 0.f, wz = 0.f;
          for (uint32_t slot = first; slot < last; ++slot) {
            const uint32_t i = members[slot];
            if (f.valid[i] == 0u)
              continue;
            energy += f.energy[i];
            wx += f.x[i] * f.energy[i];
            wy += f.y[i] * f.energy[i];
            wz += f.z[i] * f.energy[i];
            if (f.em[i] >= 0.f) {
              emWeight += f.energy[i];
              em += f.em[i] * f.energy[i];
            }
          }
          // as ROOT::Math::DisplacementVector3D::operator/: a scale by the inverse
          const float invEnergy = energy > 0.f ? 1.f / energy : 1.f;
          const float bx = wx * invEnergy;
          const float by = wy * invEnergy;
          const float bz = wz * invEnergy;
          const float groupEm = emWeight > 0.f ? em / emWeight : -1.f;
          const bool valid =
              energy > 0.f && std::isfinite(bx) && std::isfinite(by) && std::isfinite(bz) && std::isfinite(energy);
          const float eta = etaOf(bx, by, bz);
          const bool emit = valid && emits(p, energy, energy / std::cosh(eta), groupEm);
          emitted[g] = emit ? 1u : 0u;
          uint32_t nVertices = 0;
          if (emit) {
            for (uint32_t slot = first; slot < last; ++slot) {
              const uint32_t i = members[slot];
              const uint32_t c = inputs.collectionOf(i);
              nVertices += inputs.tracksters[c].vertices().count(i - inputs.offsets[c]);
            }
          }
          emittedVertices[g] = nVertices;
          emittedMembers[g] = emit ? last - first : 0u;
        }
      }
    };

    // Per emitted group: its members in the links, and the end of its layer clusters
    struct KernelFillLinks {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    uint32_t n,
                                    const uint32_t* headEnd,
                                    const uint32_t* groupSize,
                                    const uint32_t* groupEnd,
                                    const uint32_t* members,
                                    const uint32_t* emitted,
                                    const uint32_t* outputEnd,
                                    const uint32_t* verticesEnd,
                                    const uint32_t* membersEnd,
                                    TracksterDeviceCollection::View tracksters,
                                    TracksterLinksDeviceCollection::View links) const {
        const uint32_t nGroups = headEnd[n - 1];
        for (auto g : uniform_elements(acc, nGroups)) {
          if (emitted[g] == 0u)
            continue;
          const uint32_t t = outputEnd[g] - 1;
          links.offsets()[t + 1].keys_offsets() = membersEnd[g];
          tracksters.vertices().offsets()[t + 1].keys_offsets() = verticesEnd[g];
          const uint32_t first = groupEnd[g] - groupSize[g];
          uint32_t slot = membersEnd[g] - groupSize[g];
          for (uint32_t m = first; m < groupEnd[g]; ++m, ++slot) {
            links.content()[slot].values() = members[m];
          }
        }
      }
    };

    // Per link: the number of layer clusters of its input trackster
    struct KernelLinkVertexCounts {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const Inputs inputs,
                                    TracksterLinksDeviceCollection::View links,
                                    uint32_t* counts) const {
        for (auto slot : uniform_elements(acc, links.size())) {
          const uint32_t i = links.content()[slot].values();
          const uint32_t c = inputs.collectionOf(i);
          counts[slot] = inputs.tracksters[c].vertices().count(i - inputs.offsets[c]);
        }
      }
    };

    // Per link: the layer clusters of its input trackster, into the layer clusters of the linked trackster. The links
    // are in the order of the linked tracksters, so the scan of their counts places them.
    struct KernelCopyVertices {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const Inputs inputs,
                                    TracksterLinksDeviceCollection::View links,
                                    const uint32_t* countsEnd,
                                    TracksterDeviceCollection::View tracksters) const {
        auto content = tracksters.vertices().content();
        for (auto slot : uniform_elements(acc, links.size())) {
          const uint32_t i = links.content()[slot].values();
          const uint32_t c = inputs.collectionOf(i);
          const auto vertices = inputs.tracksters[c].vertices()[i - inputs.offsets[c]];
          const uint32_t first = countsEnd[slot] - static_cast<uint32_t>(vertices.size());
          // the layer clusters of the input trackster in ascending order, as the host tracksters have them
          uint32_t v = first;
          for (const uint32_t layerCluster : vertices) {
            uint32_t k = v++;
            while (k > first && content[k - 1].values() > layerCluster) {
              content[k].values() = content[k - 1].values();
              --k;
            }
            content[k].values() = layerCluster;
          }
        }
      }
    };

    // Per linked trackster: no PID, unknown = 1, as ticl::Trackster::mergeTracksters does
    struct KernelUnknownPid {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc, TracksterDeviceCollection::View tracksters) const {
        for (auto t : uniform_elements(acc, tracksters.energy().metadata().size())) {
          auto p = tracksters.id()[t].idProbabilities();
          p.setZero();
          p(kUnknown) = 1.f;
        }
      }
    };

    auto workDivFor(uint32_t n) { return make_workdiv<Acc1D>(divide_up_by(std::max(n, 1u), kBlockSize), kBlockSize); }

    // Allocates n elements and sets them to 0
    auto zeroed(Queue& queue, uint32_t n) {
      auto buffer = make_device_buffer<uint32_t[]>(queue, n);
      alpaka::memset(queue, buffer, 0x0);
      return buffer;
    }

  }  // namespace

  void Algo::group(Queue& queue, const Parameters& p, const Inputs& inputs, Scratch& scratch) const {
    const uint32_t n = inputs.size();
    scratch.nTracksters = n;
    scratch.counts = make_host_buffer<uint32_t[]>(queue, 3);
    for (uint32_t k = 0; k < 3; ++k) {
      (*scratch.counts)[k] = 0;
    }
    if (n == 0) {
      return;
    }
    const auto workDiv = workDivFor(n);

    // the features
    auto featureStore = make_device_buffer<float[]>(queue, 10 * static_cast<std::size_t>(n));
    auto valid = make_device_buffer<uint32_t[]>(queue, n);
    Features f{featureStore.data(),
               featureStore.data() + n,
               featureStore.data() + 2 * n,
               featureStore.data() + 3 * n,
               featureStore.data() + 4 * n,
               featureStore.data() + 5 * n,
               featureStore.data() + 6 * n,
               featureStore.data() + 7 * n,
               featureStore.data() + 8 * n,
               featureStore.data() + 9 * n,
               valid.data()};

    // the tiles: a bin per eta-phi cell and endcap, the tracksters of a bin contiguous in binContent
    auto bin = make_device_buffer<uint32_t[]>(queue, n);
    auto binCounts = zeroed(queue, kBins);
    auto binOffsets = zeroed(queue, kBins + 1);
    auto binContent = make_device_buffer<uint32_t[]>(queue, n);
    alpaka::exec<Acc1D>(queue, workDiv, KernelFeatures{}, p, inputs, f, bin.data(), binCounts.data());
    iterativePrefixScan<Acc1D>(queue, binCounts.data(), binOffsets.data() + 1, kBins);
    // the cursor of each bin starts at its offset
    auto binCursor = make_device_buffer<uint32_t[]>(queue, kBins);
    alpaka::memcpy(queue, binCursor, make_device_view(alpaka::getDev(queue), binOffsets.data(), kBins));
    alpaka::exec<Acc1D>(queue, workDiv, KernelFillBins{}, n, bin.data(), binCursor.data(), binContent.data());

    // the parent tree and the groups
    auto parents = make_device_buffer<uint32_t[]>(queue, n);
    alpaka::exec<Acc1D>(queue, workDiv, KernelParents{}, p, n, f, binOffsets.data(), binContent.data(), parents.data());
    scratch.label = make_device_buffer<uint32_t[]>(queue, n);
    auto labelCount = zeroed(queue, n);
    auto labelMinMember = make_device_buffer<uint32_t[]>(queue, n);
    alpaka::memset(queue, labelMinMember, 0xff);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        KernelLabels{},
                        p,
                        n,
                        f,
                        parents.data(),
                        scratch.label->data(),
                        labelCount.data(),
                        labelMinMember.data());
    // the groups in the order of their smallest member
    auto head = make_device_buffer<uint32_t[]>(queue, n);
    auto headEnd = make_device_buffer<uint32_t[]>(queue, n);
    alpaka::exec<Acc1D>(
        queue, workDiv, KernelGroupHeads{}, n, scratch.label->data(), labelMinMember.data(), head.data());
    iterativePrefixScan<Acc1D>(queue, head.data(), headEnd.data(), n);
    auto groupOfLabel = make_device_buffer<uint32_t[]>(queue, n);
    scratch.groupSize = zeroed(queue, n);
    scratch.groupEnd = make_device_buffer<uint32_t[]>(queue, n);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        KernelGroupsOfLabels{},
                        n,
                        scratch.label->data(),
                        head.data(),
                        headEnd.data(),
                        labelCount.data(),
                        groupOfLabel.data(),
                        scratch.groupSize->data());
    iterativePrefixScan<Acc1D>(queue, scratch.groupSize->data(), scratch.groupEnd->data(), n);
    auto groupCursor = zeroed(queue, n);
    auto unsorted = make_device_buffer<uint32_t[]>(queue, n);
    scratch.members = make_device_buffer<uint32_t[]>(queue, n);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        KernelFillGroups{},
                        n,
                        scratch.label->data(),
                        groupOfLabel.data(),
                        scratch.groupSize->data(),
                        scratch.groupEnd->data(),
                        groupCursor.data(),
                        unsorted.data());
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        KernelSortGroups{},
                        n,
                        scratch.label->data(),
                        groupOfLabel.data(),
                        scratch.groupSize->data(),
                        scratch.groupEnd->data(),
                        unsorted.data(),
                        scratch.members->data());

    // the emission, and the sizes of the output
    scratch.emitted = zeroed(queue, n);
    auto emittedVertices = zeroed(queue, n);
    auto emittedMembers = zeroed(queue, n);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        KernelEmission{},
                        p,
                        inputs,
                        n,
                        f,
                        headEnd.data(),
                        scratch.groupSize->data(),
                        scratch.groupEnd->data(),
                        scratch.members->data(),
                        scratch.emitted->data(),
                        emittedVertices.data(),
                        emittedMembers.data());
    scratch.outputEnd = make_device_buffer<uint32_t[]>(queue, n);
    scratch.verticesEnd = make_device_buffer<uint32_t[]>(queue, n);
    scratch.membersEnd = make_device_buffer<uint32_t[]>(queue, n);
    iterativePrefixScan<Acc1D>(queue, scratch.emitted->data(), scratch.outputEnd->data(), n);
    iterativePrefixScan<Acc1D>(queue, emittedVertices.data(), scratch.verticesEnd->data(), n);
    iterativePrefixScan<Acc1D>(queue, emittedMembers.data(), scratch.membersEnd->data(), n);
    const auto device = alpaka::getDev(queue);
    alpaka::memcpy(queue,
                   make_host_view<uint32_t>((*scratch.counts)[0]),
                   make_device_view<const uint32_t>(device, scratch.outputEnd->data()[n - 1]));
    alpaka::memcpy(queue,
                   make_host_view<uint32_t>((*scratch.counts)[1]),
                   make_device_view<const uint32_t>(device, scratch.verticesEnd->data()[n - 1]));
    alpaka::memcpy(queue,
                   make_host_view<uint32_t>((*scratch.counts)[2]),
                   make_device_view<const uint32_t>(device, scratch.membersEnd->data()[n - 1]));
    scratch.headEnd = std::move(headEnd);
  }

  void Algo::fill(Queue& queue,
                  const Inputs& inputs,
                  const Scratch& scratch,
                  float zLimitEm,
                  reco::CaloClusterDeviceCollection::ConstView layerClusters,
                  TracksterDeviceCollection::View tracksters,
                  TracksterLinksDeviceCollection::View links) const {
    // the first linked trackster starts at 0 in the links and in the layer clusters, also when there is none
    auto firstLink = make_device_view(queue, links.offsets().keys_offsets().data(), 1);
    alpaka::memset(queue, firstLink, 0x0);
    auto firstVertex = make_device_view(queue, tracksters.vertices().offsets().keys_offsets().data(), 1);
    alpaka::memset(queue, firstVertex, 0x0);
    const uint32_t n = scratch.nTracksters;
    const uint32_t nLinked = scratch.nLinked();
    if (n == 0 || nLinked == 0) {
      return;
    }
    alpaka::exec<Acc1D>(queue,
                        workDivFor(n),
                        KernelFillLinks{},
                        n,
                        scratch.headEnd->data(),
                        scratch.groupSize->data(),
                        scratch.groupEnd->data(),
                        scratch.members->data(),
                        scratch.emitted->data(),
                        scratch.outputEnd->data(),
                        scratch.verticesEnd->data(),
                        scratch.membersEnd->data(),
                        tracksters,
                        links);
    const uint32_t nMembers = scratch.nMembers();
    auto linkVertexCounts = make_device_buffer<uint32_t[]>(queue, nMembers);
    auto linkVertexEnd = make_device_buffer<uint32_t[]>(queue, nMembers);
    alpaka::exec<Acc1D>(queue, workDivFor(nMembers), KernelLinkVertexCounts{}, inputs, links, linkVertexCounts.data());
    iterativePrefixScan<Acc1D>(queue, linkVertexCounts.data(), linkVertexEnd.data(), nMembers);
    alpaka::exec<Acc1D>(
        queue, workDivFor(nMembers), KernelCopyVertices{}, inputs, links, linkVertexEnd.data(), tracksters);
    // the properties over all the layer clusters, as assignPCAtoTracksters without cleaning
    properties::fillTracksterProperties<false>(queue, zLimitEm, layerClusters, tracksters);
    alpaka::exec<Acc1D>(queue, workDivFor(nLinked), KernelUnknownPid{}, tracksters);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl::cornetto
