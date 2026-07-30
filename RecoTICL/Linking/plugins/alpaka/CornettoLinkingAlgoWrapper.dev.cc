// Cornetto trackster linking on device: tile, pair test, connected components.
//
// The three stages are the ones the CPU plugin's header advertises as the device
// mapping. Two deliberate differences from the host implementation, both of which
// must leave the emitted components IDENTICAL:
//
//  - the tiling is 2D in (endcap, eta, phi) instead of the host's phi-only tiling
//    plus an eta sort. The candidate set only shrinks by pairs the host would have
//    rejected on the eta cut anyway, so the accepted pairs are the same set.
//  - connectivity is label propagation to the minimum index in the component,
//    which is the same fixed point as the host union-find's parent[max] = min.
//
// Edges are counted before they are written, so the edge buffer is allocated to
// the exact size and there is no capacity guess and no silent truncation.

#include <cmath>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "CornettoLinkingAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  namespace {

    // HGCAL barycentre pseudorapidity range used to bound the eta binning. Values
    // outside are clamped into the edge bins, so nothing is ever dropped.
    constexpr float kEtaMin = 1.2f;
    constexpr float kEtaMax = 3.4f;
    constexpr float kPi = 3.14159265358979323846f;

    // Tile geometry, computed once on host and passed to every kernel.
    struct TileGeometry {
      int32_t nEtaBins;
      int32_t nPhiBins;
      int32_t nBins;  // 2 endcaps * nEtaBins * nPhiBins
      float etaBinWidth;
      float phiBinWidth;
    };

    ALPAKA_FN_ACC ALPAKA_FN_INLINE int32_t etaBinOf(const TileGeometry& g, float eta) {
      const int32_t b = static_cast<int32_t>((std::abs(eta) - kEtaMin) / g.etaBinWidth);
      const int32_t hi = g.nEtaBins - 1;
      return (b < 0) ? 0 : ((b > hi) ? hi : b);
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE int32_t phiBinOf(const TileGeometry& g, float phi) {
      const int32_t b = static_cast<int32_t>((phi + kPi) / g.phiBinWidth);
      const int32_t hi = g.nPhiBins - 1;
      return (b < 0) ? 0 : ((b > hi) ? hi : b);
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE int32_t binOf(const TileGeometry& g, int32_t endcap, int32_t ie, int32_t ip) {
      return (endcap * g.nEtaBins + ie) * g.nPhiBins + ip;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float deltaPhi(float a, float b) {
      float d = a - b;
      while (d > kPi)
        d -= 2.f * kPi;
      while (d <= -kPi)
        d += 2.f * kPi;
      return d;
    }

    // THE pair predicate. Kept in one place so host and device cannot drift, and
    // written as a pure function of the unordered pair (i, j): the anchor tie-break
    // on the input index is what makes that true (see the CPU plugin, same rule).
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool cornettoPair(const TracksterDeviceCollection::ConstView& ts,
                                                     const CornettoParameters& p,
                                                     int32_t i,
                                                     int32_t j) {
      const float etaI = ts[i].eta();
      const float etaJ = ts[j].eta();
      if (etaI * etaJ < 0.f)
        return false;  // same endcap only
      if (std::abs(etaI - etaJ) > p.etaWindow)
        return false;
      if (std::abs(deltaPhi(ts[i].phi(), ts[j].phi())) > p.etaWindow)
        return false;

      const float eI = ts[i].rawEnergy();
      const float eJ = ts[j].rawEnergy();
      const bool iIsAnchor = (eI > eJ) or (eI == eJ and i < j);
      const int32_t a = iIsAnchor ? i : j;
      const int32_t o = iIsAnchor ? j : i;

      const float dx = ts[o].baryX() - ts[a].baryX();
      const float dy = ts[o].baryY() - ts[a].baryY();
      const float dz = ts[o].baryZ() - ts[a].baryZ();
      const float s = dx * ts[a].axisX() + dy * ts[a].axisY() + dz * ts[a].axisZ();
      // Depth-dependent longitudinal window (same formula and float order as the CPU
      // plugin, so the two backends stay bitwise identical): widen with the anchor's
      // calorimeter depth |z|, since hadronic showers reach deeper into CE-H.
      const float zrel = std::abs(ts[a].baryZ()) - p.longitudinalZRef;
      const float maxLong = p.maxLongitudinalDistance + p.maxLongitudinalSlope * (zrel > 0.f ? zrel : 0.f);
      if (std::abs(s) > maxLong)
        return false;

      const float mag2 = dx * dx + dy * dy + dz * dz;
      const float dT2raw = mag2 - s * s;
      const float dT2 = (dT2raw > 0.f) ? dT2raw : 0.f;
      const float rT = p.transverseRadius0 + p.transverseSlope * std::abs(s);
      if (dT2 > rT * rT)
        return false;

      const float teI = ts[i].timeError();
      const float teJ = ts[j].timeError();
      if (teI > 0.f and teJ > 0.f) {
        const float sigma2 = teI * teI + teJ * teJ;
        const float dt = ts[i].time() - ts[j].time();
        if (dt * dt > p.timeCompatibilityNSigma * p.timeCompatibilityNSigma * sigma2)
          return false;
      }
      return true;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE int32_t findRoot(const int32_t* label, int32_t x) {
      while (label[x] != x)
        x = label[x];
      return x;
    }

    // Kernels


    // --- Seeded growth ---------------------------------------------------------
    // Same algorithm as the CPU plugin: cores grow by attaching satellites, satellites
    // never attach to each other, so no chain can form. Each round is a pure map over
    // the satellites judged against the core state as it was at the start of the round,
    // which is why the result does not depend on execution order and the two backends
    // stay bitwise identical.

    // Does core s accept satellite c, given the core centroid and axis of this round?
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool cornettoAttach(const TracksterDeviceCollection::ConstView& ts,
                                                       const CornettoParameters& p,
                                                       const float* coreE,
                                                       const float* coreX,
                                                       const float* coreY,
                                                       const float* coreZ,
                                                       int32_t s,
                                                       int32_t c) {
      const float invE = 1.f / coreE[s];
      const float cx = coreX[s] * invE;
      const float cy = coreY[s] * invE;
      const float cz = coreZ[s] * invE;
      if (cz * ts[c].baryZ() < 0.f)
        return false;  // same endcap only
      const float cr = std::sqrt(cx * cx + cy * cy + cz * cz);
      if (cr <= 0.f)
        return false;
      // Axis from the interaction point to the core centroid: defined for every
      // trackster, unlike the PCA eigenvector of a single layer cluster.
      const float ax = cx / cr, ay = cy / cr, az = cz / cr;
      const float cEta = std::asinh(cz / std::sqrt(cx * cx + cy * cy));
      const float cPhi = std::atan2(cy, cx);
      if (std::abs(ts[c].eta() - cEta) > p.etaWindow)
        return false;
      if (std::abs(deltaPhi(ts[c].phi(), cPhi)) > p.etaWindow)
        return false;

      const float dx = ts[c].baryX() - cx;
      const float dy = ts[c].baryY() - cy;
      const float dz = ts[c].baryZ() - cz;
      const float sProj = dx * ax + dy * ay + dz * az;
      if (p.forwardOnly and sProj < 0.f)
        return false;  // a shower develops in depth
      const float zrel = std::abs(cz) - p.longitudinalZRef;
      const float maxLong = p.maxLongitudinalDistance + p.maxLongitudinalSlope * (zrel > 0.f ? zrel : 0.f);
      if (std::abs(sProj) > maxLong)
        return false;
      const float dT2raw = dx * dx + dy * dy + dz * dz - sProj * sProj;
      const float dT2 = (dT2raw > 0.f) ? dT2raw : 0.f;
      const float rT = p.transverseRadius0 + p.transverseSlope * std::abs(sProj);
      if (dT2 > rT * rT)
        return false;

      const float teS = ts[s].timeError();
      const float teC = ts[c].timeError();
      if (teS > 0.f and teC > 0.f) {
        const float sigma2 = teS * teS + teC * teC;
        const float dt = ts[s].time() - ts[c].time();
        if (dt * dt > p.timeCompatibilityNSigma * p.timeCompatibilityNSigma * sigma2)
          return false;
      }

      // Axis stability: a satellite that would swing the core direction belongs to a
      // neighbouring shower.
      const float eC = ts[c].rawEnergy();
      const float nE = coreE[s] + eC;
      const float nx = coreX[s] + ts[c].baryX() * eC;
      const float ny = coreY[s] + ts[c].baryY() * eC;
      const float nz = coreZ[s] + ts[c].baryZ() * eC;
      const float nr = std::sqrt(nx * nx + ny * ny + nz * nz);
      if (nr <= 0.f)
        return false;
      const float invN = 1.f / nr;
      if ((nx * invN) * ax + (ny * invN) * ay + (nz * invN) * az < p.axisToleranceCos)
        return false;
      (void)nE;
      return true;
    }

    class InitCoresKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    const TracksterDeviceCollection::ConstView ts,
                                    CornettoParameters p,
                                    int32_t* owner,
                                    float* coreE,
                                    float* coreX,
                                    float* coreY,
                                    float* coreZ) const {
        for (int32_t i : uniform_elements(acc, ts.metadata().size())) {
          const float e = ts[i].rawEnergy();
          const bool isCore = e >= p.seedEnergy;
          owner[i] = isCore ? i : -1;
          coreE[i] = isCore ? e : 0.f;
          coreX[i] = isCore ? ts[i].baryX() * e : 0.f;
          coreY[i] = isCore ? ts[i].baryY() * e : 0.f;
          coreZ[i] = isCore ? ts[i].baryZ() * e : 0.f;
        }
      }
    };

    // One growth round: every unowned satellite picks the best core that accepts it.
    // Highest-energy core wins, ties by smaller index, so the choice is a function of
    // the pair alone and no thread ordering can change it.
    class GrowRoundKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    const TracksterDeviceCollection::ConstView ts,
                                    CornettoParameters p,
                                    TileGeometry g,
                                    const int32_t* tileOffset,
                                    const int32_t* tileContent,
                                    const int32_t* owner,
                                    const float* coreE,
                                    const float* coreX,
                                    const float* coreY,
                                    const float* coreZ,
                                    int32_t* claim) const {
        for (int32_t c : uniform_elements(acc, ts.metadata().size())) {
          claim[c] = -1;
          if (owner[c] != -1)
            continue;
          int32_t best = -1;
          float bestE = -1.f;
          const int32_t endcap = ts[c].baryZ() > 0.f ? 1 : 0;
          const int32_t ie = etaBinOf(g, ts[c].eta());
          const int32_t ip = phiBinOf(g, ts[c].phi());
          for (int32_t de = -1; de <= 1; ++de) {
            const int32_t e2 = ie + de;
            if (e2 < 0 or e2 >= g.nEtaBins)
              continue;
            for (int32_t dp = -1; dp <= 1; ++dp) {
              const int32_t p2 = (ip + dp + g.nPhiBins) % g.nPhiBins;
              const int32_t b = binOf(g, endcap, e2, p2);
              for (int32_t k = tileOffset[b]; k < tileOffset[b + 1]; ++k) {
                const int32_t s = tileContent[k];
                if (owner[s] != s)
                  continue;  // not a core root
                if (coreE[s] < bestE or (coreE[s] == bestE and best != -1 and s > best))
                  continue;
                if (not cornettoAttach(ts, p, coreE, coreX, coreY, coreZ, s, c))
                  continue;
                best = s;
                bestE = coreE[s];
              }
            }
          }
          claim[c] = best;
        }
      }
    };

    // Commit the round. The accumulation is atomic because several satellites of the
    // same core are committed concurrently.
    class CommitRoundKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    const TracksterDeviceCollection::ConstView ts,
                                    const int32_t* claim,
                                    int32_t* owner,
                                    float* coreE,
                                    float* coreX,
                                    float* coreY,
                                    float* coreZ,
                                    int32_t* changed) const {
        for (int32_t c : uniform_elements(acc, ts.metadata().size())) {
          const int32_t s = claim[c];
          if (s < 0)
            continue;
          owner[c] = s;
          const float e = ts[c].rawEnergy();
          alpaka::atomicAdd(acc, &coreE[s], e, alpaka::hierarchy::Blocks{});
          alpaka::atomicAdd(acc, &coreX[s], ts[c].baryX() * e, alpaka::hierarchy::Blocks{});
          alpaka::atomicAdd(acc, &coreY[s], ts[c].baryY() * e, alpaka::hierarchy::Blocks{});
          alpaka::atomicAdd(acc, &coreZ[s], ts[c].baryZ() * e, alpaka::hierarchy::Blocks{});
          alpaka::atomicAdd(acc, changed, 1, alpaka::hierarchy::Blocks{});
        }
      }
    };

    // owner -> component label, using the same convention as the union-find path: the
    // label is the smallest input index in the group. A core labels itself and its
    // satellites with min(core, satellites); an unclaimed trackster labels itself.
    class InitGroupMinKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, int32_t* groupMin, int32_t n) const {
        for (int32_t i : uniform_elements(acc, n))
          groupMin[i] = i;
      }
    };

    class OwnerToLabelKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    const int32_t* owner,
                                    int32_t* groupMin,
                                    int32_t n) const {
        for (int32_t i : uniform_elements(acc, n)) {
          const int32_t o = owner[i] >= 0 ? owner[i] : i;
          alpaka::atomicMin(acc, &groupMin[o], i, alpaka::hierarchy::Blocks{});
        }
      }
    };

    class WriteLabelKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    const int32_t* owner,
                                    const int32_t* groupMin,
                                    TracksterComponentsDeviceCollection::View components) const {
        for (int32_t i : uniform_elements(acc, components.metadata().size())) {
          const int32_t o = owner[i] >= 0 ? owner[i] : i;
          components[i].label() = groupMin[o];
        }
      }
    };

    class CountPerTileKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    const TracksterDeviceCollection::ConstView ts,
                                    TileGeometry g,
                                    int32_t* tileCount) const {
        for (int32_t i : uniform_elements(acc, ts.metadata().size())) {
          const int32_t endcap = (ts[i].eta() >= 0.f) ? 1 : 0;
          const int32_t b = binOf(g, endcap, etaBinOf(g, ts[i].eta()), phiBinOf(g, ts[i].phi()));
          alpaka::atomicAdd(acc, &tileCount[b], 1, alpaka::hierarchy::Blocks{});
        }
      }
    };

    // Serial exclusive scan. The arrays scanned here are the tile histogram (a few
    // hundred bins) and the per-trackster edge counts (O(1e4) at PU200), so a single
    // element doing the scan costs far less than the launch overhead of a parallel
    // one, and it keeps the offsets exactly reproducible.
    class ExclusiveScanKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, const int32_t* in, int32_t* out, int32_t n) const {
        if (once_per_grid(acc)) {
          int32_t sum = 0;
          for (int32_t i = 0; i < n; ++i) {
            out[i] = sum;
            sum += in[i];
          }
          out[n] = sum;
        }
      }
    };

    class FillTilesKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    const TracksterDeviceCollection::ConstView ts,
                                    TileGeometry g,
                                    const int32_t* tileOffset,
                                    int32_t* cursor,
                                    int32_t* tileContent) const {
        for (int32_t i : uniform_elements(acc, ts.metadata().size())) {
          const int32_t endcap = (ts[i].eta() >= 0.f) ? 1 : 0;
          const int32_t b = binOf(g, endcap, etaBinOf(g, ts[i].eta()), phiBinOf(g, ts[i].phi()));
          const int32_t slot = alpaka::atomicAdd(acc, &cursor[b], 1, alpaka::hierarchy::Blocks{});
          tileContent[tileOffset[b] + slot] = i;
        }
      }
    };

    // Visit the 3x3 (eta, phi) neighbourhood of trackster i within its own endcap
    // and apply f(j) to every candidate j > i.
    template <typename TFunc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void forEachCandidate(const TracksterDeviceCollection::ConstView& ts,
                                                         const TileGeometry& g,
                                                         const int32_t* tileOffset,
                                                         const int32_t* tileContent,
                                                         int32_t i,
                                                         TFunc&& f) {
      const int32_t endcap = (ts[i].eta() >= 0.f) ? 1 : 0;
      const int32_t ie = etaBinOf(g, ts[i].eta());
      const int32_t ip = phiBinOf(g, ts[i].phi());
      for (int32_t de = -1; de <= 1; ++de) {
        const int32_t e = ie + de;
        if (e < 0 or e >= g.nEtaBins)
          continue;
        for (int32_t dp = -1; dp <= 1; ++dp) {
          const int32_t p = (ip + dp + g.nPhiBins) % g.nPhiBins;
          const int32_t b = binOf(g, endcap, e, p);
          for (int32_t k = tileOffset[b]; k < tileOffset[b + 1]; ++k) {
            const int32_t j = tileContent[k];
            if (j > i)
              f(j);
          }
        }
      }
    }

    class CountEdgesKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    const TracksterDeviceCollection::ConstView ts,
                                    CornettoParameters p,
                                    TileGeometry g,
                                    const int32_t* tileOffset,
                                    const int32_t* tileContent,
                                    int32_t* perTracksterEdges) const {
        for (int32_t i : uniform_elements(acc, ts.metadata().size())) {
          int32_t n = 0;
          forEachCandidate(ts, g, tileOffset, tileContent, i, [&](int32_t j) {
            if (cornettoPair(ts, p, i, j))
              ++n;
          });
          perTracksterEdges[i] = n;
        }
      }
    };

    class FillEdgesKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    const TracksterDeviceCollection::ConstView ts,
                                    CornettoParameters p,
                                    TileGeometry g,
                                    const int32_t* tileOffset,
                                    const int32_t* tileContent,
                                    const int32_t* edgeOffset,
                                    int32_t* edgeA,
                                    int32_t* edgeB) const {
        for (int32_t i : uniform_elements(acc, ts.metadata().size())) {
          int32_t slot = edgeOffset[i];
          forEachCandidate(ts, g, tileOffset, tileContent, i, [&](int32_t j) {
            if (cornettoPair(ts, p, i, j)) {
              edgeA[slot] = i;
              edgeB[slot] = j;
              ++slot;
            }
          });
        }
      }
    };

    class InitLabelsKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, TracksterComponentsDeviceCollection::View components) const {
        for (int32_t i : uniform_elements(acc, components.metadata().size()))
          components[i].label() = i;
      }
    };

    class HookKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    const int32_t* edgeA,
                                    const int32_t* edgeB,
                                    int32_t nEdges,
                                    int32_t* label,
                                    int32_t* changed) const {
        for (int32_t e : uniform_elements(acc, nEdges)) {
          const int32_t ra = findRoot(label, edgeA[e]);
          const int32_t rb = findRoot(label, edgeB[e]);
          if (ra != rb) {
            const int32_t hi = (ra > rb) ? ra : rb;
            const int32_t lo = (ra > rb) ? rb : ra;
            alpaka::atomicMin(acc, &label[hi], lo, alpaka::hierarchy::Blocks{});
            alpaka::atomicExch(acc, changed, 1, alpaka::hierarchy::Blocks{});
          }
        }
      }
    };

    class CompressKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, int32_t* label, int32_t n) const {
        for (int32_t i : uniform_elements(acc, n))
          label[i] = findRoot(label, i);
      }
    };

  }  // namespace

  void CornettoLinkingAlgoWrapper::run(Queue& queue,
                                       const CornettoParameters& params,
                                       const TracksterDeviceCollection::ConstView tracksters,
                                       TracksterComponentsDeviceCollection::View components) const {
    const int32_t n = tracksters.metadata().size();

    // Labels are meaningful even for an empty or single-element collection: every
    // trackster is its own component until an edge says otherwise.
    const uint32_t items = 256;
    auto workDivN = make_workdiv<Acc1D>(divide_up_by(std::max(n, 1), items), items);
    alpaka::exec<Acc1D>(queue, workDivN, InitLabelsKernel{}, components);
    if (n < 2)
      return;

    TileGeometry g;
    g.nPhiBins = std::max(4, static_cast<int32_t>(2.f * kPi / params.etaWindow));
    g.phiBinWidth = 2.f * kPi / g.nPhiBins;
    g.nEtaBins = std::max(1, static_cast<int32_t>((kEtaMax - kEtaMin) / params.etaWindow));
    g.etaBinWidth = (kEtaMax - kEtaMin) / g.nEtaBins;
    g.nBins = 2 * g.nEtaBins * g.nPhiBins;

    auto tileCount = make_device_buffer<int32_t[]>(queue, g.nBins);
    auto tileOffset = make_device_buffer<int32_t[]>(queue, g.nBins + 1);
    auto cursor = make_device_buffer<int32_t[]>(queue, g.nBins);
    auto tileContent = make_device_buffer<int32_t[]>(queue, n);
    alpaka::memset(queue, tileCount, 0);
    alpaka::memset(queue, cursor, 0);

    auto workDivOne = make_workdiv<Acc1D>(1u, 1u);

    alpaka::exec<Acc1D>(queue, workDivN, CountPerTileKernel{}, tracksters, g, tileCount.data());
    alpaka::exec<Acc1D>(queue, workDivOne, ExclusiveScanKernel{}, tileCount.data(), tileOffset.data(), g.nBins);
    alpaka::exec<Acc1D>(
        queue, workDivN, FillTilesKernel{}, tracksters, g, tileOffset.data(), cursor.data(), tileContent.data());

    // Seeded growth reuses the tiles above and then replaces the connected-components
    // stage entirely: no edge list is built, because satellites never link to each
    // other and there is nothing to take a transitive closure over.
    if (params.seededGrowth) {
      auto owner = make_device_buffer<int32_t[]>(queue, n);
      auto claim = make_device_buffer<int32_t[]>(queue, n);
      auto coreE = make_device_buffer<float[]>(queue, n);
      auto coreX = make_device_buffer<float[]>(queue, n);
      auto coreY = make_device_buffer<float[]>(queue, n);
      auto coreZ = make_device_buffer<float[]>(queue, n);
      auto changed = make_device_buffer<int32_t>(queue);
      auto changedHost = make_host_buffer<int32_t>(queue);

      alpaka::exec<Acc1D>(queue,
                          workDivN,
                          InitCoresKernel{},
                          tracksters,
                          params,
                          owner.data(),
                          coreE.data(),
                          coreX.data(),
                          coreY.data(),
                          coreZ.data());

      for (int round = 0; round < kMaxGrowthRounds; ++round) {
        alpaka::memset(queue, changed, 0);
        alpaka::exec<Acc1D>(queue,
                            workDivN,
                            GrowRoundKernel{},
                            tracksters,
                            params,
                            g,
                            tileOffset.data(),
                            tileContent.data(),
                            owner.data(),
                            coreE.data(),
                            coreX.data(),
                            coreY.data(),
                            coreZ.data(),
                            claim.data());
        alpaka::exec<Acc1D>(queue,
                            workDivN,
                            CommitRoundKernel{},
                            tracksters,
                            claim.data(),
                            owner.data(),
                            coreE.data(),
                            coreX.data(),
                            coreY.data(),
                            coreZ.data(),
                            changed.data());
        // The round count is data dependent, so the loop has to see the counter. This
        // is the only host synchronisation in the growth path.
        alpaka::memcpy(queue, changedHost, changed);
        alpaka::wait(queue);
        if (*changedHost.data() == 0)
          break;
      }

      auto groupMin = make_device_buffer<int32_t[]>(queue, n);
      alpaka::exec<Acc1D>(queue, workDivN, InitGroupMinKernel{}, groupMin.data(), n);
      alpaka::exec<Acc1D>(queue, workDivN, OwnerToLabelKernel{}, owner.data(), groupMin.data(), n);
      alpaka::exec<Acc1D>(queue, workDivN, WriteLabelKernel{}, owner.data(), groupMin.data(), components);
      return;
    }

    // Count the edges before writing them: the buffer is then exactly the right
    // size, so there is no capacity parameter to tune and no truncation to detect.
    auto perTracksterEdges = make_device_buffer<int32_t[]>(queue, n);
    auto edgeOffset = make_device_buffer<int32_t[]>(queue, n + 1);
    alpaka::exec<Acc1D>(queue,
                        workDivN,
                        CountEdgesKernel{},
                        tracksters,
                        params,
                        g,
                        tileOffset.data(),
                        tileContent.data(),
                        perTracksterEdges.data());
    alpaka::exec<Acc1D>(queue, workDivOne, ExclusiveScanKernel{}, perTracksterEdges.data(), edgeOffset.data(), n);

    auto nEdgesHost = make_host_buffer<int32_t>(queue);
    auto nEdgesDevice = make_device_view<int32_t>(alpaka::getDev(queue), *(edgeOffset.data() + n));
    alpaka::memcpy(queue, nEdgesHost, nEdgesDevice);
    alpaka::wait(queue);
    const int32_t nEdges = *nEdgesHost.data();
    if (nEdges == 0)
      return;

    auto edgeA = make_device_buffer<int32_t[]>(queue, nEdges);
    auto edgeB = make_device_buffer<int32_t[]>(queue, nEdges);
    alpaka::exec<Acc1D>(queue,
                        workDivN,
                        FillEdgesKernel{},
                        tracksters,
                        params,
                        g,
                        tileOffset.data(),
                        tileContent.data(),
                        edgeOffset.data(),
                        edgeA.data(),
                        edgeB.data());

    // Label propagation. The fixed point is the minimum input index per component,
    // matching the host union-find, so the two backends are comparable elementwise.
    auto changedDevice = make_device_buffer<int32_t>(queue);
    auto changedHost = make_host_buffer<int32_t>(queue);
    auto workDivEdges = make_workdiv<Acc1D>(divide_up_by(nEdges, items), items);

    constexpr int kMaxIterations = 64;
    int iteration = 0;
    for (; iteration < kMaxIterations; ++iteration) {
      alpaka::memset(queue, changedDevice, 0);
      alpaka::exec<Acc1D>(
          queue, workDivEdges, HookKernel{}, edgeA.data(), edgeB.data(), nEdges, components.label().data(), changedDevice.data());
      alpaka::exec<Acc1D>(queue, workDivN, CompressKernel{}, components.label().data(), n);
      alpaka::memcpy(queue, changedHost, changedDevice);
      alpaka::wait(queue);
      if (*changedHost.data() == 0)
        break;
    }
    if (iteration == kMaxIterations) {
      edm::LogWarning("CornettoLinkingAlgoWrapper")
          << "label propagation did not converge in " << kMaxIterations << " iterations for " << n << " tracksters and "
          << nEdges << " edges; the emitted components may be split";
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
