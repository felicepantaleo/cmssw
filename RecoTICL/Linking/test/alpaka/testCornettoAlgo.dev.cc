// The Cornetto linking on the device: the groups, their emission, the links and the layer clusters of the linked
// tracksters, for hand-made tracksters in two collections and for random tracksters in four collections against the
// host Cornetto.

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/CaloRecHit/interface/CaloClusterHostCollection.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/TracksterHostCollection.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/HGCalReco/interface/TracksterLinksHostCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterPidHostCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterLinksDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterPidDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTICL/Linking/interface/CornettoFeatures.h"
#include "RecoTICL/Linking/interface/CornettoGeometry.h"
#include "RecoTICL/Linking/interface/CornettoGrouping.h"

// The algorithm is a plugin source, so the test compiles it for its backend
#include "RecoTICL/Linking/plugins/alpaka/CornettoAlgo.dev.cc"

using ALPAKA_ACCELERATOR_NAMESPACE::Platform;
using ALPAKA_ACCELERATOR_NAMESPACE::Queue;
namespace cornetto = ALPAKA_ACCELERATOR_NAMESPACE::ticl::cornetto;
using ALPAKA_ACCELERATOR_NAMESPACE::reco::CaloClusterDeviceCollection;
using ALPAKA_ACCELERATOR_NAMESPACE::ticl::TracksterDeviceCollection;
using ALPAKA_ACCELERATOR_NAMESPACE::ticl::TracksterLinksDeviceCollection;
using ALPAKA_ACCELERATOR_NAMESPACE::ticl::TracksterPidDeviceCollection;

namespace {

  constexpr int kPhoton = 0;
  constexpr int kChargedHadron = 4;
  constexpr int kUnknown = 7;
  constexpr float kNoTime = -99.f;
  constexpr float kNoTimeError = -1.f;
  constexpr float kZLimitEm = 400.f;
  constexpr uint32_t kLayerClustersPerTrackster = 2;

  struct Point {
    float x;
    float y;
    float z;
  };

  struct Input {
    Point barycenter;
    float energy;
    int type;  // the particle type with probability 1
    float time = kNoTime;
    float timeError = kNoTimeError;
  };

  Point along(const Point& p, float s, const Point& w, float t) {
    const float r = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    return {p.x + s * p.x / r + t * w.x, p.y + s * p.y / r + t * w.y, p.z + s * p.z / r + t * w.z};
  }

  // The defaults of TracksterLinksSoAProducer
  cornetto::Parameters defaultParameters() {
    cornetto::Parameters p;
    p.etaWindow = 0.3f;
    p.cone = {20.f, 60.f, 4.f, 0.05f};
    p.emCone = {20.f, 60.f, 1.5f, 0.05f};
    p.useTimeGate = true;
    p.timeNSigma2 = 9.f;
    p.timeFloor2 = 0.05f * 0.05f;
    p.typeVetoProbability = 0.6f;
    p.emCategories[0] = 0;
    p.emCategories[1] = 1;
    p.emCategories[2] = 3;
    p.nEmCategories = 3;
    p.minEmittedEnergy = 1.f;
    p.minEmittedPt = 0.5f;
    p.minEmittedEnergyEM = 1.f;
    p.minEmittedPtEM = 0.5f;
    p.minEmittedEnergyAnyPt = 0.f;
    return p;
  }

  // Each trackster has kLayerClustersPerTrackster layer clusters, numbered over all the collections, at its
  // barycenter shifted by -1 and +1 cm in z, with half of its energy each.
  ::ticl::TracksterHostCollection makeTracksters(Queue& queue, const std::vector<Input>& in, uint32_t firstCluster) {
    const auto n = static_cast<int32_t>(in.size());
    const auto nVertices = static_cast<int32_t>(kLayerClustersPerTrackster * in.size());
    ::ticl::TracksterHostCollection tracksters(queue, n, n, n, n, n, n, n, nVertices);
    auto view = tracksters.view();
    for (int32_t i = 0; i < n; ++i) {
      view.energy()[i].rawEnergy() = in[i].energy;
      view.energy()[i].rawEmEnergy() = 0.f;
      view.energy()[i].regressedEnergy() = in[i].energy;
      view.barycenter()[i].x() = in[i].barycenter.x;
      view.barycenter()[i].y() = in[i].barycenter.y;
      view.barycenter()[i].z() = in[i].barycenter.z;
      view.timing()[i].time() = in[i].time;
      view.timing()[i].timeError() = in[i].timeError;
      auto id = view.id()[i].idProbabilities();
      id.setZero();
      id(kChargedHadron) = 1.f;
      view.vertices().offsets()[i].keys_offsets() = kLayerClustersPerTrackster * i;
      for (uint32_t k = 0; k < kLayerClustersPerTrackster; ++k) {
        view.vertices().content()[kLayerClustersPerTrackster * i + k].values() =
            firstCluster + kLayerClustersPerTrackster * i + k;
      }
    }
    view.vertices().offsets()[n].keys_offsets() = nVertices;
    return tracksters;
  }

  ::ticl::TracksterPidHostCollection makePid(Queue& queue, const std::vector<Input>& in) {
    ::ticl::TracksterPidHostCollection pid(queue, static_cast<int32_t>(in.size()));
    for (std::size_t i = 0; i < in.size(); ++i) {
      auto p = pid.view()[i].idProbabilities();
      p.setZero();
      p(in[i].type) = 1.f;
    }
    return pid;
  }

  ::reco::CaloClusterHostCollection makeLayerClusters(Queue& queue, const std::vector<Input>& in) {
    const auto n = static_cast<int32_t>(kLayerClustersPerTrackster * in.size());
    ::reco::CaloClusterHostCollection layerClusters(queue, n, n, n, n);
    auto lcs = layerClusters.view();
    for (std::size_t t = 0; t < in.size(); ++t) {
      for (uint32_t k = 0; k < kLayerClustersPerTrackster; ++k) {
        const uint32_t i = kLayerClustersPerTrackster * t + k;
        const Point& b = in[t].barycenter;
        const float x = std::isfinite(b.x) ? b.x : 0.f;
        lcs.position()[i].x() = x;
        lcs.position()[i].y() = b.y;
        lcs.position()[i].z() = b.z + (k == 0 ? -1.f : 1.f);
        lcs.position()[i].layer() = 10 + static_cast<int>(k);
        lcs.position()[i].cells() = 3;
        lcs.energy()[i].energy() = 0.5f * in[t].energy;
        lcs.energy()[i].correctedEnergy() = 0.5f * in[t].energy;
        lcs.energy()[i].correctedEnergyUncertainty() = 0.f;
        lcs.indexes()[i].algoID() = ::reco::CaloCluster::hgcal_em;
        lcs.indexes()[i].seedID() = DetId(DetId::HGCalEE, 0);
        lcs.indexes()[i].flags() = 0;
        lcs.timing()[i].time() = kNoTime;
        lcs.timing()[i].timeError() = kNoTimeError;
      }
    }
    return layerClusters;
  }

  // A copy of host on the device, with the sizes of host
  template <typename TDevice, typename THost, typename... TSizes>
  TDevice toDevice(Queue& queue, const THost& host, TSizes... sizes) {
    TDevice device(queue, sizes...);
    alpaka::memcpy(queue, device.buffer(), host.buffer());
    return device;
  }

  // The tracksters of the two collections, by global index:
  //   0, 1   a hadronic pair on one axis, one group
  //   2      the mirror of 0 in the other endcap, a group of its own
  //   3      far in phi and below the emitted energy, a group that is not emitted
  //   4, 5   a hadronic pair on a second axis, one group
  //   6      next to 0, EM from the PID of its collection: the type veto keeps it alone
  //   7      next to 0, with a time incompatible with 0: alone
  //   8      compatible with 5 and not with its root 4: alone
  //   9      without a finite barycenter: alone and not emitted
  struct Scenario {
    std::vector<Input> first;
    std::vector<Input> second;

    Scenario() {
      const Point p0{50.f, 0.f, 350.f};
      const Point wy{0.f, 1.f, 0.f};
      const Point q0{0.f, 60.f, 380.f};
      const Point wx{1.f, 0.f, 0.f};
      const Point q1 = along(q0, 25.f, wx, 4.f);
      const float nan = std::numeric_limits<float>::quiet_NaN();
      first = {
          {p0, 20.f, kChargedHadron, 0.f, 0.02f},
          {along(p0, 10.f, wy, 1.f), 10.f, kChargedHadron},
          {{50.f, 0.f, -350.f}, 5.f, kChargedHadron},
          {{-50.f, 0.f, 350.f}, 0.5f, kChargedHadron},
          {q0, 30.f, kChargedHadron},
          {q1, 20.f, kChargedHadron},
      };
      second = {
          {along(p0, 5.f, wy, 0.5f), 8.f, kPhoton},
          {along(p0, 5.f, wy, -1.f), 6.f, kChargedHadron, 3.f, 0.02f},
          {along(q1, 25.f, wx, 4.f), 10.f, kChargedHadron},
          {{nan, 0.f, 350.f}, 50.f, kChargedHadron},
      };
    }

    std::vector<Input> all() const {
      std::vector<Input> out(first);
      out.insert(out.end(), second.begin(), second.end());
      return out;
    }
  };

}  // namespace

TEST_CASE("Cornetto linking on the device") {
  const auto& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    SKIP("no device of the backend");
  }

  const Scenario scenario;
  const std::vector<std::vector<uint32_t>> expectedLinks = {{0, 1}, {2}, {4, 5}, {6}, {7}, {8}};

  for (const auto& device : devices) {
    Queue queue(device);

    const auto firstHost = makeTracksters(queue, scenario.first, 0);
    const auto secondHost = makeTracksters(queue, scenario.second, kLayerClustersPerTrackster * scenario.first.size());
    const auto pidHost = makePid(queue, scenario.second);
    const auto layerClustersHost = makeLayerClusters(queue, scenario.all());
    const int32_t n1 = scenario.first.size();
    const int32_t n2 = scenario.second.size();
    const int32_t nClusters = kLayerClustersPerTrackster * (n1 + n2);
    const auto first = toDevice<TracksterDeviceCollection>(
        queue, firstHost, n1, n1, n1, n1, n1, n1, n1, int32_t(kLayerClustersPerTrackster) * n1);
    const auto second = toDevice<TracksterDeviceCollection>(
        queue, secondHost, n2, n2, n2, n2, n2, n2, n2, int32_t(kLayerClustersPerTrackster) * n2);
    const auto pid = toDevice<TracksterPidDeviceCollection>(queue, pidHost, n2);
    const auto layerClusters =
        toDevice<CaloClusterDeviceCollection>(queue, layerClustersHost, nClusters, nClusters, nClusters, nClusters);

    // the first collection takes its PID from the trackster SoA, the second from its PID SoA
    cornetto::Inputs inputs;
    inputs.nCollections = 2;
    inputs.offsets[0] = 0;
    inputs.offsets[1] = scenario.first.size();
    inputs.offsets[2] = scenario.first.size() + scenario.second.size();
    inputs.tracksters[0] = first.const_view();
    inputs.tracksters[1] = second.const_view();
    inputs.pid[1] = pid.const_view();
    inputs.hasPid[1] = true;

    const cornetto::Algo algo;
    cornetto::Scratch scratch;
    algo.group(queue, defaultParameters(), inputs, scratch);
    alpaka::wait(queue);

    const uint32_t nLinked = scratch.nLinked();
    REQUIRE(nLinked == expectedLinks.size());
    uint32_t nMembers = 0;
    for (const auto& links : expectedLinks) {
      nMembers += links.size();
    }
    REQUIRE(scratch.nMembers() == nMembers);
    REQUIRE(scratch.nVertices() == kLayerClustersPerTrackster * nMembers);

    TracksterDeviceCollection tracksters(
        queue, nLinked, nLinked, nLinked, nLinked, nLinked, nLinked, nLinked, scratch.nVertices());
    TracksterLinksDeviceCollection links(queue, nLinked, scratch.nMembers());
    algo.fill(queue, inputs, scratch, kZLimitEm, layerClusters.const_view(), tracksters.view(), links.view());

    ::ticl::TracksterHostCollection trackstersHost(
        queue, nLinked, nLinked, nLinked, nLinked, nLinked, nLinked, nLinked, scratch.nVertices());
    ::ticl::TracksterLinksHostCollection linksHost(queue, nLinked, scratch.nMembers());
    alpaka::memcpy(queue, trackstersHost.buffer(), tracksters.buffer());
    alpaka::memcpy(queue, linksHost.buffer(), links.buffer());
    alpaka::wait(queue);
    const auto out = trackstersHost.const_view();
    const auto outLinks = linksHost.const_view();
    const auto all = scenario.all();

    for (uint32_t t = 0; t < nLinked; ++t) {
      INFO("linked trackster " << t);
      const auto members = outLinks[t];
      REQUIRE(std::vector<uint32_t>(members.begin(), members.end()) == expectedLinks[t]);

      // the layer clusters of the members, in the order of the members
      std::vector<uint32_t> expectedVertices;
      float energy = 0.f;
      for (uint32_t m : expectedLinks[t]) {
        for (uint32_t k = 0; k < kLayerClustersPerTrackster; ++k) {
          expectedVertices.push_back(kLayerClustersPerTrackster * m + k);
        }
        energy += all[m].energy;
      }
      const auto vertices = out.vertices()[t];
      CHECK(std::vector<uint32_t>(vertices.begin(), vertices.end()) == expectedVertices);
      CHECK(out.energy()[t].rawEnergy() == Catch::Approx(energy));
      CHECK(std::isfinite(out.barycenter()[t].x()));
      CHECK(std::isfinite(out.barycenter()[t].z()));

      const auto id = out.id()[t].idProbabilities();
      for (int k = 0; k <= kUnknown; ++k) {
        CHECK(id(k) == (k == kUnknown ? 1.f : 0.f));
      }
    }

    // the barycenter of the first pair is the energy-weighted mean of its layer clusters
    const float wx =
        (all[0].barycenter.x * all[0].energy + all[1].barycenter.x * all[1].energy) / (all[0].energy + all[1].energy);
    CHECK(out.barycenter()[0].x() == Catch::Approx(wx).epsilon(1e-5));
  }
}

TEST_CASE("Cornetto linking on the device without tracksters") {
  const auto& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    SKIP("no device of the backend");
  }

  for (const auto& device : devices) {
    Queue queue(device);
    const auto emptyHost = makeTracksters(queue, {}, 0);
    const auto empty = toDevice<TracksterDeviceCollection>(queue, emptyHost, 0, 0, 0, 0, 0, 0, 0, 0);
    const auto layerClustersHost = makeLayerClusters(queue, {});
    const auto layerClusters = toDevice<CaloClusterDeviceCollection>(queue, layerClustersHost, 0, 0, 0, 0);

    cornetto::Inputs inputs;
    inputs.nCollections = 1;
    inputs.tracksters[0] = empty.const_view();

    const cornetto::Algo algo;
    cornetto::Scratch scratch;
    algo.group(queue, defaultParameters(), inputs, scratch);
    alpaka::wait(queue);
    REQUIRE(scratch.nLinked() == 0);
    REQUIRE(scratch.nVertices() == 0);
    REQUIRE(scratch.nMembers() == 0);

    TracksterDeviceCollection tracksters(queue, 0, 0, 0, 0, 0, 0, 0, 0);
    TracksterLinksDeviceCollection links(queue, 0, 0);
    algo.fill(queue, inputs, scratch, kZLimitEm, layerClusters.const_view(), tracksters.view(), links.view());

    ::ticl::TracksterLinksHostCollection linksHost(queue, 0, 0);
    alpaka::memcpy(queue, linksHost.buffer(), links.buffer());
    alpaka::wait(queue);
    CHECK(linksHost.const_view().offsets()[0].keys_offsets() == 0);
  }
}

namespace {

  // The pair test, the parents, the groups and the emission of ticl::TracksterLinkingbyCornetto without its second
  // pass. The parent search looks at all the tracksters, which gives the parents of the search in the tiles.
  std::vector<std::vector<unsigned int>> hostLinks(const std::vector<::ticl::Trackster>& tracksters,
                                                   const cornetto::Parameters& p) {
    namespace host = ::ticl::cornetto;
    const std::vector<int> emCategories(p.emCategories, p.emCategories + p.nEmCategories);
    const auto n = static_cast<unsigned int>(tracksters.size());
    host::Features f;
    f.resize(n);
    for (unsigned int i = 0; i < n; ++i) {
      const auto& b = tracksters[i].barycenter();
      f.bary[i] = b;
      f.energy[i] = tracksters[i].raw_energy();
      f.valid[i] = std::isfinite(b.x()) && std::isfinite(b.y()) && std::isfinite(b.z()) && std::isfinite(f.energy[i]);
      f.eta[i] = b.eta();
      f.phi[i] = b.phi();
      f.rad[i] = std::sqrt(b.Mag2());
      f.time[i] = tracksters[i].time();
      f.timeError[i] = tracksters[i].timeError();
      f.em[i] = host::emProbability(tracksters[i], emCategories);
    }
    const host::Cone cone{p.cone.maxBackward, p.cone.maxLongitudinal, p.cone.radius0, p.cone.slope};
    const host::Cone emCone{p.emCone.maxBackward, p.emCone.maxLongitudinal, p.emCone.radius0, p.emCone.slope};
    const auto compatible = [&](unsigned int a, unsigned int b) {
      if (!(f.bary[a].z() * f.bary[b].z() > 0.f))
        return false;
      if (!(std::abs(f.eta[b] - f.eta[a]) <= p.etaWindow) ||
          !(std::abs(reco::deltaPhi(f.phi[b], f.phi[a])) <= p.etaWindow))
        return false;
      const bool emAnchor = p.typeVetoProbability > 0.f && f.em[a] > p.typeVetoProbability;
      const auto s = host::separationInCone(emAnchor ? emCone : cone, f.bary[a], f.rad[a], f.bary[b]);
      if (!s)
        return false;
      if (p.useTimeGate && f.timeError[a] > 0.f && f.timeError[b] > 0.f) {
        const float dt = f.time[b] - f.time[a] - *s * host::kInvSpeedOfLight;
        if (!(dt * dt <=
              p.timeNSigma2 * (f.timeError[a] * f.timeError[a] + f.timeError[b] * f.timeError[b] + p.timeFloor2)))
          return false;
      }
      if (p.typeVetoProbability > 0.f && f.em[a] >= 0.f && f.em[b] >= 0.f) {
        const float hi = p.typeVetoProbability;
        const float lo = 1.f - p.typeVetoProbability;
        if ((f.em[a] > hi && f.em[b] < lo) || (f.em[a] < lo && f.em[b] > hi))
          return false;
      }
      return true;
    };
    std::vector<unsigned int> parents(n);
    for (unsigned int i = 0; i < n; ++i) {
      unsigned int best = i;
      if (f.valid[i] && f.rad[i] > 0.f) {
        for (unsigned int j = 0; j < n; ++j) {
          if (f.valid[j] && host::ranksAbove(f.energy, j, i) && (best == i || host::ranksAbove(f.energy, j, best)) &&
              compatible(j, i))
            best = j;
        }
      }
      parents[i] = best;
    }
    const auto groups = host::groups(host::roots(parents), compatible);
    const auto g = host::groupFeatures(f, groups);
    std::vector<std::vector<unsigned int>> out;
    for (unsigned int k = 0; k < groups.size(); ++k) {
      const bool isEM = p.typeVetoProbability > 0.f && g.em[k] > p.typeVetoProbability;
      const float pt = g.energy[k] / std::cosh(g.eta[k]);
      const bool emits = g.valid[k] && g.energy[k] >= (isEM ? p.minEmittedEnergyEM : p.minEmittedEnergy) &&
                         (pt >= (isEM ? p.minEmittedPtEM : p.minEmittedPt) ||
                          (p.minEmittedEnergyAnyPt > 0.f && g.energy[k] >= p.minEmittedEnergyAnyPt));
      if (emits)
        out.push_back(groups[k]);
    }
    return out;
  }

  // Random tracksters around jet axes in both endcaps, some at the phi seam, with equal energies, EM, hadronic and
  // missing PID, times with and without an error, and a few without a finite barycenter. Each trackster has one to
  // three layer clusters, listed in descending order.
  struct RandomEvent {
    std::vector<::ticl::Trackster> tracksters;
    std::vector<Input> inputs;
    std::vector<std::vector<uint32_t>> vertices;  // per trackster, as in the SoA
    std::vector<uint32_t> sizes;                  // per collection

    explicit RandomEvent(unsigned int seed) : sizes{1500, 0, 800, 900} {
      std::mt19937 gen(seed);
      std::uniform_real_distribution<float> u01(0.f, 1.f);
      std::normal_distribution<float> gauss(0.f, 1.f);
      constexpr int kJets = 120;
      std::vector<float> jetEta(kJets), jetPhi(kJets), jetSign(kJets);
      for (int j = 0; j < kJets; ++j) {
        jetSign[j] = u01(gen) < 0.5f ? -1.f : 1.f;
        jetEta[j] = 1.6f + 1.3f * u01(gen);
        jetPhi[j] = j < 15 ? (j % 2 ? 1.f : -1.f) * (std::numbers::pi_v<float> - 0.05f * u01(gen))
                           : std::numbers::pi_v<float> * (2.f * u01(gen) - 1.f);
      }
      uint32_t n = 0;
      for (auto s : sizes)
        n += s;
      uint32_t nextCluster = 0;
      for (uint32_t i = 0; i < n; ++i) {
        const int j = static_cast<int>(u01(gen) * kJets) % kJets;
        const float eta = jetEta[j] + 0.01f * gauss(gen);
        const float phi = jetPhi[j] + 0.015f * gauss(gen);
        const float z = jetSign[j] * (320.f + 150.f * u01(gen));
        const float rho = std::abs(z) / std::sinh(eta);
        Input in;
        in.barycenter = {
            i % 700 == 3 ? std::numeric_limits<float>::quiet_NaN() : rho * std::cos(phi), rho * std::sin(phi), z};
        in.energy = 0.5f * (1 + static_cast<int>(40.f * u01(gen)));
        const float kind = u01(gen);
        in.type = kind < 0.35f ? kPhoton : (kind < 0.7f ? kChargedHadron : -1);
        if (u01(gen) < 0.5f) {
          const float r = std::sqrt(rho * rho + z * z);
          in.time = r / 29.9792458f + 0.03f * gauss(gen) + (u01(gen) < 0.1f ? 1.f : 0.f);
          in.timeError = 0.03f;
        }
        ::ticl::Trackster t;
        t.setBarycenter(::ticl::Trackster::Vector(in.barycenter.x, in.barycenter.y, in.barycenter.z));
        t.setRawEnergy(in.energy);
        t.setTimeAndError(in.time, in.timeError);
        float prob[8] = {};
        if (in.type >= 0) {
          prob[in.type] = 0.7f + 0.3f * u01(gen);
          prob[in.type == kPhoton ? kChargedHadron : kPhoton] = 1.f - prob[in.type];
        }
        t.setProbabilities(prob);
        tracksters.push_back(t);
        inputs.push_back(in);
        const uint32_t nClusters = 1 + static_cast<uint32_t>(3.f * u01(gen)) % 3;
        std::vector<uint32_t> v;
        for (uint32_t k = 0; k < nClusters; ++k)
          v.push_back(nextCluster + nClusters - 1 - k);
        nextCluster += nClusters;
        vertices.push_back(v);
      }
    }
  };

}  // namespace

TEST_CASE("Cornetto linking on the device against the host Cornetto") {
  const auto& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    SKIP("no device of the backend");
  }

  const auto parameters = defaultParameters();
  for (unsigned int seed : {11u, 12u, 13u}) {
    const RandomEvent event(seed);
    const auto expected = hostLinks(event.tracksters, parameters);
    uint32_t nClusters = 0;
    for (const auto& v : event.vertices)
      nClusters += v.size();

    for (const auto& device : devices) {
      INFO("seed " << seed);
      Queue queue(device);

      // the layer clusters at the barycenter of their trackster
      ::reco::CaloClusterHostCollection layerClustersHost(queue, nClusters, nClusters, nClusters, nClusters);
      auto lcs = layerClustersHost.view();
      for (std::size_t t = 0; t < event.inputs.size(); ++t) {
        const auto& in = event.inputs[t];
        for (uint32_t lc : event.vertices[t]) {
          lcs.position()[lc].x() = std::isfinite(in.barycenter.x) ? in.barycenter.x : 0.f;
          lcs.position()[lc].y() = in.barycenter.y;
          lcs.position()[lc].z() = in.barycenter.z;
          lcs.position()[lc].layer() = 10;
          lcs.position()[lc].cells() = 3;
          lcs.energy()[lc].energy() = in.energy / event.vertices[t].size();
          lcs.energy()[lc].correctedEnergy() = lcs.energy()[lc].energy();
          lcs.energy()[lc].correctedEnergyUncertainty() = 0.f;
          lcs.indexes()[lc].algoID() = ::reco::CaloCluster::hgcal_em;
          lcs.indexes()[lc].seedID() = DetId(DetId::HGCalEE, 0);
          lcs.indexes()[lc].flags() = 0;
          lcs.timing()[lc].time() = kNoTime;
          lcs.timing()[lc].timeError() = kNoTimeError;
        }
      }
      const int32_t nc = nClusters;
      const auto layerClusters = toDevice<CaloClusterDeviceCollection>(queue, layerClustersHost, nc, nc, nc, nc);

      // four collections; the first takes its PID from the trackster SoA, the others from a PID SoA
      std::vector<TracksterDeviceCollection> collections;
      std::vector<TracksterPidDeviceCollection> pids;
      collections.reserve(event.sizes.size());
      pids.reserve(event.sizes.size());
      cornetto::Inputs inputs;
      inputs.nCollections = event.sizes.size();
      uint32_t offset = 0;
      for (uint32_t c = 0; c < event.sizes.size(); ++c) {
        const auto n = static_cast<int32_t>(event.sizes[c]);
        int32_t nVertices = 0;
        for (int32_t i = 0; i < n; ++i)
          nVertices += event.vertices[offset + i].size();
        ::ticl::TracksterHostCollection host(queue, n, n, n, n, n, n, n, nVertices);
        ::ticl::TracksterPidHostCollection pidHost(queue, n);
        auto view = host.view();
        int32_t v = 0;
        for (int32_t i = 0; i < n; ++i) {
          const auto& t = event.tracksters[offset + i];
          view.energy()[i].rawEnergy() = t.raw_energy();
          view.energy()[i].rawEmEnergy() = 0.f;
          view.energy()[i].regressedEnergy() = t.raw_energy();
          view.barycenter()[i].x() = t.barycenter().x();
          view.barycenter()[i].y() = t.barycenter().y();
          view.barycenter()[i].z() = t.barycenter().z();
          view.timing()[i].time() = t.time();
          view.timing()[i].timeError() = t.timeError();
          auto id = c == 0 ? view.id()[i].idProbabilities() : pidHost.view()[i].idProbabilities();
          for (int k = 0; k <= kUnknown; ++k)
            id(k) = t.id_probabilities(k);
          if (c != 0)
            view.id()[i].idProbabilities().setZero();
          view.vertices().offsets()[i].keys_offsets() = v;
          for (uint32_t lc : event.vertices[offset + i])
            view.vertices().content()[v++].values() = lc;
        }
        view.vertices().offsets()[n].keys_offsets() = v;
        collections.push_back(toDevice<TracksterDeviceCollection>(queue, host, n, n, n, n, n, n, n, nVertices));
        pids.push_back(toDevice<TracksterPidDeviceCollection>(queue, pidHost, n));
        inputs.offsets[c] = offset;
        inputs.tracksters[c] = collections.back().const_view();
        if (c != 0) {
          inputs.pid[c] = pids.back().const_view();
          inputs.hasPid[c] = true;
        }
        offset += n;
      }
      inputs.offsets[event.sizes.size()] = offset;

      const cornetto::Algo algo;
      cornetto::Scratch scratch;
      algo.group(queue, parameters, inputs, scratch);
      alpaka::wait(queue);
      const uint32_t nLinked = scratch.nLinked();
      REQUIRE(nLinked == expected.size());

      TracksterDeviceCollection tracksters(
          queue, nLinked, nLinked, nLinked, nLinked, nLinked, nLinked, nLinked, scratch.nVertices());
      TracksterLinksDeviceCollection links(queue, nLinked, scratch.nMembers());
      algo.fill(queue, inputs, scratch, kZLimitEm, layerClusters.const_view(), tracksters.view(), links.view());
      ::ticl::TracksterHostCollection trackstersHost(
          queue, nLinked, nLinked, nLinked, nLinked, nLinked, nLinked, nLinked, scratch.nVertices());
      ::ticl::TracksterLinksHostCollection linksHost(queue, nLinked, scratch.nMembers());
      alpaka::memcpy(queue, trackstersHost.buffer(), tracksters.buffer());
      alpaka::memcpy(queue, linksHost.buffer(), links.buffer());
      alpaka::wait(queue);

      unsigned int differences = 0;
      for (uint32_t t = 0; t < nLinked; ++t) {
        const auto members = linksHost.const_view()[t];
        if (std::vector<unsigned int>(members.begin(), members.end()) != expected[t]) {
          ++differences;
          continue;
        }
        std::vector<uint32_t> expectedVertices;
        float energy = 0.f;
        for (unsigned int m : expected[t]) {
          auto v = event.vertices[m];
          std::sort(v.begin(), v.end());
          expectedVertices.insert(expectedVertices.end(), v.begin(), v.end());
          energy += event.tracksters[m].raw_energy();
        }
        const auto vertices = trackstersHost.const_view().vertices()[t];
        if (std::vector<uint32_t>(vertices.begin(), vertices.end()) != expectedVertices ||
            !(trackstersHost.const_view().energy()[t].rawEnergy() == Catch::Approx(energy).epsilon(1e-5)))
          ++differences;
      }
      CHECK(differences == 0);
    }
  }
}
