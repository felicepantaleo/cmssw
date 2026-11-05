// The selection, the inputs and the outputs of each trackster model, on hand-made tracksters and layer clusters, through
// the same functions that TracksterInferenceProducer calls. The events of the validation have no barrel layer clusters,
// so these cases are the only ones that reach the barrel rules. The checks use REQUIRE as the PyTorch headers redefine
// CHECK.

// Eigen must precede the SoA headers to enable the Eigen columns of the TracksterSoA
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <alpaka/alpaka.hpp>

#include "DataFormats/CaloRecHit/interface/CaloClusterHostCollection.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterHostCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInference.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl;
using ALPAKA_ACCELERATOR_NAMESPACE::ticl::detail::isBarrel;
using Catch::Approx;

namespace {

  constexpr auto s_tag = "[" ALPAKA_TYPE_ALIAS_NAME(testTracksterInference) "]";

  // More than kMaxLayers, so that a layer cluster can fall outside the image
  constexpr uint32_t kLayersPerEndcap = 60;
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

  const DetId kEndcap(DetId::HGCalEE, 0);
  const DetId kEcalBarrel(DetId::Ecal, 1);
  const DetId kHcalBarrel(DetId::Hcal, 1);
  const DetId kHcalOuter(DetId::Hcal, 3);
  const DetId kHcalEndcap(DetId::Hcal, 2);

  // What a test does not set does not matter to it. The initial prediction shows what write() leaves alone.
  constexpr float kOldProbability = 0.9f;
  constexpr float kOldRegressedEnergy = 7.f;

  struct Cluster {
    float energy;
    DetId seed = kEndcap;
    int layer = 0;
    float x = 1.f, y = 1.f, z = 300.f;
    int cells = 1;
  };

  struct Input {
    std::vector<uint32_t> vertices;
    float rawEnergy = 0.f;
    float rawEmEnergy = 0.f;
    float x = 1.f, y = 1.f, z = 300.f;
  };

  // The tracksters and the layer clusters on the device
  class TestEvent {
  public:
    TestEvent(Queue& queue, std::vector<Cluster> const& clusters, std::vector<Input> const& inputs)
        : nTracksters_(inputs.size()),
          nVertices_(countVertices(inputs)),
          clusters_(queue, clusters.size(), clusters.size(), clusters.size(), clusters.size()),
          tracksters_(queue,
                      nTracksters_,
                      nTracksters_,
                      nTracksters_,
                      nTracksters_,
                      nTracksters_,
                      nTracksters_,
                      nTracksters_,
                      nVertices_) {
      ::reco::CaloClusterHostCollection clustersHost(
          queue, clusters.size(), clusters.size(), clusters.size(), clusters.size());
      clustersHost.zeroInitialise();
      auto clusterView = clustersHost.view();
      for (uint32_t i = 0; i < clusters.size(); ++i) {
        auto position = clusterView.position()[i];
        position.x() = clusters[i].x;
        position.y() = clusters[i].y;
        position.z() = clusters[i].z;
        position.layer() = clusters[i].layer;
        position.cells() = clusters[i].cells;
        clusterView.energy()[i].energy() = clusters[i].energy;
        clusterView.indexes()[i].seedID() = clusters[i].seed;
      }
      alpaka::memcpy(queue, clusters_.buffer(), clustersHost.buffer());

      ::ticl::TracksterHostCollection host = makeHost(queue);
      host.zeroInitialise();
      auto view = host.view();
      auto vertices = view.vertices();
      uint32_t offset = 0;
      for (uint32_t t = 0; t < nTracksters_; ++t) {
        auto energy = view.energy()[t];
        energy.rawEnergy() = inputs[t].rawEnergy;
        energy.rawEmEnergy() = inputs[t].rawEmEnergy;
        energy.regressedEnergy() = kOldRegressedEnergy;
        auto barycenter = view.barycenter()[t];
        barycenter.x() = inputs[t].x;
        barycenter.y() = inputs[t].y;
        barycenter.z() = inputs[t].z;
        for (int k = 0; k < 8; ++k)
          view.id()[t].idProbabilities()(k) = kOldProbability;
        vertices.offsets()[t].keys_offsets() = offset;
        for (auto vertex : inputs[t].vertices)
          vertices.content()[offset++].values() = vertex;
      }
      vertices.offsets()[nTracksters_].keys_offsets() = offset;
      alpaka::memcpy(queue, tracksters_.buffer(), host.buffer());
      alpaka::wait(queue);
    }

    TracksterModelContext context(float threshold) const {
      return {tracksters_.const_view(), clusters_.const_view(), kLayersPerEndcap, threshold};
    }

    ::ticl::TracksterHostCollection read(Queue& queue) {
      ::ticl::TracksterHostCollection host = makeHost(queue);
      alpaka::memcpy(queue, host.buffer(), tracksters_.buffer());
      alpaka::wait(queue);
      return host;
    }

    void write(Queue& queue, ::ticl::TracksterHostCollection const& host) {
      alpaka::memcpy(queue, tracksters_.buffer(), host.buffer());
      alpaka::wait(queue);
    }

    uint32_t size() const { return nTracksters_; }
    TracksterDeviceCollection& tracksters() { return tracksters_; }

  private:
    static uint32_t countVertices(std::vector<Input> const& inputs) {
      uint32_t n = 0;
      for (auto const& input : inputs)
        n += input.vertices.size();
      return n;
    }

    ::ticl::TracksterHostCollection makeHost(Queue& queue) const {
      return ::ticl::TracksterHostCollection(queue,
                                             nTracksters_,
                                             nTracksters_,
                                             nTracksters_,
                                             nTracksters_,
                                             nTracksters_,
                                             nTracksters_,
                                             nTracksters_,
                                             nVertices_);
    }

    uint32_t nTracksters_;
    uint32_t nVertices_;
    ALPAKA_ACCELERATOR_NAMESPACE::reco::CaloClusterDeviceCollection clusters_;
    TracksterDeviceCollection tracksters_;
  };

  // These host helpers take typename rather than TracksterModel: inside a host function, clang for ROCm rejects the call
  // to the device function template M::fill in the concept, which would then not be satisfied.
  struct Selection {
    device_buffer<Device, uint32_t[]> selected;
    device_buffer<Device, uint32_t[]> rows;
    std::vector<uint32_t> selectedHost;
    std::vector<uint32_t> rowsHost;
  };

  template <typename M>
  Selection runSelect(Queue& queue, TestEvent const& event, float threshold) {
    const uint32_t n = event.size();
    Selection s{make_device_buffer<uint32_t[]>(queue, n), make_device_buffer<uint32_t[]>(queue, n), {}, {}};
    TracksterInferenceAlgo<M>::select(queue, event.context(threshold), s.selected.data(), s.rows.data());
    auto selected = make_host_buffer<uint32_t[]>(queue, n);
    auto rows = make_host_buffer<uint32_t[]>(queue, n);
    alpaka::memcpy(queue, selected, s.selected);
    alpaka::memcpy(queue, rows, s.rows);
    alpaka::wait(queue);
    s.selectedHost.assign(selected.data(), selected.data() + n);
    s.rowsHost.assign(rows.data(), rows.data() + n);
    return s;
  }

  template <typename M>
  std::vector<uint32_t> selectedBy(Queue& queue, TestEvent const& event, float threshold) {
    return runSelect<M>(queue, event, threshold).selectedHost;
  }

  template <typename M>
  struct Filled {
    PortableHostCollection<typename M::Layout> inputs;
    std::vector<uint32_t> trackster;
  };

  template <typename M>
  Filled<M> runFill(Queue& queue, TestEvent const& event, float threshold) {
    auto selection = runSelect<M>(queue, event, threshold);
    const uint32_t nRows = selection.rowsHost.back();
    TracksterModelInputsDevice<M> inputs(queue, nRows);
    inputs.zeroInitialise(queue);
    auto trackster = make_device_buffer<uint32_t[]>(queue, nRows);
    TracksterInferenceAlgo<M>::fill(queue,
                                    event.context(threshold),
                                    selection.selected.data(),
                                    selection.rows.data(),
                                    trackster.data(),
                                    inputs.view());
    PortableHostCollection<typename M::Layout> host(queue, nRows);
    alpaka::memcpy(queue, host.buffer(), inputs.buffer());
    auto tracksterHost = make_host_buffer<uint32_t[]>(queue, nRows);
    alpaka::memcpy(queue, tracksterHost, trackster);
    alpaka::wait(queue);
    return {std::move(host), std::vector<uint32_t>(tracksterHost.data(), tracksterHost.data() + nRows)};
  }

  // Row r of the predictions goes to trackster rows[r], with probabilities 0.01 * (k + 1) + r and energy energies[r]
  template <typename M>
  void runWrite(Queue& queue,
                TestEvent& event,
                std::vector<uint32_t> const& rows,
                std::vector<float> const& energies,
                bool withPID,
                bool withEnergy,
                float threshold) {
    const uint32_t nRows = rows.size();
    auto trackster = make_device_buffer<uint32_t[]>(queue, nRows);
    auto tracksterHost = make_host_buffer<uint32_t[]>(queue, nRows);
    std::copy(rows.begin(), rows.end(), tracksterHost.data());
    alpaka::memcpy(queue, trackster, tracksterHost);

    TracksterPIDScoresDevice pid(queue, nRows);
    PortableHostCollection<::ticl::TracksterPIDScoresLayout<>> pidHost(queue, nRows);
    TracksterEnergyDevice energy(queue, nRows);
    PortableHostCollection<::ticl::TracksterEnergyLayout<>> energyHost(queue, nRows);
    for (uint32_t r = 0; r < nRows; ++r) {
      auto p = pidHost.view()[r];
      p.probability0() = 0.01f + r;
      p.probability1() = 0.02f + r;
      p.probability2() = 0.03f + r;
      p.probability3() = 0.04f + r;
      p.probability4() = 0.05f + r;
      p.probability5() = 0.06f + r;
      p.probability6() = 0.07f + r;
      p.probability7() = 0.08f + r;
      energyHost.view()[r].energy() = energies[r];
    }
    alpaka::memcpy(queue, pid.buffer(), pidHost.buffer());
    alpaka::memcpy(queue, energy.buffer(), energyHost.buffer());
    TracksterInferenceAlgo<M>::write(queue,
                                     trackster.data(),
                                     nRows,
                                     withPID ? &pid : nullptr,
                                     withEnergy ? &energy : nullptr,
                                     threshold,
                                     event.tracksters().view());
    alpaka::wait(queue);
  }

  template <typename Test>
  void onEachDevice(Test test) {
    auto const& devices = cms::alpakatools::devices<Platform>();
    if (devices.empty()) {
      FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
           "the test will be skipped.");
    }
    for (auto const& device : devices) {
      Queue queue(device);
      test(queue);
    }
  }

  // The layer clusters and tracksters of the selection cases, for a threshold of 2.5
  std::vector<Cluster> selectionClusters() {
    return {{3.f},               // 0
            {2.f, kEcalBarrel},  // 1
            {1.f},               // 2
            {2.f, kHcalBarrel},  // 3
            {2.f, kHcalOuter},   // 4
            {2.f, kHcalEndcap},  // 5: not barrel
            {1.f},               // 6
            {0.5f}};             // 7
  }

  std::vector<Input> selectionTracksters() {
    return {{{0}, 3.f},      // 0: endcap only
            {{1}, 1.f},      // 1: barrel only
            {{0, 1}, 2.5f},  // 2: enough without the barrel, raw energy on the threshold
            {{2, 1}, kNaN},  // 3: enough only with the barrel, NaN raw energy
            {{}, 0.f},       // 4: empty
            {{3}},           // 5: HCAL barrel
            {{4}},           // 6: HCAL outer
            {{5, 6}},        // 7: HCAL endcap and HGCAL
            {{5, 7}}};       // 8: exactly on the threshold
  }

  using Selected = std::vector<uint32_t>;

}  // namespace

// A model without the hooks is rejected at compile time
struct NotAModel {};
static_assert(not TracksterModel<NotAModel>);
static_assert(not TracksterModel<int>);

TEST_CASE("barrel seeds", s_tag) {
  REQUIRE(isBarrel(kEcalBarrel));
  REQUIRE(isBarrel(kHcalBarrel));
  REQUIRE(isBarrel(kHcalOuter));
  REQUIRE_FALSE(isBarrel(kHcalEndcap));
  REQUIRE_FALSE(isBarrel(kEndcap));
  REQUIRE_FALSE(isBarrel(DetId(DetId::HGCalHSi, 0)));
  REQUIRE_FALSE(isBarrel(DetId(DetId::HGCalHSc, 0)));
}

TEST_CASE("selection", s_tag) {
  onEachDevice([](Queue& queue) {
    TestEvent event(queue, selectionClusters(), selectionTracksters());
    // summed energy of all the layer clusters
    REQUIRE(selectedBy<TracksterCNN>(queue, event, 2.5f) == Selected{1, 0, 1, 1, 0, 0, 0, 1, 1});
    // the barrel layer clusters do not count: tracksters 3 stops passing
    REQUIRE(selectedBy<TracksterDNN>(queue, event, 2.5f) == Selected{1, 0, 1, 0, 0, 0, 0, 1, 1});
    // no barrel layer cluster at all, whatever the energy, and the empty trackster passes
    REQUIRE(selectedBy<TracksterPFN>(queue, event, 2.5f) == Selected{1, 0, 0, 0, 1, 0, 0, 1, 1});
    // the raw energy, including on the threshold and NaN
    REQUIRE(selectedBy<TracksterShowerShape>(queue, event, 2.5f) == Selected{1, 0, 1, 1, 0, 0, 0, 0, 0});
    // an empty trackster has no sum to compare, even with a threshold of 0
    REQUIRE(selectedBy<TracksterCNN>(queue, event, 0.f)[4] == 0u);
    REQUIRE(selectedBy<TracksterDNN>(queue, event, 0.f)[4] == 0u);
    // the barrel-only tracksters reach 0 with the CNN, and never with the DNN
    REQUIRE(selectedBy<TracksterCNN>(queue, event, 0.f)[1] == 1u);
    REQUIRE(selectedBy<TracksterDNN>(queue, event, 0.f)[1] == 0u);
    // rows[t] counts the selected tracksters up to t
    REQUIRE(runSelect<TracksterCNN>(queue, event, 2.5f).rowsHost == Selected{1, 1, 2, 3, 3, 3, 3, 4, 5});
  });
}

namespace {

  // Layer 3: the energies 1 to 12, one barrel layer cluster of 20, and one of the other endcap of 0.5. Layer 5: one of
  // 4. Layer 55: outside the image.
  std::vector<Cluster> imageClusters(bool withBarrel) {
    std::vector<Cluster> clusters;
    for (int i = 1; i <= 12; ++i)
      clusters.push_back({static_cast<float>(i), kEndcap, 3, 10.f + i, 5.f, 300.f});
    clusters.push_back({withBarrel ? 20.f : 0.25f, withBarrel ? kEcalBarrel : kEndcap, 3});
    clusters.push_back({4.f, kEndcap, 5, 11.f, 6.f, -320.f, 7});
    clusters.push_back({30.f, kEndcap, 55});
    clusters.push_back({0.5f, kEndcap, 3 + static_cast<int>(kLayersPerEndcap)});
    return clusters;
  }

  // Trackster 0 is not selected, so that the row and the trackster differ
  std::vector<Input> imageTracksters() {
    return {{{}, 0.f}, {{5, 12, 0, 11, 3, 14, 7, 1, 13, 9, 2, 15, 4, 10, 6, 8}, 90.f, 30.f, 3.f, 4.f, -300.f}};
  }

  template <typename Row>
  std::vector<float> energiesOf(Row const& row, int layer) {
    std::vector<float> energies;
    for (int slot = 0; slot < ::ticl::kMaxClusters; ++slot)
      energies.push_back(row.energy()(layer, slot));
    return energies;
  }

  template <typename Row>
  int filledSlots(Row const& row) {
    int n = 0;
    for (int layer = 0; layer < ::ticl::kMaxLayers; ++layer)
      for (int slot = 0; slot < ::ticl::kMaxClusters; ++slot)
        n += row.energy()(layer, slot) != 0.f;
    return n;
  }

  using Energies = std::vector<float>;

}  // namespace

TEST_CASE("image of the CNN and the DNN", s_tag) {
  onEachDevice([](Queue& queue) {
    TestEvent event(queue, imageClusters(true), imageTracksters());

    auto cnn = runFill<TracksterCNN>(queue, event, 1.f);
    REQUIRE(cnn.trackster == Selected{1});
    auto row = cnn.inputs.const_view()[0];
    // the 10 most energetic of the layer by decreasing energy, the barrel one included
    REQUIRE(energiesOf(row, 3) == Energies{20.f, 12.f, 11.f, 10.f, 9.f, 8.f, 7.f, 6.f, 5.f, 4.f});
    REQUIRE(energiesOf(row, 5)[0] == 4.f);
    REQUIRE(energiesOf(row, 5)[1] == 0.f);
    // layer 55 is outside the image
    REQUIRE(filledSlots(row) == 11);
    REQUIRE(row.absEta()(5, 0) == Approx(std::asinh(320.f / std::sqrt(11.f * 11.f + 6.f * 6.f))).epsilon(1e-5));
    REQUIRE(row.phi()(5, 0) == Approx(std::atan2(6.f, 11.f)).epsilon(1e-5));

    // the barrel layer cluster does not take a slot
    auto dnn = runFill<TracksterDNN>(queue, event, 1.f);
    REQUIRE(dnn.trackster == Selected{1});
    auto dnnRow = dnn.inputs.const_view()[0];
    REQUIRE(energiesOf(dnnRow, 3) == Energies{12.f, 11.f, 10.f, 9.f, 8.f, 7.f, 6.f, 5.f, 4.f, 3.f});
    REQUIRE(filledSlots(dnnRow) == 11);
  });
}

TEST_CASE("inputs of the PFN", s_tag) {
  onEachDevice([](Queue& queue) {
    TestEvent event(queue, imageClusters(false), imageTracksters());
    auto pfn = runFill<TracksterPFN>(queue, event, 1.f);
    // the empty trackster passes too, so the row of trackster 1 is 1
    REQUIRE(pfn.trackster == Selected{0, 1});
    auto row = pfn.inputs.const_view()[1];
    REQUIRE(energiesOf(row, 3) == Energies{12.f, 11.f, 10.f, 9.f, 8.f, 7.f, 6.f, 5.f, 4.f, 3.f});
    REQUIRE(row.x()(5, 0) == 11.f);
    REQUIRE(row.y()(5, 0) == 6.f);
    REQUIRE(row.absZ()(5, 0) == 320.f);
    REQUIRE(row.cells()(5, 0) == 7.f);
    REQUIRE(row.rawEnergy() == 90.f);
    REQUIRE(row.rawEmEnergy() == 30.f);
    REQUIRE(row.barycenterX() == 3.f);
    REQUIRE(row.barycenterY() == 4.f);
    REQUIRE(row.barycenterAbsZ() == 300.f);
    REQUIRE(row.barycenterAbsEta() == Approx(std::asinh(300.f / 5.f)).epsilon(1e-5));
    REQUIRE(row.barycenterPhi() == Approx(std::atan2(4.f, 3.f)).epsilon(1e-5));
    REQUIRE(filledSlots(pfn.inputs.const_view()[0]) == 0);
  });
}

TEST_CASE("inputs of the shower-shape model", s_tag) {
  onEachDevice([](Queue& queue) {
    TestEvent event(queue, {{1.f}, {1.f}, {1.f}}, {{{0, 1, 2}, 10.f, 4.f, 3.f, 4.f, -300.f}});
    auto host = event.read(queue);
    auto view = host.view();
    auto pca = view.pca()[0];
    pca.eigenvalues()(0) = 3.f;
    pca.eigenvalues()(1) = 2.f;
    pca.eigenvalues()(2) = 0.f;  // floored at 1e-6
    pca.sigmas()(1) = 2.f;
    pca.sigmasPCA()(2) = 5.f;
    pca.eigenvectors()(2, 0) = 1.f;
    auto profile = view.profile()[0];
    profile.maxEnergyFraction() = 0.5f;
    profile.core2EnergyFraction() = 0.25f;
    profile.core5EnergyFraction() = 0.75f;
    profile.rmsRadius() = 1.5f;
    profile.minAbsZ() = 310.f;
    profile.lengthZ() = 20.f;
    event.write(queue, host);

    auto filled = runFill<TracksterShowerShape>(queue, event, 1.f);
    REQUIRE(filled.trackster == Selected{0});
    auto row = filled.inputs.const_view()[0];
    REQUIRE(row.logRawEnergy() == Approx(std::log1p(10.f)).epsilon(1e-5));
    REQUIRE(row.emFraction() == Approx(0.4f).epsilon(1e-5));
    REQUIRE(row.barycenterAbsZ() == 300.f);
    REQUIRE(row.logNLayerClusters() == Approx(std::log1p(3.f)).epsilon(1e-5));
    REQUIRE(row.logEigenvalue0() == Approx(std::log(3.f)).epsilon(1e-5));
    REQUIRE(row.logEigenvalue2() == Approx(std::log(1e-6f)).epsilon(1e-5));
    REQUIRE(row.eigenvalue0Fraction() == Approx(0.6f).epsilon(1e-5));
    REQUIRE(row.sigma1() == 2.f);
    REQUIRE(row.sigmaPCA2() == 5.f);
    REQUIRE(row.maxEnergyFraction() == 0.5f);
    REQUIRE(row.core2EnergyFraction() == 0.25f);
    REQUIRE(row.core5EnergyFraction() == 0.75f);
    REQUIRE(row.rmsRadius() == 1.5f);
    REQUIRE(row.minAbsZ() == 310.f);
    REQUIRE(row.lengthZ() == 20.f);
  });
}

namespace {

  // Raw energies around a threshold of 5
  std::vector<Input> writeTracksters() { return {{{0}, 1.f}, {{0}, 5.f}, {{0}, 9.f}, {{0}, 2.f}}; }

}  // namespace

TEST_CASE("predictions of a model without a regression", s_tag) {
  onEachDevice([](Queue& queue) {
    TestEvent event(queue, {{1.f}}, writeTracksters());
    runWrite<TracksterCNN>(queue, event, {1, 2}, {100.f, 200.f}, true, false, 5.f);
    auto host = event.read(queue);
    auto view = host.const_view();
    REQUIRE(view.id()[1].idProbabilities()(0) == 0.01f);
    REQUIRE(view.id()[1].idProbabilities()(7) == 0.08f);
    REQUIRE(view.id()[2].idProbabilities()(3) == 1.04f);
    // only the rows change, and there is no reset or regressed energy
    REQUIRE(view.id()[0].idProbabilities()(0) == kOldProbability);
    REQUIRE(view.energy()[1].regressedEnergy() == kOldRegressedEnergy);

    // without a PID the selected tracksters keep what they had
    TestEvent untouched(queue, {{1.f}}, writeTracksters());
    runWrite<TracksterShowerShape>(queue, untouched, {1}, {100.f}, false, false, 5.f);
    auto untouchedHost = untouched.read(queue);
    REQUIRE(untouchedHost.const_view().id()[1].idProbabilities()(0) == kOldProbability);
  });
}

TEST_CASE("predictions of a model with a regression", s_tag) {
  onEachDevice([](Queue& queue) {
    // the DNN takes the prediction whatever the raw energy
    TestEvent dnn(queue, {{1.f}}, writeTracksters());
    runWrite<TracksterDNN>(queue, dnn, {0, 2}, {100.f, 200.f}, true, true, 5.f);
    auto dnnHost = dnn.read(queue);
    REQUIRE(dnnHost.const_view().energy()[0].regressedEnergy() == 100.f);
    REQUIRE(dnnHost.const_view().energy()[2].regressedEnergy() == 200.f);
    REQUIRE(dnnHost.const_view().energy()[1].regressedEnergy() == kOldRegressedEnergy);

    // the PFN keeps the raw energy up to the threshold, which is included
    TestEvent pfn(queue, {{1.f}}, writeTracksters());
    runWrite<TracksterPFN>(queue, pfn, {0, 1, 2}, {100.f, 200.f, 300.f}, true, true, 5.f);
    auto pfnHost = pfn.read(queue);
    REQUIRE(pfnHost.const_view().energy()[0].regressedEnergy() == 1.f);
    REQUIRE(pfnHost.const_view().energy()[1].regressedEnergy() == 5.f);
    REQUIRE(pfnHost.const_view().energy()[2].regressedEnergy() == 300.f);
    REQUIRE(pfnHost.const_view().energy()[3].regressedEnergy() == kOldRegressedEnergy);

    // the selected tracksters are reset first, so without predictions they are left at 0
    TestEvent reset(queue, {{1.f}}, writeTracksters());
    runWrite<TracksterDNN>(queue, reset, {1}, {100.f}, false, false, 5.f);
    auto resetHost = reset.read(queue);
    REQUIRE(resetHost.const_view().energy()[1].regressedEnergy() == 0.f);
    REQUIRE(resetHost.const_view().id()[1].idProbabilities()(0) == 0.f);
    REQUIRE(resetHost.const_view().energy()[0].regressedEnergy() == kOldRegressedEnergy);
    REQUIRE(resetHost.const_view().id()[0].idProbabilities()(0) == kOldProbability);
  });
}
