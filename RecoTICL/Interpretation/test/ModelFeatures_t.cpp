#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Core>

#include <catch2/catch_all.hpp>

#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoTICL/Interpretation/interface/ModelFeatures.h"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using ticl::Hypothesis;
using ticl::Trackster;
using namespace ticl::modelfeatures;

namespace {
  constexpr std::array<float, 8> kProbabilities{0.05f, 0.05f, 0.01f, 0.02f, 0.7f, 0.15f, 0.01f, 0.01f};

  // A trackster of three layer clusters at rho 50 cm, z 350 cm: |eta| = asinh(7).
  Trackster makeTrackster() {
    Trackster ts;
    ts.vertices() = {0, 1, 2};
    ts.vertex_multiplicity() = {1.f, 1.f, 2.f};
    ts.setBarycenter(Trackster::Vector(30.f, 40.f, 350.f));
    ts.setRawEnergy(12.5f);
    ts.setRawEmEnergy(4.f);
    ts.setRegressedEnergy(14.f);
    auto probabilities = kProbabilities;
    ts.setProbabilities(probabilities.data());
    Eigen::Matrix3f axes;
    axes << 0.6f, 0.8f, 0.f, 0.f, 0.f, 1.f, 0.8f, -0.6f, 0.f;
    ts.fillPCAVariables(Eigen::Vector3f(9.f, 4.f, 1.f),
                        axes,
                        Eigen::Vector3f(4.f, 1.f, 0.25f),
                        Eigen::Vector3f(16.f, 9.f, 4.f),
                        3,
                        Trackster::PCAOrdering::descending);
    return ts;
  }

  reco::Track makeTrack(double px, double py, double pz) {
    return reco::Track(
        10., 10., reco::Track::Point(0., 0., 0.), reco::Track::Vector(px, py, pz), 1, reco::Track::CovarianceMatrix());
  }
}  // namespace

TEST_CASE("The hypothesis features are in the slots of hypothesis_mlp_v2", "[ModelFeatures]") {
  const auto ts = makeTrackster();
  const std::vector<reco::Track> tracks{makeTrack(3., 4.2, 36.), makeTrack(1., 1., 9.)};
  const std::vector<bool> muonTrack{false, true}, electronTrack{true, false};

  SECTION("a charged hadron with a footprint") {
    Hypothesis h;
    h.type = Hypothesis::Type::ChargedHadron;
    h.score = 0.8f;
    h.trackIdx = 0;
    h.tracksterIdx = 0;
    std::array<float, 30> x{};
    fillHypothesisFeatures(h, &ts, tracks, {3, muonTrack, electronTrack}, x);
    for (int k = 0; k < 7; ++k)
      CHECK(x[k] == (k == 3 ? 1.f : 0.f));
    CHECK(x[7] == 0.8f);
    CHECK_THAT(x[8], WithinRel(std::log1p(12.5f), 1e-6f));
    CHECK_THAT(x[9], WithinRel(0.32f, 1e-6f));
    CHECK_THAT(x[10], WithinRel(std::asinh(7.f), 1e-5f));
    for (int k = 0; k < 8; ++k)
      CHECK(x[11 + k] == kProbabilities[k]);
    CHECK(x[19] == 3.f);
    CHECK(x[20] == 0.f);
    const float p = tracks[0].p();
    CHECK_THAT(x[21], WithinRel(std::log1p(p), 1e-6f));
    CHECK_THAT(x[22], WithinRel(static_cast<float>(tracks[0].pt()), 1e-6f));
    CHECK_THAT(x[23], WithinRel(static_cast<float>(std::abs(tracks[0].eta())), 1e-6f));
    CHECK_THAT(x[24], WithinRel(12.5f / p, 1e-6f));
    CHECK_THAT(x[25],
               WithinAbs(reco::deltaR(tracks[0].eta(), tracks[0].phi(), std::asinh(7.), std::atan2(40., 30.)), 1e-5));
    CHECK(x[26] == 0.f);
    CHECK(x[27] == 3.f);
    CHECK(x[28] == 0.f);
    CHECK(x[29] == 1.f);
  }

  SECTION("a jet without a footprint") {
    Hypothesis h;
    h.type = Hypothesis::Type::Jet;
    h.score = 0.5f;
    h.trackIdxs = {0, 1};
    std::array<float, 30> x{};
    fillHypothesisFeatures(h, nullptr, tracks, {1, muonTrack, electronTrack}, x);
    CHECK(x[5] == 1.f);
    for (int k = 8; k <= 10; ++k)
      CHECK(x[k] == -1.f);
    for (int k = 11; k <= 18; ++k)
      CHECK(x[k] == -1.f);
    CHECK(x[19] == 0.f);
    CHECK(x[20] == 2.f);
    const float sumP = static_cast<float>(tracks[0].p()) + static_cast<float>(tracks[1].p());
    CHECK_THAT(x[21], WithinRel(std::log1p(sumP), 1e-6f));
    CHECK_THAT(x[22], WithinRel(static_cast<float>(tracks[0].pt()) + static_cast<float>(tracks[1].pt()), 1e-6f));
    CHECK(x[24] == -1.f);
    CHECK(x[25] == -1.f);
    CHECK(x[27] == 1.f);
    CHECK(x[28] == 1.f);
    CHECK(x[29] == 1.f);
  }
}

TEST_CASE("The neutral species features are in the slots of neutralSpecies_mlp_v2", "[ModelFeatures]") {
  const auto ts = makeTrackster();
  std::array<float, 12> x{};
  fillNeutralSpeciesFeatures(ts, x);
  for (int k = 0; k < 8; ++k)
    CHECK(x[k] == kProbabilities[k]);
  CHECK_THAT(x[8], WithinRel(0.32f, 1e-6f));
  CHECK_THAT(x[9], WithinRel(std::log1p(12.5f), 1e-6f));
  CHECK_THAT(x[10], WithinRel(std::asinh(7.f), 1e-5f));
  CHECK(x[11] == 3.f);
}

TEST_CASE("The neutral share features are in the slots of neutralShare_mlp_v1", "[ModelFeatures]") {
  const auto ts = makeTrackster();
  // Three layer clusters near the barycenter, of 6, 4 and 10 GeV; the third is shared by two tracksters.
  const std::vector<reco::CaloCluster> layerClusters{reco::CaloCluster(6., math::XYZPoint(30., 40., 350.)),
                                                     reco::CaloCluster(4., math::XYZPoint(33., 44., 355.)),
                                                     reco::CaloCluster(10., math::XYZPoint(30., 40., 360.))};
  std::array<float, 31> x{};
  fillNeutralShareFeatures(ts, layerClusters, {7.f, 2, 300.f}, x);
  CHECK_THAT(x[0], WithinRel(std::log(14.f), 1e-6f));
  CHECK_THAT(x[1], WithinRel(std::log(12.5f), 1e-6f));
  CHECK_THAT(x[2], WithinRel(0.32f, 1e-6f));
  CHECK_THAT(x[3], WithinRel(std::asinh(7.f), 1e-5f));
  CHECK_THAT(x[4], WithinRel(std::log1p(3.f), 1e-6f));
  for (int k = 0; k < 8; ++k)
    CHECK(x[5 + k] == kProbabilities[k]);
  CHECK_THAT(x[13], WithinRel(std::log1p(7.f), 1e-6f));
  CHECK_THAT(x[14], WithinRel(std::log1p(300.f), 1e-6f));
  CHECK(x[15] == 2.f);
  CHECK_THAT(x[16], WithinRel(std::log(9.f), 1e-6f));
  CHECK_THAT(x[17], WithinRel(std::log(4.f), 1e-6f));
  CHECK_THAT(x[18], WithinAbs(0.f, 1e-6f));
  CHECK_THAT(x[19], WithinRel(4.f, 1e-6f));
  CHECK_THAT(x[20], WithinRel(3.f, 1e-6f));
  CHECK_THAT(x[21], WithinRel(2.f, 1e-6f));
  CHECK_THAT(x[22], WithinRel(2.f, 1e-6f));
  CHECK_THAT(x[23], WithinRel(1.f, 1e-6f));
  CHECK_THAT(x[24], WithinRel(0.5f, 1e-6f));
  // Energies 6, 4 and 10 / 2 = 5 GeV; the largest share is 6 / 15.
  CHECK_THAT(x[25], WithinRel(0.4f, 1e-6f));
  CHECK(x[26] > 0.f);
  CHECK(x[27] >= x[26]);
  CHECK(x[28] > 0.f);
  CHECK(x[29] == 350.f);
  CHECK(x[30] == 10.f);
}

TEST_CASE("The interpretation models give the logits of the training environment", "[ModelFeatures]") {
  // Inputs and logits from ONNX Runtime 1.26.0 in python.
  const std::vector<float> hypothesisInput{
      0.f,         0.f,    0.f,        1.f,  0.f,   0.f,   0.f,   0.8f,  2.6026897f, 0.32f,      2.6441208f, 0.05f,
      0.05f,       0.01f,  0.02f,      0.7f, 0.15f, 0.01f, 0.01f, 3.f,   0.f,        3.6000404f, 5.2f,       2.6390574f,
      0.34421557f, 0.012f, 0.f,        3.f,  0.f,   1.f,   0.f,   0.f,   0.f,        0.f,        0.f,        1.f,
      0.f,         0.6f,   3.9512437f, 0.1f, 2.1f,  0.02f, 0.01f, 0.f,   0.01f,      0.8f,       0.15f,      0.005f,
      0.005f,      25.f,   3.f,        4.1f, 20.f,  2.f,   0.9f,  0.05f, 0.f,        7.f,        1.f,        0.f};
  const std::vector<float> hypothesisLogits{-2.965872049331665f, 1.6499972343444824f};
  const std::vector<float> speciesInput{0.6f,  0.05f,      0.f,   0.3f,  0.03f, 0.02f,      0.f,  0.f,
                                        0.85f, 1.5040774f, 2.2f,  12.f,  0.05f, 0.02f,      0.f,  0.01f,
                                        0.2f,  0.7f,       0.01f, 0.01f, 0.15f, 3.0445225f, 1.9f, 40.f};
  const std::vector<float> speciesLogits{-3065.4287109375f,
                                         1446.5218505859375f,
                                         1200.9384765625f,
                                         -17.30788230895996f,
                                         -20.496549606323242f,
                                         13.615545272827148f};
  auto run = [](const char* model, const std::vector<float>& input, int64_t nFeatures) {
    cms::Ort::ONNXRuntime network(edm::FileInPath(model).fullPath());
    cms::Ort::FloatArrays inputs{input};
    const int64_t rows = static_cast<int64_t>(input.size()) / nFeatures;
    return network.run({"features"}, inputs, {{rows, nFeatures}}, {}, rows)[0];
  };
  const auto hyp = run("RecoTICL/Interpretation/data/arbitration/hypothesis_mlp_v2.onnx", hypothesisInput, 30);
  REQUIRE(hyp.size() == hypothesisLogits.size());
  for (std::size_t k = 0; k < hyp.size(); ++k)
    CHECK_THAT(hyp[k], WithinRel(hypothesisLogits[k], 1e-4f) || WithinAbs(hypothesisLogits[k], 1e-4f));
  const auto species = run("RecoTICL/Interpretation/data/arbitration/neutralSpecies_mlp_v2.onnx", speciesInput, 12);
  REQUIRE(species.size() == speciesLogits.size());
  for (std::size_t k = 0; k < species.size(); ++k)
    CHECK_THAT(species[k], WithinRel(speciesLogits[k], 1e-4f) || WithinAbs(speciesLogits[k], 1e-4f));
}
