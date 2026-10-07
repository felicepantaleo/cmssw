#include <cstdint>
#include <vector>

#include <catch2/catch_all.hpp>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

namespace {
  // Three neutral tracksters of a PU200 QCD event (regressed energy 1.5, 12 and 80 GeV), in the order of the 31 inputs of
  // TICLCandidateArbitrationProducer, and the shares that ONNX Runtime gives in the training environment.
  const std::vector<std::vector<float>> kInputs = {
      {0.405457586f, 0.405457586f, 1.f,          2.05007482f,  3.04452252f, 0.218951821f, 0.372752726f, 0.00608983077f,
       0.00606999174f, 0.205246583f, 0.178670049f, 0.00611568056f, 0.00610322645f, 4.19791508f, 9.01564026f, 16.f,
       3.87978864f,  1.77312136f,  0.933874369f, 5.68208694f, 2.83736038f, 4.0633502f,  3.12370753f, 2.39531755f,
       7.98165274f,  0.160519019f, 0.301226884f, 0.97993511f, 2.77077174f, 322.161011f, 28.8154907f},
      {2.48488164f, 1.79876328f, 1.f,          2.01980972f,  3.95124364f, 0.112694949f, 0.175184876f, 0.00715720002f,
       0.0071545844f, 0.369458705f, 0.314031392f, 0.00715977838f, 0.00715848012f, 3.80866623f, 9.02351189f, 13.f,
       3.79410863f,  2.10214448f,  0.570674062f, 6.4566679f,  2.11030483f, 2.8722527f,  3.25824356f, 7.36541653f,
       5.74792194f,  0.303435326f, 0.604088843f, 0.930126309f, 2.97700596f, 322.161011f, 31.2225037f},
      {4.38189745f, 4.01951122f, 0.628544331f, 2.79651022f, 5.04342508f, 0.0071782819f, 0.00574839488f, 0.00639628526f,
       0.00638419157f, 0.72756809f, 0.23390846f, 0.00641181925f, 0.00640446134f, 5.10672665f, 9.14571095f, 8.f,
       6.32232809f,  1.94180715f,  1.13070285f, 23.4772015f, 3.34315991f, 2.14035511f, 3.59464645f, 2.17879605f,
       23.2875614f,  0.0363438353f, 0.434245557f, 0.874113977f, 3.14576721f, 322.161011f, 96.25f}};
  const std::vector<float> kShares = {0.152078062f, 0.366908073f, 0.102617592f};
  constexpr char kModel[] = "RecoTICL/Interpretation/data/neutralEnergy/neutralShare_mlp_v1.onnx";
}  // namespace

TEST_CASE("The neutral share model gives the training shares", "[NeutralShareModel]") {
  cms::Ort::ONNXRuntime model(edm::FileInPath(kModel).fullPath());
  std::vector<float> x;
  for (auto const& row : kInputs)
    x.insert(x.end(), row.begin(), row.end());
  const int64_t rows = kInputs.size();
  cms::Ort::FloatArrays input{x};
  const auto out = model.run({"features"}, input, {{rows, 31}}, {}, rows);
  REQUIRE(out.size() == 1);
  REQUIRE(out[0].size() == kShares.size());
  for (size_t k = 0; k < kShares.size(); ++k)
    REQUIRE_THAT(out[0][k], Catch::Matchers::WithinAbs(kShares[k], 1e-6));
}

TEST_CASE("The neutral share model rejects another number of inputs", "[NeutralShareModel]") {
  cms::Ort::ONNXRuntime model(edm::FileInPath(kModel).fullPath());
  cms::Ort::FloatArrays input{std::vector<float>(30, 0.f)};
  REQUIRE_THROWS(model.run({"features"}, input, {{1, 30}}, {}, 1));
}
