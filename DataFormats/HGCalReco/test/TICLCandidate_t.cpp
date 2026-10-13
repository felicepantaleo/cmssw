#include <array>
#include <cmath>
#include <limits>

#include <catch2/catch_all.hpp>

#include "DataFormats/HGCalReco/interface/TICLCandidate.h"

TEST_CASE("TICLCandidate stores a probability as a 16-bit integer with 65535 for 1", "[TICLCandidate]") {
  using C = io_v1::TICLCandidate;
  REQUIRE(C::encodeProbability(0.f) == 0);
  REQUIRE(C::encodeProbability(1.f) == 65535);
  REQUIRE(C::encodeProbability(0.5f) == 32768);
  REQUIRE(C::encodeProbability(-0.2f) == 0);
  REQUIRE(C::encodeProbability(1.5f) == 65535);
  REQUIRE(C::encodeProbability(std::numeric_limits<float>::quiet_NaN()) == 0);
  REQUIRE(C::decodeProbability(0) == 0.f);
  REQUIRE(C::decodeProbability(65535) == 1.f);

  const float maxError = 0.5f / 65535.f + 1e-7f;
  for (int i = 0; i <= 100000; ++i) {
    const float p = i / 100000.f;
    REQUIRE(std::abs(C::decodeProbability(C::encodeProbability(p)) - p) <= maxError);
  }
}

TEST_CASE("TICLCandidate returns the stored probabilities in the order of ParticleType", "[TICLCandidate]") {
  using T = io_v1::TICLCandidate::ParticleType;
  io_v1::TICLCandidate cand;
  const std::array<float, 8> id{0.1f, 0.2f, 0.3f, 0.f, 0.f, 0.4f, 0.f, 0.f};
  cand.setIdProbabilities(id);
  cand.setTypeProbabilities({0.f, 0.9f, 0.f, 0.f, 0.f, 0.05f, 0.f, 0.f});
  const float maxError = 0.5f / 65535.f + 1e-7f;
  const auto back = cand.idProbabilities();
  for (unsigned int i = 0; i < id.size(); ++i)
    REQUIRE(std::abs(back[i] - id[i]) <= maxError);
  REQUIRE(std::abs(cand.id_probability(T::photon) - 0.1f) <= maxError);
  REQUIRE(std::abs(cand.typeProbability(T::electron) - 0.9f) <= maxError);
  REQUIRE(cand.typeProbabilities()[0] == 0.f);

  cand.setIdProbability(T::muon, 1.f);
  REQUIRE(cand.id_probability(T::muon) == 1.f);
  cand.zeroProbabilities();
  for (const float p : cand.idProbabilities())
    REQUIRE(p == 0.f);
}
