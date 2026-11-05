#include <vector>

#include <catch2/catch_all.hpp>

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "RecoTICL/Linking/interface/CornettoFeatures.h"

using ticl::Trackster;
using ticl::cornetto::emProbability;

namespace {
  using Type = Trackster::ParticleType;
  const std::vector<int> kPhotonElectron{static_cast<int>(Type::photon), static_cast<int>(Type::electron)};
  const std::vector<int> kPhotonElectronPi0{
      static_cast<int>(Type::photon), static_cast<int>(Type::electron), static_cast<int>(Type::neutral_pion)};

  Trackster withProbabilities(std::initializer_list<std::pair<Type, float>> values) {
    Trackster ts;
    for (auto const& [type, p] : values)
      ts.setIdProbability(type, p);
    return ts;
  }
}  // namespace

TEST_CASE("A pi0-dominated trackster is EM with the neutral pion in the categories, not without",
          "[CornettoFeatures]") {
  const auto pi0 = withProbabilities({{Type::photon, 0.10f},
                                      {Type::electron, 0.05f},
                                      {Type::neutral_pion, 0.70f},
                                      {Type::charged_hadron, 0.10f},
                                      {Type::neutral_hadron, 0.05f}});
  REQUIRE_THAT(emProbability(pi0, kPhotonElectronPi0), Catch::Matchers::WithinAbs(0.85, 1e-6));
  REQUIRE_THAT(emProbability(pi0, kPhotonElectron), Catch::Matchers::WithinAbs(0.15, 1e-6));
  // The same thresholds as the type veto: 0.85 is above 0.6, 0.15 is below 0.4.
  REQUIRE(emProbability(pi0, kPhotonElectronPi0) > 0.6f);
  REQUIRE(emProbability(pi0, kPhotonElectron) < 0.4f);
}

TEST_CASE("The EM probability is normalised to the sum of the probabilities", "[CornettoFeatures]") {
  // Probabilities that do not sum to one: the EM fraction is the ratio.
  const auto ts = withProbabilities({{Type::photon, 0.3f}, {Type::charged_hadron, 0.3f}});
  REQUIRE_THAT(emProbability(ts, kPhotonElectron), Catch::Matchers::WithinAbs(0.5, 1e-6));
  REQUIRE_THAT(emProbability(ts, {static_cast<int>(Type::charged_hadron)}), Catch::Matchers::WithinAbs(0.5, 1e-6));
}

TEST_CASE("A trackster without a PID has EM probability -1", "[CornettoFeatures]") {
  REQUIRE(emProbability(Trackster{}, kPhotonElectronPi0) == -1.f);
  // A merged trackster carries only unknown.
  const auto merged = withProbabilities({{Type::unknown, 1.f}});
  REQUIRE(emProbability(merged, kPhotonElectronPi0) == -1.f);
}
