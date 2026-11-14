#include <cmath>
#include <vector>

#include <catch2/catch_all.hpp>

#include "RecoTICL/Interpretation/interface/JetVariants.h"

namespace {
  using Footprints = std::vector<std::vector<std::vector<unsigned int>>>;
  const ticl::ExpectedDeposit deposit{0.67f, 0.04f, 0.2f, 1.5f};
  // Expected deposit of a 10 GeV track: (0.67 + 0.04 ln 10) 10 = 7.62 GeV.
  const std::vector<double> trackP{10., 5., 8.};
}  // namespace

TEST_CASE("One track left gives a recovery variant with the recovery band and score", "[JetVariants]") {
  const std::vector<ticl::JetFootprint> jets{{{0, 1}, {0, 1, 2}, {1.f, 1.f, 1.f}}};
  Footprints electrons(3);
  electrons[1] = {{5}};
  SECTION("energy in the band") {
    const std::vector<double> lcEnergy{3., 2., 2., 0., 0., 4.};
    const auto variants = ticl::jetVariantsWithoutElectrons(jets, electrons, lcEnergy, trackP, deposit, 0.f, 0.f);
    REQUIRE(variants.size() == 1);
    REQUIRE(variants[0].tracks == std::vector<int>{0});
    REQUIRE(variants[0].vertices == std::vector<unsigned int>{0, 1, 2});
    const float expected = deposit(10.f);
    REQUIRE_THAT(variants[0].score, Catch::Matchers::WithinAbs(1.f - std::abs(7.f - expected) / expected, 1e-6));
  }
  SECTION("energy above the band") {
    const std::vector<double> lcEnergy{10., 2., 2., 0., 0., 4.};
    REQUIRE(ticl::jetVariantsWithoutElectrons(jets, electrons, lcEnergy, trackP, deposit, 0.f, 0.f).empty());
  }
}

TEST_CASE("A variant equal to an earlier one is not repeated", "[JetVariants]") {
  // Three Electron hypotheses of track 1 whose footprints are outside the jet give the same variant.
  const std::vector<ticl::JetFootprint> jets{{{0, 1}, {0, 1, 2}, {1.f, 1.f, 1.f}}};
  Footprints electrons(3);
  electrons[1] = {{5}, {5}, {4, 5}};
  const std::vector<double> lcEnergy{3., 2., 2., 0., 1., 4.};
  REQUIRE(ticl::jetVariantsWithoutElectrons(jets, electrons, lcEnergy, trackP, deposit, 0.f, 0.f).size() == 1);
}

TEST_CASE("Two tracks left give a jet variant without the electron footprint", "[JetVariants]") {
  const std::vector<ticl::JetFootprint> jets{{{0, 1, 2}, {0, 1, 2}, {1.f, 2.f, 1.f}}};
  Footprints electrons(3);
  electrons[2] = {{2}};
  const std::vector<double> lcEnergy{3., 4., 6., 0., 0., 0.};
  const auto variants = ticl::jetVariantsWithoutElectrons(jets, electrons, lcEnergy, trackP, deposit, 0.f, 0.f);
  REQUIRE(variants.size() == 1);
  REQUIRE(variants[0].tracks == std::vector<int>{0, 1});
  REQUIRE(variants[0].vertices == std::vector<unsigned int>{0, 1});
  REQUIRE(variants[0].multiplicities == std::vector<float>{1.f, 2.f});
  // E = 3 + 4 / 2 = 5 GeV against p = 15 GeV.
  REQUIRE_THAT(variants[0].score, Catch::Matchers::WithinAbs(1.f - 10.f / 15.f, 1e-6));
  SECTION("the jet variant needs the minimum energy over the summed momentum") {
    REQUIRE(ticl::jetVariantsWithoutElectrons(jets, electrons, lcEnergy, trackP, deposit, 0.25f, 0.f).size() == 1);
    REQUIRE(ticl::jetVariantsWithoutElectrons(jets, electrons, lcEnergy, trackP, deposit, 0.4f, 0.f).empty());
  }
  SECTION("the minimum energy over the summed momentum applies from the minimum summed momentum") {
    REQUIRE(ticl::jetVariantsWithoutElectrons(jets, electrons, lcEnergy, trackP, deposit, 0.4f, 15.f).empty());
    REQUIRE(ticl::jetVariantsWithoutElectrons(jets, electrons, lcEnergy, trackP, deposit, 0.4f, 16.f).size() == 1);
  }
}

TEST_CASE("The jet energy test applies from the minimum summed momentum", "[JetVariants]") {
  REQUIRE(ticl::jetEnergyTooLow(5.f, 15.f, 0.4f, 0.f));
  REQUIRE(ticl::jetEnergyTooLow(5.f, 15.f, 0.4f, 15.f));
  REQUIRE_FALSE(ticl::jetEnergyTooLow(5.f, 15.f, 0.4f, 16.f));
  REQUIRE_FALSE(ticl::jetEnergyTooLow(6.f, 15.f, 0.4f, 0.f));
}
