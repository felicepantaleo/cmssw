#include <catch2/catch_all.hpp>

#include "RecoTICL/Interpretation/interface/ChargedHadronEnergy.h"

using Catch::Matchers::WithinAbs;

TEST_CASE("The margin of the charged-hadron energy test", "[ChargedHadronEnergy]") {
  SECTION("no momentum error term") {
    REQUIRE_THAT(ticl::chargedHadronEnergyMargin(20., 100., 50., 0.), WithinAbs(4., 1e-6));
    REQUIRE_THAT(ticl::chargedHadronEnergyMargin(200., 100., 50., 0.), WithinAbs(10., 1e-6));
  }
  SECTION("the momentum error term is used when it is larger") {
    REQUIRE_THAT(ticl::chargedHadronEnergyMargin(200., 1000., 100., 3.), WithinAbs(300., 1e-4));
    REQUIRE_THAT(ticl::chargedHadronEnergyMargin(200., 1000., 1., 3.), WithinAbs(10., 1e-6));
  }
  SECTION("the momentum error term is at most the momentum") {
    REQUIRE_THAT(ticl::chargedHadronEnergyMargin(200., 1000., 600., 3.), WithinAbs(1000., 1e-4));
  }
}

TEST_CASE("The combined energy of a track with a claim trackster and no other trackster", "[ChargedHadronEnergy]") {
  // Expected raw deposit of a 1 TeV track: (0.67 + 0.04 ln 1000) 1000 = 946 GeV; the minimum is 189 GeV.
  const ticl::ExpectedDeposit deposit{0.67f, 0.04f, 0.2f, 1.5f};
  const float mass = 0.1396f;
  SECTION("a claim trackster above the minimum deposit gives the combined energy") {
    REQUIRE(ticl::useCombinedClaimEnergy(700.f, mass, 400.f, 1000.f, deposit));
  }
  SECTION("a claim trackster below the minimum deposit keeps the track energy") {
    REQUIRE_FALSE(ticl::useCombinedClaimEnergy(47.6f, mass, 0.5f, 1000.f, deposit));
    REQUIRE_FALSE(ticl::useCombinedClaimEnergy(700.f, mass, 180.f, 1000.f, deposit));
  }
  SECTION("a combined energy at or below the mass keeps the track energy") {
    REQUIRE_FALSE(ticl::useCombinedClaimEnergy(mass, mass, 0.5f, 0.3f, deposit));
    REQUIRE_FALSE(ticl::useCombinedClaimEnergy(0.1f, mass, 0.5f, 0.3f, deposit));
  }
  SECTION("a minimum fraction of 0 accepts every claim trackster") {
    const ticl::ExpectedDeposit noMinimum{0.67f, 0.04f, 0.f, 1.5f};
    REQUIRE(ticl::useCombinedClaimEnergy(47.6f, mass, 0.5f, 1000.f, noMinimum));
  }
}
