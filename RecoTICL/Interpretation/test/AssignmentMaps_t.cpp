#include <vector>

#include <catch2/catch_all.hpp>

#include "RecoTICL/Interpretation/interface/AssignmentMaps.h"

namespace {
  constexpr int kElectron = static_cast<int>(ticl::TrackMode::kElectron);
  constexpr int kChargedHadron = static_cast<int>(ticl::TrackMode::kChargedHadron);
  constexpr int kNotSelected = static_cast<int>(ticl::TrackMode::kNotSelected);
}  // namespace

TEST_CASE("eGammaTracksterIndices selects electron tracksters and the listed neutral types", "[AssignmentMaps]") {
  // Tracks: an electron on trackster 4, a charged hadron on 1, an electron without a trackster, an electron on 4 again.
  const std::vector<int> trackToTrackster{4, 1, -1, 4, -1};
  const std::vector<int> trackMode{kElectron, kChargedHadron, kElectron, kElectron, kNotSelected};
  const std::vector<int> neutralIdx{0, 2, 3, 5};
  const std::vector<int> neutralPdg{22, 130, 111, 22};

  SECTION("photons only") {
    REQUIRE(ticl::eGammaTracksterIndices(trackToTrackster, trackMode, neutralIdx, neutralPdg, {22}) ==
            std::vector<int>{0, 4, 5});
  }
  SECTION("photons and neutral pions") {
    REQUIRE(ticl::eGammaTracksterIndices(trackToTrackster, trackMode, neutralIdx, neutralPdg, {22, 111}) ==
            std::vector<int>{0, 3, 4, 5});
  }
  SECTION("no neutral type") {
    REQUIRE(ticl::eGammaTracksterIndices(trackToTrackster, trackMode, neutralIdx, neutralPdg, {}) ==
            std::vector<int>{4});
  }
}

TEST_CASE("eGammaTracksterIndices is empty without electrons and listed neutrals", "[AssignmentMaps]") {
  REQUIRE(ticl::eGammaTracksterIndices({1}, {kChargedHadron}, {0}, {130}, {22, 111}).empty());
  REQUIRE(ticl::eGammaTracksterIndices({}, {}, {}, {}, {22}).empty());
}
