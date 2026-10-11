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

TEST_CASE("eGammaElectrons gives the electron track and the Electron hypotheses of each e/gamma trackster",
          "[AssignmentMaps]") {
  // E/gamma tracksters 0, 1, 2 are final tracksters 3, 5, 6. Track 0 is an electron on 5; track 1 is a charged hadron
  // whose Electron hypothesis is on 3; track 2 is an electron on 4, not an e/gamma trackster; track 3 has no hypothesis.
  const std::vector<int> selected{3, 5, 6};
  const std::vector<int> trackToTrackster{5, 1, 4, -1};
  const std::vector<int> trackMode{kElectron, kChargedHadron, kElectron, kNotSelected};
  const std::vector<int> electronTrackster{5, 3, 4, -1};
  const auto electrons = ticl::eGammaElectrons(selected, trackToTrackster, trackMode, electronTrackster);
  REQUIRE(electrons.electronTrack == std::vector<int>{-1, 0, -1});
  REQUIRE(electrons.electronHypothesis == std::vector<bool>{true, true, false});
  REQUIRE(ticl::eGammaElectrons({}, trackToTrackster, trackMode, electronTrackster).electronTrack.empty());
}

TEST_CASE("superClusterSeeding seeds from electron tracks, and from the supercluster for a tried track",
          "[AssignmentMaps]") {
  // E/gamma tracksters: 0 typed electron (track 7), 1 with the Electron hypothesis of a track, 2 neither.
  const ticl::EGammaElectrons electrons{{7, -1, -1}, {true, true, false}};
  SECTION("a trackster typed electron gives a track seed only") {
    const auto seeding = ticl::superClusterSeeding({1, 0, 0}, electrons, false);
    REQUIRE(seeding.tracks == std::vector<int>{7});
    REQUIRE(!seeding.fromSuperCluster);
  }
  SECTION("with superClusterSeedsForElectrons it also gets supercluster seeds") {
    const auto seeding = ticl::superClusterSeeding({0}, electrons, true);
    REQUIRE(seeding.tracks == std::vector<int>{7});
    REQUIRE(seeding.fromSuperCluster);
  }
  SECTION("a tried track without an electron gives supercluster seeds") {
    const auto seeding = ticl::superClusterSeeding({2, 1}, electrons, false);
    REQUIRE(seeding.tracks.empty());
    REQUIRE(seeding.fromSuperCluster);
  }
  SECTION("no electron and no tried track gives no seed") {
    const auto seeding = ticl::superClusterSeeding({2}, electrons, true);
    REQUIRE(seeding.tracks.empty());
    REQUIRE(!seeding.fromSuperCluster);
  }
}
