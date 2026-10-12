#include <cmath>
#include <limits>
#include <vector>

#include <catch2/catch_all.hpp>

#include "RecoTICL/Interpretation/interface/TypeProbabilities.h"

namespace {
  ticl::Hypothesis hypothesis(ticl::Hypothesis::Type type, int track, int trackster, std::vector<int> tracks = {}) {
    ticl::Hypothesis h;
    h.type = type;
    h.trackIdx = track;
    h.tracksterIdx = trackster;
    h.trackIdxs = std::move(tracks);
    return h;
  }
  float logistic(float w) { return 1.f / (1.f + std::exp(-w)); }
}  // namespace

TEST_CASE("trackTypeProbabilities keeps the best hypothesis of each type of each track", "[TypeProbabilities]") {
  using T = ticl::Hypothesis::Type;
  // Track 0: an electron (weight 1), two charged-hadron hypotheses (weights -1 and 0.5) and a jet with track 1.
  // Track 1: a muon (weight 2), the jet (weight 0.2) and a track-only hypothesis (weight 4).
  // Track 2: a jet vetoed on an EM footprint and a track-only hypothesis.
  const std::vector<ticl::Hypothesis> hypotheses{hypothesis(T::Electron, 0, 0),
                                                 hypothesis(T::ChargedHadron, 0, 1),
                                                 hypothesis(T::RecoveryChargedHadron, 0, 2),
                                                 hypothesis(T::Jet, -1, 3, {0, 1}),
                                                 hypothesis(T::Muon, 1, -1),
                                                 hypothesis(T::Jet, -1, 4, {2}),
                                                 hypothesis(T::Photon, -1, 5),
                                                 hypothesis(T::RecoveryChargedHadron, 1, -1),
                                                 hypothesis(T::RecoveryChargedHadron, 2, -1)};
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<float> weights{1.f, -1.f, 0.5f, 0.2f, 2.f, -inf, 3.f, 4.f, -1.f};
  const auto p = ticl::trackTypeProbabilities(hypotheses, weights, 3);
  REQUIRE(p.size() == 9);
  REQUIRE_THAT(p[0], Catch::Matchers::WithinRel(logistic(1.f), 1e-6f));   // track 0, electron
  REQUIRE(p[1] == 0.f);                                                   // track 0, muon
  REQUIRE_THAT(p[2], Catch::Matchers::WithinRel(logistic(0.5f), 1e-6f));  // track 0, charged hadron
  REQUIRE(p[3] == 0.f);
  REQUIRE_THAT(p[4], Catch::Matchers::WithinRel(logistic(2.f), 1e-6f));   // track 1, muon
  REQUIRE_THAT(p[5], Catch::Matchers::WithinRel(logistic(0.2f), 1e-6f));  // track 1, the jet
  REQUIRE(p[8] == 0.f);                                                   // track 2, the vetoed jet only
}
