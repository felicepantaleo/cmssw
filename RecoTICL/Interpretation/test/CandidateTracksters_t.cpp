#include <array>
#include <cmath>
#include <map>
#include <vector>

#include <catch2/catch_all.hpp>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoTICL/Interpretation/interface/AssignmentMaps.h"
#include "RecoTICL/Interpretation/interface/CandidateTracksters.h"
#include "RecoTICL/Interpretation/interface/TrackImpact.h"

namespace {
  constexpr int kChargedHadron = static_cast<int>(ticl::TrackMode::kChargedHadron);
  constexpr int kNotSelected = static_cast<int>(ticl::TrackMode::kNotSelected);
  constexpr float kFrontZ = 320.f;

  // The test Ptrs have no product: the fractions see them in the collection edm::ProductID().
  const edm::ProductID kTracksterId;

  reco::Track track(float px, float py, float pz) {
    return reco::Track(
        1., 10., reco::Track::Point(0., 0., 0.), reco::Track::Vector(px, py, pz), 1, reco::Track::CovarianceMatrix());
  }

  reco::CaloCluster layerCluster(float energy, float x, float y, float z) {
    return reco::CaloCluster(energy, math::XYZPoint(x, y, z));
  }

  // A hadronic trackster of the given layer clusters, each held once, with the given energies.
  ticl::Trackster trackster(const std::vector<unsigned int> &vertices, float rawEnergy, float regressedEnergy) {
    ticl::Trackster ts;
    for (auto v : vertices) {
      ts.vertices().push_back(v);
      ts.vertex_multiplicity().push_back(1.f);
    }
    ts.setRawEnergy(rawEnergy);
    ts.setRegressedEnergy(regressedEnergy);
    ts.setBarycenter({10.f, 0.f, 330.f});
    return ts;
  }

  // A straight track impact at (x, y) on the front disk.
  ticl::TrackImpact impactAt(float x, float y) {
    return ticl::TrackImpact{GlobalPoint(x, y, kFrontZ), GlobalVector(0.f, 0.f, 1.f), true};
  }

  struct Scene {
    std::vector<reco::Track> tracks;
    std::vector<ticl::Trackster> tracksters;
    std::vector<reco::CaloCluster> layerClusters;
    std::vector<TICLCandidate> candidates;
    std::vector<ticl::TrackImpact> impacts;
    std::vector<int> trackToTrackster;
    std::vector<int> trackMode;

    // A charged candidate of track t on trackster ts (-1: none) with the given impact.
    void addCharged(unsigned int t, int ts, const ticl::TrackImpact &impact) {
      edm::Ptr<ticl::Trackster> tsPtr;
      if (ts >= 0)
        tsPtr = edm::Ptr<ticl::Trackster>(&tracksters[ts], ts);
      candidates.emplace_back(edm::Ptr<reco::Track>(&tracks[t], t), tsPtr);
      impacts.push_back(impact);
    }
    // A neutral candidate on trackster ts.
    void addNeutral(unsigned int ts) {
      candidates.emplace_back(edm::Ptr<reco::Track>(), edm::Ptr<ticl::Trackster>(&tracksters[ts], ts));
      impacts.push_back({});
    }
    std::vector<std::map<unsigned int, float>> fractions(const ticl::CandidateTrackstersParameters &par) const {
      auto impactOf = [this](unsigned int c) -> const ticl::TrackImpact & { return impacts[c]; };
      return ticl::candidateLayerClusterFractions(
          candidates, kTracksterId, tracksters, trackToTrackster, trackMode, layerClusters, impactOf, par);
    }
  };

  ticl::CandidateTrackstersParameters nearestParameters(float budgetScale = 0.f) {
    ticl::CandidateTrackstersParameters par;
    par.nearest = true;
    par.radius0 = 3.f;
    par.radiusSlope = 0.05f;
    par.budgetScale = budgetScale;
    par.deposit = {0.67f, 0.04f, 0.f, 1.f};
    return par;
  }

  float fractionOf(const std::map<unsigned int, float> &m, unsigned int lc) {
    auto it = m.find(lc);
    return it == m.end() ? 0.f : it->second;
  }

  // Every candidate holds a track or a trackster after the rebuilt tracksters replace the source tracksters.
  void requireInvariant(Scene &scene,
                        const std::vector<int> &candidateIndex,
                        const std::vector<ticl::Trackster> &built) {
    for (unsigned int i = 0; i < candidateIndex.size(); ++i)
      scene.candidates[candidateIndex[i]].setTracksters({edm::Ptr<ticl::Trackster>(&built[i], i)});
    for (auto const &cand : scene.candidates)
      REQUIRE((cand.trackPtr().isNonnull() || !cand.tracksters().empty()));
  }
}  // namespace

TEST_CASE("a neutral candidate whose layer clusters all go to a track keeps its trackster", "[CandidateTracksters]") {
  Scene s;
  // Both layer clusters are within the track radius; no budget.
  s.layerClusters = {layerCluster(4.f, 11.f, 0.f, 330.f), layerCluster(3.f, 12.f, 0.f, 330.f)};
  s.tracksters = {trackster({0, 1}, 7.f, 8.f)};
  s.tracks = {track(1.f, 0.f, 30.f)};
  s.trackToTrackster = {0};
  s.trackMode = {kChargedHadron};
  s.addCharged(0, 0, impactAt(10.f, 0.f));
  s.addNeutral(0);

  const auto fractions = s.fractions(nearestParameters());
  REQUIRE(fractions[1].empty());
  std::vector<int> candidateIndex;
  const auto built = ticl::tracksterOfFractions(fractions, candidateIndex);
  REQUIRE(candidateIndex == std::vector<int>{0});
  requireInvariant(s, candidateIndex, built);
  REQUIRE(s.candidates[1].tracksters().size() == 1);
  REQUIRE(s.candidates[1].tracksters()[0].key() == 0);
}

TEST_CASE("one trackster held by one candidate goes whole to it", "[CandidateTracksters]") {
  Scene s;
  s.layerClusters = {layerCluster(2.f, 10.f, 0.f, 330.f), layerCluster(3.f, 11.f, 0.f, 340.f)};
  s.tracksters = {trackster({0, 1}, 5.f, 6.f)};
  s.tracksters[0].vertex_multiplicity()[1] = 2.f;
  s.tracks = {track(1.f, 0.f, 30.f)};
  s.trackToTrackster = {0};
  s.trackMode = {kChargedHadron};
  s.addCharged(0, 0, impactAt(10.f, 0.f));

  const auto fractions = s.fractions(nearestParameters());
  REQUIRE(fractions.size() == 1);
  REQUIRE(fractionOf(fractions[0], 0) == 1.f);
  REQUIRE(fractionOf(fractions[0], 1) == 0.5f);

  std::vector<int> candidateIndex;
  const auto built = ticl::tracksterOfFractions(fractions, candidateIndex);
  REQUIRE(candidateIndex == std::vector<int>{0});
  REQUIRE(built[0].vertices() == std::vector<unsigned int>{0, 1});
  REQUIRE(built[0].vertex_multiplicity() == std::vector<float>{1.f, 2.f});
  requireInvariant(s, candidateIndex, built);
}

TEST_CASE("a charged hadron and a residual share a trackster: inside the radius to the track, outside to the neutral",
          "[CandidateTracksters]") {
  Scene s;
  // Layer cluster 0 at 1 cm from the track, layer cluster 1 at 8 cm.
  s.layerClusters = {layerCluster(4.f, 11.f, 0.f, 330.f), layerCluster(3.f, 18.f, 0.f, 330.f)};
  s.tracksters = {trackster({0, 1}, 7.f, 8.f)};
  s.tracks = {track(1.f, 0.f, 30.f)};
  s.trackToTrackster = {0};
  s.trackMode = {kChargedHadron};
  s.addCharged(0, 0, impactAt(10.f, 0.f));
  s.addNeutral(0);

  const auto fractions = s.fractions(nearestParameters());
  REQUIRE(fractionOf(fractions[0], 0) == 1.f);
  REQUIRE(fractionOf(fractions[0], 1) == 0.f);
  REQUIRE(fractionOf(fractions[1], 0) == 0.f);
  REQUIRE(fractionOf(fractions[1], 1) == 1.f);

  std::vector<int> candidateIndex;
  const auto built = ticl::tracksterOfFractions(fractions, candidateIndex);
  REQUIRE(candidateIndex == std::vector<int>{0, 1});
  requireInvariant(s, candidateIndex, built);
}

TEST_CASE("a charged hadron whose budget is spent takes no more layer clusters", "[CandidateTracksters]") {
  Scene s;
  // Both layer clusters are 1 cm from the track; the expected deposit of a 10 GeV hadron is 7.6 GeV, the budget
  // 1.3 x 7.6 = 9.9 GeV.
  s.layerClusters = {layerCluster(9.f, 11.f, 0.f, 330.f), layerCluster(3.f, 9.f, 0.f, 330.f)};
  s.tracksters = {trackster({0, 1}, 12.f, 13.f)};
  s.tracks = {track(0.f, 0.f, 10.f)};
  s.trackToTrackster = {0};
  s.trackMode = {kChargedHadron};
  s.addCharged(0, 0, impactAt(10.f, 0.f));
  s.addNeutral(0);

  const auto fractions = s.fractions(nearestParameters(1.3f));
  REQUIRE(fractionOf(fractions[0], 0) == 1.f);
  REQUIRE(fractionOf(fractions[0], 1) == 0.f);
  REQUIRE(fractionOf(fractions[1], 1) == 1.f);
}

TEST_CASE("with a budget a shared layer cluster that no candidate can take goes to no candidate",
          "[CandidateTracksters]") {
  Scene s;
  // Track 0 is 1 cm from both layer clusters and spends its 9.9 GeV budget on the first; track 1 is 40 cm away.
  s.layerClusters = {layerCluster(9.f, 11.f, 0.f, 330.f), layerCluster(3.f, 9.f, 0.f, 330.f)};
  s.tracksters = {trackster({0, 1}, 12.f, 13.f)};
  s.tracks = {track(0.f, 0.f, 10.f), track(0.f, 0.f, 10.f)};
  s.trackToTrackster = {0, 0};
  s.trackMode = {kChargedHadron, kChargedHadron};
  s.addCharged(0, 0, impactAt(10.f, 0.f));
  s.addCharged(1, 0, impactAt(50.f, 0.f));

  const auto fractions = s.fractions(nearestParameters(1.3f));
  REQUIRE(fractionOf(fractions[0], 0) == 1.f);
  REQUIRE(fractionOf(fractions[0], 1) == 0.f);
  REQUIRE(fractions[1].empty());
  std::vector<int> candidateIndex;
  const auto built = ticl::tracksterOfFractions(fractions, candidateIndex);
  REQUIRE(candidateIndex == std::vector<int>{0});
  requireInvariant(s, candidateIndex, built);
}

TEST_CASE("energyShare splits a shared layer cluster in proportion to the candidate energies",
          "[CandidateTracksters]") {
  Scene s;
  s.layerClusters = {layerCluster(4.f, 11.f, 0.f, 330.f)};
  s.tracksters = {trackster({0}, 4.f, 6.f)};
  s.tracks = {track(1.f, 0.f, 30.f)};
  s.trackToTrackster = {0};
  s.trackMode = {kChargedHadron};
  // The charged candidate has the trackster energy 6 GeV; the neutral residual is set to 2 GeV.
  s.addCharged(0, 0, impactAt(10.f, 0.f));
  s.addNeutral(0);
  s.candidates[1].setP4(math::XYZTLorentzVector(0., 0., 2., 2.));

  auto par = nearestParameters();
  par.nearest = false;
  const auto fractions = s.fractions(par);
  REQUIRE(fractionOf(fractions[0], 0) == Catch::Approx(0.75f));
  REQUIRE(fractionOf(fractions[1], 0) == Catch::Approx(0.25f));
}

TEST_CASE("a track whose propagation fails takes no contested layer cluster and keeps its track",
          "[CandidateTracksters]") {
  Scene s;
  s.layerClusters = {layerCluster(4.f, 11.f, 0.f, 330.f), layerCluster(3.f, 18.f, 0.f, 330.f)};
  s.tracksters = {trackster({0, 1}, 7.f, 8.f)};
  s.tracks = {track(1.f, 0.f, 30.f), track(1.f, 0.f, 30.f)};
  s.trackToTrackster = {0, 0};
  s.trackMode = {kChargedHadron, kChargedHadron};
  s.addCharged(0, 0, impactAt(10.f, 0.f));
  s.addCharged(1, 0, ticl::TrackImpact{});

  const auto fractions = s.fractions(nearestParameters());
  // Layer cluster 0 is within the radius of track 0; layer cluster 1 goes to the nearest valid track.
  REQUIRE(fractionOf(fractions[0], 0) == 1.f);
  REQUIRE(fractionOf(fractions[0], 1) == 1.f);
  REQUIRE(fractions[1].empty());

  std::vector<int> candidateIndex;
  const auto built = ticl::tracksterOfFractions(fractions, candidateIndex);
  REQUIRE(candidateIndex == std::vector<int>{0});
  requireInvariant(s, candidateIndex, built);
  REQUIRE(s.candidates[1].tracksters().size() == 1);
  REQUIRE(s.candidates[1].tracksters()[0].key() == 0);
}

TEST_CASE("a track below 1 GeV has a positive budget", "[CandidateTracksters]") {
  Scene s;
  // The expected deposit of a 0.5 GeV hadron is 0.32 GeV, the budget 0.42 GeV: one 0.3 GeV layer cluster fits.
  s.layerClusters = {layerCluster(0.3f, 10.5f, 0.f, 330.f), layerCluster(0.3f, 9.5f, 0.f, 330.f)};
  s.tracksters = {trackster({0, 1}, 0.6f, 0.7f)};
  s.tracks = {track(0.f, 0.f, 0.5f)};
  s.trackToTrackster = {0};
  s.trackMode = {kChargedHadron};
  s.addCharged(0, 0, impactAt(10.f, 0.f));
  s.addNeutral(0);

  const auto par = nearestParameters(1.3f);
  REQUIRE(par.deposit(0.5f) == Catch::Approx(0.3211f).epsilon(1e-3));
  const auto fractions = s.fractions(par);
  REQUIRE(fractionOf(fractions[0], 0) + fractionOf(fractions[0], 1) == 1.f);
  REQUIRE(fractionOf(fractions[1], 0) + fractionOf(fractions[1], 1) == 1.f);
}

TEST_CASE("candidateSourceTracksters lists the held tracksters and the jet trackster of a jet member",
          "[CandidateTracksters]") {
  Scene s;
  s.tracksters = {trackster({}, 1.f, 1.f), trackster({}, 1.f, 1.f), trackster({}, 1.f, 1.f)};
  s.tracks = {track(1.f, 0.f, 30.f)};
  s.trackToTrackster = {2};
  s.trackMode = {static_cast<int>(ticl::TrackMode::kJetMember)};
  s.addCharged(0, 1, impactAt(10.f, 0.f));
  s.candidates[0].appendTrackster(edm::Ptr<ticl::Trackster>(&s.tracksters[1], 1));
  REQUIRE(ticl::candidateSourceTracksters(s.candidates[0], kTracksterId, s.trackToTrackster, s.trackMode) ==
          std::vector<unsigned int>{1, 2});
  s.trackMode = {kNotSelected};
  REQUIRE(ticl::candidateSourceTracksters(s.candidates[0], kTracksterId, s.trackToTrackster, s.trackMode) ==
          std::vector<unsigned int>{1});
}

TEST_CASE("copySourceProperties scales the regressed energy and copies the PID and the track indices",
          "[CandidateTracksters]") {
  ticl::Trackster source = trackster({0, 1}, 10.f, 12.f);
  std::array<float, 8> p{0.1f, 0.f, 0.f, 0.f, 0.7f, 0.2f, 0.f, 0.f};
  source.setProbabilities(p.data());
  source.addTrackIdx(7);
  source.addGSFTrackIdx(3);

  SECTION("one source") {
    ticl::Trackster merged = trackster({0}, 8.f, 0.f);
    ticl::copySourceProperties(merged, {&source});
    REQUIRE(merged.regressed_energy() == Catch::Approx(9.6f));
    for (unsigned int i = 0; i < p.size(); ++i)
      REQUIRE(merged.id_probabilities(i) == Catch::Approx(p[i]));
    REQUIRE(merged.trackIdxs() == std::vector<int>{7});
    REQUIRE(merged.gsftrackIdxs() == std::vector<int>{3});
  }
  SECTION("two sources") {
    ticl::Trackster other = trackster({2}, 30.f, 33.f);
    std::array<float, 8> q{0.5f, 0.f, 0.f, 0.f, 0.3f, 0.2f, 0.f, 0.f};
    other.setProbabilities(q.data());
    other.addTrackIdx(7);
    other.addTrackIdx(9);
    ticl::Trackster merged = trackster({0, 2}, 20.f, 0.f);
    ticl::copySourceProperties(merged, {&source, &other});
    // (12 + 33) x 20 / 40
    REQUIRE(merged.regressed_energy() == Catch::Approx(22.5f));
    REQUIRE(merged.id_probabilities(0) == Catch::Approx(0.4f));
    REQUIRE(merged.id_probabilities(4) == Catch::Approx(0.4f));
    REQUIRE(merged.trackIdxs() == std::vector<int>{7, 9});
  }
  SECTION("no source with raw energy") {
    ticl::Trackster merged = trackster({0}, 8.f, 0.f);
    ticl::Trackster empty;
    ticl::copySourceProperties(merged, {&empty});
    REQUIRE(merged.regressed_energy() == 0.f);
  }
}
