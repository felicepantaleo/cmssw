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
