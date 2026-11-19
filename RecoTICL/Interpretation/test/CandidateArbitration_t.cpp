#include <array>
#include <cmath>
#include <map>
#include <vector>

#include <catch2/catch_all.hpp>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoTICL/Interpretation/interface/AssignmentMaps.h"
#include "RecoTICL/Interpretation/interface/CandidateArbitration.h"
#include "RecoTICL/Interpretation/interface/CandidateTracksters.h"
#include "RecoTICL/Interpretation/interface/TrackImpact.h"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {
  constexpr float kPionMass = 0.13957f;
  const ticl::TracksterResolution kResolution{0.30f, 0.22f, 2.32f, 0.f};

  ticl::Trackster trackster(float regressedEnergy, bool hadronic) {
    ticl::Trackster ts;
    ts.setRegressedEnergy(regressedEnergy);
    std::array<float, 8> p{};
    p[static_cast<int>(hadronic ? ticl::Trackster::ParticleType::charged_hadron
                                : ticl::Trackster::ParticleType::photon)] = 1.f;
    ts.setProbabilities(p.data());
    return ts;
  }

  ticl::ChargedOnTrackster charged(std::size_t candidate, float p, float sigmaP, float claimEnergy = 0.f) {
    return {candidate, std::sqrt(p * p + kPionMass * kPionMass), sigmaP * sigmaP, false, claimEnergy, 0.f};
  }

  reco::Track track(float px, float py, float pz, int charge) {
    return reco::Track(1.,
                       10.,
                       reco::Track::Point(0., 0., 0.),
                       reco::Track::Vector(px, py, pz),
                       charge,
                       reco::Track::CovarianceMatrix());
  }

  reco::GsfTrack gsfTrack(float px, float py, float pz, int charge) {
    return reco::GsfTrack(1.,
                          10.,
                          reco::GsfTrack::Point(0., 0., 0.),
                          reco::GsfTrack::Vector(px, py, pz),
                          charge,
                          reco::GsfTrack::CovarianceMatrix());
  }
}  // namespace

TEST_CASE("combinedEnergy: inverse-variance mean when compatible, else the track energy", "[CandidateArbitration]") {
  const auto em = trackster(10.f, false);
  // sigma(trackster) at 10 GeV: 10 sqrt(0.09 / 10 + 0.0484) = 2.396.
  const float sigmaTs = kResolution.error(false, 10.f);
  REQUIRE_THAT(sigmaTs, WithinRel(2.396f, 1e-3f));
  const auto close = ticl::combinedEnergy(10.f, 1.f, 0.f, em, kResolution, 3.f);
  CHECK(close.compatible);
  const float w1 = 1.f, w2 = 1.f / (sigmaTs * sigmaTs);
  CHECK_THAT(close.energy, WithinRel((w1 * 10.f + w2 * 10.f) / (w1 + w2), 1e-5f));
  const auto far = ticl::combinedEnergy(50.f, 0.5f, 0.f, em, kResolution, 3.f);
  CHECK_FALSE(far.compatible);
  CHECK(far.energy == 50.f);
  // Zero resolution terms give the trackster energy, not a division by zero.
  const ticl::TracksterResolution exact{0.f, 0.f, 0.f, 0.f};
  const auto zero = ticl::combinedEnergy(10.5f, 1.f, 0.f, em, exact, 3.f);
  REQUIRE(zero.compatible);
  CHECK_THAT(zero.energy, WithinAbs(10.f, 1e-4f));
}

TEST_CASE("calibratedEnergy: E x q50 and a positive error for any quantile order", "[CandidateArbitration]") {
  const std::array<float, 3> ordered{0.8f, 1.1f, 1.4f};
  const auto e = ticl::calibratedEnergy(20.f, ordered);
  CHECK_THAT(e.energy, WithinRel(22.f, 1e-6f));
  CHECK_THAT(e.sigma, WithinRel(6.f, 1e-5f));
  const std::array<float, 3> crossed{1.4f, 1.1f, 0.8f};
  CHECK_THAT(ticl::calibratedEnergy(20.f, crossed).sigma, WithinRel(6.f, 1e-5f));
}

TEST_CASE("sharedTracksterEnergy: the claim energies add to the trackster energy", "[CandidateArbitration]") {
  const auto had = trackster(12.f, true);
  const std::vector<ticl::ChargedOnTrackster> tracks{charged(0, 5.f, 0.1f, 2.f), charged(1, 4.f, 0.1f, 1.f)};
  const auto plain = ticl::sharedTracksterEnergy(had, std::nullopt, tracks, kResolution);
  CHECK_THAT(plain.energy, WithinRel(15.f, 1e-6f));
  CHECK_THAT(plain.sigma, WithinRel(kResolution.error(true, tracks[0].energy + tracks[1].energy), 1e-6f));
  auto withVariance = tracks;
  withVariance[0].claimVariance = 9.f;
  const auto calibrated = ticl::sharedTracksterEnergy(had, ticl::EnergyWithError{13.f, 4.f}, withVariance, kResolution);
  CHECK_THAT(calibrated.energy, WithinRel(16.f, 1e-6f));
  CHECK_THAT(calibrated.sigma, WithinRel(5.f, 1e-6f));
}

TEST_CASE("energy closure: the charged energies and the residual sum to the regressed energy plus the claims",
          "[CandidateArbitration]") {
  // Track 0 has a claim trackster of 1.5 GeV. The calibrated energy of the trackster estimates the charged part only.
  const std::vector<ticl::ChargedOnTrackster> tracks{charged(0, 5.f, 0.5f, 1.5f), charged(1, 4.f, 1.f)};
  const float eSum = tracks[0].energy + tracks[1].energy;

  SECTION("compatible: the tracks share the calibrated energy, the residual takes the rest of the total") {
    const auto had = trackster(12.f, true);
    const auto combination = ticl::sharedTracksterEnergy(had, ticl::EnergyWithError{9.5f, 2.f}, tracks, kResolution);
    REQUIRE_THAT(combination.energy, WithinRel(11.f, 1e-6f));
    const auto shared = ticl::shareTracksterEnergy(tracks, combination, 3.f, kPionMass);
    REQUIRE(shared.combined);
    // The difference goes to the tracks in proportion to their variances.
    const float diff = combination.energy - eSum;
    const float varTotal = tracks[0].variance + tracks[1].variance + combination.sigma * combination.sigma;
    CHECK_THAT(shared.energies[0], WithinRel(tracks[0].energy + tracks[0].variance / varTotal * diff, 1e-5f));
    const ticl::TracksterEnergyBalance balance{ticl::sharedTracksterTotalEnergy(had, tracks),
                                               shared.energies[0] + shared.energies[1]};
    CHECK_THAT(balance.energy, WithinRel(13.5f, 1e-6f));
    const auto residual = ticl::neutralResidualEnergy(balance, 2.f, 0.1f);
    REQUIRE(residual.has_value());
    // The tracks take eSum + diff x (track variances) / varTotal.
    CHECK_THAT(*residual,
               WithinAbs(13.5f - eSum - diff * (varTotal - combination.sigma * combination.sigma) / varTotal, 1e-4f));
    CHECK_THAT(balance.given + *residual, WithinAbs(12.f + 1.5f, 1e-4f));
  }
  SECTION("incompatible: the tracks keep their energies and the residual takes the rest of the total") {
    const auto had = trackster(40.f, true);
    const auto combination = ticl::sharedTracksterEnergy(had, ticl::EnergyWithError{42.f, 2.f}, tracks, kResolution);
    const auto shared = ticl::shareTracksterEnergy(tracks, combination, 3.f, kPionMass);
    REQUIRE_FALSE(shared.combined);
    CHECK(shared.energies[0] == tracks[0].energy);
    CHECK(shared.energies[1] == tracks[1].energy);
    const ticl::TracksterEnergyBalance balance{ticl::sharedTracksterTotalEnergy(had, tracks),
                                               shared.energies[0] + shared.energies[1]};
    const auto residual = ticl::neutralResidualEnergy(balance, 2.f, 0.1f);
    REQUIRE(residual.has_value());
    CHECK_THAT(balance.given + *residual, WithinAbs(40.f + 1.5f, 1e-4f));
  }
  SECTION("no combination when a track would fall below the pion mass") {
    const std::vector<ticl::ChargedOnTrackster> soft{charged(0, 0.2f, 1.f), charged(1, 6.f, 0.01f)};
    const auto shared = ticl::shareTracksterEnergy(soft, {4.f, 0.5f}, 3.f, kPionMass);
    CHECK_FALSE(shared.combined);
  }
}

TEST_CASE("neutralResidualEnergy: at least max(floor, fraction x energy)", "[CandidateArbitration]") {
  CHECK_FALSE(ticl::neutralResidualEnergy({10.f, 8.5f}, 2.f, 0.1f).has_value());
  CHECK_THAT(ticl::neutralResidualEnergy({10.f, 8.f}, 2.f, 0.1f).value(), WithinAbs(2.f, 1e-6f));
  // The fraction wins above 20 GeV.
  CHECK_FALSE(ticl::neutralResidualEnergy({50.f, 46.f}, 2.f, 0.1f).has_value());
  CHECK(ticl::neutralResidualEnergy({50.f, 44.f}, 2.f, 0.1f).has_value());
}

TEST_CASE("neutralEnergyScale: the share below shareMinPt, and no candidate below minEnergy",
          "[CandidateArbitration]") {
  CHECK(ticl::neutralEnergyScale(30.f, 12.f, 0.5f, 10.f, 1.f).value() == 1.f);
  CHECK(ticl::neutralEnergyScale(30.f, 8.f, 0.5f, 10.f, 1.f).value() == 0.5f);
  CHECK(ticl::neutralEnergyScale(30.f, 12.f, 0.5f, 0.f, 1.f).value() == 0.5f);
  CHECK_FALSE(ticl::neutralEnergyScale(1.5f, 1.f, 0.5f, 10.f, 1.f).has_value());
  CHECK_FALSE(ticl::neutralEnergyScale(5.f, 1.f, 0.f, 10.f, 0.f).has_value());
}

TEST_CASE("matchGsfTrack: an e+e- pair keeps the charge of each track", "[CandidateArbitration]") {
  // Two tracks of opposite charge 0.01 apart in phi. The positive GSF track is nearer to the negative track.
  const auto electron = track(10.f, 0.f, 30.f, -1);
  const auto positron = track(10.f * std::cos(0.01f), 10.f * std::sin(0.01f), 30.f, 1);
  const std::vector<reco::GsfTrack> gsf{gsfTrack(10.f * std::cos(0.004f), 10.f * std::sin(0.004f), 30.f, 1),
                                        gsfTrack(10.f * std::cos(-0.008f), 10.f * std::sin(-0.008f), 30.f, -1)};
  std::vector<bool> used(gsf.size(), false);
  const int forElectron = ticl::matchGsfTrack(electron, gsf, used, 0.05f);
  REQUIRE(forElectron == 1);
  used[forElectron] = true;
  CHECK(ticl::matchGsfTrack(positron, gsf, used, 0.05f) == 0);
  // A used GSF track and a GSF track outside the window are not matched.
  used[0] = true;
  CHECK(ticl::matchGsfTrack(positron, gsf, used, 0.05f) == -1);
  CHECK(ticl::matchGsfTrack(track(0.f, 10.f, 30.f, 1), gsf, std::vector<bool>(gsf.size(), false), 0.05f) == -1);
}

TEST_CASE("a candidate without layer clusters holds no trackster and the raw energy counts once",
          "[CandidateArbitration]") {
  // A hadron of p = 10 GeV and a 4 GeV residual hold trackster 0 of 9.5 GeV raw energy. The budget 1.3 x 7.6 GeV
  // gives every layer cluster to the hadron; the residual gets none.
  const edm::ProductID tracksterId;
  const edm::ProductID rebuiltId(1, 2);
  std::vector<reco::CaloCluster> lcs{reco::CaloCluster(5.f, math::XYZPoint(11.f, 0.f, 330.f)),
                                     reco::CaloCluster(4.5f, math::XYZPoint(12.f, 0.f, 335.f))};
  ticl::Trackster ts;
  ts.vertices() = {0, 1};
  ts.vertex_multiplicity() = {1.f, 1.f};
  ts.setRawEnergy(9.5f);
  ts.setRegressedEnergy(14.f);
  ts.setBarycenter({11.f, 0.f, 332.f});
  std::vector<ticl::Trackster> tracksters{ts};
  std::vector<reco::Track> tracks{track(1.f, 0.f, 9.95f, 1)};
  const std::vector<int> trackToTrackster{0};
  const std::vector<int> trackMode{static_cast<int>(ticl::TrackMode::kChargedHadron)};
  std::vector<TICLCandidate> candidates;
  candidates.emplace_back(edm::Ptr<reco::Track>(&tracks[0], 0), edm::Ptr<ticl::Trackster>(&tracksters[0], 0));
  candidates.emplace_back(edm::Ptr<reco::Track>(), edm::Ptr<ticl::Trackster>(&tracksters[0], 0));
  candidates[0].setPdgId(211);
  candidates[1].setP4(math::XYZTLorentzVector(0., 0., 4., 4.));
  const std::vector<ticl::TrackImpact> impacts{
      ticl::TrackImpact{GlobalPoint(11.f, 0.f, 320.f), GlobalVector(0.f, 0.f, 1.f), true}, ticl::TrackImpact{}};
  ticl::CandidateTrackstersParameters par;
  par.budgetScale = 1.3f;
  par.deposit = {0.67f, 0.04f, 0.f, 1.f};
  const auto fractions = ticl::candidateLayerClusterFractions(
      candidates,
      tracksterId,
      tracksters,
      trackToTrackster,
      trackMode,
      lcs,
      [&](unsigned int c) -> const ticl::TrackImpact & { return impacts[c]; },
      par);
  std::vector<int> candidateIndex;
  const auto built = ticl::tracksterOfFractions(fractions, candidateIndex);
  REQUIRE(candidateIndex == std::vector<int>{0});
  std::vector<float> rawEnergies;
  for (auto const &b : built) {
    float raw = 0.f;
    for (unsigned int k = 0; k < b.vertices().size(); ++k)
      raw += lcs[b.vertices()[k]].energy() / b.vertex_multiplicity()[k];
    rawEnergies.push_back(raw);
  }
  ticl::setCandidateTracksters(candidates, candidateIndex, rawEnergies, [&](std::size_t i) {
    return edm::Ptr<ticl::Trackster>(rebuiltId, &built[i], i);
  });

  CHECK(candidates[0].tracksters().size() == 1);
  CHECK(candidates[1].tracksters().empty());
  CHECK(candidates[1].rawEnergy() == 0.f);
  // A candidate without trackster keeps its raw energy.
  TICLCandidate trackOnly(edm::Ptr<reco::Track>(&tracks[0], 0), edm::Ptr<ticl::Trackster>());
  const float trackOnlyRaw = trackOnly.rawEnergy();
  std::vector<TICLCandidate> alone{trackOnly};
  ticl::setCandidateTracksters(alone, {}, {}, [](std::size_t) { return edm::Ptr<ticl::Trackster>(); });
  CHECK(alone[0].rawEnergy() == trackOnlyRaw);
  // Every trackster Ptr is in the rebuilt product, each layer cluster is held at most once, and the raw energy of the
  // candidates is not above the layer-cluster energy.
  std::map<unsigned int, float> held;
  float rawSum = 0.f;
  for (auto const &cand : candidates) {
    rawSum += cand.rawEnergy();
    for (auto const &ptr : cand.tracksters()) {
      CHECK(ptr.id() == rebuiltId);
      for (unsigned int k = 0; k < ptr->vertices().size(); ++k)
        held[ptr->vertices()[k]] += 1.f / ptr->vertex_multiplicity()[k];
    }
  }
  for (auto const &[lc, f] : held)
    CHECK(f <= 1.f + 1e-6f);
  CHECK(rawSum <= 9.5f + 1e-4f);
  // The residual without layer clusters is removed: every candidate holds a track or a trackster.
  CHECK(ticl::removeEmptyCandidates(candidates) == std::vector<std::size_t>{0});
  REQUIRE(candidates.size() == 1);
  for (auto const &cand : candidates)
    CHECK((cand.trackPtr().isNonnull() || !cand.tracksters().empty()));
}

TEST_CASE("removeEmptyCandidates keeps the candidates with a track or a trackster", "[CandidateArbitration]") {
  std::vector<reco::Track> tracks{track(1.f, 0.f, 10.f, 1)};
  std::vector<ticl::Trackster> tracksters{trackster(5.f, false)};
  std::vector<TICLCandidate> candidates;
  candidates.emplace_back(edm::Ptr<reco::Track>(), edm::Ptr<ticl::Trackster>(&tracksters[0], 0));
  candidates.emplace_back(edm::Ptr<reco::Track>(&tracks[0], 0), edm::Ptr<ticl::Trackster>());
  candidates.emplace_back(edm::Ptr<reco::Track>(), edm::Ptr<ticl::Trackster>(&tracksters[0], 0));
  candidates[2].setTracksters({});
  candidates[1].setPdgId(211);
  CHECK(ticl::removeEmptyCandidates(candidates) == std::vector<std::size_t>{0, 1});
  REQUIRE(candidates.size() == 2);
  CHECK(candidates[0].trackPtr().isNull());
  CHECK(candidates[1].pdgId() == 211);
}
