#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include <catch2/catch_all.hpp>

#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "Validation/TruthInfo/interface/CandidateTruthMatching.h"

using namespace truth::candidate;

namespace {
  void csr(uint32_t n,
           std::vector<std::pair<uint32_t, uint32_t>> pairs,
           std::vector<uint32_t>& offsets,
           std::vector<uint32_t>& flat) {
    std::sort(pairs.begin(), pairs.end());
    offsets.assign(n + 1, 0);
    for (auto const& pr : pairs)
      ++offsets[pr.first + 1];
    for (uint32_t i = 1; i <= n; ++i)
      offsets[i] += offsets[i - 1];
    flat.assign(pairs.size(), 0);
    auto cursor = offsets;
    for (auto const& pr : pairs)
      flat[cursor[pr.first]++] = pr.second;
  }

  // p0 tau decays at v0 to p1 pi+, p2 pi0 and p3 nu; p2 decays at v1 to the photons p4 and p5; p1 interacts at v2
  // and makes p6; p7 is an unrelated pion.
  constexpr uint32_t kTau = 0, kPion = 1, kPi0 = 2, kNu = 3, kPhoton1 = 4, kPhoton2 = 5, kSecondary = 6, kOther = 7;

  truth::Graph tauGraph() {
    truth::Graph g;
    g.particles().resize(8);
    g.vertices().resize(3);
    const std::vector<std::pair<uint32_t, uint32_t>> decays{{kTau, 0}, {kPi0, 1}, {kPion, 2}};
    const std::vector<std::pair<uint32_t, uint32_t>> productions{
        {kPion, 0}, {kPi0, 0}, {kNu, 0}, {kPhoton1, 1}, {kPhoton2, 1}, {kSecondary, 2}};
    std::vector<std::pair<uint32_t, uint32_t>> incoming, outgoing;
    for (auto const& [p, v] : decays)
      incoming.emplace_back(v, p);
    for (auto const& [p, v] : productions)
      outgoing.emplace_back(v, p);
    csr(8, decays, g.particleToDecayVertexOffsets(), g.particleToDecayVertices());
    csr(8, productions, g.particleToProductionVertexOffsets(), g.particleToProductionVertices());
    csr(3, outgoing, g.vertexToOutgoingParticleOffsets(), g.vertexToOutgoingParticles());
    csr(3, incoming, g.vertexToIncomingParticleOffsets(), g.vertexToIncomingParticles());
    return g;
  }
}  // namespace

TEST_CASE("A particle projects on the member it descends from", "[CandidateTruthMatching]") {
  const auto g = tauGraph();
  REQUIRE(g.isConsistent());

  SECTION("the pi0 is a member") {
    const std::vector<unsigned int> members{kPion, kPi0};
    const LevelProjection projection(g, members);
    CHECK(projection.memberSlot(kPion) == 0);
    CHECK(projection.memberSlot(kSecondary) == 0);
    CHECK(projection.memberSlot(kPi0) == 1);
    CHECK(projection.memberSlot(kPhoton2) == 1);
    CHECK(projection.memberSlot(kTau) == LevelProjection::kNoMember);
    CHECK(projection.isAbove(kTau));
    CHECK_FALSE(projection.isAbove(kPi0));
    CHECK(projection.memberSlot(kOther) == LevelProjection::kNoMember);
    CHECK_FALSE(projection.isAbove(kOther));
  }

  SECTION("the photons are the members") {
    const std::vector<unsigned int> members{kPion, kPhoton1, kPhoton2};
    const LevelProjection projection(g, members);
    CHECK(projection.memberSlot(kPhoton2) == 2);
    CHECK(projection.memberSlot(kPi0) == LevelProjection::kNoMember);
    CHECK(projection.isAbove(kPi0));
    CHECK(projection.isAbove(kTau));
    CHECK_FALSE(projection.isAbove(kNu));
  }
}

TEST_CASE("An ancestor is found among sorted ids", "[CandidateTruthMatching]") {
  const auto g = tauGraph();
  const std::vector<uint32_t> pi0{kPi0}, pion{kPion}, tauAndOther{kTau, kOther};
  CHECK(hasAncestorIn(g, kPhoton1, pi0));
  CHECK_FALSE(hasAncestorIn(g, kPhoton1, pion));
  CHECK(hasAncestorIn(g, kPhoton1, tauAndOther));
  CHECK_FALSE(hasAncestorIn(g, kPi0, pi0));
  CHECK_FALSE(hasAncestorIn(g, kOther, tauAndOther));
}

TEST_CASE("Truth and candidate classes", "[CandidateTruthMatching]") {
  CHECK(truthClass(111, 0.) == ParticleClass::Photon);
  CHECK(truthClass(22, 0.) == ParticleClass::Photon);
  CHECK(truthClass(-11, 1.) == ParticleClass::Electron);
  CHECK(truthClass(13, -1.) == ParticleClass::Muon);
  CHECK(truthClass(-211, -1.) == ParticleClass::ChargedHadron);
  CHECK(truthClass(130, 0.) == ParticleClass::NeutralHadron);
  CHECK(truthClass(2112, 0.) == ParticleClass::NeutralHadron);
  CHECK(truthClass(15, -1.) == ParticleClass::Other);
  CHECK(candidateClass(-211) == ParticleClass::ChargedHadron);
  CHECK(candidateClass(130) == ParticleClass::NeutralHadron);
  CHECK(candidateClass(22) == ParticleClass::Photon);
  CHECK(candidateClass(11) == ParticleClass::Electron);
  CHECK(candidateClass(-13) == ParticleClass::Muon);
  CHECK(candidateClass(0) == ParticleClass::Other);
}

TEST_CASE("The ladder is read on the representative candidate", "[CandidateTruthMatching]") {
  const LadderConfig ladder{0.7, 1.3};

  SECTION("no candidate") {
    const auto out = evaluateMember(ParticleClass::ChargedHadron, 10., {}, ladder);
    CHECK_FALSE(out.reconstructed);
    CHECK(out.representative == -1);
  }

  SECTION("a charged hadron with its own track, class and energy") {
    const std::vector<MemberCandidate> cands{{0.8f, ParticleClass::ChargedHadron, true, true, 9.}};
    const auto out = evaluateMember(ParticleClass::ChargedHadron, 10., cands, ladder);
    CHECK(out.reconstructed);
    CHECK_FALSE(out.split);
    CHECK(out.trackCorrect);
    CHECK(out.pidCorrect);
    CHECK(out.energyCorrect);
  }

  SECTION("a charged hadron with the track of another particle") {
    const std::vector<MemberCandidate> cands{{0.8f, ParticleClass::ChargedHadron, true, false, 9.}};
    const auto out = evaluateMember(ParticleClass::ChargedHadron, 10., cands, ladder);
    CHECK(out.reconstructed);
    CHECK_FALSE(out.trackCorrect);
    CHECK_FALSE(out.pidCorrect);
  }

  SECTION("a photon with a track") {
    const std::vector<MemberCandidate> cands{{0.9f, ParticleClass::Electron, true, false, 10.}};
    const auto out = evaluateMember(ParticleClass::Photon, 10., cands, ladder);
    CHECK_FALSE(out.trackCorrect);
  }

  SECTION("a photon with the class of a neutral hadron") {
    const std::vector<MemberCandidate> cands{{0.9f, ParticleClass::NeutralHadron, false, false, 10.}};
    const auto out = evaluateMember(ParticleClass::Photon, 10., cands, ladder);
    CHECK(out.trackCorrect);
    CHECK_FALSE(out.pidCorrect);
  }

  SECTION("a split member reads the candidate that shares most of it") {
    const std::vector<MemberCandidate> cands{{0.3f, ParticleClass::Photon, false, false, 3.},
                                             {0.6f, ParticleClass::Photon, false, false, 6.}};
    const auto out = evaluateMember(ParticleClass::Photon, 10., cands, ladder);
    CHECK(out.split);
    CHECK(out.representative == 1);
    CHECK(out.pidCorrect);
    CHECK_FALSE(out.energyCorrect);
    CHECK(out.energySum == 9.);
  }

  SECTION("for a charged member the candidate with its track is the representative") {
    // A track-only charged candidate and the neutral candidate that took the trackster.
    const std::vector<MemberCandidate> cands{{0.9f, ParticleClass::NeutralHadron, false, false, 4.},
                                             {0.f, ParticleClass::ChargedHadron, true, true, 10.}};
    const auto out = evaluateMember(ParticleClass::ChargedHadron, 10., cands, ladder);
    CHECK(out.representative == 1);
    CHECK(out.split);
    CHECK(out.energyCorrect);
  }

  SECTION("for a neutral member the largest share is the representative, whatever the tracks") {
    // A photon that converted: an electron candidate with a leg track and a photon candidate.
    const std::vector<MemberCandidate> cands{{0.3f, ParticleClass::Electron, true, true, 3.},
                                             {0.6f, ParticleClass::Photon, false, false, 7.}};
    const auto out = evaluateMember(ParticleClass::Photon, 10., cands, ladder);
    CHECK(out.representative == 1);
    CHECK(out.pidCorrect);
    CHECK(out.energyCorrect);
  }
}
