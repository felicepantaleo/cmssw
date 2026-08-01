// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"

namespace {

  // Minimal CSR graph builder, the same one the other tests in this package use.
  struct GraphBuilder {
    explicit GraphBuilder(uint32_t nParticles, uint32_t nVertices) {
      graph.particles().resize(nParticles);
      graph.vertices().resize(nVertices);
    }
    void addDecay(uint32_t particleId, uint32_t vertexId) {
      d2v.emplace_back(particleId, vertexId);
      v2i.emplace_back(vertexId, particleId);
    }
    void addProduction(uint32_t vertexId, uint32_t particleId) {
      v2o.emplace_back(vertexId, particleId);
      p2v.emplace_back(particleId, vertexId);
    }
    static void csr(uint32_t n,
                    std::vector<std::pair<uint32_t, uint32_t>>& pairs,
                    std::vector<uint32_t>& off,
                    std::vector<uint32_t>& flat) {
      std::sort(pairs.begin(), pairs.end());
      pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
      off.assign(n + 1, 0);
      for (auto const& pr : pairs)
        ++off[pr.first + 1];
      for (uint32_t i = 1; i <= n; ++i)
        off[i] += off[i - 1];
      flat.assign(pairs.size(), 0);
      auto cur = off;
      for (auto const& pr : pairs)
        flat[cur[pr.first]++] = pr.second;
    }
    truth::Graph finish() {
      csr(graph.nParticles(), d2v, graph.particleToDecayVertexOffsets(), graph.particleToDecayVertices());
      csr(graph.nParticles(), p2v, graph.particleToProductionVertexOffsets(), graph.particleToProductionVertices());
      csr(graph.nVertices(), v2o, graph.vertexToOutgoingParticleOffsets(), graph.vertexToOutgoingParticles());
      csr(graph.nVertices(), v2i, graph.vertexToIncomingParticleOffsets(), graph.vertexToIncomingParticles());
      CPPUNIT_ASSERT(graph.isConsistent());
      return graph;
    }
    truth::Graph graph;
    std::vector<std::pair<uint32_t, uint32_t>> d2v, p2v, v2o, v2i;
  };

  // A tau decaying to a pion and a neutrino, the pion reaching the calorimeter.
  //   p0  tau, isHardProcess, decays at v0
  //   p1  pi+, status 1, records a calorimeter boundary crossing
  //   p2  nu,  status 1, no crossing
  truth::Graph buildDecay() {
    GraphBuilder b(3, 1);

    auto& tau = b.graph.particles()[0];
    tau.genNode = 100;
    tau.pdgId = 15;
    tau.status = 2;
    tau.statusFlags = truth::detail::kIsHardProcess;
    tau.momentum = math::XYZTLorentzVectorD(50., 0., 0., 60.);

    auto& pion = b.graph.particles()[1];
    pion.genNode = 101;
    pion.simNode = 201;
    pion.pdgId = 211;
    pion.status = 1;
    pion.momentum = math::XYZTLorentzVectorD(30., 0., 0., 35.);
    pion.checkpoints.push_back(truth::Checkpoint{});

    auto& nu = b.graph.particles()[2];
    nu.genNode = 102;
    nu.pdgId = 16;
    nu.status = 1;
    nu.momentum = math::XYZTLorentzVectorD(5., 0., 0., 5.);

    b.addDecay(0, 0);
    b.addProduction(0, 1);
    b.addProduction(0, 2);
    return b.finish();
  }

}  // namespace

class LevelFlags_t : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(LevelFlags_t);
  CPPUNIT_TEST(testFitsInThePaddingHole);
  CPPUNIT_TEST(testFlagsMatchTheAntichain);
  CPPUNIT_TEST(testIdempotent);
  CPPUNIT_TEST_SUITE_END();

public:
  // REQUIRED: the flags word occupies the alignment hole between genEvent and momentum,
  // so carrying it costs no memory. A change that grows ParticleData past 96 bytes has
  // moved it out of the hole and needs to be justified, not absorbed silently.
  void testFitsInThePaddingHole() { CPPUNIT_ASSERT_EQUAL(std::size_t{96}, sizeof(truth::ParticleData)); }

  // REQUIRED: a stored flag says exactly what levelAntichain() would say. This is the
  // defence against a graph written before a level definition changed, which is
  // indistinguishable from a fresh one by inspection.
  void testFlagsMatchTheAntichain() {
    truth::Graph g = buildDecay();
    truth::fillLevelFlags(g);

    for (const truth::Level level : truth::kAllLevels) {
      std::vector<bool> expected(g.nParticles(), false);
      for (const uint32_t id : truth::levelAntichain(g, level)) {
        expected[id] = true;
      }
      const truth::LevelFlag flag = truth::levelFlagOf(level);
      for (uint32_t id = 0; id < g.nParticles(); ++id) {
        CPPUNIT_ASSERT_EQUAL(static_cast<bool>(expected[id]), g.particles()[id].isAtLevel(flag));
      }
    }

    // The physics the sample encodes, so a passing test means the right thing and not
    // merely a self-consistent one: the tau is the hard process, the pion and the
    // neutrino are the stable decay products, only the pion reaches the calorimeter.
    CPPUNIT_ASSERT(g.particles()[0].isAtLevel(truth::LevelFlag::HardProcess));
    CPPUNIT_ASSERT(!g.particles()[1].isAtLevel(truth::LevelFlag::HardProcess));
    CPPUNIT_ASSERT(g.particles()[1].isAtLevel(truth::LevelFlag::StableDecayProducts));
    CPPUNIT_ASSERT(g.particles()[2].isAtLevel(truth::LevelFlag::StableDecayProducts));
    CPPUNIT_ASSERT(g.particles()[1].isAtLevel(truth::LevelFlag::CaloBoundary));
    CPPUNIT_ASSERT(!g.particles()[2].isAtLevel(truth::LevelFlag::CaloBoundary));
  }

  // REQUIRED: filling twice leaves the same answer as filling once, so a graph that
  // passes through the stamp again cannot accumulate membership it no longer has.
  void testIdempotent() {
    truth::Graph g = buildDecay();
    truth::fillLevelFlags(g);
    std::vector<uint32_t> once;
    for (auto const& p : g.particles()) {
      once.push_back(p.levelFlags);
    }
    truth::fillLevelFlags(g);
    for (uint32_t id = 0; id < g.nParticles(); ++id) {
      CPPUNIT_ASSERT_EQUAL(once[id], g.particles()[id].levelFlags);
    }
  }
};

CPPUNIT_TEST_SUITE_REGISTRATION(LevelFlags_t);
