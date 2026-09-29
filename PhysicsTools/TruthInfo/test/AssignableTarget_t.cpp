// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "PhysicsTools/TruthInfo/interface/AssignableTarget.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "PhysicsTools/TruthInfo/test/TestGraphBuilder.h"

namespace {

  using GraphBuilder = truth::test::GraphBuilder;

  // One interaction with everything the rule has to separate.
  //   p0  beam proton, no production vertex, decays at v0 (the hard scatter)
  //   p1  gluon from v0, decays at v1
  //   p2  Z from v0, decays at v2
  //   p3  pi0 from v1, decays at v3
  //   p4  photon from v3, stable
  //   p5  muon from v2, stable
  //   p6  connector, produced at the artificial InitialState vertex v4
  //   p7  pi+ from v1, stable, produced at a normal vertex like p3
  //   p8  muon produced at v4, a gun particle whose own production vertex the preset dropped
  //   p9  pi+ produced at the artificial UnderlyingEvent vertex v5, a stable spectator
  constexpr uint32_t kBeam = 0, kGluon = 1, kZ = 2, kPi0 = 3, kPhoton = 4, kMuon = 5, kConnector = 6, kPion = 7,
                     kGunSeed = 8, kSpectator = 9;

  truth::Graph buildInteraction() {
    GraphBuilder b(10, 6);
    b.graph.particles()[kBeam].pdgId = 2212;
    b.graph.particles()[kGluon].pdgId = 21;
    b.graph.particles()[kZ].pdgId = 23;
    b.graph.particles()[kPi0].pdgId = 111;
    b.graph.particles()[kPhoton].pdgId = 22;
    b.graph.particles()[kMuon].pdgId = 13;
    b.graph.particles()[kConnector].pdgId = 0;
    b.graph.particles()[kConnector].role = static_cast<uint8_t>(truth::ParticleRole::Connector);
    b.graph.particles()[kPion].pdgId = 211;
    b.graph.particles()[kGunSeed].pdgId = -13;
    b.graph.particles()[kSpectator].pdgId = 211;
    b.graph.vertices()[4].role = static_cast<uint8_t>(truth::VertexRole::InitialState);
    b.graph.vertices()[5].role = static_cast<uint8_t>(truth::VertexRole::UnderlyingEvent);

    b.addDecay(kBeam, 0);
    b.addProduction(0, kGluon);
    b.addProduction(0, kZ);
    b.addDecay(kGluon, 1);
    b.addProduction(1, kPi0);
    b.addProduction(1, kPion);
    b.addDecay(kZ, 2);
    b.addProduction(2, kMuon);
    b.addDecay(kPi0, 3);
    b.addProduction(3, kPhoton);
    b.addProduction(4, kConnector);
    b.addProduction(4, kGunSeed);
    b.addProduction(5, kSpectator);
    return b.finish();
  }

}  // namespace

class TestAssignableTarget : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestAssignableTarget);
  CPPUNIT_TEST(testDetectorParticlesAreAssignable);
  CPPUNIT_TEST(testBookkeepingNodesAreNot);
  CPPUNIT_TEST(testEachClauseCanBeTurnedOff);
  CPPUNIT_TEST(testExtraBarredPdgIdsCoverBothSigns);
  CPPUNIT_TEST(testArtificialSourceKeepsRealParticles);
  CPPUNIT_TEST_SUITE_END();

public:
  void testDetectorParticlesAreAssignable();
  void testBookkeepingNodesAreNot();
  void testEachClauseCanBeTurnedOff();
  void testExtraBarredPdgIdsCoverBothSigns();
  void testArtificialSourceKeepsRealParticles();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestAssignableTarget);

void TestAssignableTarget::testDetectorParticlesAreAssignable() {
  const auto graph = buildInteraction();
  const truth::AssignableTargetConfig config;
  // The merged-pi0 case the adaptive search exists for: the pi0 is reached by crossing a
  // decay vertex and stays a valid answer, as do its photon and the other final state.
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kPi0, config));
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kPhoton, config));
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kMuon, config));
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kPion, config));
}

void TestAssignableTarget::testBookkeepingNodesAreNot() {
  const auto graph = buildInteraction();
  const truth::AssignableTargetConfig config;
  CPPUNIT_ASSERT(!truth::isAssignableTarget(graph, kBeam, config));
  CPPUNIT_ASSERT(!truth::isAssignableTarget(graph, kGluon, config));
  CPPUNIT_ASSERT(!truth::isAssignableTarget(graph, kZ, config));
  CPPUNIT_ASSERT(!truth::isAssignableTarget(graph, kConnector, config));
  // A particle id past the end is not a target rather than an out-of-range read.
  CPPUNIT_ASSERT(!truth::isAssignableTarget(graph, graph.nParticles(), config));
}

void TestAssignableTarget::testEachClauseCanBeTurnedOff() {
  const auto graph = buildInteraction();
  truth::AssignableTargetConfig config;
  config.excludeBeamParticles = false;
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kBeam, config));
  CPPUNIT_ASSERT(!truth::isAssignableTarget(graph, kGluon, config));

  config = truth::AssignableTargetConfig();
  config.excludePartons = false;
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kGluon, config));

  config = truth::AssignableTargetConfig();
  config.excludeElectroweakBosons = false;
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kZ, config));

  config = truth::AssignableTargetConfig();
  config.excludeSynthetic = false;
  // The role is the only thing that bars an invented node.
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kConnector, config));
}

void TestAssignableTarget::testExtraBarredPdgIdsCoverBothSigns() {
  const auto graph = buildInteraction();
  truth::AssignableTargetConfig config;
  config.extraBarredPdgIds = {211};
  CPPUNIT_ASSERT(!truth::isAssignableTarget(graph, kPion, config));
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kPi0, config));

  auto negative = buildInteraction();
  negative.particles()[kPion].pdgId = -211;
  CPPUNIT_ASSERT(!truth::isAssignableTarget(negative, kPion, config));
}

// REQUIRED: a selection preset attaches a real particle whose production vertex it dropped
// to an artificial source vertex. A gun particle lands on the InitialState vertex and a
// stable spectator on the UnderlyingEvent vertex, and both are particles a detector sees,
// so both stay assignable. Only the invented node produced beside them is barred.
void TestAssignableTarget::testArtificialSourceKeepsRealParticles() {
  const auto graph = buildInteraction();
  const truth::AssignableTargetConfig config;

  CPPUNIT_ASSERT(graph.vertices()[4].isArtificial());
  CPPUNIT_ASSERT(graph.vertices()[5].isArtificial());
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kGunSeed, config));
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kSpectator, config));
  CPPUNIT_ASSERT(!truth::isAssignableTarget(graph, kConnector, config));
}
