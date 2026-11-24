// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <cstdint>
#include <vector>

#include "TBufferFile.h"
#include "TClass.h"

#include "SimDataFormats/TruthInfo/interface/DeltaCoding.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/TruthGraph.h"

namespace {
  using Index = truth::LogicalGraphHitIndex;
  using Hit = Index::Hit;

  // Writes an object to a buffer and reads it back, as a file does.
  template <typename T>
  T roundTrip(T const& in, const char* className) {
    TClass* cls = TClass::GetClass(className);
    CPPUNIT_ASSERT(cls != nullptr);
    TBufferFile buffer(TBuffer::kWrite);
    cls->WriteBuffer(buffer, const_cast<T*>(&in));
    buffer.SetReadMode();
    buffer.SetBufferOffset(0);
    T out;
    cls->ReadBuffer(buffer, &out);
    return out;
  }

  void assertSameHits(std::span<const Hit> a, std::span<const Hit> b) {
    CPPUNIT_ASSERT_EQUAL(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      CPPUNIT_ASSERT_EQUAL(a[i].detId, b[i].detId);
      CPPUNIT_ASSERT_EQUAL(a[i].recHitIndex, b[i].recHitIndex);
      CPPUNIT_ASSERT_EQUAL(a[i].energy, b[i].energy);
    }
  }
}  // namespace

class TruthStoredFormTest : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TruthStoredFormTest);
  CPPUNIT_TEST(deltaCodingIsExact);
  CPPUNIT_TEST(truthGraphReadsBack);
  CPPUNIT_TEST(logicalGraphReadsBack);
  CPPUNIT_TEST(sharedHitIndexReadsBack);
  CPPUNIT_TEST(materialisedHitIndexReadsBack);
  CPPUNIT_TEST_SUITE_END();

public:
  // A decode of an encode gives the input, for increasing, decreasing and extreme values.
  void deltaCodingIsExact() {
    const std::vector<uint32_t> unsignedValues{5, 3, 3, 4000000000u, 0, 0xFFFFFFFFu};
    std::vector<uint32_t> stored, restored;
    truth::delta::encode(unsignedValues, stored);
    truth::delta::decode(stored, restored);
    CPPUNIT_ASSERT(restored == unsignedValues);

    const std::vector<int64_t> signedValues{-7, 12, INT64_MAX, INT64_MIN, 0};
    std::vector<int64_t> storedSigned, restoredSigned;
    truth::delta::encode(signedValues, storedSigned);
    truth::delta::decode(storedSigned, restoredSigned);
    CPPUNIT_ASSERT(restoredSigned == signedValues);

    const std::vector<int32_t> indices{-1, 0, 5, -1, 2, INT32_MAX};
    std::vector<int32_t> storedIndices, restoredIndices;
    truth::delta::encodeRelativeToIndex(indices, storedIndices);
    truth::delta::decodeRelativeToIndex(storedIndices, restoredIndices);
    CPPUNIT_ASSERT(restoredIndices == indices);
  }

  // A packed TruthGraph that is written and read has the same arrays.
  void truthGraphReadsBack() {
    TruthGraph graph;
    graph.nodes().assign(3, TruthGraph::NodeRef(TruthGraph::NodeKind::SimTrack, 42));
    graph.offsets() = {0, 2, 2, 3};
    graph.edges() = {1, 2, 0};
    graph.edgeKind() = {0, 1, 2};
    graph.pdgId() = {11, -11, 22};
    graph.simTrackToVtx() = {-1, 0, 1};
    graph.pack();
    const TruthGraph read = roundTrip(graph, "TruthGraph");
    CPPUNIT_ASSERT(read.offsets() == graph.offsets());
    CPPUNIT_ASSERT(read.edges() == graph.edges());
    CPPUNIT_ASSERT(read.edgeKind() == graph.edgeKind());
    CPPUNIT_ASSERT(read.pdgId() == graph.pdgId());
    CPPUNIT_ASSERT(read.simTrackToVtx() == graph.simTrackToVtx());
    CPPUNIT_ASSERT_EQUAL(graph.nNodes(), read.nNodes());
    CPPUNIT_ASSERT(read.isConsistent());
  }

  // A packed truth::Graph that is written and read has the same adjacency.
  void logicalGraphReadsBack() {
    truth::Graph graph;
    graph.particles().resize(3);
    graph.vertices().resize(2);
    graph.particleToDecayVertexOffsets() = {0, 1, 1, 1};
    graph.particleToDecayVertices() = {1};
    graph.particleToProductionVertexOffsets() = {0, 1, 2, 3};
    graph.particleToProductionVertices() = {0, 1, 1};
    graph.vertexToOutgoingParticleOffsets() = {0, 1, 3};
    graph.vertexToOutgoingParticles() = {0, 2, 1};
    graph.vertexToIncomingParticleOffsets() = {0, 0, 1};
    graph.vertexToIncomingParticles() = {0};
    graph.pack();
    const truth::Graph read = roundTrip(graph, "truth::Graph");
    CPPUNIT_ASSERT_EQUAL(graph.nParticles(), read.nParticles());
    CPPUNIT_ASSERT(read.particleToDecayVertexOffsets() == graph.particleToDecayVertexOffsets());
    CPPUNIT_ASSERT(read.particleToDecayVertices() == graph.particleToDecayVertices());
    CPPUNIT_ASSERT(read.particleToProductionVertexOffsets() == graph.particleToProductionVertexOffsets());
    CPPUNIT_ASSERT(read.particleToProductionVertices() == graph.particleToProductionVertices());
    CPPUNIT_ASSERT(read.vertexToOutgoingParticleOffsets() == graph.vertexToOutgoingParticleOffsets());
    CPPUNIT_ASSERT(read.vertexToOutgoingParticles() == graph.vertexToOutgoingParticles());
    CPPUNIT_ASSERT(read.vertexToIncomingParticleOffsets() == graph.vertexToIncomingParticleOffsets());
    CPPUNIT_ASSERT(read.vertexToIncomingParticles() == graph.vertexToIncomingParticles());
  }

  // A hit index in the shared layout that is written and read has the same hits and ranges.
  void sharedHitIndexReadsBack() {
    std::vector<Index::Channel> channels(truth::kNumHitChannels);
    auto& tracker = channels[static_cast<std::size_t>(truth::HitChannel::Tracker)];
    tracker.directHits = {{10, 5, 1.f}, {10, 7, 2.f}, {12, 1, 3.f}, {3, Hit::kNoCell, 4.f}};
    tracker.dfsOffsets = {0, 2, 4};
    auto& mtd = channels[static_cast<std::size_t>(truth::HitChannel::MTD)];
    mtd.directHits = {{7, 9, 0.5f}};
    mtd.directHitTimes = {1.5f};
    mtd.dfsOffsets = {0, 0, 1};
    mtd.cellKeyed = true;
    const Index index(2, channels, {1, 0}, {0, 1, 2}, {{1, 1}, {0, 2}});
    const Index read = roundTrip(index, "truth::LogicalGraphHitIndex");
    CPPUNIT_ASSERT(read.sharedSubgraphStore());
    CPPUNIT_ASSERT(read.dfsPos() == index.dfsPos());
    for (uint32_t particle = 0; particle < 2; ++particle) {
      const auto ranges = index.subgraphRanges(particle);
      const auto readRanges = read.subgraphRanges(particle);
      CPPUNIT_ASSERT_EQUAL(ranges.size(), readRanges.size());
      for (std::size_t i = 0; i < ranges.size(); ++i) {
        CPPUNIT_ASSERT_EQUAL(ranges[i].firstSlot, readRanges[i].firstSlot);
        CPPUNIT_ASSERT_EQUAL(ranges[i].slotCount, readRanges[i].slotCount);
      }
      for (auto channel : {truth::HitChannel::Tracker, truth::HitChannel::MTD, truth::HitChannel::Calo}) {
        assertSameHits(index.directHits(channel, particle), read.directHits(channel, particle));
        assertSameHits(index.subgraphHits(channel, particle), read.subgraphHits(channel, particle));
      }
    }
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), read.directHits(truth::HitChannel::Tracker, 1).size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(4), read.subgraphHits(truth::HitChannel::Tracker, 1).size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), read.directHitTimes(truth::HitChannel::MTD, 0).size());
    CPPUNIT_ASSERT(read.isCellKeyed(truth::HitChannel::MTD));
  }

  // A hit index in the materialised layout that is written and read has the same hits.
  void materialisedHitIndexReadsBack() {
    std::vector<Index::Channel> channels(truth::kNumHitChannels);
    auto& calo = channels[static_cast<std::size_t>(truth::HitChannel::Calo)];
    calo.directOffsets = {0, 1, 3};
    calo.directHits = {{100, Hit::kInvalidRecHitIndex, 1.f}, {90, 4, 2.f}, {100, 5, 3.f}};
    calo.subgraphOffsets = {0, 2, 4};
    calo.subgraphHits = {{90, 4, 2.f}, {100, 5, 4.f}, {90, 4, 2.f}, {100, 5, 3.f}};
    const Index index(2, channels);
    const Index read = roundTrip(index, "truth::LogicalGraphHitIndex");
    CPPUNIT_ASSERT(!read.sharedSubgraphStore());
    for (uint32_t particle = 0; particle < 2; ++particle) {
      assertSameHits(index.directHits(truth::HitChannel::Calo, particle),
                     read.directHits(truth::HitChannel::Calo, particle));
      assertSameHits(index.subgraphHits(truth::HitChannel::Calo, particle),
                     read.subgraphHits(truth::HitChannel::Calo, particle));
    }
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), read.directHits(truth::HitChannel::Calo, 1).size());
  }
};

CPPUNIT_TEST_SUITE_REGISTRATION(TruthStoredFormTest);
