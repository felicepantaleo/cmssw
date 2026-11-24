// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <cstdint>
#include <unordered_map>

#include "PhysicsTools/TruthInfo/interface/SimKeyMap.h"
#include "SimDataFormats/TruthInfo/interface/InteractionId.h"

class SimKeyMapTest : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(SimKeyMapTest);
  CPPUNIT_TEST(testEmptyMapFindsNothing);
  CPPUNIT_TEST(testKeyZeroIsAKey);
  CPPUNIT_TEST(testAssignReplacesTheValue);
  CPPUNIT_TEST(testGrowthKeepsEveryKey);
  CPPUNIT_TEST_SUITE_END();

public:
  void testEmptyMapFindsNothing();
  void testKeyZeroIsAKey();
  void testAssignReplacesTheValue();
  void testGrowthKeepsEveryKey();
};

CPPUNIT_TEST_SUITE_REGISTRATION(SimKeyMapTest);

void SimKeyMapTest::testEmptyMapFindsNothing() {
  const truth::SimKeyMap map;
  CPPUNIT_ASSERT_EQUAL(truth::SimKeyMap::kNotFound, map.find(0));
  CPPUNIT_ASSERT_EQUAL(truth::SimKeyMap::kNotFound, map.find(42));
  CPPUNIT_ASSERT_EQUAL(std::size_t(0), map.size());
}

// The key of the first vertex of the signal sub-event is 0.
void SimKeyMapTest::testKeyZeroIsAKey() {
  truth::SimKeyMap map(4);
  CPPUNIT_ASSERT_EQUAL(truth::SimKeyMap::kNotFound, map.find(0));
  map.insertOrAssign(0, 7);
  CPPUNIT_ASSERT_EQUAL(uint32_t(7), map.find(0));
  CPPUNIT_ASSERT_EQUAL(truth::SimKeyMap::kNotFound, map.find(1));
  CPPUNIT_ASSERT_EQUAL(std::size_t(1), map.size());
}

void SimKeyMapTest::testAssignReplacesTheValue() {
  truth::SimKeyMap map;
  map.insertOrAssign(truth::simObjectKey(3, 12), 1);
  map.insertOrAssign(truth::simObjectKey(3, 12), 2);
  CPPUNIT_ASSERT_EQUAL(uint32_t(2), map.find(truth::simObjectKey(3, 12)));
  CPPUNIT_ASSERT_EQUAL(std::size_t(1), map.size());
}

// The same trackIds in many sub-events, added with no reserve, so the table grows.
void SimKeyMapTest::testGrowthKeepsEveryKey() {
  truth::SimKeyMap map;
  std::unordered_map<uint64_t, uint32_t> reference;
  uint32_t value = 0;
  for (uint64_t eventId = 0; eventId < 200; ++eventId) {
    for (uint32_t trackId = 1; trackId <= 500; trackId += 1 + (trackId % 3)) {
      const uint64_t key = truth::simObjectKey(eventId, trackId);
      map.insertOrAssign(key, value);
      reference[key] = value;
      ++value;
    }
  }
  CPPUNIT_ASSERT_EQUAL(reference.size(), map.size());
  for (auto const& [key, expected] : reference)
    CPPUNIT_ASSERT_EQUAL(expected, map.find(key));
  CPPUNIT_ASSERT_EQUAL(truth::SimKeyMap::kNotFound, map.find(truth::simObjectKey(200, 1)));
  CPPUNIT_ASSERT_EQUAL(truth::SimKeyMap::kNotFound, map.find(truth::simObjectKey(0, 501)));
}
