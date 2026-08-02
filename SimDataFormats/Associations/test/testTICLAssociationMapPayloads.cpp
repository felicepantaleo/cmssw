// Compile-time and runtime checks for the ticl::AssociationMap payload vocabulary.
//
// The payload concepts exist to make two classes of mistake impossible: reading a
// quantity a payload does not carry, and adding a payload without teaching every
// branch about it. Both used to be silent, so both are asserted here.

#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

namespace {
  using ticl::AssociationElement;
  using ticl::FractionType;
  using ticl::SharedEnergyType;
  using ticl::SharedHitsType;

  // Every payload the vocabulary admits models the concept.
  static_assert(ticl::AssociationPayload<FractionType>);
  static_assert(ticl::AssociationPayload<SharedEnergyType>);
  static_assert(ticl::AssociationPayload<SharedHitsType>);
  static_assert(ticl::AssociationPayload<std::pair<SharedHitsType, float>>);

  // Anything else does not, so AssociationElement<T> fails to instantiate rather than
  // falling through the if constexpr chains the way it silently did before.
  static_assert(!ticl::AssociationPayload<float>);
  static_assert(!ticl::AssociationPayload<std::pair<float, float>>);
  static_assert(!ticl::AssociationPayload<std::pair<FractionType, int>>);

  // Scalar vs scored classification.
  static_assert(ticl::ScalarPayload<SharedHitsType>);
  static_assert(!ticl::ScoredPayload<SharedHitsType>);
  static_assert(ticl::ScoredPayload<std::pair<SharedHitsType, float>>);

  using ElemHits = AssociationElement<SharedHitsType>;
  using ElemHitsScored = AssociationElement<std::pair<SharedHitsType, float>>;
  using ElemEnergy = AssociationElement<SharedEnergyType>;

  // Detection concepts: naming them keeps the checks readable and keeps template
  // commas out of the cppunit macros below.
  template <typename E>
  concept HasSharedHits = requires(const E& e) { e.sharedHits(); };
  template <typename E>
  concept HasSharedEnergy = requires(const E& e) { e.sharedEnergy(); };
  template <typename E>
  concept HasFraction = requires(const E& e) { e.fraction(); };
  template <typename E>
  concept HasScore = requires(const E& e) { e.score(); };

  // A named accessor is callable only for the payloads that carry that quantity.
  static_assert(HasSharedHits<ElemHits>);
  static_assert(!HasSharedEnergy<ElemHits>);
  static_assert(!HasFraction<ElemHits>);
  static_assert(!HasSharedHits<ElemEnergy>);
  static_assert(HasSharedEnergy<ElemEnergy>);

  // score() exists exactly for the scored payloads.
  static_assert(!HasScore<ElemHits>);
  static_assert(HasScore<ElemHitsScored>);
}  // namespace

class testTICLAssociationMapPayloads : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testTICLAssociationMapPayloads);
  CPPUNIT_TEST(defaultElementIsInvalid);
  CPPUNIT_TEST(accumulateAddsBothMembers);
  CPPUNIT_TEST(sharedHitsMapRoundTrip);
  CPPUNIT_TEST_SUITE_END();

public:
  // A default-constructed element must report invalid for BOTH payload families; the
  // scalar branch seeds value_.value and the scored branch value_.first.value.
  void defaultElementIsInvalid() {
    const ElemHits emptyScalar;
    const ElemHitsScored emptyScored;
    const ElemHits filled(0u, SharedHitsType(3.f));
    CPPUNIT_ASSERT(!emptyScalar.isValid());
    CPPUNIT_ASSERT(!emptyScored.isValid());
    CPPUNIT_ASSERT(filled.isValid());
  }

  void accumulateAddsBothMembers() {
    ElemHitsScored e(7u, {SharedHitsType(2.f), 0.25f});
    e.accumulate({SharedHitsType(3.f), 0.5f});
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, e.sharedHits(), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.75, e.score(), 1e-6);
    CPPUNIT_ASSERT_EQUAL(7u, e.index());
  }

  // The index-only map mode, which the associator producers use for scratch maps.
  void sharedHitsMapRoundTrip() {
    // insert takes the raw float, not the payload wrapper: the map type fixes the units.
    ticl::AssociationMap<ticl::mapWithSharedHitsAndScore> map(2);
    map.insert(0u, 11u, 4.f, 0.1f);
    map.insert(0u, 11u, 2.f, 0.0f);  // duplicate index2 accumulates
    map.insert(1u, 22u, 1.f, 0.9f);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), map[0].size());
    CPPUNIT_ASSERT_EQUAL(11u, map[0][0].index());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, map[0][0].sharedHits(), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, map[1][0].sharedHits(), 1e-6);
  }
};

CPPUNIT_TEST_SUITE_REGISTRATION(testTICLAssociationMapPayloads);
