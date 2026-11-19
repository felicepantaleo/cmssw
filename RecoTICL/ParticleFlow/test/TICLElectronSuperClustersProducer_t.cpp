#include <memory>
#include <vector>

#include <catch2/catch_all.hpp>

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "RecoTICL/Interpretation/interface/AssignmentMaps.h"

namespace {
  constexpr int kElectron = static_cast<int>(ticl::TrackMode::kElectron);
  constexpr int kChargedHadron = static_cast<int>(ticl::TrackMode::kChargedHadron);
  constexpr int kNotSelected = static_cast<int>(ticl::TrackMode::kNotSelected);

  const char *kConfig =
      "from FWCore.TestProcessor.TestProcess import *\n"
      "process = TestProcess()\n"
      "process.toTest = cms.EDProducer('TICLElectronSuperClustersProducer',\n"
      "    superClusters = cms.InputTag('sc'),\n"
      "    eGammaTracksters = cms.InputTag('eg'),\n"
      "    interpretations = cms.InputTag('interp'))\n"
      "process.moduleToTest(process.toTest)\n";

  using Ints = std::vector<int>;
}  // namespace

TEST_CASE("TICLElectronSuperClustersProducer: track seeds, and a copy for the supercluster seeding with its index",
          "[TICLElectronSuperClustersProducer]") {
  edm::test::TestProcessor::Config config{kConfig};
  const auto putSC = config.produces<reco::SuperClusterCollection>("sc");
  const auto putSCIndex = config.produces<Ints>("sc", "tracksterIndex");
  const auto putEGIndex = config.produces<Ints>("eg", "interpretationIndex");
  const auto putTrackToTrackster = config.produces<Ints>("interp", "trackToTrackster");
  const auto putTrackMode = config.produces<Ints>("interp", "trackMode");
  const auto putElectronTrackster = config.produces<Ints>("interp", "electronTrackster");
  edm::test::TestProcessor tester(config);

  // E/gamma tracksters 0, 1, 2 are the final tracksters 1, 3, 4. Track 0 is an electron on final 1; track 1 is a
  // charged hadron on final 2 with its Electron hypothesis on final 3; track 2 is not selected.
  auto superClusters = std::make_unique<reco::SuperClusterCollection>();
  for (double e : {10., 20., 30.})
    superClusters->emplace_back(e, math::XYZPoint(0., 50., 350.));
  auto run = [&](Ints tracksterIndex) {
    return tester.test(
        std::make_pair(putSC, std::make_unique<reco::SuperClusterCollection>(*superClusters)),
        std::make_pair(putSCIndex, std::make_unique<Ints>(std::move(tracksterIndex))),
        std::make_pair(putEGIndex, std::make_unique<Ints>(Ints{1, 3, 4})),
        std::make_pair(putTrackToTrackster, std::make_unique<Ints>(Ints{1, 2, -1})),
        std::make_pair(putTrackMode, std::make_unique<Ints>(Ints{kElectron, kChargedHadron, kNotSelected})),
        std::make_pair(putElectronTrackster, std::make_unique<Ints>(Ints{1, 3, -1})));
  };

  SECTION("selection") {
    // Supercluster 0 of e/gamma trackster 0, 1 of 2, 2 of 1.
    const auto event = run({0, 2, 1});
    const auto &copies = *event.get<reco::SuperClusterCollection>();
    REQUIRE(copies.size() == 1);
    CHECK(copies[0].energy() == 30.);
    CHECK(*event.get<Ints>("superClusterIndex") == Ints{2});
    CHECK(*event.get<Ints>("trackSeedSuperCluster") == Ints{0});
    CHECK(*event.get<Ints>("trackSeedTrack") == Ints{0});
  }
  SECTION("an e/gamma trackster index out of range throws") { REQUIRE_THROWS(run({0, 2, 3})); }
  SECTION("a trackster index list of another size throws") { REQUIRE_THROWS(run({0, 2})); }
}
