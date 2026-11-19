#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <catch2/catch_all.hpp>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoTICL/ParticleFlow/interface/EGammaSuperClusterFeatures.h"

namespace egsc = ticl::egammaSuperCluster;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {
  constexpr unsigned int kLastLayer = 47;
  constexpr double kZ = 350.;

  // A point at z = +-kZ on the line from the origin along (eta, phi).
  math::XYZPoint point(double eta, double phi) {
    const double rho = kZ / std::sinh(std::abs(eta));
    return math::XYZPoint(rho * std::cos(phi), rho * std::sin(phi), eta > 0. ? kZ : -kZ);
  }

  // A cluster of energy e at (eta, phi) with the hits ids (fraction 1), seed hit ids[0].
  reco::CaloCluster cluster(double e, double eta, double phi, const std::vector<uint32_t> &ids) {
    std::vector<std::pair<DetId, float>> hits;
    for (auto id : ids)
      hits.emplace_back(DetId(id), 1.f);
    return reco::CaloCluster(e,
                             point(eta, phi),
                             reco::CaloID(reco::CaloID::DET_HGCAL_ENDCAP),
                             hits,
                             reco::CaloCluster::hgcal_em,
                             DetId(ids.empty() ? 0 : ids[0]));
  }

  // A supercluster of the clusters (seed: the first one) at (eta, phi), with the widths of the arguments.
  reco::SuperCluster superCluster(const std::vector<reco::CaloCluster> &clusters,
                                  double eta,
                                  double phi,
                                  double etaWidth = 0.,
                                  double phiWidth = 0.) {
    reco::CaloClusterPtrVector members;
    double raw = 0.;
    for (std::size_t i = 0; i < clusters.size(); ++i) {
      members.push_back(reco::CaloClusterPtr(&clusters, i));
      raw += clusters[i].energy();
    }
    return reco::SuperCluster(raw, point(eta, phi), members[0], members, 0., phiWidth, etaWidth);
  }

  // A track of transverse momentum pt from the origin along (eta, phi).
  reco::Track track(double pt, double eta, double phi) {
    return reco::Track(1.,
                       10.,
                       reco::Track::Point(0., 0., 0.),
                       reco::Track::Vector(pt * std::cos(phi), pt * std::sin(phi), pt * std::sinh(eta)),
                       1,
                       reco::Track::CovarianceMatrix());
  }

  // Tiles with the layer clusters, each on its layer index.
  std::unique_ptr<TICLLayerTiles> tiles(const std::vector<reco::CaloCluster> &lcs,
                                        const std::vector<unsigned int> &layers) {
    auto result = std::make_unique<TICLLayerTiles>();
    for (std::size_t i = 0; i < lcs.size(); ++i)
      result->fill(layers[i], lcs[i].eta(), lcs[i].phi(), i);
    return result;
  }

  double et(const reco::CaloCluster &lc) { return lc.energy() / std::cosh(lc.eta()); }
}  // namespace

TEST_CASE("cellFeatures: core, wide, largest cells and RMS around the centroid of each layer", "[cellFeatures]") {
  // Layer 0: 10 GeV at x = 0 and 2 GeV (fraction 0.5) at x = 5, centroid x = 10/12 (rechit energies).
  // Layer 1: 4 GeV at (1, 1), its own centroid.
  const std::vector<egsc::CellHit> hits{{0, 0.f, 0.f, 10.f, 1.f, egsc::kHDCellSize},
                                        {0, 5.f, 0.f, 2.f, 0.5f, egsc::kLDCellSize},
                                        {1, 1.f, 1.f, 4.f, 1.f, egsc::kHDCellSize}};
  const auto f = egsc::cellFeatures(hits, 2, 20.);
  const double c = 10. / 12.;
  CHECK_THAT(f.coreFraction, WithinRel(14. / 20., 1e-6));
  CHECK_THAT(f.wideFraction, WithinRel(15. / 20., 1e-6));
  CHECK_THAT(f.maxCellFraction, WithinRel(10. / 20., 1e-6));
  CHECK_THAT(f.secondCellFraction, WithinRel(4. / 20., 1e-6));
  CHECK_THAT(f.transverseRMS, WithinRel(std::sqrt((10. * c * c + 1. * (5. - c) * (5. - c)) / 15.), 1e-5));
}

TEST_CASE("cellFeatures: the cell size adds to the core radius", "[cellFeatures]") {
  // A 1 GeV hit at 3.4 cm from a 100 GeV hit: in the core only with the LD cell size.
  const float r = 3.4f * 101.f / 100.f;
  for (const float size : {egsc::kHDCellSize, egsc::kLDCellSize}) {
    const std::vector<egsc::CellHit> hits{{0, 0.f, 0.f, 100.f, 1.f, egsc::kHDCellSize}, {0, r, 0.f, 1.f, 1.f, size}};
    const auto f = egsc::cellFeatures(hits, 1, 101.);
    CHECK_THAT(f.coreFraction, WithinRel((size == egsc::kLDCellSize ? 101. : 100.) / 101., 1e-6));
  }
}

TEST_CASE("cellFeatures: no CE-E hit, a hit beyond the last layer and a raw energy of 0 give zeros", "[cellFeatures]") {
  for (const auto &f : {egsc::cellFeatures({}, 26, 10.),
                        egsc::cellFeatures({{26, 1.f, 1.f, 5.f, 1.f, egsc::kHDCellSize}}, 26, 10.),
                        egsc::cellFeatures({{0, 1.f, 1.f, 5.f, 1.f, egsc::kHDCellSize}}, 26, 0.)}) {
    CHECK(f.coreFraction == 0.f);
    CHECK(f.wideFraction == 0.f);
    CHECK(f.maxCellFraction == 0.f);
    CHECK(f.secondCellFraction == 0.f);
    CHECK(f.transverseRMS == 0.f);
  }
}

TEST_CASE("regressionFeatures: the order of the 19 features", "[regressionFeatures]") {
  const std::vector<reco::CaloCluster> clusters{cluster(30., 2.0, 0.1, {1, 2, 3}), cluster(10., 2.05, 0.12, {4, 5})};
  const auto sc = superCluster(clusters, 2.0, 0.1, 0.02, 0.03);
  ticl::Trackster ts;
  ts.setBarycenter(ticl::Trackster::Vector(0., 0., kZ));
  ts.setRawEnergy(50.f);
  ts.setRawEmEnergy(40.f);
  ts.vertices() = {0, 1, 2, 3, 4};
  ts.fillPCAVariables(Eigen::Vector3f(1.f, 1.f, 1.f),
                      Eigen::Matrix3f::Identity(),
                      Eigen::Vector3f(1.f, 1.f, 1.f),
                      Eigen::Vector3f(4.f, 1.f, 0.25f),
                      3,
                      ticl::Trackster::PCAOrdering::descending);
  egsc::CellFeatures cell;
  cell.coreFraction = 0.8f;
  cell.wideFraction = 0.95f;
  cell.maxCellFraction = 0.3f;
  cell.secondCellFraction = 0.2f;
  cell.transverseRMS = 1.5f;
  const auto f = egsc::regressionFeatures(sc, ts, cell, 1000);
  REQUIRE(f.size() == 19);
  CHECK_THAT(f[0], WithinRel(std::log(40.f), 1e-6f));
  CHECK_THAT(f[1], WithinRel(2.f, 1e-5f));
  CHECK_THAT(f[2], WithinRel(0.02f, 1e-6f));
  CHECK_THAT(f[3], WithinRel(0.03f, 1e-6f));
  CHECK(f[4] == 2.f);
  CHECK_THAT(f[5], WithinRel(30.f / 40.f, 1e-6f));
  CHECK_THAT(f[6], WithinRel(40.f / 50.f, 1e-6f));
  CHECK_THAT(f[7], WithinRel(2.f, 1e-6f));
  CHECK_THAT(f[8], WithinRel(1.f, 1e-6f));
  CHECK_THAT(f[9], WithinRel(0.5f, 1e-6f));
  CHECK_THAT(f[10], WithinRel(std::log(6.f), 1e-6f));
  CHECK(f[11] == 0.8f);
  CHECK(f[12] == 0.3f);
  CHECK_THAT(f[13], WithinRel(std::log(50.f / 40.f), 1e-5f));
  CHECK_THAT(f[14], WithinRel(std::log(1001.f), 1e-6f));
  CHECK_THAT(f[15], WithinRel(std::log(6.f), 1e-6f));
  CHECK(f[16] == 0.2f);
  CHECK(f[17] == 1.5f);
  CHECK(f[18] == 0.95f);
}

TEST_CASE("regressionFeatures: one cluster and a trackster below the supercluster energy", "[regressionFeatures]") {
  const std::vector<reco::CaloCluster> clusters{cluster(20., -2.2, 3.1, {7})};
  const auto sc = superCluster(clusters, -2.2, 3.1);
  ticl::Trackster ts;
  ts.setRawEnergy(15.f);
  ts.setRawEmEnergy(15.f);
  const auto f = egsc::regressionFeatures(sc, ts, egsc::CellFeatures(), 0);
  CHECK_THAT(f[1], WithinRel(2.2f, 1e-5f));
  CHECK(f[4] == 1.f);
  CHECK(f[5] == 1.f);
  CHECK(f[13] == 0.f);
  CHECK(f[14] == 0.f);
}

TEST_CASE("isolationEt: inner and outer cones, the own layer clusters and the other endcap", "[isolationEt]") {
  for (const double side : {1., -1.}) {
    const double eta = 2. * side;
    const std::vector<reco::CaloCluster> clusters{cluster(40., eta, 0., {1, 2})};
    const auto sc = superCluster(clusters, eta, 0.);
    // own seed hit, dR 0.1, dR 0.2, dR 0.4, and the mirror of the inner one in the other endcap
    const std::vector<reco::CaloCluster> lcs{cluster(5., eta, 0.05, {2}),
                                             cluster(3., eta, 0.1, {10}),
                                             cluster(2., eta + 0.2 * side, 0., {11}),
                                             cluster(7., eta, 0.4, {12}),
                                             cluster(9., -eta, 0.1, {13})};
    const unsigned int layer = side > 0. ? kLastLayer + 3 : 3;
    const unsigned int mirror = side > 0. ? 3 : kLastLayer + 3;
    const auto t = tiles(lcs, {layer, layer, layer + 1, layer, mirror});
    const auto iso = egsc::isolationEt(sc, *t, lcs, kLastLayer, egsc::kIsolationInnerDR, egsc::kIsolationOuterDR);
    CHECK_THAT(iso[0], WithinRel(et(lcs[1]), 1e-6));
    CHECK_THAT(iso[1], WithinRel(et(lcs[1]) + et(lcs[2]), 1e-6));
  }
}

TEST_CASE("isolationEt: a layer cluster across the phi seam", "[isolationEt]") {
  const std::vector<reco::CaloCluster> clusters{cluster(40., 2.4, M_PI - 0.01, {1})};
  const auto sc = superCluster(clusters, 2.4, M_PI - 0.01);
  const std::vector<reco::CaloCluster> lcs{cluster(3., 2.4, -M_PI + 0.05, {10})};
  const auto iso = egsc::isolationEt(
      sc, *tiles(lcs, {kLastLayer}), lcs, kLastLayer, egsc::kIsolationInnerDR, egsc::kIsolationOuterDR);
  CHECK_THAT(iso[0], WithinRel(et(lcs[0]), 1e-6));
  CHECK_THAT(iso[1], WithinRel(et(lcs[0]), 1e-6));
}

TEST_CASE("fillTrackTiles and trackFeatures: near tracks, isolation, both endcaps and the phi seam",
          "[trackFeatures]") {
  for (const double side : {1., -1.}) {
    for (const double phi : {0.5, M_PI - 0.01}) {
      const double eta = 2. * side;
      const std::vector<reco::CaloCluster> clusters{cluster(40., eta, phi, {1})};
      const auto sc = superCluster(clusters, eta, phi);
      // near (pT 5, dR 0.02), near but pT 1.5, isolation (pT 3, dR 0.1, across the seam for phi = pi - 0.01), below
      // kTrackMinPt, below kTrackMinAbsEta, and the other endcap
      const std::vector<reco::Track> tracks{track(5., eta, phi + 0.02),
                                            track(1.5, eta + 0.03 * side, phi),
                                            track(3., eta, phi + 0.1),
                                            track(0.5, eta, phi + 0.1),
                                            track(4., 1.0 * side, phi),
                                            track(4., -eta, phi + 0.1)};
      auto trackTiles = std::make_unique<egsc::TrackTiles>();
      egsc::fillTrackTiles(tracks, *trackTiles);
      const auto f = egsc::trackFeatures(sc, tracks, *trackTiles);
      CHECK(f.nNear == 1);
      CHECK_THAT(f.isolation, WithinRel(3., 1e-6));
    }
  }
}

TEST_CASE("scoreFeatures: the order of the 35 features", "[scoreFeatures]") {
  const std::vector<reco::CaloCluster> clusters{cluster(30., 2.0, 0.1, {1, 2})};
  auto sc = superCluster(clusters, 2.0, 0.1);
  sc.setEnergy(36.);
  ticl::Trackster ts;
  for (int j = 0; j < 8; ++j)
    ts.setIdProbability(static_cast<ticl::Trackster::ParticleType>(j), 0.1f * (j + 1));
  std::array<float, egsc::kNFeatures> regression;
  for (std::size_t j = 0; j < regression.size(); ++j)
    regression[j] = 100.f + j;
  egsc::TrackFeatures tracks;
  tracks.nNear = 2;
  tracks.isolation = 4.;
  const auto f = egsc::scoreFeatures(regression, sc, ts, {1., 3.}, true, false, tracks);
  REQUIRE(f.size() == 35);
  for (std::size_t j = 0; j < egsc::kNFeatures; ++j)
    CHECK(f[j] == 100.f + j);
  for (int j = 0; j < 8; ++j)
    CHECK(f[19 + j] == 0.1f * (j + 1));
  const float sinTheta = std::sin(static_cast<float>(sc.position().theta()));
  const float rawEt = 30.f * sinTheta;
  CHECK_THAT(f[27], WithinRel(std::log(36.f * sinTheta), 1e-5f));
  CHECK_THAT(f[28], WithinRel(1.f / rawEt, 1e-5f));
  CHECK_THAT(f[29], WithinRel(3.f / rawEt, 1e-5f));
  CHECK_THAT(f[30], WithinRel(36.f / 30.f, 1e-6f));
  CHECK(f[31] == 1.f);
  CHECK(f[32] == 0.f);
  CHECK(f[33] == 2.f);
  CHECK_THAT(f[34], WithinRel(4.f / rawEt, 1e-5f));
  const auto pion = egsc::scoreFeatures(regression, sc, ts, {0., 0.}, false, true, egsc::TrackFeatures());
  CHECK(pion[31] == 0.f);
  CHECK(pion[32] == 1.f);
  CHECK(pion[28] == 0.f);
  CHECK(pion[33] == 0.f);
}
