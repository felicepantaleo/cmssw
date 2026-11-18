#ifndef DataFormats_HGCalReco_test_HGCalRecoSoATestValues_h
#define DataFormats_HGCalReco_test_HGCalRecoSoATestValues_h

#include <Eigen/Core>

#include <cstdint>
#include <string_view>

#include "DataFormats/HGCalReco/interface/HGCalSoALayerClustersExtraHostCollection.h"
#include "DataFormats/HGCalReco/interface/LayerClusterMaskHostCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterHostCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterLinksHostCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterPidHostCollection.h"
#include "FWCore/Utilities/interface/Exception.h"

// Known values for the HGCalReco SoA collections of the unit tests: fill() writes them, check() throws if one differs.
namespace hgcalrecotest {

  // The value of field k of element i: exact in float for the sizes of the tests.
  inline float value(int i, int k) { return 100.f * i + k + 0.25f; }

  // The number of vertices of n tracksters: trackster i has i + 1 layer clusters.
  inline int32_t numberOfVertices(int32_t n) { return n * (n + 1) / 2; }

  inline void require(bool condition, std::string_view what, int i) {
    if (not condition)
      throw cms::Exception("TestFailure") << what << " differs at index " << i;
  }

  inline void requireSize(int32_t size, int32_t expected, std::string_view what) {
    if (size != expected)
      throw cms::Exception("TestFailure") << what << " is " << size << ", expected " << expected;
  }

  inline ticl::TracksterHostCollection makeTracksters(int32_t n) {
    const int32_t nVertices = numberOfVertices(n);
    return ticl::TracksterHostCollection(n, n, n, n, n, n, n, nVertices);
  }

  inline void fill(ticl::TracksterSoAView view) {
    const int n = view.energy().metadata().size();
    for (int i = 0; i < n; ++i) {
      view.energy()[i].rawEnergy() = value(i, 0);
      view.energy()[i].rawEmEnergy() = value(i, 1);
      view.energy()[i].regressedEnergy() = value(i, 2);
      view.barycenter()[i].x() = value(i, 3);
      view.barycenter()[i].y() = value(i, 4);
      view.barycenter()[i].z() = value(i, 5);
      view.timing()[i].time() = value(i, 6);
      view.timing()[i].timeError() = value(i, 7);
      view.pca()[i].eigenvalues() = Eigen::Vector3f(value(i, 8), value(i, 9), value(i, 10));
      for (int k = 0; k < 9; ++k)
        view.pca()[i].eigenvectors()(k / 3, k % 3) = value(i, 11 + k);
      view.pca()[i].sigmas() = Eigen::Vector3f(value(i, 20), value(i, 21), value(i, 22));
      view.pca()[i].sigmasPCA() = Eigen::Vector3f(value(i, 23), value(i, 24), value(i, 25));
      for (int k = 0; k < ticl::kTracksterIdProbabilitiesSize; ++k)
        view.id()[i].idProbabilities()(k) = value(i, 26 + k);
      view.profile()[i].maxEnergyFraction() = value(i, 34);
      view.profile()[i].core2EnergyFraction() = value(i, 35);
      view.profile()[i].core5EnergyFraction() = value(i, 36);
      view.profile()[i].rmsRadius() = value(i, 37);
      view.profile()[i].minAbsZ() = value(i, 38);
      view.profile()[i].lengthZ() = value(i, 39);
    }
    for (int i = 0; i <= n; ++i)
      view.vertices().offsets()[i].keys_offsets() = numberOfVertices(i);
    for (int j = 0; j < view.vertices().content().metadata().size(); ++j)
      view.vertices().content()[j].values() = 1000 + j;
  }

  inline void check(ticl::TracksterSoAConstView view, int32_t n) {
    requireSize(view.energy().metadata().size(), n, "energy size");
    requireSize(view.barycenter().metadata().size(), n, "barycenter size");
    requireSize(view.timing().metadata().size(), n, "timing size");
    requireSize(view.pca().metadata().size(), n, "pca size");
    requireSize(view.id().metadata().size(), n, "id size");
    requireSize(view.profile().metadata().size(), n, "profile size");
    requireSize(view.vertices().offsets().metadata().size(), n + 1, "vertices offsets size");
    requireSize(view.vertices().keys(), n, "vertices keys");
    requireSize(view.vertices().content().metadata().size(), numberOfVertices(n), "vertices content size");
    for (int i = 0; i < n; ++i) {
      require(view.energy()[i].rawEnergy() == value(i, 0), "rawEnergy", i);
      require(view.energy()[i].rawEmEnergy() == value(i, 1), "rawEmEnergy", i);
      require(view.energy()[i].regressedEnergy() == value(i, 2), "regressedEnergy", i);
      require(view.barycenter()[i].x() == value(i, 3), "barycenter x", i);
      require(view.barycenter()[i].y() == value(i, 4), "barycenter y", i);
      require(view.barycenter()[i].z() == value(i, 5), "barycenter z", i);
      require(view.timing()[i].time() == value(i, 6), "time", i);
      require(view.timing()[i].timeError() == value(i, 7), "timeError", i);
      for (int k = 0; k < 3; ++k) {
        require(view.pca()[i].eigenvalues()(k) == value(i, 8 + k), "eigenvalues", i);
        require(view.pca()[i].sigmas()(k) == value(i, 20 + k), "sigmas", i);
        require(view.pca()[i].sigmasPCA()(k) == value(i, 23 + k), "sigmasPCA", i);
      }
      for (int k = 0; k < 9; ++k)
        require(view.pca()[i].eigenvectors()(k / 3, k % 3) == value(i, 11 + k), "eigenvectors", i);
      for (int k = 0; k < ticl::kTracksterIdProbabilitiesSize; ++k)
        require(view.id()[i].idProbabilities()(k) == value(i, 26 + k), "idProbabilities", i);
      require(view.profile()[i].maxEnergyFraction() == value(i, 34), "maxEnergyFraction", i);
      require(view.profile()[i].core2EnergyFraction() == value(i, 35), "core2EnergyFraction", i);
      require(view.profile()[i].core5EnergyFraction() == value(i, 36), "core5EnergyFraction", i);
      require(view.profile()[i].rmsRadius() == value(i, 37), "rmsRadius", i);
      require(view.profile()[i].minAbsZ() == value(i, 38), "minAbsZ", i);
      require(view.profile()[i].lengthZ() == value(i, 39), "lengthZ", i);
      require(view.vertices().count(i) == static_cast<uint32_t>(i + 1), "vertices count", i);
      for (int k = 0; k <= i; ++k)
        require(view.vertices()[i][k] == static_cast<uint32_t>(1000 + numberOfVertices(i) + k), "vertices", i);
    }
  }

  inline void fill(HGCalSoALayerClustersExtraHostCollection::View view) {
    const int n = view.metadata().size();
    for (int i = 0; i < n; ++i)
      view[i].tracksterIndex() = i % 3 - 1;
    view.numberOfTrackstersScalar() = n;
    view.numberOfVerticesScalar() = numberOfVertices(n);
  }

  inline void check(HGCalSoALayerClustersExtraHostCollection::ConstView view, int32_t n) {
    requireSize(view.metadata().size(), n, "layer clusters extra size");
    for (int i = 0; i < n; ++i)
      require(view[i].tracksterIndex() == i % 3 - 1, "tracksterIndex", i);
    require(view.numberOfTrackstersScalar() == static_cast<unsigned int>(n), "numberOfTrackstersScalar", n);
    require(
        view.numberOfVerticesScalar() == static_cast<unsigned int>(numberOfVertices(n)), "numberOfVerticesScalar", n);
  }

  inline void fill(ticl::LayerClusterMaskHostCollection::View view) {
    for (int i = 0; i < view.metadata().size(); ++i)
      view[i].mask() = static_cast<float>(i % 2);
  }

  inline void check(ticl::LayerClusterMaskHostCollection::ConstView view, int32_t n) {
    requireSize(view.metadata().size(), n, "mask size");
    for (int i = 0; i < n; ++i)
      require(view[i].mask() == static_cast<float>(i % 2), "mask", i);
  }

  inline void fill(ticl::TracksterPidHostCollection::View view) {
    for (int i = 0; i < view.metadata().size(); ++i)
      for (int k = 0; k < ticl::kTracksterIdProbabilitiesSize; ++k)
        view[i].idProbabilities()(k) = value(i, 40 + k);
  }

  inline void check(ticl::TracksterPidHostCollection::ConstView view, int32_t n) {
    requireSize(view.metadata().size(), n, "PID size");
    for (int i = 0; i < n; ++i)
      for (int k = 0; k < ticl::kTracksterIdProbabilitiesSize; ++k)
        require(view[i].idProbabilities()(k) == value(i, 40 + k), "PID idProbabilities", i);
  }

  inline ticl::TracksterLinksHostCollection makeLinks(int32_t n) {
    return ticl::TracksterLinksHostCollection(n, numberOfVertices(n));
  }

  // Linked trackster i has the input tracksters 2000 + numberOfVertices(i) + k, for k from 0 to i.
  inline void fill(ticl::TracksterLinksHostCollection::View view) {
    const int n = view.offsets().metadata().size() - 1;
    for (int i = 0; i <= n; ++i)
      view.offsets()[i].keys_offsets() = numberOfVertices(i);
    for (int j = 0; j < view.content().metadata().size(); ++j)
      view.content()[j].values() = 2000 + j;
  }

  inline void check(ticl::TracksterLinksHostCollection::ConstView view, int32_t n) {
    requireSize(view.offsets().metadata().size(), n + 1, "links offsets size");
    requireSize(view.keys(), n, "links keys");
    requireSize(view.size(), numberOfVertices(n), "links content size");
    for (int i = 0; i < n; ++i) {
      require(view.count(i) == static_cast<uint32_t>(i + 1), "links count", i);
      for (int k = 0; k <= i; ++k)
        require(view[i][k] == static_cast<uint32_t>(2000 + numberOfVertices(i) + k), "links", i);
    }
  }

}  // namespace hgcalrecotest

#endif  // DataFormats_HGCalReco_test_HGCalRecoSoATestValues_h
