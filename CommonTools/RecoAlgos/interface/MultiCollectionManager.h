// Author: Felice Pantaleo (CERN), 2025, felice.pantaleo@cern.ch
#ifndef CommonTools_RecoAlgos_MultiCollectionManager_h
#define CommonTools_RecoAlgos_MultiCollectionManager_h


#include <span>
#include <utility>
#include <vector>

#include "DataFormats/Common/interface/RefProd.h"
#include "MultiVectorManager.h"

/**
 * @brief Lightweight persistent holder for several `edm::RefProd<Collection>`
 *        objects.
 *
 * Only the vector of `RefProd`s is stored on disk.  No transient caches or
 * synchronisation primitives live inside the class, so it is trivially
 * movable.  Consumers obtain a flat view with `makeFlatView()`, which assembles
 * and *returns* a fully‑populated `MultiVectorManager` by value.
 *
 * This design avoids copy/move issues with `std::once_flag` and makes the type
 * EDM‑wrapper‑friendly while still giving fast, indexed access to the
 * concatenated elements.
 */
template <typename Collection>
class MultiCollectionManager {
 public:
  using value_type = typename Collection::value_type;

  MultiCollectionManager() = default;

  explicit MultiCollectionManager(std::initializer_list<edm::RefProd<Collection>> refs)
      : refProds_{refs} {}

  // ---------------- producer‑side API ----------------------------------
  void addCollection(edm::RefProd<Collection> const& ref) { refProds_.push_back(ref); }

  // ---------------- consumer‑side helpers ------------------------------
  /**
   * @brief Build and return a flat view that spans all referenced collections.
   *
   * The returned `MultiVectorManager` is independent of `this`, so callers may
   * move or store it locally without keeping the manager alive.
   */
  [[nodiscard]] MultiVectorManager<value_type> makeFlatView() const {
    MultiVectorManager<value_type> mv;
    for (auto const& rp : refProds_) {
      auto const& coll = *rp;  // Framework‑managed retrieval
      mv.addVector(std::span<const value_type>(coll.data(), coll.size()));
    }
    return mv;
  }

  const std::vector<edm::RefProd<Collection>>& refProds() const { return refProds_; }

 private:
  std::vector<edm::RefProd<Collection>> refProds_;
};

#endif  // CommonTools_RecoAlgos_MultiCollectionManager_h
