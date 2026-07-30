#ifndef RecoHGCal_TICL_ClusterFilterByAlgoAndSizeAndEnergy_h
#define RecoHGCal_TICL_ClusterFilterByAlgoAndSizeAndEnergy_h

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "ClusterFilterBase.h"

#include <memory>
#include <utility>

// ClusterFilterByAlgoAndSize plus a minimum layer-cluster energy.
//
// Motivation (measured, TenTau PU200 D122): one endcap event holds ~146k layer
// clusters of median energy 0.06 GeV, and the ~92k that CLUE3DHigh leaves over (36%
// of the layer-cluster energy) each become their own Recovery trackster. Those
// singletons are what the linking stage then has to carry: they are the substrate a
// connected-components linker percolates through, and they dominate its cost. Cutting
// them at 0.05 GeV removes ~60% of the leftovers while keeping 83% of their energy.
//
// Threshold guidance from that spectrum: 0.05 GeV keeps 83.5% of the leftover energy,
// 0.1 GeV keeps 67%, 0.5 GeV keeps only 26% (a 0.5 GeV LAYER CLUSTER is in the top 6%
// of the whole event, so a half-GeV cut here deletes the recovery pass rather than
// cleaning it). This is an INPUT-side cut: unlike an emit-side one it changes which
// objects can exist and therefore which links can form, so it must be A/B'd rather
// than assumed safe.
namespace ticl {
  class ClusterFilterByAlgoAndSizeAndEnergy final : public ClusterFilterBase {
  public:
    ClusterFilterByAlgoAndSizeAndEnergy(const edm::ParameterSet& ps)
        : ClusterFilterBase(ps),
          algo_number_(ps.getParameter<std::vector<int>>("algo_number")),
          min_cluster_size_(ps.getParameter<int>("min_cluster_size")),
          max_cluster_size_(ps.getParameter<int>("max_cluster_size")),
          min_cluster_energy_(ps.getParameter<double>("min_cluster_energy")) {}
    ~ClusterFilterByAlgoAndSizeAndEnergy() override {}

    void filter(const std::vector<reco::CaloCluster>& layerClusters,
                std::vector<float>& layerClustersMask,
                hgcal::RecHitTools& rhtools) const override {
      for (size_t i = 0; i < layerClusters.size(); i++) {
        if ((find(algo_number_.begin(), algo_number_.end(), layerClusters[i].algo()) == algo_number_.end()) or
            (layerClusters[i].hitsAndFractions().size() > max_cluster_size_) or
            (layerClusters[i].energy() < min_cluster_energy_) or
            ((layerClusters[i].hitsAndFractions().size() < min_cluster_size_) and
             (rhtools.isSilicon(layerClusters[i].hitsAndFractions()[0].first)))) {
          layerClustersMask[i] = 0.;
        }
      }
    }

  private:
    std::vector<int> algo_number_;
    unsigned int min_cluster_size_;
    unsigned int max_cluster_size_;
    double min_cluster_energy_;
  };
}  // namespace ticl

#endif
