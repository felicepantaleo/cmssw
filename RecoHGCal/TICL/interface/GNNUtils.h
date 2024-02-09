#ifndef RecoHGCal_TICL_interface_GNNUtils_h
#define RecoHGCal_TICL_interface_GNNUtils_h

#include <vector>
#include <map>
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"

namespace ticl {

  std::vector<float> countDegree(std::vector<float> src, std::vector<float> dst, int N) { 
      std::vector<float> degrees;
      std::map<int, int> mp;
      for (const auto& i : src) {
          mp[int(i)]++;
      }
      for (const auto& i : dst) {
          mp[int(i)]++;
      }
      for (auto x : mp){ 
          degrees.push_back(x.second/float(N-1));
      }
      return degrees;
  }

  std::unique_ptr<TICLGraph> produce_ticl_graph(const MultiVectorManager<Trackster> &trackstersclue3d, double delta_eta, double delta_phi) {
    // TICL graph producer

    TICLLayerTile tracksterTilePos;
    TICLLayerTile tracksterTileNeg;
  
    for (size_t id_t = 0; id_t < trackstersclue3d.size(); ++id_t) {
      auto t = trackstersclue3d[id_t];
      if (t.barycenter().eta() > 0.) {
        tracksterTilePos.fill(t.barycenter().eta(), t.barycenter().phi(), id_t);
      } else if (t.barycenter().eta() < 0.) {
        tracksterTileNeg.fill(t.barycenter().eta(), t.barycenter().phi(), id_t);
      }
    }
  
    std::vector<Node> allNodes;
  
    for (size_t id_t = 0; id_t < trackstersclue3d.size(); ++id_t) {
      auto t = trackstersclue3d[id_t];
  
      Node tNode(id_t);
  
      auto bary = t.barycenter();
      double del = delta_eta / 2;
  
      double eta_min = std::max(abs(bary.eta()) - del, (double)TileConstants::minEta);
      double eta_max = std::min(abs(bary.eta()) + del, (double)TileConstants::maxEta);
      
      del = delta_phi / 2;
  
      if (bary.eta() > 0.) {
        std::array<int, 4> search_box =
            tracksterTilePos.searchBoxEtaPhi(eta_min, eta_max, bary.phi() - del, bary.phi() + del);
        if (search_box[2] > search_box[3]) {
          search_box[3] += TileConstants::nPhiBins;
        }
  
        for (int eta_i = search_box[0]; eta_i <= search_box[1]; ++eta_i) {
          for (int phi_i = search_box[2]; phi_i <= search_box[3]; ++phi_i) {
            auto &neighbours = tracksterTilePos[tracksterTilePos.globalBin(eta_i, (phi_i % TileConstants::nPhiBins))];
            for (auto n : neighbours) {
              if (trackstersclue3d[n].barycenter().z() < bary.z()) {
                tNode.addInner(n);
              } else if (trackstersclue3d[n].barycenter().z() > bary.z()) {
                tNode.addOuter(n);
              }
            }
          }
        }
      }
  
      else if (bary.eta() < 0.) {
        std::array<int, 4> search_box =
            tracksterTileNeg.searchBoxEtaPhi(eta_min, eta_max, bary.phi() - del, bary.phi() + del);
        if (search_box[2] > search_box[3]) {
          search_box[3] += TileConstants::nPhiBins;
        }
  
        for (int eta_i = search_box[0]; eta_i <= search_box[1]; ++eta_i) {
          for (int phi_i = search_box[2]; phi_i <= search_box[3]; ++phi_i) {
            auto &neighbours = tracksterTileNeg[tracksterTileNeg.globalBin(eta_i, (phi_i % TileConstants::nPhiBins))];
            for (auto n : neighbours) {
              if (abs(trackstersclue3d[n].barycenter().z()) < abs(bary.z())) {
                tNode.addInner(n);
              } else if (abs(trackstersclue3d[n].barycenter().z()) > abs(bary.z())) {
                tNode.addOuter(n);
              }
            }
          }
        }
      }
      allNodes.push_back(tNode);
    }
    auto resultGraph = std::make_unique<TICLGraph>(allNodes);
  
    return resultGraph;
  }


  std::vector<float> calculate_dist(const std::vector<std::vector<float>> &lcs_i, const std::vector<std::vector<float>> &lcs_j)
  {
      std::vector<float> dists;
  
      // Iterate through each point in lcs_i and lcs_j
      for (const auto &point_i : lcs_i){
          for (const auto &point_j : lcs_j){
              // Calculate Euclidean distance between point_i and point_j and push it to dists
              dists.push_back(std::sqrt(std::pow((point_i[0] - point_j[0]), 2) + std::pow((point_i[1] - point_j[1]), 2) + std::pow((point_i[2] - point_j[2]), 2)));
          }
      }
      return dists;
  }

}  // namespace ticl

#endif
