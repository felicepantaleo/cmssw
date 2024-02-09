#ifndef RecoHGCal_TICL_interface_TICLGraph_h
#define RecoHGCal_TICL_interface_TICLGraph_h

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <unordered_set>
#include <vector>

class Node {
public:
  Node() = default;
  Node(unsigned index, bool isTrackster = true) : index_(index), isTrackster_(isTrackster), alreadyVisited_{false}{};
  
  void addInner(unsigned int trackster_id) { innerNodes_.push_back(trackster_id); addNeighbour(trackster_id);}
  void addOuter(unsigned int trackster_id) { outerNodes_.push_back(trackster_id); addNeighbour(trackster_id);}
  void addNeighbour(unsigned int trackster_id, double weight = 0.0) {
    neighboursId_.push_back(trackster_id);
    weights_.push_back(weight);
  }
  void updateWeight(unsigned int neighborIndex, float weight) {
        // Find the neighbor index in the list of neighbors
        auto it = std::find(neighboursId_.begin(), neighboursId_.end(), neighborIndex);
        if (it != neighboursId_.end()) {
            // Update the weight if the neighbor is found
            size_t index = std::distance(neighboursId_.begin(), it);
            weights_[index] = weight;
        }
    }
  

  const unsigned int getId() const { return index_; }
  std::vector<unsigned int> getNeighbours() const { return neighboursId_; }
  std::vector<float> getWeights() const { return weights_; }
  std::vector<unsigned int> getInner() const { return innerNodes_; }
  std::vector<unsigned int> getOuter() const { return outerNodes_; }
  
  void findSubComponents(std::vector<Node>& graph, std::vector<unsigned int>& subComponent, float threshold) {
    if (!alreadyVisited_) {
      alreadyVisited_ = true;
      subComponent.push_back(index_);
      
      // Using a const iterator since we don't intend to modify elements
      auto neighbourIt = neighboursId_.cbegin(); // Iterator for neighboursId_
      auto weightIt = weights_.cbegin(); // Iterator for weights_
      
      // Loop through both vectors simultaneously
      for (; neighbourIt != neighboursId_.cend() && weightIt != weights_.cend(); ++neighbourIt, ++weightIt) {
          int neighbour = *neighbourIt;
          double weight = *weightIt;
          
          if (weight >= threshold){
            graph[neighbour].findSubComponents(graph, subComponent, threshold);
          }
      }
    }
  }

  ~Node() = default;

private:
  unsigned index_;
  bool isTrackster_;

  std::vector<unsigned int> neighboursId_;
  std::vector<float> weights_;
  std::vector<unsigned int> innerNodes_;
  std::vector<unsigned int> outerNodes_;
  bool alreadyVisited_;

  //bool areCompatible(const std::vector<Node>& graph, const unsigned int& outerNode) { return true; };

};

class TICLGraph {
public:
  TICLGraph() = default;
  TICLGraph(std::vector<Node> &n) { nodes_ = n; };
  
  const std::vector<Node>& getNodes() const { return nodes_; }
  const Node& getNode(int i) const { return nodes_[i]; }
  
  void setEdgeWeight(unsigned int nodeIndexI, unsigned int nodeIndexJ, float weight) {
        // Check if the node indices are valid
        if (nodeIndexI >= nodes_.size() || nodeIndexJ >= nodes_.size()) {
            // Handle invalid node indices
            return;
        }
        // Update the weight for the edge between node i and node j
        nodes_[nodeIndexI].updateWeight(nodeIndexJ, weight);
        // If j is bidirectionally connected, update the weight for j -> i as well
        nodes_[nodeIndexJ].updateWeight(nodeIndexI, weight);
  }

  std::vector<std::vector<unsigned int>> findSubComponents(float threshold) {
    std::vector<std::vector<unsigned int>> components;
    for (auto const& node: nodes_) {
      auto const id = node.getId();
      std::vector<unsigned int> tmpSubComponents;
      nodes_[id].findSubComponents(nodes_, tmpSubComponents, threshold);
      if (!tmpSubComponents.empty()) {
        components.push_back(tmpSubComponents);
      }
    }
    return components;
  }

  ~TICLGraph() = default;

  void dfsForCC(unsigned int nodeIndex,
                std::unordered_set<unsigned int>& visited,
                std::vector<unsigned int>& component) const {
    visited.insert(nodeIndex);
    component.push_back(nodeIndex);

    for (auto const& neighbourIndex : nodes_[nodeIndex].getNeighbours()) {
      if (visited.find(neighbourIndex) == visited.end()) {
        dfsForCC(neighbourIndex, visited, component);
      }
    }
  }

  std::vector<std::vector<unsigned int>> getConnectedComponents() const {
    std::unordered_set<unsigned int> visited;
    std::vector<std::vector<unsigned int>> components;

    for (unsigned int i = 0; i < nodes_.size(); ++i) {
      if (visited.find(i) == visited.end()) {
        std::vector<unsigned int> component;
        dfsForCC(i, visited, component);
        components.push_back(component);
      }
    }

    return components;
  }

private:
  std::vector<Node> nodes_;
};

#endif