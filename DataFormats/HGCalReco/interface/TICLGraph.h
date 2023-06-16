#ifndef DataFormats_HGCalReco_TICLGraph_h
#define DataFormats_HGCalReco_TICLGraph_h

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"

class Node {
public:
  Node() = default;
  Node(unsigned index, bool isTrackster = true) : index_(index), isTrackster_(isTrackster){};

  void addInner(unsigned int trackster_id) { innerNodes_.push_back(trackster_id); }
  void addOuter(unsigned int trackster_id) { outerNodes_.push_back(trackster_id); }

  const unsigned int getId() const { return index_; }
  std::vector<unsigned int> getInner() const { return innerNodes_; }
  std::vector<unsigned int> getOuter() const { return outerNodes_; }

  ~Node() = default;

private:
  unsigned index_;
  bool isTrackster_;
  std::vector<unsigned int> innerNodes_;
  std::vector<unsigned int> outerNodes_;
};

class TICLGraph {
public:
  TICLGraph() = default;
  TICLGraph(std::vector<Node> &n) { nodes_ = n; };
  const std::vector<Node> &getNodes() const { return nodes_; }
  const Node &getNode(int i) const { return nodes_[i]; }
  ~TICLGraph() = default;

private:
  std::vector<Node> nodes_;
};

#endif