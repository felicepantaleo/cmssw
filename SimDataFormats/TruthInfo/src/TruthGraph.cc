// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "SimDataFormats/TruthInfo/interface/TruthGraph.h"

bool TruthGraph::isConsistent() const {
  if (offsets_.size() != nodes_.size() + 1)
    return false;
  if (!offsets_.empty() && offsets_.front() != 0)
    return false;
  if (!offsets_.empty() && offsets_.back() != edges_.size())
    return false;
  if (!edgeKind_.empty() && edgeKind_.size() != edges_.size())
    return false;

  for (size_t i = 1; i < offsets_.size(); ++i) {
    if (offsets_[i] < offsets_[i - 1])
      return false;
  }
  return true;
}
