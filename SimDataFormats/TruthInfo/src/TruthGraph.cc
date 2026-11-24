// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

// A compact, read-only graph representation of the truth information in an event.
// The graph is built in the TruthGraphProducer module, which also fills the node metadata and associations.
// The graph is intended to be a common data format for various use cases (e.g. validation, analysis, visualization).

#include "SimDataFormats/TruthInfo/interface/DeltaCoding.h"
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

void TruthGraph::pack() {
  truth::delta::encode(offsets_, edgeCounts_);
  truth::delta::encode(edges_, edgeDeltas_);
  truth::delta::encodeRelativeToIndex(simTrackToVtx_, simTrackToVtxRelative_);
}

void TruthGraph::unpackOffsets(std::vector<uint32_t> const& stored, std::vector<uint32_t>& offsets) {
  truth::delta::decode(stored, offsets);
}

void TruthGraph::unpackEdges(std::vector<uint32_t> const& stored, std::vector<uint32_t>& edges) {
  truth::delta::decode(stored, edges);
}

void TruthGraph::unpackSimTrackToVtx(std::vector<int32_t> const& stored, std::vector<int32_t>& simTrackToVtx) {
  truth::delta::decodeRelativeToIndex(stored, simTrackToVtx);
}
