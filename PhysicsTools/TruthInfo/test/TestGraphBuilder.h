// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#ifndef PhysicsTools_TruthInfo_test_TestGraphBuilder_h
#define PhysicsTools_TruthInfo_test_TestGraphBuilder_h

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "cppunit/extensions/HelperMacros.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"

namespace truth::test {

  // A truth::Graph built from explicit edges, for the unit tests. The vertices are GEN
  // vertices unless genVertices is false, as the decay vertices of a production graph.
  struct GraphBuilder {
    explicit GraphBuilder(uint32_t nParticles, uint32_t nVertices, bool genVertices = true) {
      graph.particles().resize(nParticles);
      graph.vertices().resize(nVertices);
      if (genVertices) {
        for (uint32_t v = 0; v < nVertices; ++v)
          graph.vertices()[v].genNode = 200 + v;
      }
    }
    void addDecay(uint32_t particleId, uint32_t vertexId) {
      d2v.emplace_back(particleId, vertexId);
      v2i.emplace_back(vertexId, particleId);
    }
    void addProduction(uint32_t vertexId, uint32_t particleId) {
      v2o.emplace_back(vertexId, particleId);
      p2v.emplace_back(particleId, vertexId);
    }
    truth::Graph finish() {
      csr(graph.nParticles(), d2v, graph.particleToDecayVertexOffsets(), graph.particleToDecayVertices());
      csr(graph.nParticles(), p2v, graph.particleToProductionVertexOffsets(), graph.particleToProductionVertices());
      csr(graph.nVertices(), v2o, graph.vertexToOutgoingParticleOffsets(), graph.vertexToOutgoingParticles());
      csr(graph.nVertices(), v2i, graph.vertexToIncomingParticleOffsets(), graph.vertexToIncomingParticles());
      CPPUNIT_ASSERT(graph.isConsistent());
      return graph;
    }

    truth::Graph graph;
    std::vector<std::pair<uint32_t, uint32_t>> d2v, p2v, v2o, v2i;

  private:
    static void csr(uint32_t n,
                    std::vector<std::pair<uint32_t, uint32_t>>& pairs,
                    std::vector<uint32_t>& off,
                    std::vector<uint32_t>& flat) {
      std::sort(pairs.begin(), pairs.end());
      pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
      off.assign(n + 1, 0);
      for (auto const& pr : pairs)
        ++off[pr.first + 1];
      for (uint32_t i = 1; i <= n; ++i)
        off[i] += off[i - 1];
      flat.assign(pairs.size(), 0);
      auto cur = off;
      for (auto const& pr : pairs)
        flat[cur[pr.first]++] = pr.second;
    }
  };

}  // namespace truth::test

#endif
