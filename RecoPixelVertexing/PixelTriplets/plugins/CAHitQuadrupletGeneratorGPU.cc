//
// Author: Felice Pantaleo, CERN
//
#include "CAHitQuadrupletGeneratorGPU.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

#include "CellularAutomaton.h"

#include "CommonTools/Utils/interface/DynArray.h"

#include "FWCore/Utilities/interface/isFinite.h"

#include <functional>

namespace {

template <typename T> T sqr(T x) { return x * x; }
} // namespace

using namespace std;

constexpr unsigned int CAHitQuadrupletGeneratorGPU::minLayers;

CAHitQuadrupletGeneratorGPU::CAHitQuadrupletGeneratorGPU(
    const edm::ParameterSet &cfg,
    edm::ConsumesCollector &iC)
    : extraHitRPhitolerance(cfg.getParameter<double>(
          "extraHitRPhitolerance")), // extra window in
                                     // ThirdHitPredictionFromCircle range
                                     // (divide by R to get phi)
      maxChi2(cfg.getParameter<edm::ParameterSet>("maxChi2")),
      fitFastCircle(cfg.getParameter<bool>("fitFastCircle")),
      fitFastCircleChi2Cut(cfg.getParameter<bool>("fitFastCircleChi2Cut")),
      useBendingCorrection(cfg.getParameter<bool>("useBendingCorrection")),
      caThetaCut(cfg.getParameter<double>("CAThetaCut")),
      caPhiCut(cfg.getParameter<double>("CAPhiCut")),
      caHardPtCut(cfg.getParameter<double>("CAHardPtCut")) {
  edm::ParameterSet comparitorPSet =
      cfg.getParameter<edm::ParameterSet>("SeedComparitorPSet");
  std::string comparitorName =
      comparitorPSet.getParameter<std::string>("ComponentName");
  if (comparitorName != "none") {
    theComparitor.reset(SeedComparitorFactory::get()->create(
        comparitorName, comparitorPSet, iC));
  }

  allocateOnGPU();
}

void CAHitQuadrupletGeneratorGPU::fillDescriptions(
    edm::ParameterSetDescription &desc) {
  desc.add<double>("extraHitRPhitolerance", 0.1);
  desc.add<bool>("fitFastCircle", false);
  desc.add<bool>("fitFastCircleChi2Cut", false);
  desc.add<bool>("useBendingCorrection", false);
  desc.add<double>("CAThetaCut", 0.00125);
  desc.add<double>("CAPhiCut", 10);
  desc.add<double>("CAHardPtCut", 0);
  desc.addOptional<bool>("CAOnlyOneLastHitPerLayerFilter")
      ->setComment(
          "Deprecated and has no effect. To be fully removed later when the "
          "parameter is no longer used in HLT configurations.");
  edm::ParameterSetDescription descMaxChi2;
  descMaxChi2.add<double>("pt1", 0.2);
  descMaxChi2.add<double>("pt2", 1.5);
  descMaxChi2.add<double>("value1", 500);
  descMaxChi2.add<double>("value2", 50);
  descMaxChi2.add<bool>("enabled", true);
  desc.add<edm::ParameterSetDescription>("maxChi2", descMaxChi2);

  edm::ParameterSetDescription descComparitor;
  descComparitor.add<std::string>("ComponentName", "none");
  descComparitor.setAllowAnything(); // until we have moved SeedComparitor too
                                     // to EDProducers
  desc.add<edm::ParameterSetDescription>("SeedComparitorPSet", descComparitor);
}

void CAHitQuadrupletGeneratorGPU::initEvent(const edm::Event &ev,
                                            const edm::EventSetup &es) {
  if (theComparitor)
    theComparitor->init(ev, es);
}

CAHitQuadrupletGeneratorGPU::~CAHitQuadrupletGeneratorGPU() {

  deallocateOnGPU();
}

namespace {
void createGraphStructure(const SeedingLayerSetsHits &layers, CAGraph &g) {
  for (unsigned int i = 0; i < layers.size(); i++) {
    for (unsigned int j = 0; j < 4; ++j) {
      auto vertexIndex = 0;
      auto foundVertex = std::find(g.theLayers.begin(), g.theLayers.end(),
                                   layers[i][j].name());
      if (foundVertex == g.theLayers.end()) {
        g.theLayers.emplace_back(layers[i][j].name(),
                                 layers[i][j].hits().size());
        vertexIndex = g.theLayers.size() - 1;
      } else {
        vertexIndex = foundVertex - g.theLayers.begin();
      }
      if (j == 0) {

        if (std::find(g.theRootLayers.begin(), g.theRootLayers.end(),
                      vertexIndex) == g.theRootLayers.end()) {
          g.theRootLayers.emplace_back(vertexIndex);
        }
      }
    }
  }
}
void clearGraphStructure(const SeedingLayerSetsHits &layers, CAGraph &g) {
  g.theLayerPairs.clear();
  for (unsigned int i = 0; i < g.theLayers.size(); i++) {
    g.theLayers[i].theInnerLayers.clear();
    g.theLayers[i].theInnerLayerPairs.clear();
    g.theLayers[i].theOuterLayers.clear();
    g.theLayers[i].theOuterLayerPairs.clear();
    for (auto &v : g.theLayers[i].isOuterHitOfCell)
      v.clear();
  }
}
void fillGraph(const SeedingLayerSetsHits &layers,
               const IntermediateHitDoublets::RegionLayerSets &regionLayerPairs,
               CAGraph &g, std::vector<const HitDoublets *> &hitDoublets) {

  for (unsigned int i = 0; i < layers.size(); i++) {
    for (unsigned int j = 0; j < 4; ++j) {
      auto vertexIndex = 0;
      auto foundVertex = std::find(g.theLayers.begin(), g.theLayers.end(),
                                   layers[i][j].name());
      if (foundVertex == g.theLayers.end()) {
        vertexIndex = g.theLayers.size() - 1;
      } else {
        vertexIndex = foundVertex - g.theLayers.begin();
      }

      if (j > 0) {

        auto innerVertex = std::find(g.theLayers.begin(), g.theLayers.end(),
                                     layers[i][j - 1].name());

        CALayerPair tmpInnerLayerPair(innerVertex - g.theLayers.begin(),
                                      vertexIndex);

        if (std::find(g.theLayerPairs.begin(), g.theLayerPairs.end(),
                      tmpInnerLayerPair) == g.theLayerPairs.end()) {
          auto found = std::find_if(
              regionLayerPairs.begin(), regionLayerPairs.end(),
              [&](const IntermediateHitDoublets::LayerPairHitDoublets &pair) {
                return pair.innerLayerIndex() == layers[i][j - 1].index() &&
                       pair.outerLayerIndex() == layers[i][j].index();
              });
          if (found != regionLayerPairs.end()) {
            hitDoublets.emplace_back(&(found->doublets()));
            g.theLayerPairs.push_back(tmpInnerLayerPair);
            g.theLayers[vertexIndex].theInnerLayers.push_back(
                innerVertex - g.theLayers.begin());
            innerVertex->theOuterLayers.push_back(vertexIndex);
            g.theLayers[vertexIndex].theInnerLayerPairs.push_back(
                g.theLayerPairs.size() - 1);
            innerVertex->theOuterLayerPairs.push_back(g.theLayerPairs.size() -
                                                      1);
          }
        }
      }
    }
  }

  std::cout << "number of layer pairs" << hitDoublets.size() << std::endl;

  for (unsigned int i = 0; i < hitDoublets.size(); ++i) {
    std::cout << i << " " << hitDoublets[i]->size() << std::endl;
  }

  for (unsigned int i = 0; i < g.theLayerPairs.size(); ++i) {
    std::cout << "layer pair " << i << " " << g.theLayers[g.theLayerPairs[i].theLayers[0]].name()
    <<  " hits: " << hitDoublets[i]->layers[0]->size()
              << " and " << g.theLayers[g.theLayerPairs[i].theLayers[1]].name()
              << " hits: " << hitDoublets[i]->layers[1]->size()
              << std::endl;
  }
}
} // namespace

void CAHitQuadrupletGeneratorGPU::hitNtuplets(
    const IntermediateHitDoublets &regionDoublets,
    std::vector<OrderedHitSeeds> &result, const edm::EventSetup &es,
    const SeedingLayerSetsHits &layers) {
  CAGraph g;

  std::vector<const HitDoublets *> hitDoublets;

  const int numberOfHitsInNtuplet = 4;
  std::vector<CACell::CAntuplet> foundQuadruplets;

  for (unsigned int lpIdx = 0; lpIdx < maxNumberOfLayerPairs; ++lpIdx) {
    h_doublets[lpIdx].size = 0;
  }
  numberOfRootLayerPairs = 0;

  for (unsigned int layerIdx = 0; layerIdx < maxNumberOfLayers; ++layerIdx) {

    h_layers[layerIdx].size = 0;
  }

  int index = 0;
  for (const auto &regionLayerPairs : regionDoublets) {

    const TrackingRegion &region = regionLayerPairs.region();
    hitDoublets.clear();
    foundQuadruplets.clear();
    if (index == 0) {
      createGraphStructure(layers, g);
    } else {
      clearGraphStructure(layers, g);
    }

    fillGraph(layers, regionLayerPairs, g, hitDoublets);
  }

  for (unsigned int i = 0; i < hitDoublets.size(); ++i) {
    h_doublets[i].size = hitDoublets[i]->size();
    h_doublets[i].innerLayerId = g.theLayerPairs[i].theLayers[0];
    h_doublets[i].outerLayerId = g.theLayerPairs[i].theLayers[1];
    for (unsigned int rl : g.theRootLayers) {
      if (rl == h_doublets[i].innerLayerId) {
        auto rootlayerPairId = numberOfRootLayerPairs;
        h_rootLayerPairs[rootlayerPairId] = i;
        numberOfRootLayerPairs++;
      }
    }

    for (unsigned int l = 0; l < hitDoublets[i]->size(); ++l) {
      auto hitId = i * maxNumberOfDoublets * 2 + 2 * l;
      h_indices[hitId] = hitDoublets[i]->indeces[l].first;
      h_indices[hitId + 1] = hitDoublets[i]->indeces[l].second;
    }

    if(h_layers[h_doublets[i].innerLayerId].size == 0)
    {
        auto & layer = h_layers[h_doublets[i].innerLayerId];
        layer.size = hitDoublets[i]->layers[0]->size();
        layer.layerId = h_doublets[i].innerLayerId;

        for (unsigned int l = 0; l < layer.size; ++l)
        {
            auto hitId = layer.layerId * maxNumberOfHits + l;

            h_x[hitId] = hitDoublets[i]->layers[0]->x[l];
            h_y[hitId] = hitDoublets[i]->layers[0]->y[l];
            h_z[hitId] = hitDoublets[i]->layers[0]->z[l];
            // h_y[hitId] = hostEvents[i].hitsLayers[j].y[l];
            // h_z[hitId] = hostEvents[i].hitsLayers[j].z[l];
        }
    }

    if(h_layers[h_doublets[i].outerLayerId].size == 0)
    {
        auto & layer = h_layers[h_doublets[i].outerLayerId];
        layer.size = hitDoublets[i]->layers[1]->size();
        layer.layerId = h_doublets[i].outerLayerId;
        for (unsigned int l = 0; l < layer.size; ++l)
        {
            auto hitId = layer.layerId * maxNumberOfHits + l;

            h_x[hitId] = hitDoublets[i]->layers[1]->x[l];
            h_y[hitId] = hitDoublets[i]->layers[1]->y[l];
            h_z[hitId] = hitDoublets[i]->layers[1]->z[l];

        }

    }

    // for (unsigned int j = 0; j < hostEvents[i].hitsLayers.size(); ++j)
    // {
    //     auto layerIdx = i * maxNumberOfLayers + j;
    //
    //     h_layers[layerIdx].size = hostEvents[i].hitsLayers[j].size;
    //     h_layers[layerIdx].layerId = hostEvents[i].hitsLayers[j].layerId;
    //     for (unsigned int l = 0; l < hostEvents[i].hitsLayers[j].size; ++l)
    //     {
    //         auto hitId = layerIdx * maxNumberOfHits + l;
    //
    //         h_x[hitId] = hostEvents[i].hitsLayers[j].x[l];
    //         h_y[hitId] = hostEvents[i].hitsLayers[j].y[l];
    //         h_z[hitId] = hostEvents[i].hitsLayers[j].z[l];
    //     }
    // }
  }
}
