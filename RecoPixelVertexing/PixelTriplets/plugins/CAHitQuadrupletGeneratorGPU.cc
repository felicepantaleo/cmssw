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
}
} // namespace
