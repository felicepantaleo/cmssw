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
void createGraphStructure(const SeedingLayerSetsHits &layers, CAGraph &g,
                          GPULayerHits *h_layers, unsigned int maxNumberOfHits,
                          float *h_x, float *h_y, float *h_z) {
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

  //   std::cout << "number of layer pairs " << hitDoublets.size() << std::endl;
  //
  // for (unsigned int i = 0; i < hitDoublets.size(); ++i) {
  //   std::cout << i << " " << hitDoublets[i]->size() << std::endl;
  // }
  //
  // for (unsigned int i = 0; i < g.theLayerPairs.size(); ++i) {
  //   std::cout << "layer pair " << i << " "
  //             << g.theLayers[g.theLayerPairs[i].theLayers[0]].name()
  //             << " hits: " << hitDoublets[i]->innerLayer().size() << " and "
  //             << g.theLayers[g.theLayerPairs[i].theLayers[1]].name()
  //             << " hits: " << hitDoublets[i]->outerLayer().size() <<
  //             std::endl;
  // }
}
} // namespace

void CAHitQuadrupletGeneratorGPU::hitNtuplets(
    const IntermediateHitDoublets &regionDoublets,
    std::vector<OrderedHitSeeds> &result, const edm::EventSetup &es,
    const SeedingLayerSetsHits &layers) {
  CAGraph g;

  std::vector<const HitDoublets *> hitDoublets;

  const int numberOfHitsInNtuplet = 4;

  for (unsigned int lpIdx = 0; lpIdx < maxNumberOfLayerPairs; ++lpIdx) {
    h_doublets[lpIdx].size = 0;
  }
  numberOfRootLayerPairs = 0;
  numberOfLayerPairs = 0;
  numberOfLayers = 0;

  for (unsigned int layerIdx = 0; layerIdx < maxNumberOfLayers; ++layerIdx) {

    h_layers[layerIdx].size = 0;
  }

  int index = 0;
  for (const auto &regionLayerPairs : regionDoublets) {

    const TrackingRegion &region = regionLayerPairs.region();
    hitDoublets.clear();
    if (index == 0) {
      createGraphStructure(layers, g, h_layers, maxNumberOfHits, h_x, h_y, h_z);
    } else {
      clearGraphStructure(layers, g);
    }

    fillGraph(layers, regionLayerPairs, g, hitDoublets);
    numberOfLayers = g.theLayers.size();

    numberOfLayerPairs = hitDoublets.size();
    std::vector<bool> layerAlreadyParsed(g.theLayers.size(), false);
    for (unsigned int i = 0; i < numberOfLayerPairs; ++i) {

      h_doublets[i].size = hitDoublets[i]->size();
      // std::cout << "layerPair " << i << " has doublets: " <<
      // h_doublets[i].size << std::endl;

      h_doublets[i].innerLayerId = g.theLayerPairs[i].theLayers[0];
      h_doublets[i].outerLayerId = g.theLayerPairs[i].theLayers[1];
      if (layerAlreadyParsed[h_doublets[i].innerLayerId] == false) {
        layerAlreadyParsed[h_doublets[i].innerLayerId] = true;

        h_layers[h_doublets[i].innerLayerId].size =
            hitDoublets[i]->innerLayer().hits().size();
        h_layers[h_doublets[i].innerLayerId].layerId =
            h_doublets[i].innerLayerId;

        for (unsigned int l = 0; l < h_layers[h_doublets[i].innerLayerId].size;
             ++l) {
          auto hitId =
              h_layers[h_doublets[i].innerLayerId].layerId * maxNumberOfHits +
              l;
          h_x[hitId] =
              hitDoublets[i]->innerLayer().hits()[l]->globalPosition().x();
          h_y[hitId] =
              hitDoublets[i]->innerLayer().hits()[l]->globalPosition().y();
          h_z[hitId] =
              hitDoublets[i]->innerLayer().hits()[l]->globalPosition().z();
        }
      }
      if (layerAlreadyParsed[h_doublets[i].outerLayerId] == false) {
        layerAlreadyParsed[h_doublets[i].outerLayerId] = true;

        h_layers[h_doublets[i].outerLayerId].size =
            hitDoublets[i]->outerLayer().hits().size();
        h_layers[h_doublets[i].outerLayerId].layerId =
            h_doublets[i].outerLayerId;

        for (unsigned int l = 0; l < h_layers[h_doublets[i].outerLayerId].size;
             ++l) {
          auto hitId =
              h_layers[h_doublets[i].outerLayerId].layerId * maxNumberOfHits +
              l;
          h_x[hitId] =
              hitDoublets[i]->outerLayer().hits()[l]->globalPosition().x();
          h_y[hitId] =
              hitDoublets[i]->outerLayer().hits()[l]->globalPosition().y();
          h_z[hitId] =
              hitDoublets[i]->outerLayer().hits()[l]->globalPosition().z();
        }
      }

      for (unsigned int rl : g.theRootLayers) {
        if (rl == h_doublets[i].innerLayerId) {
          auto rootlayerPairId = numberOfRootLayerPairs;
          h_rootLayerPairs[rootlayerPairId] = i;
          numberOfRootLayerPairs++;
        }
      }

      for (unsigned int l = 0; l < hitDoublets[i]->size(); ++l) {
        auto hitId = i * maxNumberOfDoublets * 2 + 2 * l;
        assert(maxNumberOfDoublets >= hitDoublets[i]->size());
        h_indices[hitId] = hitDoublets[i]->innerHitId(l);
        h_indices[hitId + 1] = hitDoublets[i]->outerHitId(l);
      }
    }

    for (unsigned int j = 0; j < numberOfLayers; ++j) {
      // std::cout << std::hex <<&h_layers[j] << " " << std::dec <<
      // h_layers[j].layerId << " " << h_layers[j].size << std::endl;
      for (unsigned int l = 0; l < h_layers[j].size; ++l) {
        auto hitId = h_layers[j].layerId * maxNumberOfHits + l;
        // std::cout << " " << h_x[hitId]<< " "<<h_y[hitId]<< " "<<h_z[hitId]<<
        // " "<< std::endl;
        assert(h_x[hitId] != 0);
      }
    }

    // //DEBUG!!!!
    //   for (unsigned int i = 0; i < numberOfLayerPairs; ++i) {
    //
    //
    //     std::cout << "layerPair " << i << " has doublets: " <<
    //     h_doublets[i].size
    //     << " on layers "  << h_doublets[i].innerLayerId << " and " <<
    //     h_doublets[i].outerLayerId << std::endl;
    //
    //     for (unsigned int l = 0; l < hitDoublets[i]->size(); ++l) {
    //       auto hitId = i * maxNumberOfDoublets * 2 + 2 * l;
    //       auto innerHitOffset = h_doublets[i].innerLayerId * maxNumberOfHits;
    //       auto outerHitOffset = h_doublets[i].outerLayerId * maxNumberOfHits;
    //
    //       std::cout << "Doublet " << l << " has hit " << h_indices[hitId] <<
    //       ": " << h_x[innerHitOffset+h_indices[hitId]] << " , " <<
    //       h_y[innerHitOffset+h_indices[hitId]] << " , " <<
    //       h_z[innerHitOffset+h_indices[hitId]] << " and outer hit " <<
    //       h_indices[hitId+1] << " : " <<
    //       h_x[outerHitOffset+h_indices[hitId+1]] << " , "
    //       <<h_y[outerHitOffset+h_indices[hitId+1]] << " , " <<
    //       h_z[outerHitOffset+h_indices[hitId+1]] << std::endl;
    //     }
    //   }

    for (unsigned int j = 0; j < numberOfLayerPairs; ++j) {
      tmp_layerDoublets[j] = h_doublets[j];
      tmp_layerDoublets[j].indices = &d_indices[j * maxNumberOfDoublets * 2];
      cudaMemcpyAsync(&d_indices[j * maxNumberOfDoublets * 2],
                      &h_indices[j * maxNumberOfDoublets * 2],
                      tmp_layerDoublets[j].size * 2 * sizeof(int),
                      cudaMemcpyHostToDevice, cudaStream_);
    }

    for (unsigned int j = 0; j < numberOfLayers; ++j) {
      tmp_layers[j] = h_layers[j];
      tmp_layers[j].x = &d_x[maxNumberOfHits * j];

      cudaMemcpyAsync(&d_x[maxNumberOfHits * j], &h_x[j * maxNumberOfHits],
                      tmp_layers[j].size * sizeof(float),
                      cudaMemcpyHostToDevice, cudaStream_);

      tmp_layers[j].y = &d_y[maxNumberOfHits * j];
      cudaMemcpyAsync(&d_y[maxNumberOfHits * j], &h_y[j * maxNumberOfHits],
                      tmp_layers[j].size * sizeof(float),
                      cudaMemcpyHostToDevice, cudaStream_);

      tmp_layers[j].z = &d_z[maxNumberOfHits * j];

      cudaMemcpyAsync(&d_z[maxNumberOfHits * j], &h_z[j * maxNumberOfHits],
                      tmp_layers[j].size * sizeof(float),
                      cudaMemcpyHostToDevice, cudaStream_);
    }

    cudaMemcpyAsync(d_rootLayerPairs, h_rootLayerPairs,
                    numberOfRootLayerPairs * sizeof(unsigned int),
                    cudaMemcpyHostToDevice, cudaStream_);
    cudaMemcpyAsync(d_doublets, tmp_layerDoublets,
                    numberOfLayerPairs * sizeof(GPULayerDoublets),
                    cudaMemcpyHostToDevice, cudaStream_);
    cudaMemcpyAsync(d_layers, tmp_layers, numberOfLayers * sizeof(GPULayerHits),
                    cudaMemcpyHostToDevice, cudaStream_);

    auto foundQuads = launchKernels(region);
    unsigned int numberOfFoundQuadruplets = foundQuads.size();
    const QuantityDependsPtEval maxChi2Eval = maxChi2.evaluator(es);

    // re-used thoughout
    std::array<float, 4> bc_r;
    std::array<float, 4> bc_z;
    std::array<float, 4> bc_errZ2;
    std::array<GlobalPoint, 4> gps;
    std::array<GlobalError, 4> ges;
    std::array<bool, 4> barrels;
    // Loop over quadruplets
    for (unsigned int quadId = 0; quadId < numberOfFoundQuadruplets; ++quadId) {

      auto isBarrel = [](const unsigned id) -> bool {
        return id == PixelSubdetector::PixelBarrel;
      };
      for (unsigned int i = 0; i < 3; ++i) {
        auto layerPair = foundQuads[quadId][i].first;
        auto doubletId = foundQuads[quadId][i].second;

        auto const &ahit =
            hitDoublets[layerPair]->hit(doubletId, HitDoublets::inner);
        gps[i] = ahit->globalPosition();
        ges[i] = ahit->globalPositionError();
        barrels[i] = isBarrel(ahit->geographicalId().subdetId());
      }
      auto layerPair = foundQuads[quadId][2].first;
      auto doubletId = foundQuads[quadId][2].second;

      auto const &ahit =
          hitDoublets[layerPair]->hit(doubletId, HitDoublets::outer);
      gps[3] = ahit->globalPosition();
      ges[3] = ahit->globalPositionError();
      barrels[3] = isBarrel(ahit->geographicalId().subdetId());
      // TODO:
      // - if we decide to always do the circle fit for 4 hits, we don't
      //   need ThirdHitPredictionFromCircle for the curvature; then we
      //   could remove extraHitRPhitolerance configuration parameter
      ThirdHitPredictionFromCircle predictionRPhi(gps[0], gps[2],
                                                  extraHitRPhitolerance);
      const float curvature = predictionRPhi.curvature(
          ThirdHitPredictionFromCircle::Vector2D(gps[1].x(), gps[1].y()));
      const float abscurv = std::abs(curvature);
      const float thisMaxChi2 = maxChi2Eval.value(abscurv);
      if (theComparitor) {
        SeedingHitSet tmpTriplet(
            hitDoublets[foundQuads[quadId][0].first]->hit(
                foundQuads[quadId][0].second, HitDoublets::inner),
            hitDoublets[foundQuads[quadId][2].first]->hit(
                foundQuads[quadId][2].second, HitDoublets::inner),
            hitDoublets[foundQuads[quadId][2].first]->hit(
                foundQuads[quadId][2].second, HitDoublets::outer));
        if (!theComparitor->compatible(tmpTriplet)) {
          continue;
        }
      }

      float chi2 = std::numeric_limits<float>::quiet_NaN();
      // TODO: Do we have any use case to not use bending correction?
      if (useBendingCorrection) {
        // Following PixelFitterByConformalMappingAndLine
        const float simpleCot = (gps.back().z() - gps.front().z()) /
                                (gps.back().perp() - gps.front().perp());
        const float pt = 1.f / PixelRecoUtilities::inversePt(abscurv, es);
        for (int i = 0; i < 4; ++i) {
          const GlobalPoint &point = gps[i];
          const GlobalError &error = ges[i];
          bc_r[i] = sqrt(sqr(point.x() - region.origin().x()) +
                         sqr(point.y() - region.origin().y()));
          bc_r[i] += pixelrecoutilities::LongitudinalBendingCorrection(pt, es)(
              bc_r[i]);
          bc_z[i] = point.z() - region.origin().z();
          bc_errZ2[i] =
              (barrels[i]) ? error.czz() : error.rerr(point) * sqr(simpleCot);
        }
        RZLine rzLine(bc_r, bc_z, bc_errZ2, RZLine::ErrZ2_tag());
        chi2 = rzLine.chi2();
      } else {
        RZLine rzLine(gps, ges, barrels);
        chi2 = rzLine.chi2();
      }
      if (edm::isNotFinite(chi2) || chi2 > thisMaxChi2) {
        continue;
      }
      // TODO: Do we have any use case to not use circle fit? Maybe
      // HLT where low-pT inefficiency is not a problem?
      if (fitFastCircle) {
        FastCircleFit c(gps, ges);
        chi2 += c.chi2();
        if (edm::isNotFinite(chi2))
          continue;
        if (fitFastCircleChi2Cut && chi2 > thisMaxChi2)
          continue;
      }
      result[index].emplace_back(
          hitDoublets[foundQuads[quadId][0].first]->hit(
              foundQuads[quadId][0].second, HitDoublets::inner),
          hitDoublets[foundQuads[quadId][1].first]->hit(
              foundQuads[quadId][1].second, HitDoublets::inner),
          hitDoublets[foundQuads[quadId][2].first]->hit(
              foundQuads[quadId][2].second, HitDoublets::inner),
          hitDoublets[foundQuads[quadId][2].first]->hit(
              foundQuads[quadId][2].second, HitDoublets::outer));
    }

    index++;
  }
}
