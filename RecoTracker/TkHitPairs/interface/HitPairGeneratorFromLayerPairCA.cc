#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPairCA.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkHitPairs/src/InnerDeltaPhi.h"

#include "FWCore/Framework/interface/Event.h"

#define greatz 5E3

using namespace GeomDetEnumerators;
using namespace std;

typedef PixelRecoRange<float> Range;

namespace {
  template<class T> inline T sqr( T t) {return t*t;}
}


#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"

HitPairGeneratorFromLayerPairCA::HitPairGeneratorFromLayerPairCA(
							     unsigned int inner,
							     unsigned int outer,
							     LayerFKDTreeCache* treeCache,LayerDoubletsCache* doubletsCache,
							     unsigned int max)
  : theKDTreeCache(*treeCache),theDoubletsCache(doubletsCache) , theOuterLayer(outer), theInnerLayer(inner), theMaxElement(max)
{
}

HitPairGeneratorFromLayerPairCA::~HitPairGeneratorFromLayerPairCA() {}

// devirtualizer
#include<tuple>
namespace {

  template<typename Algo>
  struct Kernel {
    using  Base = HitRZCompatibility;
    void set(Base const * a) {
      assert( a->algo()==Algo::me);
      checkRZ=reinterpret_cast<Algo const *>(a);
    }
    
      void operator()(std::vector<unsigned int> & hitsToBeChecked, const std::vector<Hit> & hits, const std::vector<bool> & ok,bool innerBarrel) const {
      constexpr float nSigmaRZ = 3.46410161514f; // std::sqrt(12.f);
      for (int i=0; i!=hitsToBeChecked.size(); ++i) {
          
          unsigned int index = hitsToBeChecked[i];
          auto const & gs = static_cast<BaseTrackerRecHit const &>(hits[index]).globalState();
          auto loc = gs.position-origin.basicVector();
          
          auto u = innerBarrel ? loc.perp() : gs.position.z();
          auto v = innerBarrel ? gs.position.z() : loc.perp();
          auto dv = innerBarrel ? gs.errorZ : gs.errorR;
          
          Range allowed = checkRZ->range(u);
          float vErr = nSigmaRZ * innerHitsMap.dv[i];
          Range hitRZ(innerHitsMap.v-vErr, innerHitsMap.v+vErr);
          Range crossRange = allowed.intersection(hitRZ);
          ok.push_back = ! crossRange.empty() ;
          
      }
    }
    Algo const * checkRZ;
    
  };


  template<typename ... Args> using Kernels = std::tuple<Kernel<Args>...>;

}

/*
void HitPairGeneratorFromLayerPair::hitPairs(
					     const TrackingRegion & region, OrderedHitPairs & result,
					     const edm::Event& iEvent, const edm::EventSetup& iSetup, Layers layers) {

  auto const & ds = doublets(region, iEvent, iSetup, layers);
  for (std::size_t i=0; i!=ds.size(); ++i) {
    result.push_back( OrderedHitPair( ds.hit(i,HitDoublets::inner),ds.hit(i,HitDoublets::outer) ));
  }
  if (theMaxElement!=0 && result.size() >= theMaxElement){
     result.clear();
    edm::LogError("TooManyPairs")<<"number of pairs exceed maximum, no pairs produced";
  }
}*/


HitDoubletsCA HitPairGeneratorFromLayerPairCA::doublets (const TrackingRegion& reg,
                                                         const edm::Event & ev,  const edm::EventSetup& es,const SeedingLayer& innerLayer,const SeedingLayer& outerLayer, const FKDTree<float,2> & innerTree, const FKDTree<float,2> & outerTree) {

  typedef OrderedHitPair::InnerRecHit InnerHit;
  typedef OrderedHitPair::OuterRecHit OuterHit;
    
  FKDTree<float,2>* innerLayerKDTree;
  FKDTree<float,2>* outerLayerKDTree;

  /*
  Layer innerLayerObj = layers[0];
  Layer outerLayerObj = layers[1];
    
  const innerLayerKDTree = (*theKDTReeCache)(&innerLayerObj,region,ev,es); innerLayerKDTree.FKDTree::build();
  const outerLayerKDTree = (*theKDTReeCache)(&outerLayerObj,region,ev,es); outerLayerKDTree.FKDTree::build();*/

  HitDoubletsCA result((&innerLayer,&outerLayer);
                       //result.reserve(std::max(innerHitsMap.size(),outerHitsMap.size()));

  InnerDeltaPhi deltaPhi(outerLayerObj.detLayer(), innerLayerObj.detLayer(), region, iSetup);

  // std::cout << "layers " << theInnerLayer.detLayer()->seqNum()  << " " << outerLayer.detLayer()->seqNum() << std::endl;

  // constexpr float nSigmaRZ = std::sqrt(12.f);
  constexpr float nSigmaPhi = 3.f;
  for (int io = 0; io!=int(outerLayerObj.hits().size()); ++io) {
      
    Hit const & ohit = outerLayerObj.hits()[io];
    auto const & gs = static_cast<BaseTrackerRecHit const &>(ohit).globalState();
    auto loc = gs.position-origin.basicVector();
      
    float oX = gs.position.x();
    float oY = gs.position.y();
    float oZ = gs.position.z();
    float oRv = loc.perp();
      
    float oDrphi = gs.errorRPhi;
    float oDr = gs.errorR;
    float oDz = gs.errorZ;
      
    if (!deltaPhi.prefilter(oX,oY)) continue;
      
    PixelRecoRange<float> phiRange = deltaPhi(oX,oY,oZ,nSigmaPhi*drphi);

    if (phiRange.empty()) continue;

    const HitRZCompatibility *checkRZ = region.checkRZ(innerLayerObj.detLayer(), ohit, iSetup, outerLayerObj.detLayer(), oRv, oZ, oDr, oDz);
    if(!checkRZ) continue;

    Kernels<HitZCheck,HitRCheck,HitEtaCheck> kernels;
    
    auto v = layer.detLayer()->isBarrel() ? gs.position.z() : gs.r;
      
    FKDPoint<float,2> minPoint(phiRange.min(),-bigValue);
    FKDPoint<float,2> maxPoint(phiRange.max(),bigValue);
      
    std::vector<unsigned int>& foundHitsInRange;
      
    search_in_the_box(&minPoint, &maxPoint,foundHitsInRange);
    
    /*auto innerRange = innerHitsMap.doubleRange(phiRange.min(), phiRange.max());
      
    LogDebug("HitPairGeneratorFromLayerPair")<<
      "preparing for combination of: "<< innerRange[1]-innerRange[0]+innerRange[3]-innerRange[2]
				      <<" inner and: "<< outerHitsMap.theHits.size()<<" outter";*/
      
      std::vector<bool>& ok;
      
      switch (checkRZ->algo()) {
          case (HitRZCompatibility::zAlgo) :
              std::get<0>(kernels).set(checkRZ);
              std::get<0>(kernels)(foundHitsInRange,innerLayerObj.hits(),ok,innerLayerObj.detLayer()->isBarrel());
              break;
          case (HitRZCompatibility::rAlgo) :
              std::get<1>(kernels).set(checkRZ);
              std::get<1>(kernels)(foundHitsInRange,innerLayerObj.hits(),ok,innerLayerObj.detLayer()->isBarrel());
              break;
          case (HitRZCompatibility::etaAlgo) :
              std::get<2>(kernels).set(checkRZ);
              std::get<2>(kernels)(foundHitsInRange,innerLayerObj.hits(),ok,innerLayerObj.detLayer()->isBarrel());
              break;
      }
      
      for (int i=0; i!=ok.size(); ++i) {
          if (!ok[i]) continue;
          if (theMaxElement!=0 && result.size() >= theMaxElement){
              result.clear();
              edm::LogError("TooManyPairs")<<"number of pairs exceed maximum, no pairs produced";
              delete checkRZ;
              return result;
          }
          result.add(foundHitsInRange[i],io);
      }
      delete checkRZ;
  
  }
  LogDebug("HitPairGeneratorFromLayerPairCA")<<" total number of pairs provided back: "<<result.size();
  result.shrink_to_fit();
  return result;
}
