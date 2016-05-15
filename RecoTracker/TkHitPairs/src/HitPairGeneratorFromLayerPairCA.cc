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
#include "RecoPixelVertexing/PixelTriplets/plugins/FKDPoint.h"


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
    
      void operator()(LayerTree& innerTree,const SeedingLayerSetsHits::SeedingLayer& innerLayer,const PixelRecoRange<float>& phiRange,std::vector<unsigned int>& foundHits,Range searchRange) const {
          /*
          constexpr float nSigmaRZ = 3.46410161514f; // std::sqrt(12.f);
          
          //const BarrelDetLayer& layerBarrelGeometry = static_cast<const BarrelDetLayer&>(*innerLayer.detLayer());
          
          //std::cout<<"BarrelDet : done!"<<std::endl;
          
          float vErr = 0.0;
          
          for(auto hit : innerLayer.hits()){
              auto const & gs = static_cast<BaseTrackerRecHit const &>(*hit).globalState();
              auto dv = innerLayer.detLayer()->isBarrel() ? gs.errorZ : gs.errorR;
              auto max = std::max(vErr,dv);
              vErr = max;
          }
          std::cout<<"vErrMax : calculated!"<<std::endl;
          vErr *= nSigmaRZ;
          */
          float rmax,rmin,zmax,zmin;
          /*
          auto thickness = innerLayer.detLayer()->surface().bounds().thickness();
          auto u = innerLayer.detLayer()->isBarrel() ? uLayer : innerLayer.detLayer()->position().z(); //BARREL? Raggio //FWD? z
		  thickness *= (u)/(std::fabs(u));
          std::cout<<"U & thickness : done! "<<u<<" - "<<thickness<<std::endl;
		 
          Range upperRange = checkRZ->range(u+thickness);
          Range lowerRange = checkRZ->range(u-thickness);
		  
		  std::cout<<"Lower at "<<u+thickness<<std::endl;
		  std::cout<<lowerRange.min()<<" - "<<lowerRange.max()<<std::endl;
		  std::cout<<"Upper at "<<u-thickness<<std::endl;
		  std::cout<<upperRange.min()<<" - "<<upperRange.max()<<std::endl;
          std::cout<<"Ranges : done!"<<std::endl;
          */
          if(innerLayer.detLayer()->isBarrel()){
              /*
              zmax = std::max(upperRange.max(),lowerRange.max());
              zmin = -std::max(-upperRange.min(),-lowerRange.min());
			  rmax = 1000;//+(u+thickness+vErr;
			  rmin = -1000;//u-thickness-vErr;*/
			  
			  zmax = searchRange.max();
			  zmin = searchRange.min();
			  rmax = 1000;//+(u+thickness+vErr;
			  rmin = -1000;//u-thickness-vErr;
			  
          }else{
              /*
              rmax = std::max(upperRange.max(),lowerRange.max());
              rmin = -std::max(-upperRange.min(),-lowerRange.min());
			  zmin = -1000;//u+thickness+vErr;
			  zmax = 1000;//u-thickness-vErr;*/
			  
			  rmax = searchRange.max();
			  rmin = searchRange.min();
			  zmin = -1000;//u+thickness+vErr;
			  zmax = 1000;//u-thickness-vErr;
			  
          }
		   
          std::cout<<"Rs & Zs : done!"<<std::endl;
          std::cout<<"Phi min : "<<phiRange.min()<<std::endl;
          std::cout<<"Phi max : "<<phiRange.max()<<std::endl;
          std::cout<<"r min : "<<rmin<<std::endl;
          std::cout<<"r max : "<<rmax<<std::endl;
          std::cout<<"z min : "<<zmin<<std::endl;
          std::cout<<"z max : "<<zmax<<std::endl;
		   
		  
          LayerPoint minPoint(phiRange.min(),zmin,rmin,0);
          //LayerPoint minPoint(-10000,-10000,-10000,0);
          std::cout<<"LayerPoint Min : done!"<<std::endl;
          LayerPoint maxPoint(phiRange.max(),zmax,rmax,100000);
          std::cout<<"LayerPoint Max : done!"<<std::endl;
          
          innerTree.LayerTree::search_in_the_box(minPoint,maxPoint,foundHits);
          
          std::cout<<"FKDTree Search : done!"<<std::endl;
          
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
                                                         const edm::Event & ev,  const edm::EventSetup& es,const SeedingLayerSetsHits::SeedingLayer& innerLayer,
                                                         const SeedingLayerSetsHits::SeedingLayer& outerLayer, LayerTree & innerTree) {
    
  std::cout<<"Hit Doublets CA Generator : in!  -  ";
  HitDoubletsCA result(innerLayer,outerLayer);
  //std::cout<<"Results initialised : done!"<<std::endl;
  InnerDeltaPhi deltaPhi(*outerLayer.detLayer(),*innerLayer.detLayer(), reg, es);
  //std::cout<<"Delta phi : done!"<<std::endl;
  // std::cout << "layers " << theInnerLayer.detLayer()->seqNum()  << " " << outerLayer.detLayer()->seqNum() << std::endl;
  bool rangesDone = false;
  float upperLimit = -10000;
  float lowerLimit = 10000;
	
  constexpr float nSigmaPhi = 3.f;
  for (int io = 0; io!=int(outerLayer.hits().size()); ++io) {
    std::cout<<"  Outer hit cylce : in!("<<io<<")"<<std::endl;
    Hit const & ohit = outerLayer.hits()[io];
    auto const & gs = static_cast<BaseTrackerRecHit const &>(*ohit).globalState();
    auto loc = gs.position-reg.origin().basicVector();
      
    float oX = gs.position.x();
    float oY = gs.position.y();
    float oZ = gs.position.z();
    float oRv = loc.perp();
      
    float oDrphi = gs.errorRPhi;
    float oDr = gs.errorR;
    float oDz = gs.errorZ;
    //std::cout<<"Outer Hit""Parameters : done!"<<"("<<io<<")"<<std::endl;
    if (!deltaPhi.prefilter(oX,oY)) continue;
      
    PixelRecoRange<float> phiRange = deltaPhi(oX,oY,oZ,nSigmaPhi*oDrphi);

    const HitRZCompatibility *checkRZ = reg.checkRZ(innerLayer.detLayer(), ohit, es, outerLayer.detLayer(), oRv, oZ, oDr, oDz);

    if(!checkRZ) continue;
	 
	  if(!rangesDone){
		  for(int ii = 0; ii!=int(innerLayer.hits().size()); ++ii){
			  
			  Hit const & ihit = innerLayer.hits()[io];
			  auto const & gsInner = static_cast<BaseTrackerRecHit const &>(*ihit).globalState();
			  auto locInner = gsInner.position-reg.origin().basicVector();

			  auto uInner = innerLayer.detLayer()->isBarrel() ? locInner.perp() : gs.position.z();
			  
			  Range bufferrange = checkRZ->range(uInner);
			  
			  upperLimit = std::max(bufferrange.min(),upperLimit);
			  upperLimit = std::max(bufferrange.max(),upperLimit);
			  
			  lowerLimit = std::min(bufferrange.max(),lowerLimit);
			  lowerLimit = std::min(bufferrange.min(),lowerLimit);
			  
		  }
		  
		}
	
	Range rangeSearch(lowerLimit,upperLimit);
	  
    std::cout<<"  -  HitRZ Check : done!"<<"("<<io<<")   ";
    Kernels<HitZCheck,HitRCheck,HitEtaCheck> kernels;
      
    std::vector<unsigned int> foundHitsInRange;
	
	  
      switch (checkRZ->algo()) {
          case (HitRZCompatibility::zAlgo) :
              std::cout<<" -  HitRZ Check : zAlgo!"<<"("<<io<<")  ";
              std::get<0>(kernels).set(checkRZ);
              std::get<0>(kernels)(innerTree,innerLayer,phiRange,foundHitsInRange,rangeSearch);
              break;
          case (HitRZCompatibility::rAlgo) :
              std::cout<<"  -  HitRZ Check : rAlgo!"<<"("<<io<<")  ";
              std::get<1>(kernels).set(checkRZ);
              std::get<0>(kernels)(innerTree,innerLayer,phiRange,foundHitsInRange,rangeSearch);
              break;
          case (HitRZCompatibility::etaAlgo) :
              //std::cout<<"HitRZ Check : etaAlgo CAZZO!"<<"("<<io<<")"<<std::endl;
              break;
      }
      std::cout<<"Found hits : "<<foundHitsInRange.size()<<" ("<<io<<")"<<std::endl;
      for (auto i=0; i!=(int)foundHitsInRange.size(); ++i) {

          if (theMaxElement!=0 && result.size() >= theMaxElement){
              result.clear();
              edm::LogError("TooManyPairs")<<"number of pairs exceed maximum, no pairs produced";
              delete checkRZ;
			  std::cout<<"  -  CheckRX : deleted!"<<"("<<io<<")  ";
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
