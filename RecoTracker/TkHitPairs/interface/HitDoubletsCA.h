#ifndef HitDoubletsCA_h
#define HitDoubletsCA_h

#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"


typedef SeedingLayerSetsHits::SeedingLayer Layer;
typedef BaseTrackerRecHit const * Hit;

class DetLayer;
class TrackingRegion;

class HitDoubletsCA{
public:
    enum layer { inner=0, outer=1};
    
    typedef BaseTrackerRecHit const * Hit;
    
    HitDoubletsCA(Layer const & innerLayer,
                Layer const & outerLayer) :
    layers{{&innerLayer,&outerLayer}}{}
    
    HitDoubletsCA(HitDoubletsCA && rh) : layers(std::move(rh.layers)), indeces(std::move(rh.indeces)){}
    
    void reserve(std::size_t s) { indeces.reserve(2*s);}
    std::size_t size() const { return indeces.size()/2;}
    bool empty() const { return indeces.empty();}
    void clear() { indeces.clear();}
    void shrink_to_fit() { indeces.shrink_to_fit();}
    
    void add (int il, int ol) { indeces.push_back(il);indeces.push_back(ol);}
    
    DetLayer const * detLayer(layer l) const { return layers[l]->detLayer();}
    
    int innerHitId(int i) const {return indeces[2*i];}
    int outerHitId(int i) const {return indeces[2*i+1];}
    
    Hit const & hit(int i, layer l) const {return layers[l]->hits()[i];}
    
    float       phi(int i, layer l) const { return layers[l]->hits()[i]->globalState().position.phi();}
    float       r(int i, layer l) const { return layers[l]->hits()[i]->globalState().r;}
    float       eta(int i, layer l) const { return layers[l]->hits()[i]->globalState().position.eta();}
    float        z(int i, layer l) const { return layers[l]->hits()[i]->globalState().position.z();}
    float        x(int i, layer l) const { return layers[l]->hits()[i]->globalState().position.x();}
    float        y(int i, layer l) const { return layers[l]->hits()[i]->globalState().position.y();}
    GlobalPoint gp(int i, layer l) const { return GlobalPoint(x(i,l),y(i,l),z(i,l));}
    
private:
    
    std::array<Layer const *,2> layers;
    std::vector<int> indeces;
    
};

#endif
