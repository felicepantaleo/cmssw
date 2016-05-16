
#ifndef FKDTREE_KDPOINT_H_
#define FKDTREE_KDPOINT_H_
#include <array>
#include <utility>
#include <vector>
#include <iostream>

#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"

template<class TYPE, int numberOfDimensions>
class FKDPoint
{
    
public:
    FKDPoint() :
    theElements(), theId(0)
    {
    }
    
    FKDPoint(const FKDPoint<TYPE, numberOfDimensions>& other) :
    theElements(other.theElements),theId(other.theId)
    {
        
    }
    
    template<class T>
    FKDPoint<TYPE, numberOfDimensions> & operator=(
                                                   const FKDPoint<TYPE, numberOfDimensions> & other)
    {
        if (this != &other)
        {
            theId = other.theId;
            theElements = other.theElements;
        }
        return *this;
        
    }
    
    FKDPoint(TYPE x, TYPE y, unsigned int id)
    {
        static_assert( numberOfDimensions == 2, "Point dimensionality differs from the number of passed arguments." );
        theId = id;
        theElements[0] = x;
        theElements[1] = y;
    }
    
    FKDPoint(TYPE x, TYPE y, TYPE z, unsigned int id)
    {
        static_assert( numberOfDimensions == 3, "Point dimensionality differs from the number of passed arguments." );
        theId = id;
        theElements[0] = x;
        theElements[1] = y;
        theElements[2] = z;
    }
    
    FKDPoint(TYPE x, TYPE y, TYPE z, TYPE w, unsigned int id)
    {
        static_assert( numberOfDimensions == 4, "Point dimensionality differs from the number of passed arguments." );
        theId = id;
        theElements[0] = x;
        theElements[1] = y;
        theElements[2] = z;
        theElements[3] = w;
    }
    
    // the user should check that i < numberOfDimensions
    TYPE& operator[](int const i)
    {
        return theElements[i];
    }
    
    TYPE const& operator[](int const i) const
    {
        return theElements[i];
    }
    
    void setDimension(int i, const TYPE& value)
    {
        theElements[i] = value;
    }
    
    void setId(const unsigned int id)
    {
        theId = id;
    }
    
    unsigned int getId() const
    {
        return theId;
    }
    
    void print()
    {
        std::cout << "point id: " << theId << std::endl;
        for (auto i : theElements)
        {
            std::cout << i << std::endl;
        }
    }
    
private:
    std::array<TYPE, numberOfDimensions> theElements;
    unsigned int theId;
};

/* Utility functions to create 1-, 2-, 3-, or 4-Points from values. */
template<typename TYPE>
FKDPoint<TYPE, 1> make_FKDPoint(TYPE x, unsigned int id)
{
    FKDPoint<TYPE, 1> result;
    result.setDimension(0, x);
    result.setId(id);
    return result;
}

template<typename TYPE>
FKDPoint<TYPE, 2> make_FKDPoint(TYPE x, TYPE y, unsigned int id)
{
    FKDPoint<TYPE, 2> result;
    result.setDimension(0, x);
    result.setDimension(1, y);
    result.setId(id);
    return result;
}

template<typename TYPE>
FKDPoint<TYPE, 3> make_FKDPoint(TYPE x, TYPE y, TYPE z, unsigned int id)
{
    FKDPoint<TYPE, 3> result;
    result.setDimension(0, x);
    result.setDimension(1, y);
    result.setDimension(2, z);
    result.setId(id);
    return result;
}

template<typename TYPE>
FKDPoint<TYPE, 4> make_FKDPoint(TYPE x, TYPE y, TYPE z, TYPE w, unsigned int id)
{
    FKDPoint<TYPE, 4> result;
    result.setDimension(0, x);
    result.setDimension(1, y);
    result.setDimension(2, z);
    result.setDimension(3, w);
    result.setId(id);
    return result;
}


template<typename TYPE>
void setDimensions(FKDPoint<TYPE, 2>& point, TYPE x, TYPE y)
{
    point.setDimension(0, x);
    point.setDimension(1, y);
}


//const template<typename TYPE> std::vector<FKDPoint<TYPE, 2>> &
//operator()
/*
template<typename TYPE> std::vector<FKDPoint<TYPE, 2>> make_FKDPointVectorFromHits(const std::vector<HitWithPhi> *Hits, TYPE x, TYPE y, TYPE z, TYPE w, unsigned int id)
{
    std::vector<FKDPoint<TYPE, 2>> result;
    result.setDimension(0, x);
    result.setDimension(1, y);
    result.setDimension(2, z);
    result.setDimension(3, w);
    result.setId(id);
    return result;
}


    
    template<typename TYPE> std::vector<FKDPoint<TYPE, 2>> make_FKDPointVectorFromHits (const std::vector<Hit>& hits, DetLayer const * il, float &minv, float &maxv, float &maxErr)
{
    std::vector<FKDPoint<TYPE, 2>> result;
    for (std::vector<Hit>::const_iterator i=hits.begin(); i!=hits.end(); i++) {
        minv = std::min(minv,v);  maxv = std::max(maxv,v);
        float myerr = hits.dv[i];
        float        gv(int i) const { return isBarrel ? z[i] : gp(i).perp();}
        layerTree.emplace_back(i, angle, v);
    }
}
    layer(il),
    isBarrel(il->isBarrel()),
    x(hits.size()),y(hits.size()),z(hits.size()),drphi(hits.size()),
    u(hits.size()),v(hits.size()),du(hits.size()),dv(hits.size()),
    lphi(hits.size())
    {
        
        // standard region have origin as 0,0,z (not true!!!!0
        // cosmic region never used here
        // assert(origin.x()==0 && origin.y()==0);
        
        for (std::vector<Hit>::const_iterator i=hits.begin(); i!=hits.end(); i++) {
            theHits.push_back(HitWithPhi(*i));
        }
        std::sort( theHits.begin(), theHits.end(), HitLessPhi());
        
        for (unsigned int i=0; i!=theHits.size(); ++i) {
            auto const & h = *theHits[i].hit();
            auto const & gs = static_cast<BaseTrackerRecHit const &>(h).globalState();
            auto loc = gs.position-origin.basicVector();
            float lr = loc.perp();
            // float lr = gs.position.perp();
            float lz = gs.position.z();
            float dr = gs.errorR;
            float dz = gs.errorZ;
            // r[i] = gs.position.perp();
            // phi[i] = gs.position.barePhi();
            x[i] = gs.position.x();
            y[i] = gs.position.y();
            z[i] = lz;
            drphi[i] = gs.errorRPhi;
            u[i] = isBarrel ? lr : lz;
            v[i] = isBarrel ? lz : lr;
            du[i] = isBarrel ? dr : dz;
            dv[i] = isBarrel ? dz : dr;
            lphi[i] = loc.barePhi();
        }
        
    }
*/

#endif /* FKDTREE_KDPOINT_H_ */