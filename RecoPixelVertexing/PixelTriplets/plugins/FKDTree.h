/*
 * FKDTree.h
 *
 *  Created on: Jan 28, 2016
 *      Author: fpantale
 */

#ifndef FKDTREE_FKDTREE_H_
#define FKDTREE_FKDTREE_H_

#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <utility>
#include <iostream>
#include <cassert>
#include <deque>
#include "FKDPoint.h"
#include "FQueue.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

typedef BaseTrackerRecHit const * Hit;

template<class TYPE, int numberOfDimensions>
class FKDTree
{
    
public:
    
    FKDTree(const unsigned int nPoints)
    {
        theNumberOfPoints = 0;
        theDepth = 0;
        for (auto& x : theDimensions)
            x.reserve(theNumberOfPoints);
        theIntervalLength.reserve(theNumberOfPoints);
        theIntervalMin.reserve(theNumberOfPoints);
        theIds.reserve(theNumberOfPoints);
        thePoints.reserve(theNumberOfPoints);
    }
    
    void reserve(const unsigned int nPoints)
    {
        for (auto& x : theDimensions)
            x.reserve(nPoints);
        theIntervalLength.reserve(nPoints);
        theIntervalMin.reserve(nPoints);
        theIds.reserve(nPoints);
        thePoints.reserve(nPoints);
        
    }
    
    FKDTree(const std::vector<FKDPoint<TYPE, numberOfDimensions> >& points)
    {
        theNumberOfPoints = points.size();
        theDepth = std::floor(log2(theNumberOfPoints));
        for (auto& x : theDimensions)
            x.resize(theNumberOfPoints);
        theIntervalLength.resize(theNumberOfPoints, 0);
        theIntervalMin.resize(theNumberOfPoints, 0);
        theIds.resize(theNumberOfPoints, 0);
        thePoints = points;
        
    }
    
    FKDTree()
    {
        theNumberOfPoints = 0;
        theDepth = 0;
        for (auto& x : theDimensions)
            x.clear();
        theIntervalLength.clear();
        theIntervalMin.clear();
        theIds.clear();
        thePoints.clear();
        theIndecesToVisit.clear();
    }
    
    FKDTree(const FKDTree<TYPE, numberOfDimensions>& other)
    {
        theNumberOfPoints(other.theNumberOfPoints);
        theDepth(other.theDepth);
        
        theIntervalLength.clear();
        theIntervalMin.clear();
        theIds.clear();
        thePoints.clear();
        for (auto& x : theDimensions)
            x.clear();
        
        theIntervalLength = other.theIntervalLength;
        theIntervalMin = other.theIntervalMin;
        theIds = other.theIds;
        
        thePoints = other.thePoints;
        for (int i = 0; i < numberOfDimensions; ++i)
            theDimensions = other.theDimensions;
        
    }
    
    FKDTree(FKDTree<TYPE, numberOfDimensions> && other)
    {
        theNumberOfPoints(std::move(other.theNumberOfPoints));
        theDepth(std::move(other.theDepth));
        
        theIntervalLength.clear();
        theIntervalMin.clear();
        theIds.clear();
        thePoints.clear();
        for (auto& x : theDimensions)
            x.clear();
        
        theIntervalLength = std::move(other.theIntervalLength);
        theIntervalMin = std::move(other.theIntervalMin);
        theIds = std::move(other.theIds);
        
        thePoints = std::move(other.thePoints);
        for (int i = 0; i < numberOfDimensions; ++i)
            theDimensions = std::move(other.theDimensions);
    }
    
    
    FKDTree<TYPE, numberOfDimensions>& operator=(
                                                 FKDTree<TYPE, numberOfDimensions> && other)
    {
        
        if (this != &other)
        {
            theNumberOfPoints(std::move(other.theNumberOfPoints));
            theDepth(std::move(other.theDepth));
            
            theIntervalLength.clear();
            theIntervalMin.clear();
            theIds.clear();
            thePoints.clear();
            for (auto& x : theDimensions)
                x.clear();
            
            theIntervalLength = std::move(other.theIntervalLength);
            theIntervalMin = std::move(other.theIntervalMin);
            theIds = std::move(other.theIds);
            
            thePoints = std::move(other.thePoints);
            for (int i = 0; i < numberOfDimensions; ++i)
                theDimensions = std::move(other.theDimensions);
        }
        return *this;
        
    }
    
    void resize(unsigned int nPoints)
    {
        theNumberOfPoints = nPoints;
        theDepth = std::floor(log2(nPoints));
        for (auto& x : theDimensions)
            x.resize(theNumberOfPoints);
        theIntervalLength.resize(theNumberOfPoints, 0);
        theIntervalMin.resize(theNumberOfPoints, 0);
        theIds.resize(theNumberOfPoints);
        thePoints.reserve(theNumberOfPoints);
    }
    
    void clear()
    {
        
        theNumberOfPoints = 0;
        theDepth = 0;
        for (auto& x : theDimensions)
            x.clear();
        theIntervalLength.clear();
        theIntervalMin.clear();
        theIds.clear();
        thePoints.clear();
    }
    
    bool empty()
    {
        return theNumberOfPoints == 0;
    }
    
    void push_back(const FKDPoint<TYPE, numberOfDimensions>& point)
    {
        
        theNumberOfPoints++;
        thePoints.push_back(point);
        for (int i = 0; i < numberOfDimensions; ++i)
            theDimensions.at(i).push_back(point[i]);
        theIds.push_back(point.getId());
        
    }
    
    void push_back(FKDPoint<TYPE, numberOfDimensions> && point)
    {
        theNumberOfPoints++;
        thePoints.push_back(point);
        for (int i = 0; i < numberOfDimensions; ++i)
            theDimensions.at(i).push_back(point[i]);
        theIds.push_back(point.getId());
    }
    
    void add_at_position(const FKDPoint<TYPE, numberOfDimensions>& point,
                         const unsigned int position)
    {
        for (int i = 0; i < numberOfDimensions; ++i)
            theDimensions[i][position] = point[i];
        theIds[position] = point.getId();
        
    }
    
    void add_at_position(FKDPoint<TYPE, numberOfDimensions> && point,
                         const unsigned int position)
    {
        for (int i = 0; i < numberOfDimensions; ++i)
            theDimensions[i][position] = point[i];
        theIds[position] = point.getId();
        
    }
    
    FKDPoint<TYPE, numberOfDimensions> getPoint(unsigned int index) const
    {
        
        FKDPoint<TYPE, numberOfDimensions> point;
        
        for (int i = 0; i < numberOfDimensions; ++i)
            point.setDimension(i, theDimensions[i][index]);
        
        point.setId(theIds[index]);
        
        return point;
    }
    
    void search_in_the_box(const FKDPoint<TYPE, numberOfDimensions>& minPoint,
                           const FKDPoint<TYPE, numberOfDimensions>& maxPoint,
                           std::vector<unsigned int>& foundPoints)
    {
        theIndecesToVisit.clear();
        theIndecesToVisit.push_back(0);
        for (int depth = 0; depth < theDepth + 1; ++depth)
        {
            
            int dimension = depth % numberOfDimensions;
            unsigned int numberOfIndecesToVisitThisDepth =
            theIndecesToVisit.size();
            for (unsigned int visitedIndecesThisDepth = 0;
                 visitedIndecesThisDepth < numberOfIndecesToVisitThisDepth;
                 visitedIndecesThisDepth++)
            {
                
                unsigned int index = theIndecesToVisit[visitedIndecesThisDepth];
                bool intersection = intersects(index, minPoint, maxPoint,
                                               dimension);
                
                if (intersection && is_in_the_box(index, minPoint, maxPoint))
                    foundPoints.emplace_back(theIds[index]);
                
                bool isLowerThanBoxMin = theDimensions[dimension][index]
                < minPoint[dimension];
                
                int startSon = isLowerThanBoxMin; //left son = 0, right son =1
                
                int endSon = isLowerThanBoxMin || intersection;
                
                for (int whichSon = startSon; whichSon < endSon + 1; ++whichSon)
                {
                    unsigned int indexToAdd = leftSonIndex(index) + whichSon;
                    
                    if (indexToAdd < theNumberOfPoints)
                    {
                        theIndecesToVisit.push_back(indexToAdd);
                    }
                    
                }
                
            }
            
            theIndecesToVisit.pop_front(numberOfIndecesToVisitThisDepth);
        }
    }
    
    
    void search_in_the_box_recursive(const FKDPoint<TYPE, numberOfDimensions>& minPoint,
                                     const FKDPoint<TYPE, numberOfDimensions>& maxPoint, std::vector<unsigned int>& foundPoints,unsigned int index=0, int dimension=0) const
    {
        
        unsigned int firstSonToVisitNext = leftSonIndex(index);
        int maxNumberOfSonsToVisitNext = (firstSonToVisitNext < theNumberOfPoints) + ((firstSonToVisitNext+1) < theNumberOfPoints);
        bool intersection = intersects(index, minPoint, maxPoint,
                                       dimension);
        int numberOfSonsToVisitNext;
        if (intersection)
        {
            if(is_in_the_box(index, minPoint, maxPoint))
            {
                foundPoints.emplace_back(theIds[index]);
            }
            numberOfSonsToVisitNext = maxNumberOfSonsToVisitNext;
        }
        else
        {
            bool isLowerThanBoxMin = theDimensions[dimension][index]
            < minPoint[dimension];
            numberOfSonsToVisitNext = isLowerThanBoxMin && (maxNumberOfSonsToVisitNext==1)? 0: std::min(maxNumberOfSonsToVisitNext,1);
            firstSonToVisitNext += isLowerThanBoxMin;
        }
        
        if(numberOfSonsToVisitNext != 0)
            
        {
            auto nextDimension = (dimension+1) % numberOfDimensions;
            for(int whichSon = 0; whichSon < numberOfSonsToVisitNext; ++whichSon)
                search_in_the_box_recursive(minPoint, maxPoint, foundPoints,firstSonToVisitNext+whichSon,nextDimension);
        }
        
    }
    
    bool test_correct_build(unsigned int index = 0, int dimension = 0) const
    {
        
        unsigned int leftSonIndexInArray = 2 * index + 1;
        unsigned int rightSonIndexInArray = leftSonIndexInArray + 1;
        if (rightSonIndexInArray >= theNumberOfPoints
            && leftSonIndexInArray >= theNumberOfPoints)
        {
            return true;
        }
        else
        {
            if (leftSonIndexInArray < theNumberOfPoints)
            {
                if (theDimensions[dimension][index]
                    >= theDimensions[dimension][leftSonIndexInArray])
                {
                    test_correct_build(leftSonIndexInArray,
                                       (dimension + 1) % numberOfDimensions);
                    
                }
                else
                    return false;
            }
            
            if (rightSonIndexInArray < theNumberOfPoints)
            {
                if (theDimensions[dimension][index]
                    <= theDimensions[dimension][rightSonIndexInArray])
                {
                    test_correct_build(rightSonIndexInArray,
                                       (dimension + 1) % numberOfDimensions);
                    
                }
                else
                    return false;
            }
            
        }
        
    }
    
    bool test_correct_search(const std::vector<unsigned int> foundPoints,
                             const FKDPoint<TYPE, numberOfDimensions>& minPoint,
                             const FKDPoint<TYPE, numberOfDimensions>& maxPoint) const
    {
        bool testGood = true;
        for (unsigned int i = 0; i < theNumberOfPoints; ++i)
        {
            
            bool shouldBeInTheBox = true;
            for (int dim = 0; dim < numberOfDimensions; ++dim)
            {
                shouldBeInTheBox &= (thePoints[i][dim] <= maxPoint[dim]
                                     && thePoints[i][dim] >= minPoint[dim]);
            }
            
            bool foundToBeInTheBox = std::find(foundPoints.begin(),
                                               foundPoints.end(), thePoints[i].getId())
            != foundPoints.end();
            
            if (foundToBeInTheBox == shouldBeInTheBox)
            {
                
                testGood &= true;
            }
            else
            {
                if (foundToBeInTheBox)
                    std::cerr << "Point " << thePoints[i].getId()
                    << " was wrongly found to be in the box."
                    << std::endl;
                else
                    std::cerr << "Point " << thePoints[i].getId()
                    << " was wrongly found to be outside the box."
                    << std::endl;
                
                testGood &= false;
                
            }
        }
        
        if (testGood)
            std::cout << "Search correctness test completed successfully."
            << std::endl;
        return testGood;
    }
    
    std::vector<TYPE> getDimensionVector(const int dimension) const
    {
        if (dimension < numberOfDimensions)
            return theDimensions[dimension];
    }
    
    std::vector<unsigned int> getIdVector() const
    {
        return theIds;
    }
    void build()
    {
        theDepth = std::floor(log2(theNumberOfPoints));
        for (auto& x : theDimensions)
            x.resize(theNumberOfPoints);
        theIntervalLength.resize(theNumberOfPoints, 0);
        theIntervalMin.resize(theNumberOfPoints, 0);
        theIds.resize(theNumberOfPoints);
        thePoints.reserve(theNumberOfPoints);
        
        //gather kdtree building
        int dimension;
        theIntervalMin[0] = 0;
        theIntervalLength[0] = theNumberOfPoints;
        
        for (int depth = 0; depth < theDepth; ++depth)
        {
            
            dimension = depth % numberOfDimensions;
            unsigned int firstIndexInDepth = (1 << depth) - 1;
            for (int indexInDepth = 0; indexInDepth < (1 << depth);
                 ++indexInDepth)
            {
                unsigned int indexInArray = firstIndexInDepth + indexInDepth;
                unsigned int leftSonIndexInArray = 2 * indexInArray + 1;
                unsigned int rightSonIndexInArray = leftSonIndexInArray + 1;
                
                unsigned int whichElementInInterval = partition_complete_kdtree(
                                                                                theIntervalLength[indexInArray]);
                std::nth_element(
                                 thePoints.begin() + theIntervalMin[indexInArray],
                                 thePoints.begin() + theIntervalMin[indexInArray]
                                 + whichElementInInterval,
                                 thePoints.begin() + theIntervalMin[indexInArray]
                                 + theIntervalLength[indexInArray],
                                 [dimension](const FKDPoint<TYPE,numberOfDimensions> & a, const FKDPoint<TYPE,numberOfDimensions> & b) -> bool
                                 {
                                     if(a[dimension] == b[dimension])
                                         return a.getId() < b.getId();
                                     else
                                         return a[dimension] < b[dimension];
                                 });
                add_at_position(
                                thePoints[theIntervalMin[indexInArray]
                                          + whichElementInInterval], indexInArray);
                
                if (leftSonIndexInArray < theNumberOfPoints)
                {
                    theIntervalMin[leftSonIndexInArray] =
                    theIntervalMin[indexInArray];
                    theIntervalLength[leftSonIndexInArray] =
                    whichElementInInterval;
                }
                
                if (rightSonIndexInArray < theNumberOfPoints)
                {
                    theIntervalMin[rightSonIndexInArray] =
                    theIntervalMin[indexInArray]
                    + whichElementInInterval + 1;
                    theIntervalLength[rightSonIndexInArray] =
                    (theIntervalLength[indexInArray] - 1
                     - whichElementInInterval);
                }
            }
        }
        
        dimension = theDepth % numberOfDimensions;
        unsigned int firstIndexInDepth = (1 << theDepth) - 1;
        for (unsigned int indexInArray = firstIndexInDepth;
             indexInArray < theNumberOfPoints; ++indexInArray)
        {
            add_at_position(thePoints[theIntervalMin[indexInArray]],
                            indexInArray);
            
        }
        
    }
    
    
   
    FKDTree<TYPE, numberOfDimensions>& make_FKDTreeFromRegionLayer(const SeedingLayerSetsHits::SeedingLayer& layer, const TrackingRegion & region, const edm::Event & iEvent, const edm::EventSetup & iSetup)
    {
        static_assert( numberOfDimensions == 2, "Dimensions 2" );
        //static_assert( (typeof(TYPE) == float), "Float." );
        const float maxDelphi = region.ptMin() < 0.3f ? float(M_PI)/4.f : float(M_PI)/8.f;
        const float safePhi = M_PI-maxDelphi;
        
        FKDPoint<TYPE, numberOfDimensions> result;
        
        std::vector<Hit>& hits = region.hits(iEvent,iSetup,layer);
        unsigned int pointID = 0;
        for (std::vector<Hit>::const_iterator i=hits.begin(); i!=hits.end(); i++) {
            
            auto const & gs = static_cast<BaseTrackerRecHit const &>(*i).globalState();
            auto phi = gs.position.barePhi();
            auto v = layer.detLayer()->isBarrel() ? gs.position.z() : gs.r;
            
            result.push_back(make_FKDPoint(phi,v,pointID)); pointID++;
            
            if (phi>safePhi) {result.push_back(make_FKDPoint(phi-Geom::ftwoPi(), v,pointID)); pointID++;}
            else if (phi<-safePhi) {result.push_back(make_FKDPoint(phi+Geom::ftwoPi(), v,pointID));pointID++;}
            
        }
        
        return result;
        
    }

    
private:
    
    unsigned int partition_complete_kdtree(unsigned int length)
    {
        if (length == 1)
            return 0;
        unsigned int index = 1 << ((int) log2(length));
        
        if ((index / 2) - 1 <= length - index)
            return index - 1;
        else
            return length - index / 2;
        
    }
    
    unsigned int leftSonIndex(unsigned int index) const
    {
        return 2 * index + 1;
    }
    
    unsigned int rightSonIndex(unsigned int index) const
    {
        return 2 * index + 2;
    }
    
    bool intersects(unsigned int index,
                    const FKDPoint<TYPE, numberOfDimensions>& minPoint,
                    const FKDPoint<TYPE, numberOfDimensions>& maxPoint,
                    int dimension) const
    {
        return (theDimensions[dimension][index] <= maxPoint[dimension]
                && theDimensions[dimension][index] >= minPoint[dimension]);
    }
    
    bool is_in_the_box(unsigned int index,
                       const FKDPoint<TYPE, numberOfDimensions>& minPoint,
                       const FKDPoint<TYPE, numberOfDimensions>& maxPoint) const
    {
        
        for (int i = 0; i < numberOfDimensions; ++i)
        {
            if (!(theDimensions[i][index] <= maxPoint[i]
                  && theDimensions[i][index] >= minPoint[i]))
                return false;
        }
        
        return true;
    }
    
    unsigned int theNumberOfPoints;
    int theDepth;
    std::vector<FKDPoint<TYPE, numberOfDimensions> > thePoints;
    std::array<std::vector<TYPE>, numberOfDimensions> theDimensions;
    FQueue<unsigned int> theIndecesToVisit;
    std::vector<unsigned int> theIntervalLength;
    std::vector<unsigned int> theIntervalMin;
    std::vector<unsigned int> theIds;
    
};

#endif /* FKDTREE_FKDTREE_H_ */

/*
#ifndef FKDTREE_H_
#define FKDTREE_H_
#include <array>
#include <utility>

template <int numberOfDimensions>
struct KDRange
{

	using 1DRange = std::pair<float, float>;
	std::array<1DRange, numberOfDimensions> theKDRange;

  public:

  KDTreeBox(float d1min, float d1max,
	    float d2min, float d2max)
    : dim1min (d1min), dim1max(d1max)
    , dim2min (d2min), dim2max(d2max)
  {}

  KDTreeBox()
    : dim1min (0), dim1max(0)
    , dim2min (0), dim2max(0)
  {}
};



#endif /* CMSSW_8_0_0_PRE4_SRC_RECOPIXELVERTEXING_PIXELTRIPLETS_PLUGINS_FKDTREE_H_ */
