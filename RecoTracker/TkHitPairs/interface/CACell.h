/*
 * CACell.h
 *
 *  Created on: Jan 29, 2016
 *      Author: fpantale
 */

#ifndef CACELL_H_
#define CACELL_H_
// tbb headers
#include <tbb/concurrent_vector.h>
#include <cassert>

//TRIVIAL GIT CHECK
//#include "RecHitsKDTree.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/GeometryVector/interface/Basic2DVector.h"

#include <cmath>

class CACell
{
public:
	CACell() { }
	/*CACell(const RecHitsKDTree* hitsKDTree, int innerHitId, int outerHitId, int layerId, const GlobalPoint& beamSpot ) : theHitsKDTree(hitsKDTree), theCAState(0),
     theInnerHitId(innerHitId), theOuterHitId(outerHitId), theLayerId(layerId), hasFriends(false)*/
    /*CACell(const RecHitsKDTree* hitsKDTree, int innerHitId, int outerHitId, int innerLayerId,int outerLayerId,const GlobalPoint& beamSpot ) : theHitsKDTree(hitsKDTree), theCAState(0),
    theInnerHitId(innerHitId), theOuterHitId(outerHitId),theInnerLayerId(layerId),theOuterLayerId(layerId), hasFriends(false){
		if(!areAlmostAligned(beamSpot, 1e-3))
		{
			cellAxesCircleRadius(beamSpot);

		}

    }

	CACell(const RecHitsKDTree* hitsKDTree, int innerHitId, int outerHitId, int layerId, const GlobalPoint& beamSpot, const float tip) : theHitsKDTree(hitsKDTree), theCAState(0),
			theInnerHitId(innerHitId), theOuterHitId(outerHitId), theLayerId(layerId), hasFriends(false) {



	}

	void setCellId(const int id)
	{
		theCellId(id);
		theHitsKDTree[theOuterHitId].pushInnerCell(theCellId);
		theHitsKDTree[theInnerHitId].pushOuterCell(theCellId);
	}


	void tagNeighbors(const CACells& CACellsOnOuterLayer, float maxDeltaZAtBeamLine, float maxDeltaRadius)
	{
		for(auto& outerCellId: theHitsKDTree[theOuterHitId].getOuterCells())
		{
			if(areCompatible(CACellsOnOuterLayer[outerCellId],maxDeltaZAtBeamLine,maxDeltaRadius))
			{


				theOuterNeighbors.push_back()



			}
		}

	}

	// three points are collinear if the area of the triangle having the points as vertices is 0
	bool areAlmostAligned(const GlobalPoint& vtxHypothesis, const float epsilon)
	{
		auto x1 = theHitsKDTree[theInnerHitId].x();
		auto y1 = theHitsKDTree[theInnerHitId].y();
		auto x2 = theHitsKDTree[theOuterHitId].x();
		auto y2 = theHitsKDTree[theOuterHitId].y();
		auto x3 = vtxHypothesis.x();
		auto y3 = vtxHypothesis.y();

		return x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)<= epsilon;
	}

	void isRootCell(tbb::concurrent_vector<int>* rootCells)
	{
		if(theInnerNeighbors.size()== 0 && theCAState >= 2)
		{
			rootCells->push_back(theId);
		}
	}

	// if there is at least one left neighbor with the same state (friend), the state has to be increased by 1.
	int getCAState  () const
	{
		return theCAState;
	}

	void evolve(const tbb::concurrent_vector<int>* cells) {
		hasFriends = false;
		for(auto i =0; i < theOuterNeighbors.size(); ++i)
		{
			if(cells->at(theOuterNeighbors.at(i)).getCAState() == theCAState)
			{
				hasFriends = true;
				break;
			}
		}
	}



	inline
	void hasSameStateNeighbor()
	{
		if(hasFriends)
		{
			theCAState++;
		}
	}

	bool
	void isHighPt() const
	{
		return isHighPtCell;
	}

	bool areCompatible(const CACell& a, float maxDeltaZAtBeamLine, float maxDeltaRadius)
	{
		if((fabs(a.getZAtBeamLine() - zAtBeamLine) > maxDeltaZAtBeamLine))
			return false;
		else
			if (isHighPtCell != a.isHighPt())
				return false;
			else
				if(!isHighPtCell and fabs(a.getRadius()-theRadius)>maxDeltaRadius)
		return false;
				else return true;
	}




	// trying to free the track building process from hardcoded layers, leaving the visit of the graph
	// based on the neighborhood connections between cells.
	void findTracks ( tbb::concurrent_vector<CATrack>& foundTracks, const tbb::concurrent_vector<CACell>& cells, CATrack& tmpTrack) {

		// the building process for a track ends if:
		// it has no right neighbor
		// it has no compatible neighbor

		// the track is then saved if the number of hits it contains is greater than a threshold
		if(theOuterNeighbors.size() == 0 )
		{
			if( tmpTrack.size() >= c_minHitsPerTrack-1)
				foundTracks.push_back(tmpTrack);
			else
				return;
		}
		else
		{
			bool hasOneCompatibleNeighbor = false;
			for( auto i=0 ; i < theOuterNeighbors.size(); ++i)
			{
				if(tmpTrack.size() <= 2 || areCompatible(cells.at(theOuterNeighbors.at(i)), innermostTripletChargeHypothesis) )
				{
					hasOneCompatibleNeighbor = true;
					tmpTrack.push_back(theOuterNeighbors.at(i));
					cells.at(theOuterNeighbors.at(i)).findTracks(foundTracks,cells, tmpTrack );
					tmpTrack.pop_back();
				}
			}

			if (!hasOneCompatibleNeighbor && tmpTrack.size() >= c_minHitsPerTrack-1)
			{
				foundTracks.push(tmpTrack);
			}
		}

	}
    
    //Returns the radius of the circumference through the beamspot and the cell hits (WITHOUT error = 0.0)
    void cellAxesCircleRadius(GlobalPoint beamSpot){
        
        theRadius = 0.0;
        theSigmaR = 0.0; //No tip
        
        //Cell hits coordinates
        float x1 = theHitsKDTree->theHits[theInnerHitId].x();
        float y1 = theHitsKDTree->theHits[theInnerHitId].y();
        float x2 = theHitsKDTree->theHits[theOuterHitId].x();
        float y2 = theHitsKDTree->theHits[theOuterHitId].y();
        
        //Trivial check
        if ((x1 == x2) && (y1 == y2)) return;
        
        //Rotating the cell line so that it's vertical (reducing the error reducing the slope)
        float deltaPhi = 0.0;
        deltaPhi = (x2==x1)? Geom::fhalfPi() : std::atan2((y2-y1),(x2-x1));
        deltaPhi *= -1.0;
        deltaPhi += Geom::fhalfPi();
        
        x1 = theHitsKDTree->theHits[theInnerHitId].x()*std::cos(deltaPhi) - theHitsKDTree->theHits[theInnerHitId].y()*std::sin(deltaPhi);
        x2 = theHitsKDTree->theHits[theOuterHitId].x()*std::cos(deltaPhi) - theHitsKDTree->theHits[theOuterHitId].y()*std::sin(deltaPhi);
        
        y1 = theHitsKDTree->theHits[theOuterHitId].y()*std::cos(deltaPhi) + theHitsKDTree->theHits[theOuterHitId].x()*std::sin(deltaPhi);
        y2 = theHitsKDTree->theHits[theInnerHitId].y()*std::cos(deltaPhi) + theHitsKDTree->theHits[theInnerHitId].x()*std::sin(deltaPhi);
        
        //Beamspot coordinates
        float xBeam = beamSpot.x()*std::cos(deltaPhi) - beamSpot.y()*std::sin(deltaPhi);
        float yBeam = beamSpot.y()*std::cos(deltaPhi) + beamSpot.x()*std::sin(deltaPhi);
        
        //Trivial check
        if (((x1 == xBeam) && (y1 == yBeam))||((x2 == xBeam) && (y2 == yBeam))) return;
        
        float slopeOrthogonalXYBeam;
        
        //Slope of the line orthogonal to the line passing through the beam spot and the inner hit - some checks to avoid slope = infinite and if the points are coincident (error)
        slopeOrthogonalXYBeam = (x1 == xBeam) ?  0.0 : (y1 == yBeam) ? HUGE_VALF : -(x1-xBeam)/(y1-yBeam);
        //Slope of the line passing through the beam spot and the inner hit - some checks to avoid slope = infinite
        
        //Midpoints : [theInnerHit,theOuterHit] & [theInnerHit,beamSpot]
        Basic2DVectorF midpointCell ((x1+x2)/2.0,(y1+y2)/2.0);
        Basic2DVectorF midpointBeam ((x1+xBeam)/2.0,(y1+yBeam)/2.0);
        
        //The intersection point of the two axes and checks on the slope
        Basic2DVectorF intersectionPoint (0.0,midpointCell.y());
        intersectionPoint.v[0] =(slopeOrthogonalXYBeam == HUGE_VALF) ? midpointBeam.x() : (((midpointCell.y() - midpointBeam.y())/slopeOrthogonalXYBeam) + midpointBeam.x());
        
        //The radius : distance between theOuterHit and the intersectionPoint
        theRadius = std::sqrt((intersectionPoint.x()-x2)*(intersectionPoint.x()-x2)+(intersectionPoint.y()-y2)*(intersectionPoint.y()-y2));
        
        //The radius sign: positive if the curvature is clockwise, negative if the curvature is anticlockwise
        if(x1>xBeam) theRadius *= -1.0;
        
    }

    //Returns the radius of the circumference through the beamspot and the cell hits (WITH error)
    void cellAxesCircleRadius(const GlobalPoint& beamSpot, float tip){
        
        theRadius = 0.0;
        theSigmaR = 0.0;
        
        //Cell hits coordinates
        float x1 = theHitsKDTree->theHits[theInnerHitId].x();
        float y1 = theHitsKDTree->theHits[theInnerHitId].y();
        float x2 = theHitsKDTree->theHits[theOuterHitId].x();
        float y2 = theHitsKDTree->theHits[theOuterHitId].y();
        
        //Trivial check
        if ((x1 == x2) && (y1 == y2)) return;
        
        //Rotating the cell line so that it's vertical (reducing the error reducing the slope)
        float deltaPhi = 0.0;
        deltaPhi = (x2==x1)? Geom::fhalfPi() : std::atan2((y2-y1),(x2-x1));
        deltaPhi *= -1.0;
        deltaPhi += Geom::fhalfPi();
        
        x1 = theHitsKDTree->theHits[theInnerHitId].x()*std::cos(deltaPhi) - theHitsKDTree->theHits[theInnerHitId].y()*std::sin(deltaPhi);
        x2 = theHitsKDTree->theHits[theOuterHitId].x()*std::cos(deltaPhi) - theHitsKDTree->theHits[theOuterHitId].y()*std::sin(deltaPhi);
        
        y1 = theHitsKDTree->theHits[theOuterHitId].y()*std::cos(deltaPhi) + theHitsKDTree->theHits[theOuterHitId].x()*std::sin(deltaPhi);
        y2 = theHitsKDTree->theHits[theInnerHitId].y()*std::cos(deltaPhi) + theHitsKDTree->theHits[theInnerHitId].x()*std::sin(deltaPhi);
        
        //Beamspot coordinates
        float xBeam = beamSpot.x()*std::cos(deltaPhi) - beamSpot.y()*std::sin(deltaPhi);
        float yBeam = beamSpot.y()*std::cos(deltaPhi) + beamSpot.x()*std::sin(deltaPhi);
        
        //Trivial Check
        if (((x1 == xBeam) && (y1 == yBeam))||((x2 == xBeam) && (y2 == yBeam))) return;
        
        float slopeOrthogonalXYBeam;
        
        float slopeOrthogonalExtremeRight, slopeOrthogonalExtremeLeft;
        float diameterLimitDeltaX = 0.0;
        float radiusErrorNear,radiusErrorFar;
        
        //Slope of the line orthogonal to the line passing through the beam spot and the inner hit - some checks to avoid slope = infinite and if the points are coincident (error)
        slopeOrthogonalXYBeam = (x1 == xBeam) ? 0.0 : (y1 == yBeam) ? HUGE_VALF : -(x1-xBeam)/(y1-yBeam);
        //Slope of the line passing through the beam spot and the inner hit - some checks to avoid slope = infinite
        //slopeXYBeam = (slopeOrthogonalXYBeam == HUGE_VALF) ? 0.0 : (slopeOrthogonalXYBeam == 0.0) ? HUGE_VALF : (-1.0/slopeOrthogonalXYBeam);
        
        //Midpoints : [theInnerHit,theOuterHit] & [theInnerHit,beamSpot]
        Basic2DVectorF midpointCell ((x1+x2)/2.0,(y1+y2)/2.0);
        Basic2DVectorF midpointBeam ((x1+xBeam)/2.0,(y1+yBeam)/2.0);
        
        //The intersection point of the two axes and checks on the slope
        Basic2DVectorF intersectionPoint (0.0,midpointCell.y());
        intersectionPoint.v[0] =(slopeOrthogonalXYBeam == HUGE_VALF) ? midpointBeam.x() : (((midpointCell.y() - midpointBeam.y())/slopeOrthogonalXYBeam) + midpointBeam.x());
        
        //The radius : distance between theOuterHit and the intersectionPoint
        theRadius = std::sqrt((intersectionPoint.x()-x2)*(intersectionPoint.x()-x2)+(intersectionPoint.y()-y2)*(intersectionPoint.y()-y2));
        
        //The radius sign: positive if the curvature is clockwise, negative if the curvature is anticlockwise
        if(x1>xBeam) theRadius *= -1.0;
        //////
        //EVALUATING RADIUS SIGMA
        //////
        
        //Diameter slope = line orthogonal to the line from beam to inner hit slope == slopeOrthogonalXYBeam
        diameterLimitDeltaX = (slopeOrthogonalXYBeam == HUGE_VALF) ? 0.0 : std::sqrt(tip*tip/(1+slopeOrthogonalXYBeam*slopeOrthogonalXYBeam));
        
        //The diameter extremes
        Basic2DVectorF diameterExtremeRight(xBeam+diameterLimitDeltaX,0.0);
        diameterExtremeRight.v[1] = (slopeOrthogonalXYBeam == HUGE_VALF) ? yBeam + tip : yBeam + (diameterExtremeRight.x() - xBeam)*slopeOrthogonalXYBeam;
        Basic2DVectorF diameterExtremeLeft(xBeam-diameterLimitDeltaX,0.0);
        diameterExtremeLeft.v[1] = (slopeOrthogonalXYBeam == HUGE_VALF) ? yBeam - tip : yBeam + (diameterExtremeLeft.x() - xBeam)*slopeOrthogonalXYBeam;
        
        //Midpoints of [theInnerHit,theDiameterExtreme(s)]
        Basic2DVectorF theHitAndExtremeMidpointRight ((diameterExtremeRight.x()+x1)*0.5,(diameterExtremeRight.y()+y1)*0.5);
        Basic2DVectorF theHitAndExtremeMidpointLeft ((diameterExtremeLeft.x()+x1)*0.5,(diameterExtremeLeft.y()+y1)*0.5);
        
        //Trivial Check
        if (((x1 == diameterExtremeRight.x()) && (y1 == diameterExtremeRight.y()) )||((x1 == diameterExtremeLeft.x()) && (y1 == diameterExtremeLeft.y()) )) return;
        
        //Slope of the line(s) orthogonal to the line(s) [theInnerHit,theDiameterExtreme(s)]
        slopeOrthogonalExtremeRight = (x1 == diameterExtremeRight.x()) ?  0.0 : (y1 == diameterExtremeRight.y()) ? HUGE_VALF : -(x1-diameterExtremeRight.x())/(y1-diameterExtremeRight.y());
        slopeOrthogonalExtremeLeft = (x1 == diameterExtremeLeft.x()) ?  0.0 : (y1 == diameterExtremeLeft.y()) ? HUGE_VALF : -(x1-diameterExtremeLeft.x())/(y1-diameterExtremeLeft.y());
        
        //Intersection points between the cell axis and the [theInnerHit,theDiameterExtreme(s)] axes
        Basic2DVectorF intersectionPointErrorNear (0.0,midpointCell.y());
        Basic2DVectorF  intersectionPointErrorFar (0.0,midpointCell.y());
        
        if (slopeOrthogonalXYBeam>0){
            
            intersectionPointErrorNear.v[0]= (((midpointCell.y()-theHitAndExtremeMidpointRight.y())/slopeOrthogonalExtremeRight) + theHitAndExtremeMidpointRight.x());
            intersectionPointErrorFar.v[0]= (((midpointCell.y()-theHitAndExtremeMidpointLeft.y())/slopeOrthogonalExtremeLeft )+theHitAndExtremeMidpointLeft.x());
        }else{
            
            intersectionPointErrorFar.v[0]= (((midpointCell.y()-theHitAndExtremeMidpointRight.y())/slopeOrthogonalExtremeRight )+theHitAndExtremeMidpointRight.x());
            intersectionPointErrorNear.v[0]= (((midpointCell.y()-theHitAndExtremeMidpointLeft.y())/slopeOrthogonalExtremeLeft )+theHitAndExtremeMidpointLeft.x());
            
        }
        
        //As sigmaR is chosen the biggest of the asymmetrical errors
        radiusErrorNear = fabs(theRadius - sqrt((intersectionPointErrorNear.x()-x2)*(intersectionPointErrorNear.x()-x2)+(intersectionPointErrorNear.y()-y2)*(intersectionPointErrorNear.y()-y2)));
        
        radiusErrorFar = fabs(theRadius - sqrt((intersectionPointErrorFar.x()-x2)*(intersectionPointErrorFar.x()-x2)+(intersectionPointErrorFar.y()-y2)*(intersectionPointErrorFar.y()-y2)));
        
        theSigmaR = std::max(radiusErrorFar,radiusErrorNear);
        
    }
    
    //Returns the z of the intersection of the beam axis with of the line passing through cell hits
    float cellZOnBeam(float beamPhi,float beamR){
        
        //Cell hits z
        float z1 = theHitsKDTree->theHits[theInnerHitId].z();
        float z2 = theHitsKDTree->theHits[theOuterHitId].z();
        //Vertical Cell in y-z plane
        if(z1==z2) return z1;
        
        //Cell hits y
        float y1 =  theHitsKDTree->theHits[theInnerHitId].y();
        float y2 = theHitsKDTree->theHits[theOuterHitId].y();
        //Horizontal Cell in y-z plane
        if(y1==y2) return HUGE_VALF;
        
        float beamHeight = beamR*std::sin(beamPhi);
        
        return (beamHeight-y2)*((z2-z1)/(y2-y1))+z2;
        
    }
    
    //Return the angle between the innner hit radius and the line passing through cell hits
    float cellPhiAngle(){
        
        //Hits parameters
        float phi1 = theHitsKDTree->theHits[theInnerHitId].phi();
        float phi2 = theHitsKDTree->theHits[theOuterHitId].phi();
        
        //r alligned hits
        if (phi1==phi2) return 0.0;
        //NOTA : per l'allienamento in x ci vorrebbe un
        //x alligned hits
        float x1 = theHitsKDTree->theHits[theInnerHitId].x();
        float x2 = theHitsKDTree->theHits[theOuterHitId].x();
        if (x1==x2) return Geom::fpi() - phi1;
        //y alligned hits
        float y1 = theHitsKDTree->theHits[theInnerHitId].y();
        float y2 = theHitsKDTree->theHits[theOuterHitId].y();
        
        if (y1==y2) return phi1;
        
        float gamma = std::atan2((y2-y1),(x2-x1));
        
        return Geom::fpi()-gamma-phi1;
    }
    

	float getZAtBeamLine() const
	{
		return zAtBeamLine;
	}

	float getRadius() const
	{
		return theRadius;

	}

	float getSigmaR() const
	{
		return theSigmaR;
	}
    */
    
    int getInnerLayerId() const{
        
        return theInnerLayerId;
    }

    
    int getOuterLayerId() const{
        
        return theOuterLayerId;
    }


private:

	tbb::concurrent_vector<int> theInnerNeighbors;
	tbb::concurrent_vector<int> theOuterNeighbors;
	int theCellId;
	int theInnerHitId;
	int theOuterHitId;
	float theRadius;
	float theSigmaR;
	float zAtBeamLine;
	//short int theLayerId; //There should be trace of both layers
    short int theInnerLayerId;
    short int theOuterLayerId;
	short int theCAState;
	bool isHighPtCell;
	bool hasFriends;
	//RecHitsKDTree* theHitsKDTree;

};

/*
class CACells
{
public:
    CACells() { };
    
    CACells(std::vector<CACell>& cellVector){
        
        for (std::vector<CACell>::iterator it = cellVector.begin() ; it != cellVector.end() - 1; ++it)
        {
            assert(((*it).getInnerLayerId() == (*(it+1)).getInnerLayerId())&&((*it).getOuterLayerId() == (*(it+1)).getOuterLayerId()));//, "CACell(s) in the constructor vector must refer to the same layer couple" );
        }
        
        theCells = std::move(cellVector);
        theNumberOfCells = cellVector.size();
        theTwoLayersIndeces = std::make_pair (cellVector[0].getInnerLayerId(),cellVector[0].getOuterLayerId());
        
    }
    
    void push_back(const CACell& cell)
    {
        assert((cell.getInnerLayerId() == theTwoLayersIndeces.first)&&(cell.getOuterLayerId() == theTwoLayersIndeces.second));//, "The CACell pushed must refer to the same layer couple" );
    
        theNumberOfCells++;
        theCells.push_back(cell);
        
    }
    
    CACells& operator=(CACells && other)
    {
        
        if (this != &other)
        {
            theCells = std::move(other.theCells);
            theNumberOfCells(std::move(other.theNumberOfCells));
            theTwoLayersIndeces = std:move(other.theTwoLayersIndeces);
        }
        return *this;
        
    }

    
private:
    std::vector<CACell> theCells;
    unsigned int theNumberOfCells;
    std::pair<int,int> theTwoLayersIndeces;

};
*/
#endif /*CACELL_H_ */
