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


#include "RecHitsKDTree.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"





class CACell
{
public:
	CACell() { }
	CACell(const RecHitsKDTree* hitsKDTree, int innerHitId, int outerHitId, int layerId, const GlobalPoint& beamSpot ) : theHitsKDTree(hitsKDTree), theCAState(0),
			theInnerHitId(innerHitId), theOuterHitId(outerHitId), theLayerId(layerId), hasFriends(false) {



	}

	CACell(const RecHitsKDTree* hitsKDTree, int innerHitId, int outerHitId, int layerId, const GlobalPoint& beamSpot, const float tip) : theHitsKDTree(hitsKDTree), theCAState(0),
			theInnerHitId(innerHitId), theOuterHitId(outerHitId), theLayerId(layerId), hasFriends(false) {



	}

	void tagNeighbors(const CACells& CACellsOnOuterLayer, float maxDeltaZAtBeamLine, float maxDeltaRadius)
	{



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


	//
	//
	//	//check whether a Cell and the root have compatible parameters.
	//	inline
	//	bool areCompatible(const Cell& a, const int innerTripletChargeHypothesis)
	//	{
	//
	//		return (a.thechargeHypothesis == innerTripletChargeHypothesis) || (a.thechargeHypothesis == 0) || (innerTripletChargeHypothesis ==0) ;
	//
	//	}
	//
	//


	// trying to free the track building process from hardcoded layers, leaving the visit of the graph
	// based on the neighborhood connections between cells.
	inline
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
    
    //Returns the radius of the circumference through the beamspot and the cell hits
    void cellAxesCircleRadius(GlobalPoint beamSpot){
        
        theSigmaR = 0.0; //No tip
        
        float x1 = theHitsKDTree->hits[theInnerHitId].x();
        float x2 = theHitsKDTree->hits[theOuterHitId].x();
        float y1 = theHitsKDTree->hits[theInnerHitId].y();
        float y2 = theHitsKDTree->hits[theOuterHitId].y();
        
        float slopeXYCell,slopeOrthogonalXYCell,interceptOrthogonalXYCell;
        
        float slopeXYBeam,slopeOrthogonalXYBeam,interceptOrthogonalXYBeam;
        float intersectionPointX,intersectionPointY;
        
        //Slope of the line orthogonal to the line passing through cell hits - some checks to avoid slope = infinite and if the points are coincident (error)
        slopeOrthogonalXYCell = (x1 == x2) ? (y1 == y2) ? return 1 : 0.0 : (y1 == y2) ? HUGE_VALF : -(x1-x2)/(y1-y2);
        //Slope of the line orthogonal to the line passing through the beam spot and the inner hit - some checks to avoid slope = infinite and if the points are coincident (error)
        slopeOrthogonalXYBeam = (x1 == beamSpot.x()) ? (y1 == beamSpot.y()) ? return 1 : 0.0 : (y1 == beamSpot.y()) ? HUGE_VALF : -(x1-beamSpot.x())/(y1-beamSpot.y());
        
        
        //Slope of the line passing through cell hits - some checks to avoid slope = infinite
        slopeXYCell = (slopeOrthogonalXYCell == HUGE_VALF) ? 0.0 : (slopeOrthogonalXYCell == 0.0) ? HUGE_VALF : (1.0/slopeOrthogonalXYCell);
        //Slope of the line passing through the beam spot and the inner hit - some checks to avoid slope = infinite
        slopeXYBeam = (slopeOrthogonalXYCell == HUGE_VALF) ? 0.0 : (slopeOrthogonalXYCell == 0.0) ? HUGE_VALF : (1.0/slopeOrthogonalXYCell);
        
        //Midpoints : [theInnerHit,theOuterHit] & [theInnerHit,beamSpot]
        Basic2DVector midpointCell ((x1+x2)/2.0,(y1+y2;)/2.0);
        Basic2DVector midpointBeam ((x1+beamSpot.x())/2.0,(y1+beamSpot.())/2.0);
        
        //X and Y of the intersection point of the two axes and checks on the slope
        intersectionPointX = (slopeOrthogonalXYCell == HUGE_VALF) ? midpointCell.x() : (slopeOrthogonalXYBeam == HUGE_VALF) midpointBeam.x() ? :(midpointCell.y()-midpointBeam.y()-slopeOrthogonalXYCell*midpointCell.x()+slopeOrthogonalXYBeam*midpointCell.y())/(slopeOrthogonalXYBeam-slopeOrthogonalXYCell);
        
        intersectionPointY = (slopeOrthogonalXYCell == HUGE_VALF) ? slopeOrthogonalXYBeam*(intersectionPointX-midpointBeam.x())+midpointBeam.y() : slopeOrthogonalXYCell*(intersectionPointX-midpointCell.x())+midpointCell.y();
        
        //The radius : distance between theOuterHit and the intersectionPoint
        theRadius = std::sqrt((intersectionPointX-x2)*(intersectionPointX-x2)+(intersectionPointY-y2)*(intersectionPointY-y2));
        //The radius sign: positive if the curvature is clockwise, negative if the curvature is anticlockwise
        theRadius *= (slopeXYBeam == HUGE_VALF) ? (x2>beamSpot.x()) ? 1.0 : -1.0 : (y2>slopeXYBeam*(x2-beamSpot.x())+beamSpot.y())? -1.0 : 1.0;
        
    }


	tbb::concurrent_vector<int> theInnerNeighbors;
	tbb::concurrent_vector<int> theOuterNeighbors;

	int theInnerHitId;
	int theOuterHitId;
	float theRadius;
	float theSigmaR;
	float zAtBeamLine;
	short int theLayerId;
	short int theCAState;
	bool isHighPtCell;
	bool hasFriends;
	RecHitsKDTree* theHitsKDTree;

};


class HitDoublets {
public:
  enum layer { inner=0, outer=1};

  using Hit=RecHitsKDTree::Hit;


  HitDoublets(  RecHitsKDTree const & in,
		  RecHitsKDTree const & out) :
    layers{{&in,&out}}{}

  HitDoublets(HitDoublets && rh) : layers(std::move(rh.layers)), indeces(std::move(rh.indeces)){}

  void reserve(std::size_t s) { indeces.reserve(2*s);}
  std::size_t size() const { return indeces.size()/2;}
  bool empty() const { return indeces.empty();}
  void clear() { indeces.clear();}
  void shrink_to_fit() { indeces.shrink_to_fit();}

  void add (int il, int ol) { indeces.push_back(il);indeces.push_back(ol);}

  DetLayer const * detLayer(layer l) const { return layers[l]->layer; }

  Hit const & hit(int i, layer l) const { return layers[l]->theHits[indeces[2*i+l]].hit();}
  float       phi(int i, layer l) const { return layers[l]->phi(indeces[2*i+l]);}
  float       rv(int i, layer l) const { return layers[l]->rv(indeces[2*i+l]);}
  float        z(int i, layer l) const { return layers[l]->z[indeces[2*i+l]];}
  float        x(int i, layer l) const { return layers[l]->x[indeces[2*i+l]];}
  float        y(int i, layer l) const { return layers[l]->y[indeces[2*i+l]];}
  GlobalPoint gp(int i, layer l) const { return GlobalPoint(x(i,l),y(i,l),z(i,l));}

private:

  std::array<RecHitsSortedInPhi const *,2> layers;


  std::vector<int> indeces;

};


class CACells
{
public:
	  using Hit=RecHitsKDTree::Hit;
	  void neighborSearch(const CACells& CACellsOnOuterLayer)
		{
			const float c_maxParAbsDifference[parNum]= {0.06, 0.07};
//TODO parallelize this
			for(auto& cell: theCACells )
			{

				cell.tagNeighbors(CACellsOnOuterLayer, maxDeltaZAtBeamLine, maxDeltaRadius);

			}

			int neighborNum = 0;

			for (auto i= 0; i < outerCells.size(); ++i)
			{
				if(thecellsArray[ outerCells[i]].theInnerHitId != theOuterHitId)
					continue;
				bool isNeighbor = true;
				isNeighbor = isNeighbor && (fabs((theparams.thedata[0] - thecellsArray[rightCells.thedata[i]].theparams.thedata[0]))  < c_maxParAbsDifference[0]);
				isNeighbor = isNeighbor && areAlmostAligned(thehitsArray[theInnerHitId], thehitsArray[theOuterHitId], thehitsArray[thecellsArray[rightCells.thedata[i]].theOuterHitId], 40);
				if(!isNeighbor)
					break;
				double delta = fabs((theparams.thedata[1] - thecellsArray[rightCells.thedata[i]].theparams.thedata[1]));
				double phiDistance=  delta< 0.5*two_pi ? delta : two_pi-delta;
				isNeighbor = isNeighbor && (phiDistance < c_maxParAbsDifference[1]);
				if(!isNeighbor)
					break;

				// if all the parameters are inside the range the right cell is a right neighbor.
				// viceversa this cell will be the left neighbors for rightNeighbor(i)
				if (isNeighbor)
				{
					thecellsArray[rightCells.thedata[i]].theInnerNeighbors.push_back(theId);
					theOuterNeighbors.push_back(thecellsArray[rightCells.thedata[i]].theId);
					++neighborNum;
				}

			}

		}
private:

	  tbb::concurrent_vector<CACell> theCACells;


};


#endif /*CACELL_H_ */
