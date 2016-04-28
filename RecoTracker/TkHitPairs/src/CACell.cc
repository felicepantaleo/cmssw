#include "RecoTracker/TkHitPairs/interface/CACell.h"

void CACell::tagNeighbor(CACell* otherCell)
{

	if(areCompatible(otherCell,maxDeltaZAtBeamLine,maxDeltaRadius))
		{
			theOuterNeighbors.push_back();
		}

}

void CACell::evolve(const std::vector<unsigned int>& cells) {
	hasFriends = false;
	for(auto i =0; i < theOuterNeighbors.size(); ++i)
	{
		if(cells->at(theOuterNeighbors.at(i)).getCAState() == theCAState)
		{
			hasSameStateNeighbors = true;
			break;
		}
	}
}

bool CACell::areAlignedRZ(Cell* otherCell) const
{
	theInnerLayer->


	auto x1 = theHitsKDTree[theInnerHitId].x();
	auto y1 = theHitsKDTree[theInnerHitId].y();
	auto x2 = theHitsKDTree[theOuterHitId].x();
	auto y2 = theHitsKDTree[theOuterHitId].y();
	auto x3 = vtxHypothesis.x();
	auto y3 = vtxHypothesis.y();

	return x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)<= epsilon;
}



void CACell::isRootCell(std::vector<unsigned int>& rootCells) const
{
	if(theInnerNeighbors.size()== 0 && theCAState >= 2)
	{
		rootCells->push_back(theId);
	}
}

void CACell::findNtuplets ( std::vector<CAntuplet>& foundNtuplets, const std::vector<CACell>& cells, CAntuplet& tmpNtuplet, const int minHitsPerNtuplet) const
{

	// the building process for a track ends if:
	// it has no right neighbor
	// it has no compatible neighbor

	// the ntuplets is then saved if the number of hits it contains is greater than a threshold
	if(theOuterNeighbors.size() == 0 )
	{
		if( tmpNtuplet.size() >= minHitsPerNtuplet-1)
			foundNtuplets.push_back(tmpNtuplet);
		else
			return;
	}
	else
	{
		bool hasOneCompatibleNeighbor = false;
		for( auto i=0 ; i < theOuterNeighbors.size(); ++i)
		{
			if(tmpNtuplet.size() <= 2 || areCompatible(cells.at(theOuterNeighbors.at(i)), innermostTripletChargeHypothesis) )
			{
				hasOneCompatibleNeighbor = true;
				tmpNtuplet.push_back(cells.at(theOuterNeighbors.at(i)));
				cells.at(theOuterNeighbors.at(i)).findNtuplets(foundNtuplets,cells, tmpNtuplet, minHitsPerNtuplet );
				tmpNtuplet.pop_back();
			}
		}

		if (!hasOneCompatibleNeighbor && tmpNtuplet.size() >= minHitsPerNtuplet-1)
		{
			foundNtuplets.push_back(tmpNtuplet);
		}
	}

}



void CACell::cellAxesCircleRadius(GlobalPoint beamSpot){

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
