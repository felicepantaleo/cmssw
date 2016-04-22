#ifndef RECOTRACKER_TKHITPAIRS_INTERFACE_CELLULARAUTOMATON_H_
#define RECOTRACKER_TKHITPAIRS_INTERFACE_CELLULARAUTOMATON_H_
#include <array>
#include "CACell.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"

class CellularAutomaton
{
public:
	CellularAutomaton()
	{

	}


	CellularAutomaton(const std::vector<FKDTree<float,2>* >& layersHitsTree, const GlobalPoint* beamSpot )
	{
		theNumberOfLayers = layersHitsTree.size();
		theRootCells.clear();
		theFoundCellsPerLayer.clear();
		theNtuplets.clear();
		theBeamSpot = beamSpot;

	}



	int create_cells(const HitDoublets&, const unsigned int);
	void create_graph();
	int evolve();
	int find_ntuplets();

	void neighborSearch(const CACells& CACellsOnOuterLayer)
	{

		const float c_maxParAbsDifference[parNum] =
		{ 0.06, 0.07 };
		//TODO parallelize this
		for (auto& cell : theCACells)
		{
			cell.tagNeighbors(CACellsOnOuterLayer, maxDeltaZAtBeamLine,
					maxDeltaRadius);
		}

	}

	CACell& cell(int id)
	{
		return theCACells.at(id);


	}

private:
	unsigned int theNumberOfLayers;


	//for each hit in each layer, store the pointers of the Cells of which it is outerHit
	std::vector<std::vector<std::vector<CACell*> > > isOuterHitOfCell;
	std::vector<std::vector<CACell>> theFoundCellsPerLayer;

	std::vector<std::size_t> theRootCells;
	std::vector<std::vector<std::size_t> > theNtuplets;
	GlobalPoint* theBeamSpot;

};

#endif /* RECOTRACKER_TKHITPAIRS_INTERFACE_CELLULARAUTOMATON_H_ */
