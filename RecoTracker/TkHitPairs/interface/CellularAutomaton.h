#ifndef RECOTRACKER_TKHITPAIRS_INTERFACE_CELLULARAUTOMATON_H_
#define RECOTRACKER_TKHITPAIRS_INTERFACE_CELLULARAUTOMATON_H_
#include <array>
#include "CACell.h"
template<int theNumberOfLayers>
class CellularAutomaton
{
public:

	int find_cells();
	void create_graph();
	int evolve();
	int find_ntuplets();

	void neighborSearch(const CACells& CACellsOnOuterLayer)
	{

		const float c_maxParAbsDifference[parNum]= {0.06, 0.07};
		//TODO parallelize this
		for(auto& cell: theCACells )
		{
			cell.tagNeighbors(CACellsOnOuterLayer, maxDeltaZAtBeamLine, maxDeltaRadius);
		}

	}


	CACell& cell(int id) {
		return theCACells.at(id);
	}


private:


	std::vector<CACell> theCACells;
	std::array<std::size_t,theNumberOfLayers> theFirstCellIdOfLayer;
	std::vector<std::size_t> theRootCells;
	std::vector< std::array<std::size_t, theNumberOfLayers> > theNtuplets;
    std::vector<std::vector<CACell>* > foundCellsPerLayer;



};

#endif /* RECOTRACKER_TKHITPAIRS_INTERFACE_CELLULARAUTOMATON_H_ */
