#include "RecoTracker/TkHitPairs/interface/CellularAutomaton.h"


int CellularAutomaton::create_cells(const HitDoublets& doublets, const unsigned int layerIdInFourLayers, SeedingLayer* innerLayer, SeedingLayer* outerLayer )
{
	auto numberOfDoublets = doublets.size();
	isOuterHitOfCell.at(layerIdInFourLayers).resize(numberOfDoublets);
	theFoundCellsPerLayer.at(layerIdInFourLayers).resize(numberOfDoublets);


	if(layerIdInFourLayers == 0)
	{
		for(auto i = 0; i<numberOfDoublets; ++i)
		{
			CACell tmpCell(doublets.innerHitId[i], doublets.outerHitId[i], &theFoundCellsPerLayer, innerLayer, outerLayer);
			theFoundCellsPerLayer.at(layerIdInFourLayers).at(i) = tmpCell;
			isOuterHitOfCell.at(layerIdInFourLayers).at(doublets.outerHitId[i]).push_back(&theFoundCellsPerLayer.at(layerIdInFourLayers).at(i));
		}
	}
	//if the layer is not the innermost one we check the compatibility between the two cells that share the same hit: one in the inner layer, previously created,
	// and the one we are about to create. If these two cells meet the neighboring conditions, they become one the neighbor of the other.
	else
	{
		for(auto i = 0; i<numberOfDoublets; ++i)
		{
			CACell tmpCell(doublets.innerHitId[i], doublets.outerHitId[i], &theFoundCellsPerLayer, innerLayer, outerLayer);
			theFoundCellsPerLayer.at(layerIdInFourLayers).at(i) = tmpCell;
			isOuterHitOfCell.at(layerIdInFourLayers).at(doublets.outerHitId[i]).push_back(&(theFoundCellsPerLayer.at(layerIdInFourLayers).at(i)));


			for(auto neighboringCell: isOuterHitOfCell.at(layerIdInFourLayers-1).at(doublets.innerHitId[i]))
				theFoundCellsPerLayer.at(layerIdInFourLayers).at(i).tag_neighbor(neighboringCell);


		}


	}
}




