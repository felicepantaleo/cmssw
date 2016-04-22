#include "RecoTracker/TkHitPairs/interface/CellularAutomaton.h"


int CellularAutomaton::create_cells(const HitDoublets& doublets, const unsigned int layerIdInFourLayers, SeedingLayer* innerLayer, SeedingLayer* outerLayer )
{
	auto numberOfDoublets = doublets.size();
	isOuterHitOfCell.at(layerIdInFourLayers).resize(numberOfDoublets);
	theFoundCellsPerLayer.at(layerIdInFourLayers).resize(numberOfDoublets);
	for(auto i = 0; i<numberOfDoublets; ++i)
	{
		CACell tmpCell(doublets.innerHitId[i], doublets.outerHitId[i], &theFoundCellsPerLayer, innerLayer, outerLayer);
		theFoundCellsPerLayer.at(layerIdInFourLayers).at(i) = tmpCell;
		isOuterHitOfCell.at(layerIdInFourLayers).at(doublets.outerHitId[i]).push_back(&theFoundCellsPerLayer.at(layerIdInFourLayers).at(i));

	}




}
