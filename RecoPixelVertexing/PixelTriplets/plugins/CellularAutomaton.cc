	
#include "CellularAutomaton.h"


void CellularAutomaton::create_and_connect_cells (std::vector<const HitDoublets*> doublets, const SeedingLayerSetsHits::SeedingLayerSet& fourLayers, const float pt_min)
{

  unsigned int cellId = 0;


  unsigned int numberOfLayerPairs = doublets.size();

  for (unsigned int layerPairId = 0; layerPairId < numberOfLayerPairs; ++layerPairId)
  {
    auto innerLayerId = layerPairId;
    auto outerLayerId = innerLayerId + 1;
    auto numberOfDoublets = doublets[layerPairId]->size ();


    isOuterHitOfCell[outerLayerId].resize(fourLayers[outerLayerId].hits().size());

    theFoundCellsPerLayer[layerPairId].reserve (numberOfDoublets);

    if (layerPairId == 0)
    {
      for (unsigned int i = 0; i < numberOfDoublets; ++i)
      {
        //        std::cout << "pushing cell: " << doublets.at(layerId)->innerHitId(i) << " " << doublets.at(layerId)->outerHitId(i) <<  std::endl;
        CACell tmpCell (doublets[layerPairId], i,  cellId++,  doublets[layerPairId]->innerHitId(i), doublets[layerPairId]->outerHitId(i));
        theFoundCellsPerLayer[layerPairId].push_back (tmpCell);
        //        std::cout << "adding cell to outerhit: " << doublets.at(layerId)->outerHitId(i) << " on layer " << layerId << std::endl;
        //        std::cout << "cell outer hit coordinates: " << tmpCell.get_outer_x() << " " <<tmpCell.get_outer_y() << " " <<tmpCell.get_outer_z() << tmpCell.get_inner_r() << std::endl;


        isOuterHitOfCell[outerLayerId][doublets[layerPairId]->outerHitId(i)].push_back (&(theFoundCellsPerLayer[layerPairId][i]));
      }
    }// if the layer is not the innermost one we check the compatibility between the two cells that share the same hit: one in the inner layer, previously created,
      // and the one we are about to create. If these two cells meet the neighboring conditions, they become one the neighbor of the other.
    else
    {
      for (unsigned int i = 0; i < numberOfDoublets; ++i)
      {

        CACell tmpCell(doublets[layerPairId], i, cellId++, doublets[layerPairId]->innerHitId(i), doublets[layerPairId]->outerHitId(i));
        theFoundCellsPerLayer[layerPairId].push_back (tmpCell);

        isOuterHitOfCell[outerLayerId][doublets[layerPairId]->outerHitId(i)].push_back (&(theFoundCellsPerLayer[layerPairId][i]));

//        std::cout << "\n\n\nchecking alignment of cell: " << std::endl;
     //   theFoundCellsPerLayer[layerPairId][i].print_cell();

        
        for (auto neigCell : isOuterHitOfCell[innerLayerId][doublets[layerPairId]->innerHitId(i)])
        {
  //        std::cout << "\nwith cell: " << std::endl;
   //       isOuterHitOfCell.at (innerLayerId).at (doublets.at(layerPairId)->innerHitId(i)).at(neigCellId)->print_cell();

		//	std::cout << "checking: " << std::endl;
          theFoundCellsPerLayer[layerPairId][i].check_alignment_and_tag (neigCell, pt_min);

        }

      }
    }
  }
}





void
CellularAutomaton::evolve ()
{

  for (unsigned int iteration = 0; iteration < theNumberOfLayers - 2; ++iteration)
  {    
    for (unsigned int innerLayerId = 0; innerLayerId < theNumberOfLayers - iteration - 2; ++innerLayerId)
    {
      unsigned int numberOfCellsFound = theFoundCellsPerLayer.at (innerLayerId).size();

      for (unsigned int cellId = 0; cellId < numberOfCellsFound; ++cellId)
      {


        theFoundCellsPerLayer[innerLayerId][cellId].evolve();
      }
    }

    for (unsigned int innerLayerId = 0; innerLayerId < theNumberOfLayers - iteration - 2; ++innerLayerId)
    {

      for (auto& cell :theFoundCellsPerLayer[innerLayerId])
      {
        cell.update_state();


      }
    }
  }
}


void
CellularAutomaton::find_root_cells (const unsigned int minimumCAState)
{


  for (CACell& cell : theFoundCellsPerLayer.at (0))
  {
    if (cell.is_root_cell (minimumCAState))
    {
      theRootCells.push_back (&cell);
       
    }
  }
}

void
CellularAutomaton::find_ntuplets(std::vector<CACell::CAntuplet>& foundNtuplets,  const unsigned int minHitsPerNtuplet)
{
  std::vector<CACell> tmpNtuplet;
  tmpNtuplet.reserve(4);

  for (CACell* root_cell : theRootCells)
  {
    tmpNtuplet.clear();
	 tmpNtuplet.push_back(*root_cell);
    root_cell->find_ntuplets (foundNtuplets, tmpNtuplet, minHitsPerNtuplet);
  }

}
