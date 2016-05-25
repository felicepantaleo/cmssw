
#include "CellularAutomaton.h"

void CellularAutomaton::create_and_connect_cells (std::vector<const HitDoubletsCA*> doublets, const SeedingLayerSetsHits::SeedingLayerSet& fourLayers)
{
  std::cout << "entering create and connect" << std::endl;
  unsigned int cellId = 0;
  for (unsigned int layerId = 0; layerId < doublets.size(); ++layerId)
  {

    auto numberOfDoublets = doublets.at(layerId)->size ();
    std::cout << "\n\n\nstarting to create "<< numberOfDoublets << " cells " << fourLayers[layerId].name() << " " << fourLayers[layerId+1].name() << std::endl;

    isOuterHitOfCell.at (layerId).resize(fourLayers[layerId + 1].hits().size());
    theFoundCellsPerLayer.at (layerId).reserve (numberOfDoublets);

    if (layerId == 0)
    {
      for (unsigned int i = 0; i < numberOfDoublets; ++i)
      {
//        std::cout << "pushing cell: " << doublets.at(layerId)->innerHitId(i) << " " << doublets.at(layerId)->outerHitId(i) <<  std::endl;
        CACell tmpCell (doublets.at (layerId), fourLayers[layerId], fourLayers[layerId + 1], cellId++,  doublets.at(layerId)->innerHitId(i), doublets.at(layerId)->outerHitId(i));
        theFoundCellsPerLayer.at (layerId).push_back (tmpCell);
//        std::cout << "adding cell to outerhit: " << doublets.at(layerId)->outerHitId(i) << " on layer " << layerId << std::endl;
//        std::cout << "cell outer hit coordinates: " << tmpCell.get_outer_x() << " " <<tmpCell.get_outer_y() << " " <<tmpCell.get_outer_z() << tmpCell.get_inner_r() << std::endl;


        isOuterHitOfCell.at (layerId).at(doublets.at(layerId)->outerHitId(i)).push_back (&(theFoundCellsPerLayer.at (layerId).at (i)));
      }
    }// if the layer is not the innermost one we check the compatibility between the two cells that share the same hit: one in the inner layer, previously created,
      // and the one we are about to create. If these two cells meet the neighboring conditions, they become one the neighbor of the other.
    else
    {
      for (unsigned int i = 0; i < numberOfDoublets; ++i)
      {
        std::cout << "pushing cell: " << doublets.at(layerId)->innerHitId(i) << " " << doublets.at(layerId)->outerHitId(i) << std::endl;
        CACell tmpCell(doublets.at (layerId), fourLayers[layerId], fourLayers[layerId + 1], cellId++, doublets.at(layerId)->innerHitId(i), doublets.at(layerId)->outerHitId(i));
        theFoundCellsPerLayer.at (layerId).push_back (tmpCell);
        std::cout << "adding cell to outerhit: " << doublets.at(layerId)->outerHitId(i) << " on layer " << layerId << std::endl;
        std::cout << "cell outer hit coordinates: " << theFoundCellsPerLayer.at (layerId).at(i).get_outer_x()  << " " << theFoundCellsPerLayer.at (layerId).at(i).get_outer_y() << " " << theFoundCellsPerLayer.at (layerId).at(i).get_outer_z()  << " " <<theFoundCellsPerLayer.at (layerId).at(i).get_outer_r() <<   std::endl;

        isOuterHitOfCell.at (layerId).at (doublets.at(layerId)->outerHitId(i)).push_back (&(theFoundCellsPerLayer.at (layerId).at (i)));

        std::cout << "checking alignment with cell on previous layer with outer hit " << doublets.at(layerId)->innerHitId(i) << std::endl;
        for (unsigned int neigCellId = 0; neigCellId < isOuterHitOfCell.at (layerId - 1).at (doublets.at(layerId)->innerHitId(i)).size(); neigCellId++)
        {
          theFoundCellsPerLayer.at (layerId).at (i).check_alignment_and_tag (isOuterHitOfCell.at (layerId - 1).at (doublets.at(layerId)->innerHitId(i)).at(neigCellId));

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
      std::cout << numberOfCellsFound << " cells found on layer "<<  innerLayerId << std::endl;
      for (unsigned int cellId = 0; cellId < numberOfCellsFound; ++cellId)
      {
//        theFoundCellsPerLayer.at (innerLayerId).at(cellId).print_cell();
        std::cout << "cell " << theFoundCellsPerLayer.at (innerLayerId).at(cellId).get_cell_id() << " is evolving. iteration: " << iteration << " innerlayerId " << innerLayerId << " cellidinlayer " << cellId << std::endl;
        theFoundCellsPerLayer.at (innerLayerId).at(cellId).evolve();
      }
    }
    std::cout << "starting to update state " << std::endl;
    for (unsigned int innerLayerId = 0; innerLayerId < theNumberOfLayers - iteration - 2; ++innerLayerId)
    {
      unsigned int numberOfCellsFound = theFoundCellsPerLayer.at (innerLayerId).size();
      for (unsigned int cellId = 0; cellId < numberOfCellsFound; ++cellId)
      {
        theFoundCellsPerLayer.at(innerLayerId).at(cellId).update_state();
//        theFoundCellsPerLayer.at (innerLayerId).at(cellId).print_cell();

      }
    }
  }
}

void
CellularAutomaton::find_root_cells (const unsigned int minimumCAState)
{
      std::cout << "entering find root cells" << std::endl;

  for (CACell& cell : theFoundCellsPerLayer.at (0))
  {
    if (cell.is_root_cell (minimumCAState))
    {
      theRootCells.push_back (&cell);
       cell.print_cell();
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
    root_cell->find_ntuplets (foundNtuplets, tmpNtuplet, minHitsPerNtuplet);
  }

}
