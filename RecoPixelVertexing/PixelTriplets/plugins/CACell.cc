#include "CACell.h"

void CACell::tag_as_outer_neighbor(CACell* otherCell)
{
  theOuterNeighbors.push_back(otherCell);
}

void CACell::tag_as_inner_neighbor(CACell* otherCell)
{
  theInnerNeighbors.push_back(otherCell);
}

void CACell::evolve()
{

  hasSameStateNeighbors = 0;
  unsigned int numberOfNeighbors = theOuterNeighbors.size();
  
  for (unsigned int i =0 ; i< numberOfNeighbors; ++i)
  {
    std::cout << "checking compatibility with cell " << theOuterNeighbors.at(i)->get_cell_id() << std::endl;
    if (theOuterNeighbors.at(i)->get_CA_state() == theCAState)
    {

      std::cout << get_cell_id() << " and are found compatible " << theOuterNeighbors.at(i)->get_cell_id() << std::endl;

      hasSameStateNeighbors = 1;

      break;
    }
  }
  std::cout << "cell " << get_cell_id() << " has evolved." << std::endl;
}

void CACell::update_state()
{
  theCAState +=hasSameStateNeighbors;
  std::cout << "cell " << theCellId <<  " has now state " << theCAState  << std::endl;

}

void CACell::check_alignment_and_tag(CACell* innerCell)
{
  std::cout << "checking alignment" << std::endl;
  if (are_aligned_RZ(innerCell))
  {
    std::cout << "cells " << theCellId << " " << innerCell->get_cell_id() << " are neighbors" << std::endl;

    tag_as_inner_neighbor(innerCell);
    innerCell->tag_as_outer_neighbor(this);

  }
  else
  {
        std::cout << "cells are not aligned" << std::endl;

  }
}

bool CACell::are_aligned_RZ(const CACell* otherCell) const
{

//   std::cout << otherCell->get_inner_x();
   
  auto r1 = get_inner_r();
  auto r2 = get_outer_r();
  
// std::cout << " got own r" << std::endl;
// std::cout << " the Inner and outer hits id of the other cells are: " << otherCell->get_inner_hit_id()<<  " " << otherCell->get_outer_hit_id()<< std::endl;
// std::cout << otherCell->get_inner_x();
  auto r3 = otherCell->get_inner_r();
//  std::cout << "cells r copied " << r1 << " " << r2 << " " << r3 <<  std::endl;


  float z1 = get_inner_z();
  float z2 = get_outer_z();
  float z3 = otherCell->get_inner_z();
//  std::cout << "cells z copied " << z1 << " " << z2 << " " << z3 <<  std::endl;

//  std::cout << "result: "  << fabs(z1 * (r2 - r3) + z2 * (r3 - r1) +z3 * (r1 - r2))/(z2*z2 + r2*r2) << std::endl;
  return fabs(z1 * (r2 - r3) + z2 * (r3 - r1) +z3 * (r1 - r2))/(z2*z2 + r2*r2) <= 10.f;
}

bool CACell::is_root_cell(const unsigned int minimumCAState) const
{
  return (theInnerNeighbors.size() == 0 && theCAState >= minimumCAState);

}

void CACell::find_ntuplets ( std::vector<CAntuplet>& foundNtuplets, CAntuplet& tmpNtuplet, const unsigned int minHitsPerNtuplet) const
{

  // the building process for a track ends if:
  // it has no right neighbor
  // it has no compatible neighbor
  print_cell();
  std::cout << "number of cells " << tmpNtuplet.size() << std::endl;
  // the ntuplets is then saved if the number of hits it contains is greater than a threshold
  if (theOuterNeighbors.size() == 0 )
  {
    if ( tmpNtuplet.size() >= minHitsPerNtuplet - 1)
      foundNtuplets.push_back(tmpNtuplet);
    else
      return;
  } else
  {
  //  bool hasOneCompatibleNeighbor = false;
    for ( unsigned int i=0 ; i < theOuterNeighbors.size(); ++i)
    {
//      if (tmpNtuplet.size() <= 2 )
  //    {
  //      hasOneCompatibleNeighbor = true;
        tmpNtuplet.push_back(*(theOuterNeighbors.at(i)));
        theOuterNeighbors.at(i)->find_ntuplets(foundNtuplets, tmpNtuplet, minHitsPerNtuplet );
        tmpNtuplet.pop_back();
//      }
    }

 //   if (!hasOneCompatibleNeighbor && tmpNtuplet.size() >= minHitsPerNtuplet - 1)
 //   {
   //   foundNtuplets.push_back(tmpNtuplet);
   // }
  }

}

