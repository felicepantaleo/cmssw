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

    if (theOuterNeighbors.at(i)->get_CA_state() == theCAState)
    {



      hasSameStateNeighbors = 1;

      break;
    }
  }

}

void CACell::update_state()
{
  theCAState +=hasSameStateNeighbors;


}

void CACell::check_alignment_and_tag(CACell* innerCell)
{

  if (are_aligned_RZ(innerCell))
  {


    tag_as_inner_neighbor(innerCell);
    innerCell->tag_as_outer_neighbor(this);

  }
  else
  {


  }
}

bool CACell::are_aligned_RZ(const CACell* otherCell) const
{


  float r1 = otherCell->get_inner_r();   
  float r2 = get_inner_r();
  float r3 = get_outer_r();
    

  float z1 = otherCell->get_inner_z();
  float z2 = get_inner_z();
  float z3 = get_outer_z();

  float distance_13_squared = (r1-r3)*(r1-r3) + (z1-z3)*(z1-z3);  
  float tan_12_13 = 2*fabs(z1 * (r2 - r3) + z2 * (r3 - r1) +z3 * (r1 - r2))/distance_13_squared;
    
  std::cout <<   "result of alignment " <<  tan_12_13  << std::endl;
//  
  return tan_12_13 <= 1e-2;
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

