#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
namespace cudaCompat {
  thread_local dim3 blockIdx;
  thread_local dim3 gridDim;

 struct InitGrid {
   InitGrid() { resetGrid();}
 };
 
 InitGrid initGrid;

}
