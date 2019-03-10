#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
namespace cudaCompat {
  thread_local dim3 blockIdx;
  thread_local dim3 gridDim;
}

namespace {
 struct InitGrid {
   InitGrid() { cudaCompat::resetGrid();}
 };
 
 const InitGrid initGrid;

}
