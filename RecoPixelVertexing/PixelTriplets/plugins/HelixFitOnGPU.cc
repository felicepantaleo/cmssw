#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HelixFitOnGPU.h"

void HelixFitOnGPU::allocateOnGPU(Tuples const *tuples,
                     TupleMultiplicity const *tupleMultiplicity,
                     OutputSoA *helix_fit_results);
  tuples_d = tuples;
  tupleMultiplicity_d = tupleMultiplicity;
  helix_fit_results_d = helix_fit_results;

  assert(tuples_d);
  assert(tupleMultiplicity_d);
  assert(helix_fit_results_d);
}

void HelixFitOnGPU::deallocateOnGPU() {}
