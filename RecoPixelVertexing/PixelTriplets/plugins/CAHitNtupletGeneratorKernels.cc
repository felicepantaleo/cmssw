#include "RecoPixelVertexing/PixelTriplets/plugins/CAHitNtupletGeneratorKernelsImpl.h"
template<>
void CAHitNtupletGeneratorKernelsCPU::printCounters(Counters const * counters) {
   kernel_printCounters(counters);
}

