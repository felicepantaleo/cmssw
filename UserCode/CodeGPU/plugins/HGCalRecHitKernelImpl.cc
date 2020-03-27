

__global__
void HGCeeRecHitKernel(HGCUncalibratedRecHit* oldhits, 
		       HGCUncalibratedRecHit* newhits, size_t length)
{
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x)
    {
      newhits[i] = oldhits[i];
    }
}
