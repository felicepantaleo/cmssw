#include <limits>
#include <float>

//texture<float, 1, cudaReadModeElementType> tex;

__constant__ PixelClusterUtils utility;

// specialize for float to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<float>
{
    __device__ inline operator       float *()
    {
        extern __shared__ float __smem_d[];
        return (float *)__smem_d;
    }

    __device__ inline operator const float *() const
    {
        extern __shared__ float __smem_d[];
        return (float *)__smem_d;
    }
};


__device__ float pixelWeight(int* mapcharge, int* count,
		int* totalcharge, int clx, int cly, int x, int y, int direction) {

	if (x - clx + 10 < - utility.BinsX) return 0;
	if (y - cly + (utility.size_y + 1) / 2 < 0) return 0;
	if (x - clx + 10 >= utility.BinsX) return 0;
	if (y - cly + (utility.size_y + 1) / 2 >= utility.BinsY) return 0;

	int caseX = direction / 2;
	direction = direction % 2 + 1;

	unsigned int binX = clx * utility.BinsXposition / 160;
	unsigned int sizeY = utility.size_ysizeY + (direction - 1);
	// fact e' la percentuale di carica attesa in quel pixel dato un cluster
	// mapcharge e' la carica media rilasciata da un cluster in quel pixel
	// count e' il numero totale di cluster su quel pixel

	float x = mapcharge[(y - cly + (sizeY - 1) / 2) + utility.BinsY*(x - clx + 10 + caseX)+utility.BinsX*utility.BinsY*direction+utility.BinsDirections*utility.BinsX*utility.BinsY*binX+utility.BinsXposition*utility.BinsDirections*utility.BinsX*utility.BinsY*utility.binjetZOverRho)];

	float fact = x / totalcharge[direction + binX*utility.BinsDirections + utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections] *
			count[direction + binX*utility.BinsDirections + utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections];

	return fact;
}

inline
__device__ float* min(float* a, float* b)
{
	return (*a < *b)? a : b;

}

template <unsigned int blockSize>
__global__ void
minReduce(float *g_idata, float *g_odata, unsigned int n)
{
	float **sdata = SharedMemory<float*>();
	float inf = FLT_MAX;
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	float* myMin = &inf;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread


	while (i < n)
	{
		myMin = min(&g_idata[i], myMin);

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		// We always work with powerof2 sized arrays
		//if (nIsPow2 || i + blockSize < n)
			myMin = min(&g_idata[i+blockSize],myMin) ;

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = myMin;
	__syncthreads();


	// do reduction in shared mem
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			sdata[tid] = myMin = min(sdata[tid + 256], myMin);
		}

		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			sdata[tid] = myMin = min(sdata[tid + 128], myMin);
		}

		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tid <  64)
		{
			sdata[tid] = myMin = min(sdata[tid + 64], myMin);
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile float *smem = sdata;

		if (blockSize >=  64)
		{
			smem[tid] = myMin = min(sdata[tid + 32], myMin);
		}

		if (blockSize >=  32)
		{
			smem[tid] = myMin = min(sdata[tid + 16], myMin);
		}

		if (blockSize >=  16)
		{
			smem[tid] = myMin = min(sdata[tid + 8], myMin);
		}

		if (blockSize >=   8)
		{
			smem[tid] = myMin = min(sdata[tid + 4], myMin);
		}

		if (blockSize >=   4)
		{
			smem[tid] = myMin = min(sdata[tid + 2], myMin);
		}

		if (blockSize >=   2)
		{
			smem[tid] = myMin = min(sdata[tid + 1], myMin);
		}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

__global__ void kernel3(GPUPixelSoA* pixels, int* mapcharge, int* count,
		int* totalcharge, int numPositions, float* blocksChi2s)
{
	__shared__ float chi2 = FLT_MAX;
	__syncthreads();
	if(blockIdx.x < numPositions && blockIdx.y <= blockIdx.x && blockIdx.z <= blockIdx.x)
	{
		__shared__ float prob = 0.;
		__shared__ float chi2 = 0.;
		__shared__ unsigned int comb[3];
		__shared__ int clusterPosX;
		__shared__ int clusterPosY;
		__shared__ int direction;
		comb[0] = blockIdx.x;
		comb[1] = blockIdx.y;
		comb[2] = blockIdx.z;
		//	combination[0] = blockIdx.x;
		//	combination[1] = blockIdx.y;
		//	combination[2] = threadIdx.x;
		int x = threadIdx.x;
		int y = threadIdx.y;
		// TODO: invece di caricare tutti gli adc (anche quelli zero) converrebbe caricare solo quelli diversi da zero
		uint16_t originaladc = originalADC[blockDim.x*threadIdx.y + threadIdx.x];
		uint16_t adc = originaladc;

		for (unsigned int clusterIdx = 0; clusterIdx < 3; clusterIdx++)
		{
			if(threadIdx.x == 0 && threadIdx.y == 0)
			{

				clusterPosX = pixels.x[comb[clusterIdx] / 4];
				clusterPosY = pixels.y[comb[clusterIdx] / 4];
				direction = comb[clusterIdx] % 4;
			}
			__syncthreads();

			if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500)
					&& (y>= utility.ymin - (utility.size_y +1)/2) && y>=0
					&& (y<=utility.ymax + (utility.size_y +1)/2) && y < 500 )
			{
				float fact = pixelWeight(mapcharge, count, totalcharge, clusterPosX, clusterPosY, x, y, direction);

				if (fact > 0)
					adc -= fact * utility.expectedADC;
			}

			atomicAdd(&prob, count[direction%2+1 + clusterPosX / 32 *utility.BinsDirections + utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections]);
		}
		if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500)
				&& (y>= utility.ymin - (utility.size_y +1)/2) && y>=0
				&& (y<=utility.ymax + (utility.size_y +1)/2) && y < 500 )
		{
			float res = adc;
			float charge = fabs(originaladc - adc); // charge assigned to this pixel
			float chargeMeasured = originaladc;



			if (chargeMeasured < 5000 && charge < 5000) {  // threshold effect
				res = 0;
			}

			if (chargeMeasured <= 2000) chargeMeasured = 2000;
			if (charge < 2000) charge = 2000;



			atomicAdd(&chi2, (res * res) / (charge * charge));


		}
		__syncthreads();

		if(threadIdx.x == 0 && threadIdx.y == 0)
		{
			prob = prob/3; // 3 is the number of expected clusters
			chi2 = chi2/prob;
		}


	}

}



extern "C" void cudaClusterSplitter_(GPUPixelSoA* pixels, uint16_t* originalADC, int* gpu_mapcharge, int* gpu_count_array,
		int* gpu_totalcharge_array, PixelClusterUtils* constantDataNeededOnGPU, int numClusters, int numPositions) {

	// bind texture to buffer
	//	cudaBindTexture(0, tex, gpu_mapcharge, 168000*sizeof(int));
	cudaMemcpyToSymbol(  utility,  constantDataNeededOnGPU,   sizeof(PixelClusterUtils) );
	if(numClusters == 3)
	{
		dim3 block(128,1,1);
		dim3 grid(block.x,1,1);
		kernel3<<<grid,block>>>(originalADC, numPositions);

	}

	//   cudaUnbindTexture(tex);

}


prob /= expectedClusters;
chi2 /= prob;

if (chi2 < chiminlocal) {
	chiminlocal = chi2;
}
if (chi2 < chimin) {
	chiN = chi2 * prob / aCluster.size();
	chimin = chi2;
	bestcomb = comb;
	bestExpCluster = expectedClusters;
}
}
