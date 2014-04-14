#include <limits>
#include <cuda.h>
#include "RecoLocalTracker/SubCollectionProducers/interface/GPUPixel.h"

//texture<float, 1, cudaReadModeElementType> tex;

struct Chi2Comb {
	int16_t chi2;
	int8_t comb[6];
};


__constant__ PixelClusterUtils utility;


template<class T>
struct SharedMemory
{
    __device__ inline operator T*()
    {
     	extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
     	extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

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
	unsigned int sizeY = utility.size_y + (direction - 1);
	// fact e' la percentuale di carica attesa in quel pixel dato un cluster
	// mapcharge e' la carica media rilasciata da un cluster in quel pixel
	// count e' il numero totale di cluster su quel pixel

	float mc = mapcharge[(y - cly + (sizeY - 1) / 2) + utility.BinsY*(x - clx + 10 + caseX)+utility.BinsX*utility.BinsY*direction+utility.BinsDirections*utility.BinsX*utility.BinsY*binX+utility.BinsXposition*utility.BinsDirections*utility.BinsX*utility.BinsY* utility.binjetZOverRho];

	float fact = mc / totalcharge[direction + binX*utility.BinsDirections + utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections] *
			count[direction + binX*utility.BinsDirections + utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections];

	return fact;
}

inline
__device__ float* min(float* a, float* b)
{
	return (*a < *b)? a : b;

}

__device__ short atomicAddShort(short* address, short val)

{

    unsigned int *base_address = (unsigned int *) ((char *)address - ((size_t)address & 2));

    unsigned int long_val = ((size_t)address & 2) ? ((unsigned int)val << 16) : (unsigned short)val;



    unsigned int long_old = atomicAdd(base_address, long_val);

    if((size_t)address & 2) {

        return (short)(long_old >> 16);

    } else {

        unsigned int overflow = ((long_old & 0xffff) + long_val) & 0xffff0000;

        if (overflow)

            atomicSub(base_address, overflow);

        return (short)(long_old & 0xffff);

    }

}
__device__ Chi2Comb atomicMinChi2(Chi2Comb* address, Chi2Comb val)
{
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long  old = *addr_as_ull;
    unsigned long long  assumed;
    do
    {
        assumed = old;
        Chi2Comb* temp = (Chi2Comb*)&assumed;
        if (val.chi2 < temp->chi2 )
            old = atomicCAS(addr_as_ull, assumed, *(unsigned long long*)&val);
        else
            break;
    }
    while(assumed != old);
    return *((Chi2Comb*)&old);
}


//template <unsigned int blockSize>
//__global__ void
//minReduce(float *g_idata, float *g_odata, unsigned int n)
//{
//	float **sdata = SharedMemory<float*>();
//	float inf = FLT_MAX;
//	// perform first level of reduction,
//	// reading from global memory, writing to shared memory
//	unsigned int tid = threadIdx.x;
//	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
//	unsigned int gridSize = blockSize*2*gridDim.x;
//
//	float* myMin = &inf;
//
//	// we reduce multiple elements per thread.  The number is determined by the
//	// number of active thread blocks (via gridDim).  More blocks will result
//	// in a larger gridSize and therefore fewer elements per thread
//
//
//	while (i < n)
//	{
//		myMin = min(&g_idata[i], myMin);
//
//		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
//		// We always work with powerof2 sized arrays
//		//if (nIsPow2 || i + blockSize < n)
//			myMin = min(&g_idata[i+blockSize],myMin) ;
//
//		i += gridSize;
//	}
//
//	// each thread puts its local sum into shared memory
//	sdata[tid] = myMin;
//	__syncthreads();
//
//
//	// do reduction in shared mem
//	if (blockSize >= 512)
//	{
//		if (tid < 256)
//		{
//			sdata[tid] = myMin = min(sdata[tid + 256], myMin);
//		}
//
//		__syncthreads();
//	}
//
//	if (blockSize >= 256)
//	{
//		if (tid < 128)
//		{
//			sdata[tid] = myMin = min(sdata[tid + 128], myMin);
//		}
//
//		__syncthreads();
//	}
//
//	if (blockSize >= 128)
//	{
//		if (tid <  64)
//		{
//			sdata[tid] = myMin = min(sdata[tid + 64], myMin);
//		}
//
//		__syncthreads();
//	}
//
//	if (tid < 32)
//	{
//		// now that we are using warp-synchronous programming (below)
//		// we need to declare our shared memory volatile so that the compiler
//		// doesn't reorder stores to it and induce incorrect behavior.
//		volatile float *smem = sdata;
//
//		if (blockSize >=  64)
//		{
//			smem[tid] = myMin = min(sdata[tid + 32], myMin);
//		}
//
//		if (blockSize >=  32)
//		{
//			smem[tid] = myMin = min(sdata[tid + 16], myMin);
//		}
//
//		if (blockSize >=  16)
//		{
//			smem[tid] = myMin = min(sdata[tid + 8], myMin);
//		}
//
//		if (blockSize >=   8)
//		{
//			smem[tid] = myMin = min(sdata[tid + 4], myMin);
//		}
//
//		if (blockSize >=   4)
//		{
//			smem[tid] = myMin = min(sdata[tid + 2], myMin);
//		}
//
//		if (blockSize >=   2)
//		{
//			smem[tid] = myMin = min(sdata[tid + 1], myMin);
//		}
//	}
//
//	// write result for this block to global mem
//	if (tid == 0)
//		g_odata[blockIdx.x] = sdata[0];
//}
//



__global__ void kernel3(GPUPixelSoA* pixels, int* originalADC, int* mapcharge, int* count,
		int* totalcharge, int numPositions, Chi2Comb* reducedChi2s)
{
	__shared__ Chi2Comb combinationChi2;

	const unsigned int numClusters = 3;

	if(blockIdx.x < numPositions && blockIdx.y <= blockIdx.x && blockIdx.z <= blockIdx.y)
	{
		__shared__ float prob;

		__shared__ int clusterPosX;
		__shared__ int clusterPosY;
		__shared__ int direction;
		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			combinationChi2.chi2 = 0;
			combinationChi2.comb[0] = blockIdx.x;
			combinationChi2.comb[1] = blockIdx.y;
			combinationChi2.comb[2] = blockIdx.z;
			combinationChi2.comb[3] = -1;
			combinationChi2.comb[4] = -1;
			combinationChi2.comb[5] = -1;
			prob = 0.f;

		}
		__syncthreads();



		int x = threadIdx.x;
		// TODO: invece di caricare tutti gli adc (anche quelli zero) converrebbe caricare solo quelli diversi da zero
		int originaladc = originalADC[blockDim.x*threadIdx.y + threadIdx.x];
		int adc = originaladc;

		for (unsigned int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++)
		{
			__syncthreads();
			if(threadIdx.x == 0 && threadIdx.y == 0)
			{
				clusterPosX = pixels->x[combinationChi2.comb[clusterIdx] / 4];
				clusterPosY = pixels->y[combinationChi2.comb[clusterIdx] / 4];
				direction = combinationChi2.comb[clusterIdx] % 4;
			}
			__syncthreads();
			for(int y = threadIdx.y; (y>= utility.ymin - (utility.size_y + 1)/2) && y>=0
			&& (y<=utility.ymax + (utility.size_y + 1)/2) && y < 500; y+= blockDim.y)
			{
				if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500))
				{
					float fact = pixelWeight(mapcharge, count, totalcharge, clusterPosX, clusterPosY, x, y, direction);

					if (fact > 0)
						adc -= fact * utility.expectedADC;
				}
			}
			atomicAdd(&prob, count[direction%2+1 + clusterPosX / 32 *utility.BinsDirections +
			                       utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections]);
		}
		for(int y = threadIdx.y; (y>= utility.ymin - (utility.size_y +1)/2) && y>=0
		&& (y<=utility.ymax + (utility.size_y +1)/2) && y < 500; y+= blockDim.y)
		{
			if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500)  )
			{
				float res = adc;
				float charge = (originaladc - adc) > 0? (originaladc - adc): -(originaladc - adc) ; // charge assigned to this pixel
				float chargeMeasured = originaladc;



				if (charge < 5000 && chargeMeasured < 5000 ) {  // threshold effect
					res = 0;
				}

				if (chargeMeasured <= 2000) chargeMeasured = 2000;
				if (charge < 2000) charge = 2000;



				atomicAddShort(&combinationChi2.chi2, (int16_t)((res * res) / (charge * charge)));


			}
			__syncthreads();
		}

		if(threadIdx.x == 0 && threadIdx.y == 0)
		{
			prob = prob/numClusters;
			combinationChi2.chi2 = combinationChi2.chi2/prob;
			atomicMinChi2(reducedChi2s,combinationChi2 );
		}

		__syncthreads();


	}

}


__global__ void kernel4(GPUPixelSoA* pixels, int* originalADC, int* mapcharge, int* count,
		int* totalcharge, int numPositions, Chi2Comb* reducedChi2s)
{
	__shared__ Chi2Comb combinationChi2;

	const unsigned int numClusters = 4;

	if(blockIdx.x < numPositions && blockIdx.y <= blockIdx.x && blockIdx.z <= blockIdx.y)
	{
		__shared__ float prob;
		__shared__ int clusterPosX;
		__shared__ int clusterPosY;
		__shared__ int direction;
		int x = threadIdx.x;
		// TODO: invece di caricare tutti gli adc (anche quelli zero) converrebbe caricare solo quelli diversi da zero
		int originaladc = originalADC[blockDim.x*threadIdx.y + threadIdx.x];
		int adc = originaladc;
		for(int comb3 = 0; comb3 <= blockIdx.z; ++comb3)
		{

			__syncthreads();

			if (threadIdx.x == 0 && threadIdx.y == 0)
			{
				combinationChi2.chi2 = 0;
				combinationChi2.comb[0] = blockIdx.x;
				combinationChi2.comb[1] = blockIdx.y;
				combinationChi2.comb[2] = blockIdx.z;
				combinationChi2.comb[3] = comb3;
				prob = 0.f;

			}
			__syncthreads();





			for (unsigned int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++)
			{
				__syncthreads();

				if(threadIdx.x == 0 && threadIdx.y == 0)
				{

					clusterPosX = pixels->x[combinationChi2.comb[clusterIdx] / 4];
					clusterPosY = pixels->y[combinationChi2.comb[clusterIdx] / 4];
					direction = combinationChi2.comb[clusterIdx] % 4;
				}
				__syncthreads();
				for(int y = threadIdx.y; (y>= utility.ymin - (utility.size_y + 1)/2) && y>=0
				&& (y<=utility.ymax + (utility.size_y + 1)/2) && y < 500; y+= blockDim.y)
				{
					if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500))
					{
						float fact = pixelWeight(mapcharge, count, totalcharge, clusterPosX, clusterPosY, x, y, direction);

						if (fact > 0)
							adc -= fact * utility.expectedADC;
					}
				}
				atomicAdd(&prob, count[direction%2+1 + clusterPosX / 32 *utility.BinsDirections +
				                       utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections]);
			}
			for(int y = threadIdx.y; (y>= utility.ymin - (utility.size_y +1)/2) && y>=0
			&& (y<=utility.ymax + (utility.size_y +1)/2) && y < 500; y+= blockDim.y)
			{
				if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500)  )
				{
					float res = adc;
					float charge = (originaladc - adc) > 0? (originaladc - adc): -(originaladc - adc) ; // charge assigned to this pixel
					float chargeMeasured = originaladc;



					if (charge < 5000 && chargeMeasured < 5000 ) {  // threshold effect
						res = 0;
					}

					if (chargeMeasured <= 2000) chargeMeasured = 2000;
					if (charge < 2000) charge = 2000;



					atomicAddShort(&combinationChi2.chi2, (int16_t)((res * res) / (charge * charge)));


				}
				__syncthreads();
			}

			if(threadIdx.x == 0 && threadIdx.y == 0)
			{
				prob = prob/numClusters;
				combinationChi2.chi2 = combinationChi2.chi2/prob;
				atomicMinChi2(reducedChi2s,combinationChi2 );
			}

			__syncthreads();


		}
	}

}


__global__ void kernel5(GPUPixelSoA* pixels, int* originalADC, int* mapcharge, int* count,
		int* totalcharge, int numPositions, Chi2Comb* reducedChi2s)
{
	__shared__ Chi2Comb combinationChi2;

	const unsigned int numClusters = 5;

	if(blockIdx.x < numPositions && blockIdx.y <= blockIdx.x && blockIdx.z <= blockIdx.y)
	{
		__shared__ float prob;
		__shared__ int clusterPosX;
		__shared__ int clusterPosY;
		__shared__ int direction;
		int x = threadIdx.x;
		// TODO: invece di caricare tutti gli adc (anche quelli zero) converrebbe caricare solo quelli diversi da zero
		int originaladc = originalADC[blockDim.x*threadIdx.y + threadIdx.x];
		int adc = originaladc;
		for(int comb3 = 0; comb3 <= blockIdx.z; ++comb3)
		{
			for (int comb4 = 0; comb4 <= comb3; ++comb4)
			{
				__syncthreads();

				if (threadIdx.x == 0 && threadIdx.y == 0)
				{
					combinationChi2.chi2 = 0;
					combinationChi2.comb[0] = blockIdx.x;
					combinationChi2.comb[1] = blockIdx.y;
					combinationChi2.comb[2] = blockIdx.z;
					combinationChi2.comb[3] = comb3;
					combinationChi2.comb[4] = comb4;
					combinationChi2.comb[5] = -1;
					prob = 0.f;

				}
				__syncthreads();





				for (unsigned int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++)
				{
					__syncthreads();

					if(threadIdx.x == 0 && threadIdx.y == 0)
					{

						clusterPosX = pixels->x[combinationChi2.comb[clusterIdx] / 4];
						clusterPosY = pixels->y[combinationChi2.comb[clusterIdx] / 4];
						direction = combinationChi2.comb[clusterIdx] % 4;
					}
					__syncthreads();
					for(int y = threadIdx.y; (y>= utility.ymin - (utility.size_y + 1)/2) && y>=0
					&& (y<=utility.ymax + (utility.size_y + 1)/2) && y < 500; y+= blockDim.y)
					{
						if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500))
						{
							float fact = pixelWeight(mapcharge, count, totalcharge, clusterPosX, clusterPosY, x, y, direction);

							if (fact > 0)
								adc -= fact * utility.expectedADC;
						}
					}
					atomicAdd(&prob, count[direction%2+1 + clusterPosX / 32 *utility.BinsDirections +
					                       utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections]);
				}
				for(int y = threadIdx.y; (y>= utility.ymin - (utility.size_y +1)/2) && y>=0
				&& (y<=utility.ymax + (utility.size_y +1)/2) && y < 500; y+= blockDim.y)
				{
					if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500)  )
					{
						float res = adc;
						float charge = (originaladc - adc) > 0? (originaladc - adc): -(originaladc - adc) ; // charge assigned to this pixel
						float chargeMeasured = originaladc;



						if (charge < 5000 && chargeMeasured < 5000 ) {  // threshold effect
							res = 0;
						}

						if (chargeMeasured <= 2000) chargeMeasured = 2000;
						if (charge < 2000) charge = 2000;



						atomicAddShort(&combinationChi2.chi2, (int16_t)((res * res) / (charge * charge)));


					}
					__syncthreads();
				}

				if(threadIdx.x == 0 && threadIdx.y == 0)
				{
					prob = prob/numClusters;
					combinationChi2.chi2 = combinationChi2.chi2/prob;
					atomicMinChi2(reducedChi2s,combinationChi2 );
				}

				__syncthreads();


			}
		}
	}

}

__global__ void kernel6(GPUPixelSoA* pixels, int* originalADC, int* mapcharge, int* count,
		int* totalcharge, int numPositions, Chi2Comb* reducedChi2s)
{
	__shared__ Chi2Comb combinationChi2;

	const unsigned int numClusters = 6;

	if(blockIdx.x < numPositions && blockIdx.y <= blockIdx.x && blockIdx.z <= blockIdx.y)
	{
		__shared__ float prob;
		__shared__ int clusterPosX;
		__shared__ int clusterPosY;
		__shared__ int direction;
		int x = threadIdx.x;
		// TODO: invece di caricare tutti gli adc (anche quelli zero) converrebbe caricare solo quelli diversi da zero
		int originaladc = originalADC[blockDim.x*threadIdx.y + threadIdx.x];
		int adc = originaladc;
		for(int comb3 = 0; comb3 <= blockIdx.z; ++comb3)
		{
			for (int comb4 = 0; comb4 <= comb3; ++comb4)
			{
				for (int comb5 = 0; comb5 <= comb4; ++comb5)
				{
					__syncthreads();

					if (threadIdx.x == 0 && threadIdx.y == 0)
					{
						combinationChi2.chi2 = 0;
						combinationChi2.comb[0] = blockIdx.x;
						combinationChi2.comb[1] = blockIdx.y;
						combinationChi2.comb[2] = blockIdx.z;
						combinationChi2.comb[3] = comb3;
						combinationChi2.comb[4] = comb4;
						combinationChi2.comb[5] = comb5;
						prob = 0.f;

					}
					__syncthreads();





					for (unsigned int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++)
					{
						__syncthreads();

						if(threadIdx.x == 0 && threadIdx.y == 0)
						{

							clusterPosX = pixels->x[combinationChi2.comb[clusterIdx] / 4];
							clusterPosY = pixels->y[combinationChi2.comb[clusterIdx] / 4];
							direction = combinationChi2.comb[clusterIdx] % 4;
						}
						__syncthreads();
						for(int y = threadIdx.y; (y>= utility.ymin - (utility.size_y + 1)/2) && y>=0
						&& (y<=utility.ymax + (utility.size_y + 1)/2) && y < 500; y+= blockDim.y)
						{
							if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500))
							{
								float fact = pixelWeight(mapcharge, count, totalcharge, clusterPosX, clusterPosY, x, y, direction);

								if (fact > 0)
									adc -= fact * utility.expectedADC;
							}
						}
						atomicAdd(&prob, count[direction%2+1 + clusterPosX / 32 *utility.BinsDirections +
						                       utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections]);
					}
					for(int y = threadIdx.y; (y>= utility.ymin - (utility.size_y +1)/2) && y>=0
					&& (y<=utility.ymax + (utility.size_y +1)/2) && y < 500; y+= blockDim.y)
					{
						if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500)  )
						{
							float res = adc;
							float charge = (originaladc - adc) > 0? (originaladc - adc): -(originaladc - adc) ; // charge assigned to this pixel
							float chargeMeasured = originaladc;



							if (charge < 5000 && chargeMeasured < 5000 ) {  // threshold effect
								res = 0;
							}

							if (chargeMeasured <= 2000) chargeMeasured = 2000;
							if (charge < 2000) charge = 2000;



							atomicAddShort(&combinationChi2.chi2, (int16_t)((res * res) / (charge * charge)));


						}
						__syncthreads();
					}
					if(threadIdx.x == 0 && threadIdx.y == 0)
					{
						prob = prob/numClusters;
						combinationChi2.chi2 = combinationChi2.chi2/prob;
						atomicMinChi2(reducedChi2s,combinationChi2 );
					}

					__syncthreads();

				}
			}
		}
	}

}

extern "C" void cudaClusterSplitter_(GPUPixelSoA* pixels, int* originalADC, int* gpu_mapcharge, int* gpu_count_array,
		int* gpu_totalcharge_array, PixelClusterUtils* constantDataNeededOnGPU, unsigned int numClusters, int numPositions) {

	// bind texture to buffer
	//	cudaBindTexture(0, tex, gpu_mapcharge, 168000*sizeof(int));
	cudaMemcpyToSymbol(  utility,  constantDataNeededOnGPU,   sizeof(PixelClusterUtils) );
	Chi2Comb* gpu_bestChi2Combination;
	Chi2Comb* host_bestChi2Combination;
	cudaMallocHost((void**)&host_bestChi2Combination, sizeof(Chi2Comb));
	cudaMalloc((void**)&gpu_bestChi2Combination, sizeof(Chi2Comb));

	host_bestChi2Combination->chi2 = SHRT_MAX;
	for(int i = 0; i<6; ++i)
		host_bestChi2Combination->comb[i] = -1;
	cudaMemcpyAsync(gpu_bestChi2Combination, host_bestChi2Combination,sizeof(Chi2Comb), cudaMemcpyHostToDevice, 0);

	dim3 block(512,2,1);
	dim3 grid(numPositions,numPositions,numPositions);
	size_t SmemSize = sizeof(Chi2Comb)+4*sizeof(float);

	if(numClusters == 3)
	{
//		float* blocksChi2s;
//		cudaMalloc((void**)&blocksChi2s, (64<<numClusters)*sizeof(float));


		kernel3<<<grid,block,SmemSize,0>>>(pixels, originalADC, gpu_mapcharge,
				gpu_count_array, gpu_totalcharge_array, numPositions, gpu_bestChi2Combination);
//		//TODO: evaluate the amount of shared memory needed
//		int numblocksReduction = 2 << (6*numClusters-10);
//		minReduce<<<numblocksReduction,2048,2048 * sizeof(float),0>>>(blocksChi2s, reducedChi2, blockDim.x*blockDim.y*blockDim.z);
	}
	else if (numClusters == 4)
	{
		kernel4<<<grid,block,SmemSize,0>>>(pixels, originalADC, gpu_mapcharge,
				gpu_count_array, gpu_totalcharge_array, numPositions, gpu_bestChi2Combination);
	}
	else if (numClusters == 5)
	{
		kernel5<<<grid,block,SmemSize,0>>>(pixels, originalADC, gpu_mapcharge,
				gpu_count_array, gpu_totalcharge_array, numPositions, gpu_bestChi2Combination);
	}
	else if (numClusters == 6)
	{
		kernel6<<<grid,block,SmemSize,0>>>(pixels, originalADC, gpu_mapcharge,
				gpu_count_array, gpu_totalcharge_array, numPositions, gpu_bestChi2Combination);
	}
	cudaMemcpyAsync(host_bestChi2Combination, gpu_bestChi2Combination,sizeof(Chi2Comb), cudaMemcpyDeviceToHost, 0);
	cudaDeviceSynchronize();
	cudaFreeHost(host_bestChi2Combination);
	cudaFree(gpu_bestChi2Combination);
	//   cudaUnbindTexture(tex);

}
