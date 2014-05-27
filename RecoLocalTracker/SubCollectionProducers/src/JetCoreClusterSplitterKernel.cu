#include <limits>
#include <cuda.h>
#include "RecoLocalTracker/SubCollectionProducers/interface/GPUPixel.h"
#include <iostream>
#include <stdio.h>
#include "RecoLocalTracker/SubCollectionProducers/interface/cuPrintf.cu"

//texture<float, 1, cudaReadModeElementType> tex;


#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt,  \
		blockIdx.x,	   \
		threadIdx.x,     \
		__VA_ARGS__)


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


// specialize for Chi2Comb to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<Chi2Comb>
{
	__device__ inline operator       Chi2Comb *()
    						{
		extern __shared__ Chi2Comb __smem_d[];
		return (Chi2Comb *)__smem_d;
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

	int binX = int(clx * utility.BinsXposition / 160);
	int sizeY = utility.size_y + (direction - 1);
	// fact e' la percentuale di carica attesa in quel pixel dato un cluster
	// mapcharge e' la carica media rilasciata da un cluster in quel pixel
	// count e' il numero totale di cluster su quel pixel

	float mc = mapcharge[(y - cly + (sizeY - 1) / 2) + utility.BinsY*(x - clx + 10 + caseX)+utility.BinsX*utility.BinsY*direction+utility.BinsDirections*utility.BinsX*utility.BinsY*binX+utility.BinsXposition*utility.BinsDirections*utility.BinsX*utility.BinsY* utility.binjetZOverRho];

	float fact = mc / (float)totalcharge[direction + binX*utility.BinsDirections + utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections] *
			(float)count[direction + binX*utility.BinsDirections + utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections];

	return fact;
}

inline
__device__ float* min(float* a, float* b)
{
	return (*a < *b)? a : b;

}

inline
__device__ void min(Chi2Comb& a, Chi2Comb& b)
{
	if (a.chi2 > b.chi2)
	{
		a.chi2 = b.chi2;
		for(int i = 0; i< 6; ++i)
			a.comb[i] = b.comb[i];
	}


}
inline
__device__ void min(Chi2Comb& a, volatile Chi2Comb& b)
{

	if (a.chi2 > b.chi2)
	{
		a.chi2 = b.chi2;
#pragma unroll
		for(int i = 0; i< 6; ++i)
			a.comb[i] = b.comb[i];
	}
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


// inside warp reduction
__device__ void sharedMinReduction(Chi2Comb* chisquares, int blockSize)
{
	unsigned int tid = threadIdx.x;

	if (tid < 32)
	{
		Chi2Comb myChi2 = chisquares[tid];

		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile Chi2Comb *smem = chisquares;

		if (tid + 32 > blockSize)
		{
			smem[tid].chi2 = myChi2.chi2;
#pragma unroll
			for(int idx = 0; idx < 6 ; ++idx)
				smem[tid].comb[idx] = myChi2.comb[idx];
		}
		else
		{
			min(myChi2,  smem[tid  +  32]);
			smem[tid].chi2 = myChi2.chi2;
#pragma unroll
			for(int idx = 0; idx < 6 ; ++idx)
				smem[tid].comb[idx] = myChi2.comb[idx];
		}

		if (tid + 16 > blockSize)
		{
			smem[tid].chi2 = myChi2.chi2;
#pragma unroll
			for(int idx = 0; idx < 6 ; ++idx)
				smem[tid].comb[idx] = myChi2.comb[idx];
		}
		else
		{
			min(myChi2,  smem[tid  +  16]);
			smem[tid].chi2 = myChi2.chi2;
#pragma unroll
			for(int idx = 0; idx < 6 ; ++idx)
				smem[tid].comb[idx] = myChi2.comb[idx];
		}

		if (tid + 8 > blockSize)
		{
			smem[tid].chi2 = myChi2.chi2;
#pragma unroll
			for(int idx = 0; idx < 6 ; ++idx)
				smem[tid].comb[idx] = myChi2.comb[idx];
		}
		else
		{
			min(myChi2,  smem[tid  +  8]);
			smem[tid].chi2 = myChi2.chi2;
#pragma unroll
			for(int idx = 0; idx < 6 ; ++idx)
				smem[tid].comb[idx] = myChi2.comb[idx];
		}

		if (tid + 4 > blockSize)
		{
			smem[tid].chi2 = myChi2.chi2;
#pragma unroll
			for(int idx = 0; idx < 6 ; ++idx)
				smem[tid].comb[idx] = myChi2.comb[idx];
		}
		else
		{
			min(myChi2,  smem[tid  +  4]);
			smem[tid].chi2 = myChi2.chi2;
#pragma unroll
			for(int idx = 0; idx < 6 ; ++idx)
				smem[tid].comb[idx] = myChi2.comb[idx];
		}

		if (tid + 2 > blockSize)
		{
			smem[tid].chi2 = myChi2.chi2;
#pragma unroll
			for(int idx = 0; idx < 6 ; ++idx)
				smem[tid].comb[idx] = myChi2.comb[idx];
		}
		else
		{
			min(myChi2,  smem[tid  +  2]);
			smem[tid].chi2 = myChi2.chi2;
#pragma unroll
			for(int idx = 0; idx < 6 ; ++idx)
				smem[tid].comb[idx] = myChi2.comb[idx];
		}

		if (tid + 1 > blockSize)
		{
			smem[tid].chi2 = myChi2.chi2;
#pragma unroll
			for(int idx = 0; idx < 6 ; ++idx)
				smem[tid].comb[idx] = myChi2.comb[idx];
		}
		else
		{
			min(myChi2,  smem[tid  +  1]);
			smem[tid].chi2 = myChi2.chi2;
#pragma unroll
			for(int idx = 0; idx < 6 ; ++idx)
				smem[tid].comb[idx] = myChi2.comb[idx];
		}
	}
//
//		if (blockSize >=  64)
//		{
//			min(myChi2,  smem[tid  +  32]);
//			smem[tid].chi2 = myChi2.chi2;
//
//		}
//
//		if (blockSize >=  32)
//		{
//			min(myChi2,  smem[tid  + 16]);
//			smem[tid].chi2 = myChi2.chi2;
//			for(int idx = 0; idx < 6 ; ++idx)
//				smem[tid].comb[idx] = myChi2.comb[idx];
//		}
//
//		if (blockSize >=  16)
//		{
//			min(myChi2,  smem[tid  +  8]);
//			smem[tid].chi2 = myChi2.chi2;
//			for(int idx = 0; idx < 6 ; ++idx)
//				smem[tid].comb[idx] = myChi2.comb[idx];
//		}
//
//		if (blockSize >=   8)
//		{
//			min(myChi2,  smem[tid  +  4]);
//			smem[tid].chi2 = myChi2.chi2;
//			for(int idx = 0; idx < 6 ; ++idx)
//				smem[tid].comb[idx] = myChi2.comb[idx];
//		}
//
//		if (blockSize >=   4)
//		{
//			min(myChi2,  smem[tid  + 2]);
//			smem[tid].chi2 = myChi2.chi2;
//			for(int idx = 0; idx < 6 ; ++idx)
//				smem[tid].comb[idx] = myChi2.comb[idx];
//		}
//
//		if (blockSize >=   2)
//		{
//			min(myChi2,  smem[tid  + 1]);
//			smem[tid].chi2 = myChi2.chi2;
//			for(int idx = 0; idx < 6 ; ++idx)
//				smem[tid].comb[idx] = myChi2.comb[idx];
//		}
//	}


}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T *g_idata, T *g_odata, unsigned int n)
{
	T *sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	T mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i];

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += g_idata[i+blockSize];

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 256];
		}

		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 128];
		}

		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tid <  64)
		{
			sdata[tid] = mySum = mySum + sdata[tid +  64];
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile T *smem = sdata;

		if (blockSize >=  64)
		{
			smem[tid] = mySum = mySum + smem[tid + 32];
		}

		if (blockSize >=  32)
		{
			smem[tid] = mySum = mySum + smem[tid + 16];
		}

		if (blockSize >=  16)
		{
			smem[tid] = mySum = mySum + smem[tid +  8];
		}

		if (blockSize >=   8)
		{
			smem[tid] = mySum = mySum + smem[tid +  4];
		}

		if (blockSize >=   4)
		{
			smem[tid] = mySum = mySum + smem[tid +  2];
		}

		if (blockSize >=   2)
		{
			smem[tid] = mySum = mySum + smem[tid +  1];
		}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}







//__global__ void analyseCluster(Chi2Comb combination, GPUPixelSoA* pixels, int* originalADC, PixelClusterUtils utility,
//		int* mapcharge, int* count,	int* totalcharge, int numClusters)
//{
//
//	int idx = blockIdx.x*blockDim.x + threadIdx.x;
//	int idy = blockIdx.y*blockDim.y + threadIdx.y;
//	__shared__ int clusterPosX;
//	__shared__ int clusterPosY;
//	__shared__ int direction;
//	__shared__ float prob;
//	__shared__ float chi2;
//
//	if (threadIdx.x == 0 && threadIdx.y == 0)
//	{
//		prob = 0.f;
//	}
//	for (unsigned int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++)
//	{
//		if(threadIdx.x == 0 && threadIdx.y == 0)
//		{
//			clusterPosX = pixels->x[combinationChi2.comb[clusterIdx] / 4];
//			clusterPosY = pixels->y[combinationChi2.comb[clusterIdx] / 4];
//			direction = combinationChi2.comb[clusterIdx] % 4;
//			prob+=count[direction%2+1 + clusterPosX / 32 *utility.BinsDirections +
//	                       utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections];
//
//		}
//		__syncthreads();
//		if (idx < 500 && idy < 500)
//		{
//			// TODO: invece di caricare tutti gli adc (anche quelli zero) converrebbe caricare solo quelli diversi da zero
//			int originaladc = originalADC[500*idy + idx];
//			int adc = originaladc;
//			if((idx >= utility.xmin - 5) && (idx<= utility.xmax + 5) &&
//					(idy >= utility.ymin - (utility.size_y + 1)/2) && (idy<=utility.ymax + (utility.size_y + 1)/2))
//			{
//				float fact = pixelWeight(mapcharge, count, totalcharge, clusterPosX, clusterPosY, x, y, direction);
//
//				adc -= fact * utility.expectedADC;
//			}
//
//		}
//
//	}
//
//
//
////		atomicAdd(&prob, count[direction%2+1 + clusterPosX / 32 *utility.BinsDirections +
////		                       utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections]);
////	}
////	for(int y = threadIdx.y; (y>= utility.ymin - (utility.size_y +1)/2) && y>=0
////	&& (y<=utility.ymax + (utility.size_y +1)/2) && y < 500; y+= blockDim.y)
////	{
////		if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500)  )
////		{
////			float res = adc;
////			float charge = (originaladc - adc) > 0? (originaladc - adc): -(originaladc - adc) ; // charge assigned to this pixel
////			float chargeMeasured = originaladc;
////
////
////
////			if (charge < 5000 && chargeMeasured < 5000 ) {  // threshold effect
////				res = 0;
////			}
////
////			if (chargeMeasured <= 2000) chargeMeasured = 2000;
////			if (charge < 2000) charge = 2000;
////
////
////
////			atomicAdd(&chi2, ((res * res) / (charge * charge)));
////
////			__syncthreads();
////
////
////		}
////		__syncthreads();
////	}
//
//}

__global__ void kernel3threads(GPUPixelSoA* pixels, int* originalADC, int* mapcharge, int* count,
		int* totalcharge, int numPositions, Chi2Comb* reducedChi2s)
{

	if(blockIdx.x < numPositions && blockIdx.y <= blockIdx.x && threadIdx.x <= blockIdx.y)
	{
//		Chi2Comb *shared_combinationChi2 = SharedMemory<Chi2Comb>();
		__shared__ Chi2Comb shared_combinationChi2[64];

		// perform first level of reduction,
		// reading from global memory, writing to shared memory
		unsigned int tid = threadIdx.x;
		Chi2Comb combinationChi2;


		const unsigned int numClusters = 3;


		float chi2=0;
		combinationChi2.chi2 = 0;
		combinationChi2.comb[0] = blockIdx.x;
		combinationChi2.comb[1] = blockIdx.y;
		combinationChi2.comb[2] = threadIdx.x;
		combinationChi2.comb[3] = -1;
		combinationChi2.comb[4] = -1;
		combinationChi2.comb[5] = -1;
		//			CUPRINTF("\n numPositions = %d, comb = %d %d %d %d %d %d", numPositions,combinationChi2.comb[0],combinationChi2.comb[1],combinationChi2.comb[2],combinationChi2.comb[3],combinationChi2.comb[4],combinationChi2.comb[5]);
		float prob = 0.f;

		//		CUPRINTF("\nCOMBINATION %d, %d, %d\n",combinationChi2.comb[0], combinationChi2.comb[1], combinationChi2.comb[2]);

		for (unsigned int clusterIdx = 0; clusterIdx < numClusters; ++clusterIdx)
		{

			int clusterPosX = pixels[combinationChi2.comb[clusterIdx] / 4].x;
//			int clusterPosY = pixels->y[combinationChi2.comb[clusterIdx] / 4];
			int direction = combinationChi2.comb[clusterIdx] % 4;
			prob += count[direction%2+1 + clusterPosX / 32 *utility.BinsDirections +
			              utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections];
		}

		//			CUPRINTF("\n%d, %d, %d\n",clusterPosX, clusterPosY, direction);
		int xmin = utility.xmin-5 >= 0 ? utility.xmin-5 : 0;
		int xmax = utility.xmax+5 <= 500 ? utility.xmax+5 : 500;
		int ymin = utility.ymin-(utility.size_y + 1)/2 >= 0 ? utility.ymin-(utility.size_y + 1)/2 : 0;
		int ymax = utility.ymax + (utility.size_y + 1)/2 <=500 ? utility.ymax + (utility.size_y + 1)/2 : 500;

		for(int y = ymin; y<ymax; ++y)
		{
			for (int x = xmin; x<xmax; ++x)
			{
				int originaladc = 0;
				int adc = 0;
				//					CUPRINTF("\nHEY %d!!!!!!\n", x);
				for (unsigned int clusterIdx = 0; clusterIdx < numClusters; ++clusterIdx)
				{



					int clusterPosX = pixels[combinationChi2.comb[clusterIdx] / 4].x;
					int clusterPosY = pixels[combinationChi2.comb[clusterIdx] / 4].y;
					int direction = combinationChi2.comb[clusterIdx] % 4;


					if(clusterIdx == 0)
					{
						originaladc = originalADC[500*y + x];
						adc = originaladc;

					}
					float fact = pixelWeight(mapcharge, count, totalcharge, clusterPosX, clusterPosY, x, y, direction);

					if (fact > 0)
						adc -= fact * utility.expectedADC;

					if(clusterIdx == numClusters-1)
					{
						float res = adc;
						float charge = (originaladc - adc) > 0? (originaladc - adc): -(originaladc - adc) ; // charge assigned to this pixel
						float chargeMeasured = originaladc;



						if (charge < 5000 && chargeMeasured < 5000 )
						{  // threshold effect
							res = 0;
						}

						if (chargeMeasured <= 2000) chargeMeasured = 2000;
						if (charge < 2000) charge = 2000;


						chi2 +=((res * res) / (charge * charge));
						//						CUPRINTF("\n\n\n  chi2 %f = !!!!!\n", chi2);

					}
				}
			}
		}



	prob = prob/numClusters;
	combinationChi2.chi2 = (1024*(chi2/prob))<USHRT_MAX?(1024*(chi2/prob)):USHRT_MAX;
	__syncthreads();

	shared_combinationChi2[tid] = combinationChi2;
	__syncthreads();

	sharedMinReduction(shared_combinationChi2, combinationChi2.comb[1]);
	__syncthreads();

	if(tid==0)
		atomicMinChi2(reducedChi2s,shared_combinationChi2[0]);
//	CUPRINTF("\n block chi2 = %hd, global chi2 = %hd", shared_combinationChi2[0].chi2, reducedChi2s->chi2);

	}
}





__global__ void kernel4threads(GPUPixelSoA* pixels, int* originalADC, int* mapcharge, int* count,
		int* totalcharge, int numPositions, Chi2Comb* reducedChi2s)
{

	if(blockIdx.x < numPositions && blockIdx.y <= blockIdx.x && blockIdx.z <= blockIdx.y && threadIdx.x <= blockIdx.z)
	{
//		Chi2Comb *shared_combinationChi2 = SharedMemory<Chi2Comb>();

		// perform first level of reduction,
		// reading from global memory, writing to shared memory
		unsigned int tid = threadIdx.x;
		Chi2Comb combinationChi2;

		__shared__ Chi2Comb shared_combinationChi2[64];

		const unsigned int numClusters = 4;


		float chi2=0;
		combinationChi2.chi2 = 0;
		combinationChi2.comb[0] = blockIdx.x;
		combinationChi2.comb[1] = blockIdx.y;
		combinationChi2.comb[2] = blockIdx.z;
		combinationChi2.comb[3] = threadIdx.x;
		combinationChi2.comb[4] = -1;
		combinationChi2.comb[5] = -1;
		//			CUPRINTF("\n numPositions = %d, comb = %d %d %d %d %d %d", numPositions,combinationChi2.comb[0],combinationChi2.comb[1],combinationChi2.comb[2],combinationChi2.comb[3],combinationChi2.comb[4],combinationChi2.comb[5]);
		float prob = 0.f;

		//		CUPRINTF("\nCOMBINATION %d, %d, %d, %d\n",combinationChi2.comb[0], combinationChi2.comb[1], combinationChi2.comb[2], combinationChi2.comb[3]);

		for (unsigned int clusterIdx = 0; clusterIdx < numClusters; ++clusterIdx)
		{

			int clusterPosX  = pixels[combinationChi2.comb[clusterIdx] / 4].x;
//			int clusterPosY = pixels->y[combinationChi2.comb[clusterIdx] / 4];
			int direction = combinationChi2.comb[clusterIdx] % 4;
			prob += count[direction%2+1 + clusterPosX / 32 *utility.BinsDirections +
			              utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections];
		}

		//			CUPRINTF("\n%d, %d, %d\n",clusterPosX, clusterPosY, direction);
		int xmin = utility.xmin-5 >= 0 ? utility.xmin-5 : 0;
		int xmax = utility.xmax+5 <= 500 ? utility.xmax+5 : 500;
		int ymin = utility.ymin-(utility.size_y + 1)/2 >= 0 ? utility.ymin-(utility.size_y + 1)/2 : 0;
		int ymax = utility.ymax + (utility.size_y + 1)/2 <=500 ? utility.ymax + (utility.size_y + 1)/2 : 500;

		for(int y = ymin; y<ymax; ++y)
		{
			for (int x = xmin; x<xmax; ++x)
			{
				int originaladc = 0;
				int adc = 0;
				//					CUPRINTF("\nHEY %d!!!!!!\n", x);
				for (unsigned int clusterIdx = 0; clusterIdx < numClusters; ++clusterIdx)
				{

					int clusterPosX =  pixels[combinationChi2.comb[clusterIdx] / 4].x;
					int clusterPosY =  pixels[combinationChi2.comb[clusterIdx] / 4].y;
					int direction = combinationChi2.comb[clusterIdx] % 4;


					if(clusterIdx == 0)
					{
						originaladc = originalADC[500*y + x];
						adc = originaladc;

					}
					float fact = pixelWeight(mapcharge, count, totalcharge, clusterPosX, clusterPosY, x, y, direction);

					if (fact > 0)
						adc -= fact * utility.expectedADC;

					if(clusterIdx == numClusters-1)
					{
						float res = adc;
						float charge = (originaladc - adc) > 0? (originaladc - adc): -(originaladc - adc) ; // charge assigned to this pixel
						float chargeMeasured = originaladc;



						if (charge < 5000 && chargeMeasured < 5000 )
						{  // threshold effect
							res = 0;
						}

						if (chargeMeasured <= 2000) chargeMeasured = 2000;
						if (charge < 2000) charge = 2000;


						chi2 +=((res * res) / (charge * charge));
						//						CUPRINTF("\n\n\n  chi2 %f = !!!!!\n", chi2);

					}
				}
			}

		}

		prob = prob/numClusters;

		combinationChi2.chi2 = (1024*(chi2/prob))<USHRT_MAX?(1024*(chi2/prob)):USHRT_MAX;
		__syncthreads();
		shared_combinationChi2[tid] = combinationChi2;
		__syncthreads();

		sharedMinReduction(shared_combinationChi2, combinationChi2.comb[2]);
		__syncthreads();

		if(tid==0)
			atomicMinChi2(reducedChi2s,shared_combinationChi2[0]);
	//	CUPRINTF("\n block chi2 = %hd, global chi2 = %hd", shared_combinationChi2[0].chi2, reducedChi2s->chi2);

	}

}

__global__ void kernel5threads(GPUPixelSoA* pixels, int* originalADC, int* mapcharge, int* count,
		int* totalcharge, int numPositions, Chi2Comb* reducedChi2s)
{

	if(blockIdx.x < numPositions && blockIdx.y <= blockIdx.x && blockIdx.z <= blockIdx.y)
	{

//		Chi2Comb *shared_combinationChi2 = SharedMemory<Chi2Comb>();
		for(int comb3 = 0; comb3 <= blockIdx.z; ++comb3)
		{
			if(threadIdx.x <= comb3)
			{
				__shared__ Chi2Comb shared_combinationChi2[64];

				// perform first level of reduction,
				// reading from global memory, writing to shared memory
				unsigned int tid = threadIdx.x;
				Chi2Comb combinationChi2;


				const unsigned int numClusters = 5;


				float chi2=0;
				combinationChi2.chi2 = 0;
				combinationChi2.comb[0] = blockIdx.x;
				combinationChi2.comb[1] = blockIdx.y;
				combinationChi2.comb[2] = blockIdx.z;
				combinationChi2.comb[3] = comb3;
				combinationChi2.comb[4] = threadIdx.x;
				combinationChi2.comb[5] = -1;
				//			CUPRINTF("\n numPositions = %d, comb = %d %d %d %d %d %d", numPositions,combinationChi2.comb[0],combinationChi2.comb[1],combinationChi2.comb[2],combinationChi2.comb[3],combinationChi2.comb[4],combinationChi2.comb[5]);
				float prob = 0.f;

				//				CUPRINTF("\nCOMBINATION %d, %d, %d, %d, %d\n",combinationChi2.comb[0], combinationChi2.comb[1], combinationChi2.comb[2], combinationChi2.comb[3], combinationChi2.comb[4]);

				for (unsigned int clusterIdx = 0; clusterIdx < numClusters; ++clusterIdx)
				{

					int clusterPosX = pixels[combinationChi2.comb[clusterIdx] / 4].x;
//					int clusterPosY = pixels->y[combinationChi2.comb[clusterIdx] / 4];
					int direction = combinationChi2.comb[clusterIdx] % 4;
					prob += count[direction%2+1 + clusterPosX / 32 *utility.BinsDirections +
					              utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections];
				}

				//					CUPRINTF("\n%d, %d, %d\n",clusterPosX, clusterPosY, direction);
				int xmin = utility.xmin-5 >= 0 ? utility.xmin-5 : 0;
				int xmax = utility.xmax+5 <= 500 ? utility.xmax+5 : 500;
				int ymin = utility.ymin-(utility.size_y + 1)/2 >= 0 ? utility.ymin-(utility.size_y + 1)/2 : 0;
				int ymax = utility.ymax + (utility.size_y + 1)/2 <=500 ? utility.ymax + (utility.size_y + 1)/2 : 500;

				for(int y = ymin; y<ymax; ++y)
				{
					for (int x = xmin; x<xmax; ++x)
					{
						int originaladc = 0;
						int adc = 0;
						//					CUPRINTF("\nHEY %d!!!!!!\n", x);
						for (unsigned int clusterIdx = 0; clusterIdx < numClusters; ++clusterIdx)
						{

							int clusterPosX =  pixels[combinationChi2.comb[clusterIdx] / 4].x;
							int clusterPosY =  pixels[combinationChi2.comb[clusterIdx] / 4].y;
							int direction = combinationChi2.comb[clusterIdx] % 4;
							if(clusterIdx == 0)
							{
								originaladc = originalADC[500*y + x];
								adc = originaladc;

							}
							float fact = pixelWeight(mapcharge, count, totalcharge, clusterPosX, clusterPosY, x, y, direction);


							if (fact > 0)
								adc -= fact * utility.expectedADC;

							if(clusterIdx == numClusters-1)
							{
								float res = adc;
								float charge = (originaladc - adc) > 0? (originaladc - adc): -(originaladc - adc) ; // charge assigned to this pixel
								float chargeMeasured = originaladc;



								if (charge < 5000 && chargeMeasured < 5000 )
								{  // threshold effect
									res = 0;
								}

								if (chargeMeasured <= 2000) chargeMeasured = 2000;
								if (charge < 2000) charge = 2000;


								chi2 +=((res * res) / (charge * charge));

							}
						}
					}

				}
				prob = prob/numClusters;

				combinationChi2.chi2 = (1024*(chi2/prob))<USHRT_MAX?(1024*(chi2/prob)):USHRT_MAX;

				shared_combinationChi2[tid] = combinationChi2;
				__syncthreads();

				sharedMinReduction(shared_combinationChi2, combinationChi2.comb[3]);
				__syncthreads();

				if(tid==0)
					atomicMinChi2(reducedChi2s,shared_combinationChi2[0]);
//				CUPRINTF("\n block chi2 = %hd, global chi2 = %hd", shared_combinationChi2[0].chi2, reducedChi2s->chi2);

			}
		}
	}

}


__global__ void kernel6threads(GPUPixelSoA* pixels, int* originalADC, int* mapcharge, int* count,
		int* totalcharge, int numPositions, Chi2Comb* reducedChi2s)
{

	if(blockIdx.x < numPositions && blockIdx.y <= blockIdx.x && blockIdx.z <= blockIdx.y)
	{

//		Chi2Comb *shared_combinationChi2 = SharedMemory<Chi2Comb>();

		for(int comb3 = 0; comb3 <= blockIdx.z; ++comb3)
		{
			for(int comb4 = 0; comb4 <= comb3; ++comb4)
			{

				if(threadIdx.x <= comb4)
				{
					__shared__ Chi2Comb shared_combinationChi2[64];

					// perform first level of reduction,
					// reading from global memory, writing to shared memory
					unsigned int tid = threadIdx.x;
					Chi2Comb combinationChi2;


					const unsigned int numClusters = 6;


					float chi2=0;
					combinationChi2.chi2 = 0;
					combinationChi2.comb[0] = blockIdx.x;
					combinationChi2.comb[1] = blockIdx.y;
					combinationChi2.comb[2] = blockIdx.z;
					combinationChi2.comb[3] = comb3;
					combinationChi2.comb[4] = comb4;
					combinationChi2.comb[5] = threadIdx.x;
					//			CUPRINTF("\n numPositions = %d, comb = %d %d %d %d %d %d", numPositions,combinationChi2.comb[0],combinationChi2.comb[1],combinationChi2.comb[2],combinationChi2.comb[3],combinationChi2.comb[4],combinationChi2.comb[5]);
					float prob = 0.f;

					//					CUPRINTF("\nCOMBINATION %d, %d, %d, %d, %d, %d\n",combinationChi2.comb[0], combinationChi2.comb[1], combinationChi2.comb[2], combinationChi2.comb[3], combinationChi2.comb[4],combinationChi2.comb[5]);

					for (unsigned int clusterIdx = 0; clusterIdx < numClusters; ++clusterIdx)
					{

						int clusterPosX = pixels[combinationChi2.comb[clusterIdx] / 4].x;
//						int clusterPosY = pixels->y[combinationChi2.comb[clusterIdx] / 4];
						int direction = combinationChi2.comb[clusterIdx] % 4;
						prob += count[direction%2+1 + clusterPosX / 32 *utility.BinsDirections +
						              utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections];
					}

					//						CUPRINTF("\n%d, %d, %d\n",clusterPosX, clusterPosY, direction);
					int xmin = utility.xmin-5 >= 0 ? utility.xmin-5 : 0;
					int xmax = utility.xmax+5 <= 500 ? utility.xmax+5 : 500;
					int ymin = utility.ymin-(utility.size_y + 1)/2 >= 0 ? utility.ymin-(utility.size_y + 1)/2 : 0;
					int ymax = utility.ymax + (utility.size_y + 1)/2 <=500 ? utility.ymax + (utility.size_y + 1)/2 : 500;

					for(int y = ymin; y<ymax; ++y)
					{
						for (int x = xmin; x<xmax; ++x)
						{

							int originaladc = 0;
							int adc = 0;
							//					CUPRINTF("\nHEY %d!!!!!!\n", x);
							for (unsigned int clusterIdx = 0; clusterIdx < numClusters; ++clusterIdx)
							{

								int clusterPosX = pixels[combinationChi2.comb[clusterIdx] / 4].x;
								int clusterPosY = pixels[combinationChi2.comb[clusterIdx] / 4].y;
								int direction = combinationChi2.comb[clusterIdx] % 4;
								if(clusterIdx == 0)
								{
									originaladc = originalADC[500*y + x];
									adc = originaladc;

								}
								float fact = pixelWeight(mapcharge, count, totalcharge, clusterPosX, clusterPosY, x, y, direction);

								if (fact > 0)
									adc -= fact * utility.expectedADC;

								if(clusterIdx == numClusters-1)
								{
									float res = adc;
									float charge = (originaladc - adc) > 0? (originaladc - adc): -(originaladc - adc) ; // charge assigned to this pixel
									float chargeMeasured = originaladc;



									if (charge < 5000 && chargeMeasured < 5000 )
									{  // threshold effect
										res = 0;
									}

									if (chargeMeasured <= 2000) chargeMeasured = 2000;
									if (charge < 2000) charge = 2000;


									chi2 +=((res * res) / (charge * charge));

								}
							}
						}

					}


					prob = prob/numClusters;
					combinationChi2.chi2 = (1024*(chi2/prob))<USHRT_MAX?(1024*(chi2/prob)):USHRT_MAX;
					__syncthreads();

					shared_combinationChi2[tid] = combinationChi2;
					__syncthreads();

					sharedMinReduction(shared_combinationChi2, combinationChi2.comb[4]);
					__syncthreads();
					if(tid==0)
						atomicMinChi2(reducedChi2s,shared_combinationChi2[0]);
	//				CUPRINTF("\n block chi2 = %hd, global chi2 = %hd", shared_combinationChi2[0].chi2, reducedChi2s->chi2);


				}
			}
		}
	}

}


//
//__global__ void kernel3(GPUPixelSoA* pixels, int* originalADC, int* mapcharge, int* count,
//		int* totalcharge, int numPositions, Chi2Comb* reducedChi2s)
//{
//
//	if(blockIdx.x < numPositions && blockIdx.y <= blockIdx.x && blockIdx.z <= blockIdx.y)
//	{
//		__shared__ Chi2Comb combinationChi2;
//		const unsigned int numClusters = 3;
//
//		if(blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
//			CUPRINTF("\n blocco %d !!!!!\n",blockIdx.x);
//
//
//		if (threadIdx.x == 0 && threadIdx.y == 0)
//		{
//
//			chi2=0;
//			combinationChi2.chi2 = 0;
//			combinationChi2.comb[0] = blockIdx.x;
//			combinationChi2.comb[1] = blockIdx.y;
//			combinationChi2.comb[2] = blockIdx.z;
//			combinationChi2.comb[3] = -1;
//			combinationChi2.comb[4] = -1;
//			combinationChi2.comb[5] = -1;
//			CUPRINTF("\n numPositions = %d, comb = %d %d %d %d %d %d", numPositions,combinationChi2.comb[0],combinationChi2.comb[1],combinationChi2.comb[2],combinationChi2.comb[3],combinationChi2.comb[4],combinationChi2.comb[5]);
//			prob = 0.f;
//
//		}
//		__syncthreads();
//
//
//
//
//// CREARE KERNEL CHE FACCIA QUESTO CON DYNAMIC PARALLELISM
//		for (unsigned int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++)
//		{
//			__syncthreads();
//			if(threadIdx.x == 0 && threadIdx.y == 0)
//			{
//				clusterPosX = pixels->x[combinationChi2.comb[clusterIdx] / 4];
//				clusterPosY = pixels->y[combinationChi2.comb[clusterIdx] / 4];
//				direction = combinationChi2.comb[clusterIdx] % 4;
//				atomicAdd(&prob, count[direction%2+1 + clusterPosX / 32 *utility.BinsDirections +
//				                       utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections]);
//			}
//			__syncthreads();
//			for(int y = threadIdx.y; (y>= utility.ymin - (utility.size_y + 1)/2) && y>=0
//			&& (y<=utility.ymax + (utility.size_y + 1)/2) && y < 500; y+= blockDim.y)
//			{
//				if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500))
//				{
//					float fact = pixelWeight(mapcharge, count, totalcharge, clusterPosX, clusterPosY, x, y, direction);
//
//					if (fact > 0)
//						adc -= fact * utility.expectedADC;
//
//				}
//			}
//
//		}
//		for(int y = threadIdx.y; (y>= utility.ymin - (utility.size_y +1)/2) && y>=0
//		&& (y<=utility.ymax + (utility.size_y +1)/2) && y < 500; y+= blockDim.y)
//		{
//			if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500)  )
//			{
//				float res = adc;
//				float charge = (originaladc - adc) > 0? (originaladc - adc): -(originaladc - adc) ; // charge assigned to this pixel
//				float chargeMeasured = originaladc;
//
//
//
//				if (charge < 5000 && chargeMeasured < 5000 ) {  // threshold effect
//					res = 0;
//				}
//
//				if (chargeMeasured <= 2000) chargeMeasured = 2000;
//				if (charge < 2000) charge = 2000;
//
//
//
//				atomicAdd(&chi2, ((res * res) / (charge * charge)));
//
//				__syncthreads();
//
//
//			}
//			__syncthreads();
//		}
//
//		if(threadIdx.x == 0 && threadIdx.y == 0)
//		{
//
//			prob = prob/numClusters;
//			combinationChi2.chi2 = chi2/prob;
//			atomicMinChi2(reducedChi2s,combinationChi2 );
//		}
//
//		__syncthreads();
//
//
//	}
//
//}

//
//__global__ void kernel4(GPUPixelSoA* pixels, int* originalADC, int* mapcharge, int* count,
//		int* totalcharge, int numPositions, Chi2Comb* reducedChi2s)
//{
//
//	if(blockIdx.x < numPositions && blockIdx.y <= blockIdx.x && blockIdx.z <= blockIdx.y)
//	{
//
//		const unsigned int numClusters = 4;
//
//		for(int comb3 = 0; comb3 <= blockIdx.z; ++comb3)
//		{
//			__shared__ Chi2Comb combinationChi2;
//			__shared__ float prob;
//			__shared__ int clusterPosX;
//			__shared__ int clusterPosY;
//			__shared__ int direction;
//			__shared__ float chi2;
//			int x = threadIdx.x;
//			// TODO: invece di caricare tutti gli adc (anche quelli zero) converrebbe caricare solo quelli diversi da zero
//			int originaladc = originalADC[blockDim.x*threadIdx.y + threadIdx.x];
//			int adc = originaladc;
//			__syncthreads();
//
//			if (threadIdx.x == 0 && threadIdx.y == 0)
//			{
//				chi2=0;
//				combinationChi2.chi2 = 0;
//				combinationChi2.comb[0] = blockIdx.x;
//				combinationChi2.comb[1] = blockIdx.y;
//				combinationChi2.comb[2] = blockIdx.z;
//				combinationChi2.comb[3] = comb3;
//				prob = 0.f;
//
//				CUPRINTF("\n numPositions = %d, comb = %d %d %d %d %d %d\n", numPositions,combinationChi2.comb[0],combinationChi2.comb[1],combinationChi2.comb[2],combinationChi2.comb[3],combinationChi2.comb[4],combinationChi2.comb[5]);
//
//			}
//			__syncthreads();
//
//
//
//
//
//			for (unsigned int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++)
//			{
//				__syncthreads();
//
//				if(threadIdx.x == 0 && threadIdx.y == 0)
//				{
//
//					clusterPosX = pixels->x[combinationChi2.comb[clusterIdx] / 4];
//					clusterPosY = pixels->y[combinationChi2.comb[clusterIdx] / 4];
//					direction = combinationChi2.comb[clusterIdx] % 4;
//				}
//				__syncthreads();
//				for(int y = threadIdx.y; (y>= utility.ymin - (utility.size_y + 1)/2) && y>=0
//				&& (y<=utility.ymax + (utility.size_y + 1)/2) && y < 500; y+= blockDim.y)
//				{
//					if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500))
//					{
//						float fact = pixelWeight(mapcharge, count, totalcharge, clusterPosX, clusterPosY, x, y, direction);
//
//						if (fact > 0)
//							adc -= fact * utility.expectedADC;
//					}
//				}
//				atomicAdd(&prob, count[direction%2+1 + clusterPosX / 32 *utility.BinsDirections +
//				                       utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections]);
//			}
//			for(int y = threadIdx.y; (y>= utility.ymin - (utility.size_y +1)/2) && y>=0
//			&& (y<=utility.ymax + (utility.size_y +1)/2) && y < 500; y+= blockDim.y)
//			{
//				if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500)  )
//				{
//					float res = adc;
//					float charge = (originaladc - adc) > 0? (originaladc - adc): -(originaladc - adc) ; // charge assigned to this pixel
//					float chargeMeasured = originaladc;
//
//
//
//					if (charge < 5000 && chargeMeasured < 5000 ) {  // threshold effect
//						res = 0;
//					}
//
//					if (chargeMeasured <= 2000) chargeMeasured = 2000;
//					if (charge < 2000) charge = 2000;
//
//
//
//					atomicAdd(&chi2, ((res * res) / (charge * charge)));
//
//					CUPRINTF("chi2 = %f", chi2);
//					__syncthreads();
//
//
//				}
//			}
//			__syncthreads();
//
//			if(threadIdx.x == 0 && threadIdx.y == 0)
//			{
//				prob = prob/numClusters;
//				combinationChi2.chi2 = chi2/prob;
//				atomicMinChi2(reducedChi2s,combinationChi2 );
//			}
//
//			__syncthreads();
//
//
//		}
//	}
//
//}
//
//
//__global__ void kernel5(GPUPixelSoA* pixels, int* originalADC, int* mapcharge, int* count,
//		int* totalcharge, int numPositions, Chi2Comb* reducedChi2s)
//{
//
//	if(blockIdx.x < numPositions && blockIdx.y <= blockIdx.x && blockIdx.z <= blockIdx.y)
//	{
//		const unsigned int numClusters = 5;
//
//		for(int comb3 = 0; comb3 <= blockIdx.z; ++comb3)
//		{
//			for (int comb4 = 0; comb4 <= comb3; ++comb4)
//			{
//				__shared__ Chi2Comb combinationChi2;
//				__shared__ float chi2;
//				__shared__ float prob;
//				__shared__ int clusterPosX;
//				__shared__ int clusterPosY;
//				__shared__ int direction;
//				int x = threadIdx.x;
//				// TODO: invece di caricare tutti gli adc (anche quelli zero) converrebbe caricare solo quelli diversi da zero
//				int originaladc = originalADC[blockDim.x*threadIdx.y + threadIdx.x];
//				int adc = originaladc;
//
//				__syncthreads();
//
//
//
//				if (threadIdx.x == 0 && threadIdx.y == 0)
//				{
//					chi2=0;
//					combinationChi2.chi2 = 0;
//					combinationChi2.comb[0] = blockIdx.x;
//					combinationChi2.comb[1] = blockIdx.y;
//					combinationChi2.comb[2] = blockIdx.z;
//					combinationChi2.comb[3] = comb3;
//					combinationChi2.comb[4] = comb4;
//					combinationChi2.comb[5] = -1;
//					prob = 0.f;
//					//					if(blockIdx.x == 0)
//					//						CUPRINTF("\n blocco %d !!!!!\n",blockIdx.x);
//
//					//			CUPRINTF("\n numPositions = %d, comb = %d %d %d %d %d %d", numPositions,combinationChi2.comb[0],combinationChi2.comb[1],combinationChi2.comb[2],combinationChi2.comb[3],combinationChi2.comb[4],combinationChi2.comb[5]);
//
//				}
//				__syncthreads();
//
//
//
//
//
//				for (unsigned int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++)
//				{
//					__syncthreads();
//
//					if(threadIdx.x == 0 && threadIdx.y == 0)
//					{
//
//						clusterPosX = pixels->x[combinationChi2.comb[clusterIdx] / 4];
//						clusterPosY = pixels->y[combinationChi2.comb[clusterIdx] / 4];
//						direction = combinationChi2.comb[clusterIdx] % 4;
//					}
//					__syncthreads();
//					for(int y = threadIdx.y; (y>= utility.ymin - (utility.size_y + 1)/2) && y>=0
//					&& (y<=utility.ymax + (utility.size_y + 1)/2) && y < 500; y+= blockDim.y)
//					{
//						if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500))
//						{
//							float fact = pixelWeight(mapcharge, count, totalcharge, clusterPosX, clusterPosY, x, y, direction);
//
//							if (fact > 0)
//								adc -= fact * utility.expectedADC;
//						}
//					}
//					atomicAdd(&prob, count[direction%2+1 + clusterPosX / 32 *utility.BinsDirections +
//					                       utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections]);
//				}
//				for(int y = threadIdx.y; (y>= utility.ymin - (utility.size_y +1)/2) && y>=0
//				&& (y<=utility.ymax + (utility.size_y +1)/2) && y < 500; y+= blockDim.y)
//				{
//					if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500)  )
//					{
//						float res = adc;
//						float charge = (originaladc - adc) > 0? (originaladc - adc): -(originaladc - adc) ; // charge assigned to this pixel
//						float chargeMeasured = originaladc;
//
//
//
//						if (charge < 5000 && chargeMeasured < 5000 ) {  // threshold effect
//							res = 0;
//						}
//
//						if (chargeMeasured <= 2000) chargeMeasured = 2000;
//						if (charge < 2000) charge = 2000;
//
//
//
//						atomicAdd(&chi2, ((res * res) / (charge * charge)));
//						CUPRINTF("chi2 = %f", chi2);
//						__syncthreads();
//
//					}
//					__syncthreads();
//				}
//
//				if(threadIdx.x == 0 && threadIdx.y == 0)
//				{
//					prob = prob/numClusters;
//					combinationChi2.chi2 = chi2/prob;
//					atomicMinChi2(reducedChi2s,combinationChi2 );
//				}
//
//				__syncthreads();
//
//
//			}
//		}
//	}
//
//}
//
//__global__ void kernel6(GPUPixelSoA* pixels, int* originalADC, int* mapcharge, int* count,
//		int* totalcharge, int numPositions, Chi2Comb* reducedChi2s)
//{
//
//	if(blockIdx.x < numPositions && blockIdx.y <= blockIdx.x && blockIdx.z <= blockIdx.y)
//	{
//
//		for(int comb3 = 0; comb3 <= blockIdx.z; ++comb3)
//		{
//			for (int comb4 = 0; comb4 <= comb3; ++comb4)
//			{
//				for (int comb5 = 0; comb5 <= comb4; ++comb5)
//				{
//					__shared__ Chi2Comb combinationChi2;
//
//					const unsigned int numClusters = 6;
//
//					__syncthreads();
//					__shared__ float chi2;
//					__shared__ float prob;
//					__shared__ int clusterPosX;
//					__shared__ int clusterPosY;
//					__shared__ int direction;
//					int x = threadIdx.x;
//					// TODO: invece di caricare tutti gli adc (anche quelli zero) converrebbe caricare solo quelli diversi da zero
//					int originaladc = originalADC[blockDim.x*threadIdx.y + threadIdx.x];
//					int adc = originaladc;
//
//					if (threadIdx.x == 0 && threadIdx.y == 0)
//					{
//						chi2=0;
//						combinationChi2.chi2 = 0;
//						combinationChi2.comb[0] = blockIdx.x;
//						combinationChi2.comb[1] = blockIdx.y;
//						combinationChi2.comb[2] = blockIdx.z;
//						combinationChi2.comb[3] = comb3;
//						combinationChi2.comb[4] = comb4;
//						combinationChi2.comb[5] = comb5;
//						prob = 0.f;
//
//						if(blockIdx.x == 0)
//							CUPRINTF("\n blocco %d !!!!!\n",blockIdx.x);
//
//						CUPRINTF("\n numPositions = %d, comb = %d %d %d %d %d %d", numPositions,combinationChi2.comb[0],combinationChi2.comb[1],combinationChi2.comb[2],combinationChi2.comb[3],combinationChi2.comb[4],combinationChi2.comb[5]);
//
//					}
//					__syncthreads();
//
//
//
//
//
//					for (unsigned int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++)
//					{
//						__syncthreads();
//
//						if(threadIdx.x == 0 && threadIdx.y == 0)
//						{
//
//							clusterPosX = pixels->x[combinationChi2.comb[clusterIdx] / 4];
//							clusterPosY = pixels->y[combinationChi2.comb[clusterIdx] / 4];
//							direction = combinationChi2.comb[clusterIdx] % 4;
//						}
//						__syncthreads();
//						for(int y = threadIdx.y; (y>= utility.ymin - (utility.size_y + 1)/2) && y>=0
//						&& (y<=utility.ymax + (utility.size_y + 1)/2) && y < 500; y+= blockDim.y)
//						{
//							if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500))
//							{
//								float fact = pixelWeight(mapcharge, count, totalcharge, clusterPosX, clusterPosY, x, y, direction);
//
//								if (fact > 0)
//									adc -= fact * utility.expectedADC;
//							}
//						}
//						atomicAdd(&prob, count[direction%2+1 + clusterPosX / 32 *utility.BinsDirections +
//						                       utility.binjetZOverRho*utility.BinsXposition*utility.BinsDirections]);
//					}
//					for(int y = threadIdx.y; (y>= utility.ymin - (utility.size_y +1)/2) && y>=0
//					&& (y<=utility.ymax + (utility.size_y +1)/2) && y < 500; y+= blockDim.y)
//					{
//						if((x >= utility.xmin - 5) && (x>=0) && (x<= utility.xmax + 5) && (x<500)  )
//						{
//							float res = adc;
//							float charge = (originaladc - adc) > 0? (originaladc - adc): -(originaladc - adc) ; // charge assigned to this pixel
//							float chargeMeasured = originaladc;
//
//
//
//							if (charge < 5000 && chargeMeasured < 5000 ) {  // threshold effect
//								res = 0;
//							}
//
//							if (chargeMeasured <= 2000) chargeMeasured = 2000;
//							if (charge < 2000) charge = 2000;
//
//
//
//							atomicAdd(&chi2, ((res * res) / (charge * charge)));
//							CUPRINTF("chi2 = %f", chi2);
//							__syncthreads();
//
//						}
//						__syncthreads();
//					}
//					if(threadIdx.x == 0 && threadIdx.y == 0)
//					{
//						prob = prob/numClusters;
//						combinationChi2.chi2 = chi2/prob;
//						atomicMinChi2(reducedChi2s,combinationChi2 );
//					}
//
//					__syncthreads();
//
//				}
//			}
//		}
//	}
//
//}

extern "C" Chi2Comb cudaClusterSplitter_(GPUPixelSoA* pixels, int* originalADC, int* gpu_mapcharge, int* gpu_count_array,
		int* gpu_totalcharge_array, PixelClusterUtils* constantDataNeededOnGPU, unsigned int numClusters, int numPositions, cudaStream_t& CUDAstream) {

	// bind texture to buffer
	//	cudaBindTexture(0, tex, gpu_mapcharge, 168000*sizeof(int));
	cudaMemcpyToSymbol(  utility,  constantDataNeededOnGPU,   sizeof(PixelClusterUtils) );
	Chi2Comb* gpu_bestChi2Combination;
	Chi2Comb* host_bestChi2Combination;
	cudaMallocHost((void**)&host_bestChi2Combination, sizeof(Chi2Comb));
	cudaMalloc((void**)&gpu_bestChi2Combination, sizeof(Chi2Comb));

	host_bestChi2Combination->chi2 = USHRT_MAX;
	for(int i = 0; i<6; ++i)
		host_bestChi2Combination->comb[i] = -1;
	cudaMemcpyAsync(gpu_bestChi2Combination, host_bestChi2Combination,sizeof(Chi2Comb), cudaMemcpyHostToDevice, CUDAstream);


	size_t SmemSize = 2*sizeof(Chi2Comb)*numPositions;
	std::cout << "running on the GPU for " << numClusters << "clusters and " << numPositions << "positions." << std::endl;
	if(numClusters == 3)
	{
		//		float* blocksChi2s;
		//		cudaMalloc((void**)&blocksChi2s, (64<<numClusters)*sizeof(float));

		dim3 block(64,1,1);
		dim3 grid(numPositions,numPositions,1);
		kernel3threads<<<grid,block,SmemSize,CUDAstream>>>(pixels, originalADC, gpu_mapcharge,
				gpu_count_array, gpu_totalcharge_array, numPositions, gpu_bestChi2Combination);
		//		//TODO: evaluate the amount of shared memory needed
		//		int numblocksReduction = 2 << (6*numClusters-10);
		//		minReduce<<<numblocksReduction,2048,2048 * sizeof(float),0>>>(blocksChi2s, reducedChi2, blockDim.x*blockDim.y*blockDim.z);
	}
	else if (numClusters == 4)
	{
		dim3 block(64,1,1);
		dim3 grid(numPositions,numPositions,numPositions);
		kernel4threads<<<grid,block,SmemSize,CUDAstream>>>(pixels, originalADC, gpu_mapcharge,
				gpu_count_array, gpu_totalcharge_array, numPositions, gpu_bestChi2Combination);
	}
	else if (numClusters == 5)
	{
		dim3 block(64,1,1);
		dim3 grid(numPositions,numPositions,numPositions);
		kernel5threads<<<grid,block,SmemSize,CUDAstream>>>(pixels, originalADC, gpu_mapcharge,
				gpu_count_array, gpu_totalcharge_array, numPositions, gpu_bestChi2Combination);
	}
	else if (numClusters == 6)
	{
		dim3 block(64,1,1);
		dim3 grid(numPositions,numPositions,numPositions);
		kernel6threads<<<grid,block,SmemSize,CUDAstream>>>(pixels, originalADC, gpu_mapcharge,
				gpu_count_array, gpu_totalcharge_array, numPositions, gpu_bestChi2Combination);
	}
	cudaMemcpyAsync(host_bestChi2Combination, gpu_bestChi2Combination,sizeof(Chi2Comb), cudaMemcpyDeviceToHost, CUDAstream);
	cudaStreamSynchronize(CUDAstream);
	Chi2Comb bestcomb;
	bestcomb.chi2 =  host_bestChi2Combination->chi2;
	printf("Best comb chi2: %hd \n", host_bestChi2Combination->chi2);

	for(int i = 0; i<numClusters; ++i)
	{
		bestcomb.comb[i] = host_bestChi2Combination->comb[i];
		printf("%hd \n", host_bestChi2Combination->comb[i]);
	}



	cudaFreeHost(host_bestChi2Combination);
	cudaFree(gpu_bestChi2Combination);
	//   cudaUnbindTexture(tex);
	return bestcomb;

}
