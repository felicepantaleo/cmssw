#ifndef RecoPixelVertexing_PixelTriplets_GPUHitsAndDoublets_h
#define RecoPixelVertexing_PixelTriplets_GPUHitsAndDoublets_h

#include <cuda_runtime.h>

#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"

struct GPULayerHits
{
	size_t size;
	float * x;
	float * y;
	float * z;
};

struct GPULayerDoublets
{
	size_t size;
	int * indices;
	GPULayerHits layers[2];
};

inline GPULayerHits copy_hits_to_gpu(RecHitsSortedInPhi const & hits)
{
	GPULayerHits d_hits;
	d_hits.size = hits.size();
	auto memsize = d_hits.size * sizeof(float);
	cudaMalloc(&d_hits.x, memsize);
	cudaMalloc(&d_hits.y, memsize);
	cudaMalloc(&d_hits.z, memsize);
	cudaMemcpy(d_hits.x, hits.x.data(), memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hits.y, hits.y.data(), memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hits.z, hits.z.data(), memsize, cudaMemcpyHostToDevice);
	return d_hits;
}

inline
void free_gpu_hits(GPULayerHits & hits)
{
	cudaFree(hits.x);
	hits.x = nullptr;
	cudaFree(hits.y);
	hits.y = nullptr;
	cudaFree(hits.z);
	hits.z = nullptr;
}

inline GPULayerDoublets copy_doublets_to_gpu(HitDoublets const & doublets,
		GPULayerHits const & inner, GPULayerHits const & outer)
{
	GPULayerDoublets d_doublets;
	d_doublets.size = doublets.size();
	d_doublets.layers[0] = inner;
	d_doublets.layers[1] = outer;
	auto memsize = d_doublets.size * sizeof(int) * 2;
	cudaMalloc(&d_doublets.indices, memsize);
	cudaMemcpy(d_doublets.indices, doublets.indices().data(), memsize,
			cudaMemcpyHostToDevice);
//  std::cout << "CPU doublet 0 " << doublets.r(0, HitDoublets::inner) << " " << doublets.r(0, HitDoublets::outer) << " inner, outer " <<doublets.innerHitId(0)
//		  << " " <<doublets.outerHitId(0) << std::endl;
	return d_doublets;
}

inline
void free_gpu_doublets(GPULayerDoublets & doublets)
{
	cudaFree(doublets.indices);
	doublets.indices = nullptr;
}

#endif // not defined RecoPixelVertexing_PixelTriplets_GPUHitsAndDoublets_h
