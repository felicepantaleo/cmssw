#ifndef RecoPixelVertexing_PixelTriplets_GPUHitsAndDoublets_h
#define RecoPixelVertexing_PixelTriplets_GPUHitsAndDoublets_h

#include <cuda_runtime.h>

#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/CAGraph.h"

struct GPULayerHits
{
	int layerId;
	size_t size;
	float * x;
	float * y;
	float * z;
};

struct GPULayerDoublets
{
	size_t size;
	int innerLayerId;
	int outerLayerId;
	int * indices;
};

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

inline void copy_hits_and_doublets_to_gpu(
		const std::vector<const RecHitsSortedInPhi *>& host_hitsOnLayer,
		const std::vector<HitDoublets>& host_doublets, const CAGraph& graph,
		std::vector<GPULayerHits>& gpu_hitsOnLayer,
		std::vector<GPULayerDoublets>& gpu_doublets	)
{
	GPULayerDoublets tmpDoublets;
	for (std::size_t i = 0; i < graph.theLayerPairs.size(); ++i)
	{
		tmpDoublets.size = host_doublets[i].size();
		auto & currentLayerPairRef = graph.theLayerPairs[i];
		tmpDoublets.innerLayerId = currentLayerPairRef.theLayers[0];
		tmpDoublets.outerLayerId = currentLayerPairRef.theLayers[1];
		auto memsize = tmpDoublets.size * sizeof(int) * 2;
		cudaMalloc(&tmpDoublets.indices, memsize);
		cudaMemcpy(tmpDoublets.indices, host_doublets[i].indices().data(), memsize,
					cudaMemcpyHostToDevice);
		gpu_doublets.push_back(tmpDoublets);

	}

	GPULayerHits tmpHits;

	for (std::size_t i = 0; i < graph.theLayers.size(); ++i)
	{
		tmpHits.layerId = i;
		tmpHits.size = host_hitsOnLayer[i]->size();

		auto memsize = tmpHits.size * sizeof(float);
		cudaMalloc(&tmpHits.x, memsize);
		cudaMalloc(&tmpHits.y, memsize);
		cudaMalloc(&tmpHits.z, memsize);
		cudaMemcpy(tmpHits.x, host_hitsOnLayer[i]->x.data(), memsize, cudaMemcpyHostToDevice);
		cudaMemcpy(tmpHits.y, host_hitsOnLayer[i]->y.data(), memsize, cudaMemcpyHostToDevice);
		cudaMemcpy(tmpHits.z, host_hitsOnLayer[i]->z.data(), memsize, cudaMemcpyHostToDevice);

		gpu_hitsOnLayer.push_back(tmpHits);

	}

}

inline
void free_gpu_doublets(GPULayerDoublets & doublets)
{
	cudaFree(doublets.indices);
	doublets.indices = nullptr;
}

#endif // not defined RecoPixelVertexing_PixelTriplets_GPUHitsAndDoublets_h
