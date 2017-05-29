#ifndef RecoPixelVertexing_PixelTriplets_GPUHitsAndDoublets_h
#define RecoPixelVertexing_PixelTriplets_GPUHitsAndDoublets_h

#include <cuda_runtime.h>

#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/CAGraph.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUMemoryManager.h"

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
		const std::vector<const HitDoublets *>& host_doublets, const CAGraph& graph,
		std::vector<GPULayerHits>& gpu_hitsOnLayer,
		std::vector<GPULayerDoublets>& gpu_doublets, GPUMemoryManager& gpuMem)
{
//    std::cout << "copying hits and doublets for "<< graph.theLayerPairs.size() << " " << host_doublets.size() << " layer pairs and "<< graph.theLayers.size() << " " << host_hitsOnLayer.size()<< " layers "<<  std::endl;
    gpu_doublets.clear();
    gpu_hitsOnLayer.clear();
	for (std::size_t i = 0; i < graph.theLayerPairs.size(); ++i)
	{
	    GPULayerDoublets tmpDoublets;

		tmpDoublets.size = host_doublets[i]->size();
		auto & currentLayerPairRef = graph.theLayerPairs[i];
		tmpDoublets.innerLayerId = currentLayerPairRef.theLayers[0];
		tmpDoublets.outerLayerId = currentLayerPairRef.theLayers[1];
		auto memsize = tmpDoublets.size * sizeof(int) * 2;
//        std::cout << "copying doublets "<< host_doublets[i]->size() << " layerpair " << i << " inner and outer layers " << tmpDoublets.innerLayerId
//                << " " << tmpDoublets.outerLayerId  << std::endl;

		tmpDoublets.indices = (int*)(gpuMem.requestMemory( memsize, GPUMemoryManager::device) );
//        for(unsigned int j = 0; j< host_doublets[i]->indeces.size(); j++)
//        {
//            std::cout << "\t hits "<< j << " " << host_doublets[i]->indeces[j].first << " " << host_doublets[i]->indeces[j].second << std::endl;
//
//        }
		cudaMemcpy(tmpDoublets.indices, host_doublets[i]->indeces.data(), memsize,
					cudaMemcpyHostToDevice);
		gpu_doublets.push_back(tmpDoublets);



	}
//    std::cout << "===================START CPU ====================" << std::endl;
//    int numLayerPairs = graph.theLayerPairs.size();
//    printf("CPU: numberOfLayerPairs: %d\n", numLayerPairs);
//
//    for(unsigned int layerPairIndex = 0; layerPairIndex < graph.theLayerPairs.size(); ++layerPairIndex)
//    {
//        int numberOfDoublets = host_doublets[layerPairIndex]->size();
//        auto & currentLayerPairRef = graph.theLayerPairs[layerPairIndex];
//        printf("\t numberOfDoublets: %d \n", numberOfDoublets);
//        printf("\t innerLayer: %d outerLayer: %d \n",currentLayerPairRef.theLayers[0], currentLayerPairRef.theLayers[1]);
//
//
//        for(unsigned int cellIndexInLayerPair = 0;cellIndexInLayerPair< host_doublets[layerPairIndex]->size();++cellIndexInLayerPair )
//        {
//            printf("\t \t innerHit: %d outerHit: %d \n",host_doublets[layerPairIndex]->innerHitId(cellIndexInLayerPair), host_doublets[layerPairIndex]->outerHitId(cellIndexInLayerPair));
//
//
//        }
//
//
//    }
//
//
//    std::cout << "===================END CPU ====================" << std::endl;








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

//inline
//void free_gpu_doublets(GPULayerDoublets & doublets)
//{
//	cudaFree(doublets.indices);
//	doublets.indices = nullptr;
//}

#endif // not defined RecoPixelVertexing_PixelTriplets_GPUHitsAndDoublets_h
