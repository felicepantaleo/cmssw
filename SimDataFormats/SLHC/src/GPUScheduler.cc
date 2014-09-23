#include "GPUScheduler.h"
#include "GPUGeometry.h"
using namespace gpu;
void GPUL1TScheduler::init(int queueCapacity)
{

		JobQueue.set_capacity(queueCapacity);
		err = cudaGetDeviceCount(&gpus);
		if (err != cudaSuccess) {
			fprintf(stderr, "Error getting number of devices in process %u\n", pid);
			exit(-1);
			}

		// TODO waiting for an implementation of the CPU consumer
//		CPUThreads.push_back(std::thread(ConsumeCPU));

		for(auto i : ngpus)
		{
			for (int j = 0; j < 3; ++j)
				GPUThreads.push_back(std::thread(ConsumeGPU, i));
		}

}


// TODO to be implemented
void GPUL1TScheduler::ConsumeCPU()
{
	GPUJobDescriptor job;
	while(true)
	{

		JobQueue.pop(&job);


	}
}


void GPUL1TScheduler::ConsumeGPU(const int gpuId)
{

	cudaError_t err;

	//// Move to the right device
	err = cudaSetDevice(gpuId);

	if (err != cudaSuccess) {
		fprintf(stderr, "Error setting device %u\n", gpuId);
		exit(-1);
	}

	cudaStream_t GPUstream;
	cudaStreamCreate ( &GPUstream );

	// creating one StubsSoAVector per sector per layer with maximum size

	// for each sector there are numberOfLayers layers.
	auto numberOfSectors = 24;

	// for the moment I'll consider only the barrel (6 layers)
	auto numberOfLayers = 6;
	StubsSoAVector StubsVector[numberOfSectors*numberOfLayers];
	for (int i = 0; i< numberOfSectors*numberOfLayers; ++i)
		StubsVector[i].create(100, i);

	GPUJobDescriptor* job, jobGPU;
	GPUL1TIntermediate* gpuPersistentData;
	cudaMallocHost((void**)&job, sizeof(GPUJobDescriptor));
	cudaMalloc((void**)&jobGPU, sizeof(GPUJobDescriptor));
	cudaMalloc((void**)&gpuPersistentData, sizeof(GPUL1Tintermediate));
	const int lookupLayer[10] = { 0, 1, 2, 2, 3, 3, 4, 4, 5, 5 };

	job->intermediateData = gpuPersistentData;

	while(true)
	{
		JobQueue.pop(job);
	    job->device = 1;
	  	job->gpuId = gpuId;
		SLHCEvent* event = (SLHCEvent*)(job.inputData);
		//this is the geometry of the barrel
		//	      L[0]=new L1TBarrel(20.0,30.0,125.0,NSector);
		//	      L[1]=new L1TBarrel(30.0,40.0,125.0,NSector);
		//	      L[2]=new L1TBarrel(40.0,60.0,125.0,NSector);
		//	      L[3]=new L1TBarrel(60.0,80.0,125.0,NSector);
		//	      L[4]=new L1TBarrel(80.0,100.0,125.0,NSector);
		//	      L[5]=new L1TBarrel(100.0,120.0,125.0,NSector);

		//Now we need to add each stub in event in the correct layer of the correct sector

		//		This is how a stub is usually added to a layer in a specific sector
		//		  bool addStub(const L1TStub& aStub){
		//		    if (aStub.r()<rmin_||aStub.r()>rmax_||fabs(aStub.z())>zmax_) return false;
		//		    double phi=aStub.phi();
		//		    if (phi<0) phi+=two_pi;
		//		    int nSector=NSector_*phi/two_pi;
		//		    assert(nSector>=0);
		//		    assert(nSector<NSector_);
		//
		//		    L1TStub tmp=aStub;
		//		    tmp.lorentzcor(-40.0/10000.0);
		//
		//		    stubs_[nSector].push_back(tmp);
		//		    return true;
		//		  }
		auto numberOfStubs = event->nstubs();
	    for (auto j = 0; j < numberOfStubs; ++j)
	    {
	    	auto phi=event->stub(j).phi();

	    	if (phi<0)
	    		phi+=two_pi;
	    	int sectorId=(int)(phi*(numberOfSectors/two_pi));

	    	if(event->stub(j).z() > 125.f)
	    	{
	    		int layerKey = (int)((event->stub(j).r() - 20.f )/ 10);
	    		int layerId = lookupLayer[layerKey];
	    		auto index = sectorId*numberOfLayers + layerId;
	    		StubsVector[index].push_back(event->stub(j));

	    	}


	    }

	    /////////////////////////////////////////////////////
	    // fill the missing parts of the Job descriptor
	    // and move the stubs to the GPU memory
	    /////////////////////////////////////////////////////
	    for (auto j = 0; j < numberOfSectors*numberOfLayers; ++j)
	    {
	    	job->StubsGPU[j] = StubsVector[j].beginGPU();
	    	job->numberOfElements[j] = StubsVector[j].size();
	    	StubsVector[j].copyHtoDAsync(GPUstream);


	    }
	    cudaMemcpyAsync(jobGPU, job, sizeof(*job), cudaMemcpyHostToDevice, GPUstream);


	}


}
