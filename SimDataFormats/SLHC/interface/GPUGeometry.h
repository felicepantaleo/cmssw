#pragma once
#include <algorithm>
#include <cstddef>
#include <memory>
#include <thread>
#include <mutex>
#include <chrono>
#include <vector>


#include "L1TStub.hh"
#include "SimDataFormats/SLHC/interface/slhcevent.hh"


namespace gpu {
// Questo ci servira' per lo scheduler
//cudaError_t err;
//int gpus;
//
//err = cudaGetDeviceCount(&gpus);
//if (err != cudaSuccess) {
//    fprintf(stderr, "Error getting number of devices in process %u\n", pid);
//    exit(-1);
//}
//
//// Move to the right device
//err = cudaSetDevice(job->gpuId % gpus);
//
//if (err != cudaSuccess) {
//    fprintf(stderr, "Error setting device %u in process %u\n", job->gpuId % gpus, pid);
//    exit(-1);
//}
//
//err = cudaMalloc((void **)&d_A0, job->size * sizeof(float));
//if (err != cudaSuccess) {
//    fprintf(stderr, "Error allocating device memory for volume 1 in process %u\n", pid);
//    exit(-1);
//}
//err = cudaMalloc((void **)&d_Anext, job->size * sizeof(float));
//if (err != cudaSuccess) {
//    fprintf(stderr, "Error allocating device memory for volume 2 in process %u\n", pid);
//    exit(-1);
//}


typedef struct {
	double rmin;
	double rmax;
	double zmax;

} Barrel;

typedef struct {
	double zmin;
	double zmax;

} Disk;

// TP geometry
typedef struct {
	Barrel barrels[6];
	Disk frontDisk[5];
	Disk rearDisk[5];
} Sector;


typedef struct{
	Sector sectors[24];
} GPUGeometry;

typedef struct{
	int32_t simtrackid[32];
	uint32_t iphi[32];
	uint32_t iz[32];
	uint32_t layer[32];
	uint32_t ladder[32];
	uint32_t module[32];
	double x[32];
	double y[32];
	double z[32];
	double sigmax[32];
	double sigmaz[32];
	double pt[32];
	int32_t nStubsInElement;
} StubsSoAElement;


typedef struct {
	int32_t  simtrackid[8];
	uint32_t iphi[8];
	uint32_t iz[8];
	uint32_t layer[8];
	uint32_t ladder[8];
	uint32_t module[8];
	double   x[8];
	double   y[8];
	double   z[8];
	double   sigmax[8];
	double   sigmaz[8];
	double   pt[8];
	// check if the tracklet has a stub in a layer
	bool     StubInLayer[8];
	int      nStubs;
	double   rinv;
	double   phi0;
	double   z0;
	double   t;

} GPUL1TTracklet;

typedef struct {
	GPUL1TTracklet seed;
	double rinv;
	double phi0;
	double z0;
	double t;
	bool isCombinatorics;
	int SimTrackID;
	double rinvfit;
	double phi0fit;
	double z0fit;
	double tfit;

	int irinvfit;
	int iphi0fit;
	int iz0fit;
	int itfit;

	double chisq1;
	double chisq2;

	int ichisq1;
	int ichisq2;

	double D[4][40];

	double M[4][8];

	double MinvDt[4][40];
} GPUL1TTrack;


constexpr inline
bool pushStubInElement (StubsSoAElement* Element, const L1TStub& Stub)
{
	if(Element->nStubsInElement < 32)
	{
		Element->nStubsInElement++;
		Element->simtrackid[Element.nStubsInElement] = Stub.simtrackid();
		Element->iphi[Element.nStubsInElement]       = Stub.iphi();
		Element->iz[Element.nStubsInElement]         = Stub.iz();
		Element->layer[Element.nStubsInElement]      = Stub.layer();
		Element->ladder[Element.nStubsInElement]     = Stub.ladder();
		Element->module[Element.nStubsInElement]     = Stub.module();
		Element->x[Element.nStubsInElement]          = Stub.x();
		Element->y[Element.nStubsInElement]          = Stub.y();
		Element->z[Element.nStubsInElement]          = Stub.z();
		Element->sigmax[Element.nStubsInElement]     = Stub.sigmax();
		Element->sigmaz[Element.nStubsInElement]     = Stub.sigmaz();
		Element->pt[Element.nStubsInElement]         = Stub.pt();

		return true;
	}
	else return false;
}

class StubsSoAVector{
public:
	typedef StubsSoAElement* iterator;
	typedef const StubsSoAElement* const_iterator;
	typedef StubsSoAElement value_type;
	typedef StubsSoAElement& reference;
	typedef const StubsSoAElement& const_reference;
	StubsSoAVector() {create();}
	//	explicit StubsSoAVector(size_t n, const StubsSoAElement& t = StubsSoAElement()) { create(n,t); }
	//	StubsSoAVector(const StubsSoAVector& v) { create(v.begin(), v.end()); }
	explicit StubsSoAVector(size_t n, int layer) { create(n, layer); }
	~StubsSoAVector() { uncreate(); }

	//da modificare
	StubsSoAElement& operator[](size_t i) { return data[i]; }
	const StubsSoAElement& operator[](size_t i) const { return data[i]; }

	size_t size() const { return numElements_; }  // changed
	site_t sizeBytes() const {return numElements_*sizeof(StubsSoAElement)}
	size_t getNumStubs() const { return numStubs_; }  // changed

	inline bool push_back(const L1TStub& stub)
	{
		std::unique_lock<std::mutex> lock (mtx);
		if(numStubs_ == maxNumStubs_ )
			return false;
		else
		{
			while(!pushStubInElement(avail, stub))
			{
				if(numElements_ < maxNumElements_)
				{
					avail++;
					numElements_++;
					avail->nStubsInElement = 0;
				}
				else
				{
					return false;
				}
			}
			numStubs_++;
		}
	}

	//	inline bool push_back_event(const SLHCEvent* ev)
	//	{
	//		std::unique_lock<std::mutex> lock (mtx);
	//		if(ev->nstubs() > maxNumStubs_ )
	//		{
	//			fprintf(stderr, "Error: number of stubs in event greater than maximum number of stubs\n");
	//			return false;
	//		}
	//		else
	//		{
	//			numStubs_ = ev->nstubs();
	//			numElements_ = 1 + (numStubs_-1)/ 32;
	//			for (int i = 0; i< numElements_-1 ; ++i)
	//			{
	//				data[i]->nStubsInElement = 32;
	//
	//				for (int j = 0; j < 32; ++j)
	//				{
	//					auto index = j + i*32;
	//					data[i]->simtrackid[j] = ev->stub(index).simtrackid();
	//					data[i]->iphi[j]       = ev->stub(index).iphi();
	//					data[i]->iz[j]         = ev->stub(index).iz();
	//					data[i]->layer[j]      = ev->stub(index).layer();
	//					data[i]->ladder[j]     = ev->stub(index).ladder();
	//					data[i]->module[j]     = ev->stub(index).module();
	//					data[i]->x[j]          = ev->stub(index).x();
	//					data[i]->y[j]          = ev->stub(index).y();
	//					data[i]->z[j]          = ev->stub(index).z();
	//					data[i]->sigmax[j]     = ev->stub(index).sigmax();
	//					data[i]->sigmaz[j]     = ev->stub(index).sigmaz();
	//					data[i]->pt[j]         = ev->stub(index).pt();
	//				}
	//			}
	//			//fill the last element
	//			for ( int j = 0; j < (numStubs_ % 32); ++j)
	//			{
	//				auto index = j + (numElements_-1)*32;
	//				data[numElements_]->nStubsInElement = numStubs_%32;
	//				data[numElements_]->simtrackid[j] = ev->stub(index).simtrackid();
	//				data[numElements_]->iphi[j]       = ev->stub(index).iphi();
	//				data[numElements_]->iz[j]         = ev->stub(index).iz();
	//				data[numElements_]->layer[j]      = ev->stub(index).layer();
	//				data[numElements_]->ladder[j]     = ev->stub(index).ladder();
	//				data[numElements_]->module[j]     = ev->stub(index).module();
	//				data[numElements_]->x[j]          = ev->stub(index).x();
	//				data[numElements_]->y[j]          = ev->stub(index).y();
	//				data[numElements_]->z[j]          = ev->stub(index).z();
	//				data[numElements_]->sigmax[j]     = ev->stub(index).sigmax();
	//				data[numElements_]->sigmaz[j]     = ev->stub(index).sigmaz();
	//				data[numElements_]->pt[j]         = ev->stub(index).pt();
	//			}
	//
	//		}
	//	}



	iterator begin() { return data; }
	const_iterator begin() const { return data; }
	iterator end() { return avail; }                 // changed
	const_iterator end() const { return avail; }     // changed
	void clear() { uncreate(); }
	bool empty() const { return numStubs_ == 0; }
	void reset ();
	iterator beginGPU() { return data_gpu ; }
	inline	cudaError_t copyHtoDAsync(const cudaStream_t& streamGPU)	{ return cudaMemcpyAsync(data_gpu, data, sizeBytes(), cudaMemcpyHostToDevice, streamGPU );}

private:
	iterator data;	// first element in the Vector
	iterator avail;	// (one past) the last element in the Vec
	iterator limit;	// (one past) the allocated memory
	iterator data_gpu;
	int layer_;

	int numElements_;
	int numStubs_;

	int maxNumElements_;
	int maxNumStubs_;

	std::mutex mtx;

	void create(size_t, int);
	void uncreate();

};

void StubsSoAVector::create(size_t n, int layer)
{
	layer_          = layer;
	data            = nullptr;
	data_gpu        = nullptr;
	maxNumStubs_    = n;
	maxNumElements_ = 1 + (n-1)/32;
	numElements_    = 0;
	numStubs_       = 0;
	CudaSafeCall( cudaMallocHost( (void**)&data , sizeof(StubsSoAElement)*maxNumElements_));
	CudaSafeCall( cudaMalloc((void**)&data_gpu, sizeof(StubsSoAElement)*maxNumElements_));

}

void StubsSoAVector::uncreate()
{
	if(data)
	{
		cudaFreeHost(data);
		cudaFree(data_gpu);

		data            = 0;
		data_gpu        = 0;
		numElements_    = 0;
		numStubs_       = 0;
		maxNumStubs_    = 0;
		maxNumElements_ = 0;

	}
}
inline
void StubsSoAVector::reset()
{
	numElements_ = 0;
	numStubs_    = 0;

}


}
//
//
//void Stream_Queue::uncreate()
//{
//	if(data)
//	{
//		cudaFreeHost(data);
//		cudaFree(data_gpu);
//
//		data = front = rear = 0;
//		front_gpu = rear_gpu = data_gpu = 0;
//		count = 0;
//		max_size = 0;
//		gpu_count = 0;
//	}
//	if(r_data)
//	{
//		cudaFreeHost(r_data);
//		cudaFree(r_data_gpu);
//		cudaFree(utility);
//		r_data_gpu = 0;
//		utility = 0;
//		r_front_gpu = r_rear_gpu = r_data = r_front = r_rear = 0;
//		r_count=0;
//	}
//	close(fd);
//}
//
//
//
//Element* Stream_Queue::pop(cudaStream_t& stream_id)
//{
//	pthread_mutex_lock(&mtx);
////	clock_gettime(CLOCK_REALTIME, &timer);
////	timer.tv_nsec +=50000;
//	int rc = 0;
//	while(count == 0 && rc == 0)
//    {
//
//			rc = pthread_cond_wait(&element_available_on_cpu, &mtx);
//		 //rc = pthread_cond_timedwait(&element_available_on_cpu, &mtx, &timer);
//    }
//    --count;
//	void* front_lock = (void*)front;
//    ++front;
//    if(front == data + max_size)
//		front = data;
//    pthread_mutex_unlock(&mtx);
//	pthread_mutex_lock(&gpu_mtx);
//
//	++gpu_count; //to be put after the copy if used to check how much data there is on the GPU
//	//std::cout << "Packet ready to be sent to the GPU " << std::endl;
//	if(gpu_count + NSTREAMS == max_size)
//		std::cerr << "The GPU is backpressuring, try increasing the buffer size" << std::endl;
//	Element* gpu_ptr = (Element*)((void*)rear_gpu);
//	rear_gpu = (unsigned char*)rear_gpu + pitch;
//    if(rear_gpu == (unsigned char*)data_gpu + max_size*pitch)
//    	rear_gpu = data_gpu;
//    pthread_mutex_unlock(&gpu_mtx);
//
//
//
//	CudaSafeCall(cudaMemcpyAsync((void*)gpu_ptr, (void*)front_lock, sizeof(Element),cudaMemcpyHostToDevice, stream_id));
//	CudaSafeCall(cudaStreamSynchronize(stream_id));
//
//
//    return gpu_ptr;
//
//}
//
