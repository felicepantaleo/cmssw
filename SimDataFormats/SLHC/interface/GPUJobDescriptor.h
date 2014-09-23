#pragma once
#include <mutex>
#include <condition_variable>
#include "SimDataFormats/SLHC/interface/slhcevent.hh"
#include "SimDataFormats/SLHC/interface/GPUGeometry.h"

namespace gpu {

enum RunningDevice { CPU, GPU } ;

struct GPUL1TIntermediate
{
	GPUL1TTracklet Tracklet[144][160];
	GPUL1TTrack Track[24][100];
	double phiWindowSF;
	double cutrz[16];
	double cutrphi[16];
};


struct GPUJobDescriptor {
	RunningDevice device;
	int gpuId;
	SLHCevent* inputData;
	// array of pointers on the GPU
	StubsSoAElement* StubsGPU[144];
	// size of each array of elements
	size_t numberOfElements[144];
	// barrel LUTs

	//todo fill these LUTs
	// NEED to build the following LUT
	//
	//findMatches(L[0],phiWindowSF_,0.04,0.5);
	//findMatches(L[1],phiWindowSF_,0.025,3.0);
	//findMatches(L[2],phiWindowSF_,0.075,0.5);
	//
	//findMatches(L[3],phiWindowSF_,0.075,3.0);
	//findMatches(L[4],phiWindowSF_,0.1,3.0);
	//findMatches(L[5],phiWindowSF_,0.15,3.0);
	//
	//
	GPUL1TIntermediate* intermediateData;

	double cutrphi[6];
	double cutrz[6];
	double phiWindowSF;
	char* outputData;
	std::condition_variable* cv;
	std::mutex* mtx;

	bool ready;


};



}
