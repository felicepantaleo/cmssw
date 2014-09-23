#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cuda.h>
#include "tbb/concurrent_queue.h"



#include "GPUJobDescriptor.h"
#include "SimDataFormats/SLHC/interface/slhcevent.hh"



namespace gpu{

class GPUL1TScheduler
{

public:
	static GPUL1TScheduler * getInstance()
	{
		if(theOnlyInstance == null)
		{
			std::unique_lock<std::mutex> lock(m);
			if(theOnlyInstance == null)
			{
				theOnlyInstance = new GPUL1TScheduler();
			}
		}
		return theOnlyInstance;
	}
// 3 arguments: number of CPU consumer threads, GPU consumer threads, maximum capacity of the Job Queue
bool initScheduler(int, int, int );
bool initJob( GPUJobDescriptor&, std::condition_variable*, std::mutex*, bool*);

void ConsumeGPU(const int);
void ConsumeCPU();

private:

/////////////////////////////////////////////////////////////
// Singleton members
/////////////////////////////////////////////////////////////
GPUL1TScheduler();
~GPUL1TScheduler();
static GPUL1TScheduler * theOnlyInstance;
std::mutex m;
std::mutex mtxGPURoundRobin;
/////////////////////////////////////////////////////////////


//The CPU threads that will be polling the job queue and compute data on GPU or CPU
std::vector<std::thread> GPUThreads;
std::vector<std::thread> CPUThreads;


private:
// The Job queue
	tbb::concurrent_bounded_queue<GPUJobDescriptor> JobQueue;

	int gpus;


};





}
