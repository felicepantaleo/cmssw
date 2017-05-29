#ifndef GPU_MEMORYMANAGER_H_
#define GPU_MEMORYMANAGER_H_
#include <iostream>
#include <cuda_runtime.h>
#include <array>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


struct cudaBuffer
{
        std::size_t allocatedSize;
        std::size_t usedSize;
        void * data;
};


class GPUMemoryManager
{
    public:
        enum location { host=0, device=1};

        GPUMemoryManager()
        {
            for(auto& b: buffer)
            {
                b.data = 0;
            }

        }

        void allocate(std::size_t bytes, location l)
        {
            if(l == host)
            {
                std::cout << "allocating " << bytes << " bytes on the host" << std::endl;

                gpuErrchk(cudaMallocHost(&(buffer[l].data), bytes));
                buffer[l].allocatedSize = bytes;
                buffer[l].usedSize = 0;
            }
            else if(l == device)
            {
                std::cout << "allocating " << bytes << " bytes on the device" << std::endl;

                gpuErrchk(cudaMalloc(&(buffer[l].data), bytes));
                buffer[l].allocatedSize = bytes;
                buffer[l].usedSize = 0;
            }


        }

        void* requestMemory(std::size_t bytes, location l)
        {
//            std::cout << "requesting " << bytes << " bytes on " << l << std::endl;

            std::size_t aligned512bytes = bytes + (512 - (bytes & 511 ));
//            std::cout << "assigning" << aligned256bytes << " bytes on " << l << std::endl;

            assert((buffer[l].usedSize + aligned512bytes <= buffer[l].allocatedSize) && "GPUMemoryManager: Requesting more memory than the amount preallocated.");
            buffer[l].usedSize += aligned512bytes;
//            std::cout << "memory usage on " << l << " " << memoryUsage(l)  << std::endl;
            return ((unsigned char*)buffer[l].data + buffer[l].usedSize);
        }

        std::size_t memoryUsage(location l) const
        {
            return buffer[l].usedSize;
        }


        void freeMemory( location l)
        {
            std::cout << "freeing " << buffer[l].allocatedSize << " bytes from " << l  << std::endl;

            if(l == host)
             {
                gpuErrchk(cudaFreeHost(buffer[l].data));

             }
             else if(l == device)
             {
                 gpuErrchk(cudaFree(buffer[l].data));

             }
        }


    private:

        std::array<cudaBuffer,2> buffer;



};


#endif
