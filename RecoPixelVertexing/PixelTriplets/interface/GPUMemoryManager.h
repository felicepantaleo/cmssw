#ifndef GPU_MEMORYMANAGER_H_
#define GPU_MEMORYMANAGER_H_
#include <iostream>
#include <cuda_runtime.h>
#include <array>


struct cudaBuffer
{
        std::size_t allocatedSize;
        std::size_t usedSize;
        unsigned char * data;
};


class GPUMemoryManager
{
    public:
        enum location { host=0, device=1};

        GPUMemoryManager()
        {

        }

        void allocate(std::size_t bytes, location l)
        {

            if(l == host)
            {
                cudaMallocHost(&buffer[l].data, bytes);
                buffer[l].allocatedSize = bytes;
                buffer[l].usedSize = 0;
            }
            else if(l == device)
            {
                cudaMalloc(&buffer[l].data, bytes);
                buffer[l].allocatedSize = bytes;
                buffer[l].usedSize = 0;
            }


        }

        unsigned char* requestMemory(unsigned int bytes, location l)
        {
            assert((buffer[l].usedSize + bytes <= buffer[l].allocatedSize) && "GPUMemoryManager: Requesting more memory than the amount preallocated.");
            buffer[l].usedSize += bytes;
            return (buffer[l].data + buffer[l].usedSize);
        }

        std::size_t memoryUsage(location l) const
        {
            return buffer[l].usedSize;
        }


        void freeMemory( location l)
        {
            if(l == host)
             {
                 cudaFreeHost(&buffer[l].data);

             }
             else if(l == device)
             {
                 cudaFree(&buffer[l].data);

             }
        }


    private:

        std::array<cudaBuffer,2> buffer;



};


#endif
