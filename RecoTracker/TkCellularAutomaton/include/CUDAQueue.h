/*
 * CUDAQueue.h
 *
 *  Created on: Jun 17, 2015
 *      Author: fpantale
 */

#ifndef INCLUDE_CUDAQUEUE_H_
#define INCLUDE_CUDAQUEUE_H_

// CUDAQueue is a single-block queue.
// One may want to use it as a __shared__ struct, and have multiple threads
// pushing data into it.
template< int maxSize, class T>
struct CUDAQueue
{
	CUDAQueue(int N=0 ) : m_size(N){ }

	__inline__ __device__
	bool push(const T& element) {

		auto previousSize = atomicAdd (&m_size, 1);
		if(previousSize<maxSize)
		{
			m_data[previousSize] = element;
			return true;
		} else
			return false;
	};

	__inline__ __device__
	T pop_back() {
		if(m_size > 0)
		{
			auto previousSize = atomicAdd (&m_size, -1);
			return m_data[previousSize];
		}

	};


	__inline__ __device__
	bool insertArray(T* array, int numElements)
	{
		if(numElements + m_size <= maxSize && numElements < blockDim.x && threadIdx.x < numElements)
		{
			auto previousSize = atomicAdd(&m_size,1);
			m_data[previousSize+threadIdx.x] = array[threadIdx.x];
			return true;
		} else
			return false;
	}

	T m_data[maxSize];
	int m_size;
};



#endif /* INCLUDE_CUDAQUEUE_H_ */
