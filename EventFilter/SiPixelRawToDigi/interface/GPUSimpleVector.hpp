//  author: Felice Pantaleo, CERN, 2018
#ifndef GPU_SIMPLEVECTOR_HPP_
#define GPU_SIMPLEVECTOR_HPP_

namespace GPU {
template <class T> struct SimpleVector {
  // Constructors
  __host__ __device__ SimpleVector(unsigned int maxSize, T *m_data = nullptr)
      : m_size(0), m_data(m_data), maxSize(static_cast<int>(maxSize)) {}

  __host__ __device__ SimpleVector() : SimpleVector(0) {}

  __inline__ __host__ __device__ int push_back(const T &element) {

    auto previousSize = m_size;
    m_size++;
    if (previousSize < maxSize) {
      m_data[previousSize] = element;
      return previousSize;
    } else {
      --m_size;
      return -1;
    }
  }

#if defined(__NVCC__) || defined(__CUDACC__)
  __device__ int push_back_ts(const T &element) {
    auto previousSize = atomicAdd(&m_size, 1);
    if (previousSize < maxSize) {
      m_data[previousSize] = element;
      return previousSize;
    } else {
      atomicSub(&m_size, 1);
      return -1;
    }
  }
#endif

  __inline__ __host__ __device__ T pop_back() {
    if (m_size > 0) {
      auto previousSize = m_size--;
      return m_data[previousSize - 1];
    } else
      return T();
  }

  __inline__ __host__ __device__ T operator[](int i) { return m_data[i]; }

  __inline__ __host__ __device__ T at(int i) {
    if (i < m_size)
      return m_data[i];
    else
      return T();
  }

  __inline__ __host__ __device__ void reset() { m_size = 0; }

  __inline__ __host__ __device__ int size() const { return m_size; }

  __inline__ __host__ __device__ int capacity() const { return maxSize; }



private:
  int m_size;
  int maxSize;

  T *m_data;
};
} // namespace GPU

#endif
