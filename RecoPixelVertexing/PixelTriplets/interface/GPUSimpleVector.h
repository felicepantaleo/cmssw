#ifndef GPU_SIMPLEVECTOR_H_
#define GPU_SIMPLEVECTOR_H_

template< int maxSize, class T>
struct GPUSimpleVector
{
	__inline__ __device__
	int push_back(const T& element) {

		auto previousSize = m_size++;
		if(previousSize<maxSize)
		{
			m_data[previousSize] = element;
			return previousSize;
		} else
			return -1;
	};

	__inline__ __device__
	T pop_back() {
		if(m_size > 0)
		{
			auto previousSize = m_size--;
			return m_data[previousSize-1];
		} else
                        return T();
	};

	__inline__ __host__ __device__
	void reset()
	{
		m_size = 0;
	};


	__inline__ __host__ __device__
	int size() const
	{
		return m_size;
	};


	T m_data[maxSize];
	int m_size;


};



#endif
