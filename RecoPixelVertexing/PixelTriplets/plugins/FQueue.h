#ifndef FKDTREE_QUEUE_H_
#define FKDTREE_QUEUE_H_

#include <vector>
template<class T>
class FQueue
{
public:
	FQueue()
	{
		theSize = 0;
		theFront = 0;
		theTail = 0;
	}

	FQueue(unsigned int initialCapacity)
	{
		theBuffer.resize(initialCapacity);
		theSize = 0;
		theFront = 0;
		theTail = 0;
	}

	FQueue(const FQueue<T> & v)
	{
		theSize = v.theSize;
		theBuffer = v.theBuffer;
		theFront = v.theFront;
		theTail = v.theTail;
	}

	FQueue(FQueue<T> && other) :
			theSize(0), theFront(0), theTail(0)
	{
		theBuffer.clear();
		theSize = other.theSize;
		theFront = other.theFront;
		theTail = other.theTail;
		theBuffer = other.theBuffer;
		other.theSize = 0;
		other.theFront = 0;
		other.theTail = 0;
	}

	FQueue<T>& operator=(FQueue<T> && other)
	{

		if (this != &other)
		{
			theBuffer.clear();
			theSize = other.theSize;
			theFront = other.theFront;
			theTail = other.theTail;
			theBuffer = other.theBuffer;
			other.theSize = 0;
			other.theFront = 0;
			other.theTail = 0;
		}
		return *this;

	}

	FQueue<T> & operator=(const FQueue<T>& v)
	{
		if (this != &v)
		{
			theBuffer.clear();
			theSize = v.theSize;
			theBuffer = v.theBuffer;
			theFront = v.theFront;
			theTail = v.theTail;
		}
		return *this;

	}
	~FQueue()
	{

	}

	unsigned int capacity() const
	{
		return theBuffer.capacity();
	}
	unsigned int size() const
	{
		return theSize;
	}
	bool empty() const;
	T & front()
	{

		return theBuffer[theFront];

	}
	T & tail()
	{
		return theBuffer[theTail];
	}
	void push_back(const T & value)
	{
		if (theSize >= theBuffer.size())
		{
			auto oldCapacity = theBuffer.size();
			theBuffer.reserve(oldCapacity + theTail);

			if (theFront != 0)
			{
				for (unsigned int i = 0; i < theTail; ++i)
				{
					theBuffer.push_back(theBuffer[i]);
				}
				theTail = 0;

			}
			else
			{
				theBuffer.resize(oldCapacity + 16);
				theTail += oldCapacity;
			}

		}

		theBuffer[theTail] = value;
		theTail = (theTail + 1) % theBuffer.size();
		theSize++;

	}
	void pop_front()
	{
		if (theSize > 0)
		{
			theFront = (theFront + 1) % theBuffer.size();
			theSize--;
		}
	}

	void pop_front(const unsigned int numberOfElementsToPop)
	{
		unsigned int elementsToErase =
				theSize > numberOfElementsToPop ?
						numberOfElementsToPop : theSize;
		theSize -= elementsToErase;
		theFront = (theFront + elementsToErase) % theBuffer.size();
	}

	void reserve(unsigned int capacity)
	{
		theBuffer.reserve(capacity);
	}
	void resize(unsigned int capacity)
	{
		theBuffer.resize(capacity);

	}

	T & operator[](unsigned int index)
	{
		return theBuffer[(theFront + index) % theBuffer.size()];
	}

	void clear()
	{
		theSize = 0;
		theFront = 0;
		theTail = 0;
	}
private:
	unsigned int theSize;
	unsigned int theFront;
	unsigned int theTail;
	std::vector<T> theBuffer;

};

#endif
