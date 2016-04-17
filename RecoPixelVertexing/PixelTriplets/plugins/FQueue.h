/*
 * Queue.h
 *
 *  Created on: Mar 1, 2016
 *      Author: fpantale
 */

#ifndef FKDTREE_QUEUE_H_
#define FKDTREE_QUEUE_H_

template<class T>

class FQueue
{
public:
	FQueue()
	{

		theSize = 0;
		theFront = 0;
		theTail = 0;
		theCapacity = theBuffer.size();

	}

	FQueue(const FQueue<T> & v)
	{
		theSize = v.theSize;
		theBuffer = v.theBuffer;
		theFront = v.theFront;
		theTail = v.theTail;
		theCapacity = v.theCapacity;
	}

	FQueue(FQueue<T> && other) :
			theSize(0), theFront(0), theTail(0)
	{

		theCapacity = other.theCapacity;
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

	unsigned int size() const
	{
		return theSize;
	}
	bool empty() const
	{
		return theSize == 0;
	}
	T & front()
	{

		return theBuffer[theFront];

	}
	T & tail()
	{
		return theBuffer[theTail];
	}

  constexpr unsigned int wrapIndex(unsigned int i) {
    return i & (theBuffer.size()-1);
  }

	void push_back(const T & value)
	{
		assert(theSize < theBuffer.size());
		theBuffer[theTail] = value;
		theTail = wrapIndex(theTail + 1);

		theSize++;


	}

	T pop_front()
	{

		if (theSize > 0)
		{
			T element = theBuffer[theFront];
			theFront = wrapIndex(theFront + 1);
			theSize--;


			return element;
		}
	}

	void pop_front(const unsigned int numberOfElementsToPop)
	{
		unsigned int elementsToErase =
				theSize > numberOfElementsToPop ?
						numberOfElementsToPop : theSize;
		theSize -= elementsToErase;
		theFront = wrapIndex(theFront + elementsToErase);
	}


	T & operator[](unsigned int index)
	{
		return theBuffer[wrapIndex(theFront + index)];
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
	//the size should always be a power of 2 to get good performance from not using the % operator
    std::array<T, 256> theBuffer;
	unsigned int theCapacity;

};

#endif
