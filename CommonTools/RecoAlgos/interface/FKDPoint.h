#ifndef FKDTREE_KDPOINT_H_
#define FKDTREE_KDPOINT_H_
#include <array>
#include <utility>

template<class TYPE, int numberOfDimensions>
class FKDPoint
{

public:
	FKDPoint() :
			theElements(), theId(0)
	{
	}

	FKDPoint(const FKDPoint<TYPE, numberOfDimensions>& other) :
			theElements(other.theElements),theId(other.theId)
	{

	}

	template<class T>
	FKDPoint<TYPE, numberOfDimensions> & operator=(
			const FKDPoint<TYPE, numberOfDimensions> & other)
	{
		if (this != &other)
		{
			theId = other.theId;
			theElements = other.theElements;
		}
		return *this;

	}

	FKDPoint(TYPE x, TYPE y, unsigned int id=0)
	{
      static_assert(numberOfDimensions==2,"FKDPoint number of arguments does not match the number of dimensions");
      	
		theId = id;
		theElements[0] = x;
		theElements[1] = y;
	}

	FKDPoint(TYPE x, TYPE y, TYPE z, unsigned int id=0)
	{
      static_assert(numberOfDimensions==3,"FKDPoint number of arguments does not match the number of dimensions");
      	
		theId = id;
		theElements[0] = x;
		theElements[1] = y;
		theElements[2] = z;
	}

	FKDPoint(TYPE x, TYPE y, TYPE z, TYPE w, unsigned int id=0)
	{
      static_assert(numberOfDimensions==4,"FKDPoint number of arguments does not match the number of dimensions");
		theId = id;
		theElements[0] = x;
		theElements[1] = y;
		theElements[2] = z;
		theElements[3] = w;
	}

// the user should check that i < numberOfDimensions
	TYPE& operator[](unsigned int const i)
	{
		return theElements[i];
	}

	TYPE const& operator[](unsigned int const i) const
	{
		return theElements[i];
	}

	void setDimension(unsigned int i, const TYPE& value)
	{
		theElements[i] = value;
	}

	void setId(const unsigned int id)
	{
		theId = id;
	}

	unsigned int getId() const
	{
		return theId;
	}


private:
	std::array<TYPE, numberOfDimensions> theElements;
	unsigned int theId;
};



#endif /* FKDTREE_KDPOINT_H_ */