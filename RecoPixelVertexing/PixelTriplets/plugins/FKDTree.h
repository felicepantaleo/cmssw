#ifndef FKDTREE_FKDTREE_H_
#define FKDTREE_FKDTREE_H_

#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <utility>
#include <iostream>
#include <cassert>
#include <deque>
#include "FKDPoint.h"
#include "FQueue.h"


#define FLOOR_LOG2(X) ((unsigned int) (31 - __builtin_clz(X | 1)))


template<class TYPE, int numberOfDimensions>
class FKDTree
{

public:

	FKDTree(const unsigned int nPoints)
	{
		theNumberOfPoints = 0;
		theDepth = 0;
		for (auto& x : theDimensions)
			x.reserve(theNumberOfPoints);
		theIntervalLength.reserve(theNumberOfPoints);
		theIntervalMin.reserve(theNumberOfPoints);
		theIds.reserve(theNumberOfPoints);
		thePoints.reserve(theNumberOfPoints);
	}

	void reserve(const unsigned int nPoints)
	{
		for (auto& x : theDimensions)
			x.reserve(nPoints);
		theIntervalLength.reserve(nPoints);
		theIntervalMin.reserve(nPoints);
		theIds.reserve(nPoints);
		thePoints.reserve(nPoints);

	}

	FKDTree(const std::vector<FKDPoint<TYPE, numberOfDimensions> >& points)
	{
		theNumberOfPoints = points.size();
		theDepth = FLOOR_LOG2(theNumberOfPoints);
		for (auto& x : theDimensions)
			x.resize(theNumberOfPoints);
		theIntervalLength.resize(theNumberOfPoints, 0);
		theIntervalMin.resize(theNumberOfPoints, 0);
		theIds.resize(theNumberOfPoints, 0);
		thePoints = points;

	}

	FKDTree()
	{
		theNumberOfPoints = 0;
		theDepth = 0;
		for (auto& x : theDimensions)
			x.clear();
		theIntervalLength.clear();
		theIntervalMin.clear();
		theIds.clear();
		thePoints.clear();
		theIndecesToVisit.clear();
	}

	FKDTree(const FKDTree<TYPE, numberOfDimensions>& other)
	{
		theNumberOfPoints(other.theNumberOfPoints);
		theDepth(other.theDepth);

		theIntervalLength.clear();
		theIntervalMin.clear();
		theIds.clear();
		thePoints.clear();
		for (auto& x : theDimensions)
			x.clear();

		theIntervalLength = other.theIntervalLength;
		theIntervalMin = other.theIntervalMin;
		theIds = other.theIds;

		thePoints = other.thePoints;
		for (int i = 0; i < numberOfDimensions; ++i)
			theDimensions = other.theDimensions;

	}


	FKDTree(FKDTree<TYPE, numberOfDimensions> && other)
	{
		theNumberOfPoints(std::move(other.theNumberOfPoints));
		theDepth(std::move(other.theDepth));

		theIntervalLength.clear();
		theIntervalMin.clear();
		theIds.clear();
		thePoints.clear();
		for (auto& x : theDimensions)
			x.clear();

		theIntervalLength = std::move(other.theIntervalLength);
		theIntervalMin = std::move(other.theIntervalMin);
		theIds = std::move(other.theIds);

		thePoints = std::move(other.thePoints);
		for (int i = 0; i < numberOfDimensions; ++i)
			theDimensions = std::move(other.theDimensions);
	}

	FKDTree<TYPE, numberOfDimensions>& operator=(
			FKDTree<TYPE, numberOfDimensions> && other)
	{

		if (this != &other)
		{
			theNumberOfPoints(std::move(other.theNumberOfPoints));
			theDepth(std::move(other.theDepth));

			theIntervalLength.clear();
			theIntervalMin.clear();
			theIds.clear();
			thePoints.clear();
			for (auto& x : theDimensions)
				x.clear();

			theIntervalLength = std::move(other.theIntervalLength);
			theIntervalMin = std::move(other.theIntervalMin);
			theIds = std::move(other.theIds);

			thePoints = std::move(other.thePoints);
			for (int i = 0; i < numberOfDimensions; ++i)
				theDimensions = std::move(other.theDimensions);
		}
		return *this;

	}

	void resize(unsigned int nPoints)
	{
		theNumberOfPoints = nPoints;
		theDepth = FLOOR_LOG2(nPoints);
		for (auto& x : theDimensions)
			x.resize(theNumberOfPoints);
		theIntervalLength.resize(theNumberOfPoints, 0);
		theIntervalMin.resize(theNumberOfPoints, 0);
		theIds.resize(theNumberOfPoints);
		thePoints.reserve(theNumberOfPoints);
	}

	unsigned int size()
	{
		return theNumberOfPoints;
	}

	void clear()
	{
		theIndecesToVisit.clear();
		theNumberOfPoints = 0;
		theDepth = 0;
		for (auto& x : theDimensions)
			x.clear();
		theIntervalLength.clear();
		theIntervalMin.clear();
		theIds.clear();
		thePoints.clear();
	}

	bool empty()
	{
		return theNumberOfPoints == 0;
	}

	void push_back(const FKDPoint<TYPE, numberOfDimensions>& point)
	{

		theNumberOfPoints++;
		thePoints.push_back(point);
		for (int i = 0; i < numberOfDimensions; ++i)
			theDimensions.at(i).push_back(point[i]);
		theIds.push_back(point.getId());

	}

	void push_back(FKDPoint<TYPE, numberOfDimensions> && point)
	{
		theNumberOfPoints++;
		thePoints.push_back(point);
		for (int i = 0; i < numberOfDimensions; ++i)
			theDimensions.at(i).push_back(point[i]);
		theIds.push_back(point.getId());
	}

	void add_at_position(const FKDPoint<TYPE, numberOfDimensions>& point,
			const unsigned int position)
	{
		for (int i = 0; i < numberOfDimensions; ++i)
			theDimensions[i][position] = point[i];
		theIds[position] = point.getId();

	}

	void add_at_position(FKDPoint<TYPE, numberOfDimensions> && point,
			const unsigned int position)
	{
		for (int i = 0; i < numberOfDimensions; ++i)
			theDimensions[i][position] = point[i];
		theIds[position] = point.getId();

	}

	FKDPoint<TYPE, numberOfDimensions> getPoint(unsigned int index) const
	{

		FKDPoint<TYPE, numberOfDimensions> point;

		for (int i = 0; i < numberOfDimensions; ++i)
			point.setDimension(i, theDimensions[i][index]);

		point.setId(theIds[index]);

		return point;
	}

	void search_in_the_box(const FKDPoint<TYPE, numberOfDimensions>& minPoint,
			const FKDPoint<TYPE, numberOfDimensions>& maxPoint,
			std::vector<unsigned int>& foundPoints)
	{
		theIndecesToVisit.clear();
		theIndecesToVisit.push_back(0);
		for (int depth = 0; depth < theDepth + 1; ++depth)
		{

			int dimension = depth % numberOfDimensions;
			unsigned int numberOfIndecesToVisitThisDepth =
					theIndecesToVisit.size();
			for (unsigned int visitedIndecesThisDepth = 0;
					visitedIndecesThisDepth < numberOfIndecesToVisitThisDepth;
					visitedIndecesThisDepth++)
			{

				unsigned int index = theIndecesToVisit[visitedIndecesThisDepth];
				bool intersection = intersects(index, minPoint, maxPoint,
						dimension);

				if (intersection && is_in_the_box(index, minPoint, maxPoint))
					foundPoints.emplace_back(theIds.at(index));

				bool isLowerThanBoxMin = theDimensions[dimension][index]
						< minPoint[dimension];

				int startSon = isLowerThanBoxMin; //left son = 0, right son =1

				int endSon = isLowerThanBoxMin || intersection;

				for (int whichSon = startSon; whichSon < endSon + 1; ++whichSon)
				{
					unsigned int indexToAdd = leftSonIndex(index) + whichSon;

					if (indexToAdd < theNumberOfPoints)
					{
						theIndecesToVisit.push_back(indexToAdd);
					}

				}

			}

			theIndecesToVisit.pop_front(numberOfIndecesToVisitThisDepth);
		}
	}

	void search_in_the_box_branchless(
			const FKDPoint<TYPE, numberOfDimensions>& minPoint,
			const FKDPoint<TYPE, numberOfDimensions>& maxPoint, std::vector<unsigned int>& foundPoints)
	{
		theIndecesToVisit.clear();
		theIndecesToVisit.push_back(0);
		unsigned int index;
		bool intersection;
		unsigned int dimension;
		int numberOfIndecesToVisitThisDepth;
		int maxNumberOfSonsToVisitNext;
		int numberOfSonsToVisitNext;
		unsigned int firstSonToVisitNext;
		for (unsigned int depth = 0; depth < theDepth + 1; ++depth)
		{

			dimension = depth % numberOfDimensions;
			numberOfIndecesToVisitThisDepth = theIndecesToVisit.size();
			for (int visitedIndecesThisDepth = 0;
					visitedIndecesThisDepth < numberOfIndecesToVisitThisDepth;
					visitedIndecesThisDepth++)
			{

				index = theIndecesToVisit[visitedIndecesThisDepth];
				intersection = intersects(index, minPoint, maxPoint, dimension);
				firstSonToVisitNext = leftSonIndex(index);
				maxNumberOfSonsToVisitNext = (firstSonToVisitNext
						< theNumberOfPoints)
						+ ((firstSonToVisitNext + 1) < theNumberOfPoints);

				if (intersection)
				{
					if (is_in_the_box(index, minPoint, maxPoint))
					{
						foundPoints.emplace_back(theIds[index]);
					}
					numberOfSonsToVisitNext = maxNumberOfSonsToVisitNext;
				}
				else
				{

					numberOfSonsToVisitNext = std::min(
							maxNumberOfSonsToVisitNext, 1);
					firstSonToVisitNext += (theDimensions[dimension][index]
							< minPoint[dimension]);
				}

				for ( int whichSon = 0; whichSon < numberOfSonsToVisitNext;
						++whichSon)
					theIndecesToVisit.push_back(firstSonToVisitNext + whichSon);
			}

			theIndecesToVisit.pop_front(numberOfIndecesToVisitThisDepth);
		}
	}
	void search_in_the_box_recursive(const FKDPoint<TYPE, numberOfDimensions>& minPoint,
			const FKDPoint<TYPE, numberOfDimensions>& maxPoint, std::vector<unsigned int>& foundPoints,unsigned int index=0, int dimension=0) const
	{

		unsigned int firstSonToVisitNext = leftSonIndex(index);
		int maxNumberOfSonsToVisitNext = (firstSonToVisitNext < theNumberOfPoints) + ((firstSonToVisitNext+1) < theNumberOfPoints);
		bool intersection = intersects(index, minPoint, maxPoint,
				dimension);
		int numberOfSonsToVisitNext;
		if (intersection)
		{
			if(is_in_the_box(index, minPoint, maxPoint))
			{
				foundPoints.emplace_back(theIds.at(index));
			}
			numberOfSonsToVisitNext = maxNumberOfSonsToVisitNext;
		}
		else
		{
			bool isLowerThanBoxMin = theDimensions[dimension][index]
					< minPoint[dimension];
			numberOfSonsToVisitNext = isLowerThanBoxMin && (maxNumberOfSonsToVisitNext==1)? 0: std::min(maxNumberOfSonsToVisitNext,1);
			firstSonToVisitNext += isLowerThanBoxMin;
		}

		if(numberOfSonsToVisitNext != 0)

		{
		auto nextDimension = (dimension+1) % numberOfDimensions;
		for(int whichSon = 0; whichSon < numberOfSonsToVisitNext; ++whichSon)
			search_in_the_box_recursive(minPoint, maxPoint, foundPoints,firstSonToVisitNext+whichSon,nextDimension);
		}

	}

	bool test_correct_build(unsigned int index = 0, int dimension = 0) const
	{

		unsigned int leftSonIndexInArray = 2 * index + 1;
		unsigned int rightSonIndexInArray = leftSonIndexInArray + 1;
		if (rightSonIndexInArray >= theNumberOfPoints
				&& leftSonIndexInArray >= theNumberOfPoints)
		{
			return true;
		}
		else
		{
			if (leftSonIndexInArray < theNumberOfPoints)
			{
				if (theDimensions[dimension][index]
						>= theDimensions[dimension][leftSonIndexInArray])
				{
					test_correct_build(leftSonIndexInArray,
							(dimension + 1) % numberOfDimensions);

				}
				else
					return false;
			}

			if (rightSonIndexInArray < theNumberOfPoints)
			{
				if (theDimensions[dimension][index]
						<= theDimensions[dimension][rightSonIndexInArray])
				{
					test_correct_build(rightSonIndexInArray,
							(dimension + 1) % numberOfDimensions);

				}
				else
					return false;
			}

		}
		return true;

	}

	bool test_correct_search(const std::vector<unsigned int>& foundPoints,
			const FKDPoint<TYPE, numberOfDimensions>& minPoint,
			const FKDPoint<TYPE, numberOfDimensions>& maxPoint) const
	{
		bool testGood = true;
		for (unsigned int i = 0; i < theNumberOfPoints; ++i)
		{

			bool shouldBeInTheBox = true;
			for (int dim = 0; dim < numberOfDimensions; ++dim)
			{
				shouldBeInTheBox &= (thePoints[i][dim] <= maxPoint[dim]
						&& thePoints[i][dim] >= minPoint[dim]);
			}

			bool foundToBeInTheBox = std::find(foundPoints.begin(),
					foundPoints.end(), thePoints[i].getId())
					!= foundPoints.end();

			if (foundToBeInTheBox == shouldBeInTheBox)
			{

				testGood &= true;
			}
			else
			{
				if (foundToBeInTheBox)
					std::cerr << "Point " << thePoints[i].getId()
							<< " was wrongly found to be in the box."
							<< std::endl;
				else
					std::cerr << "Point " << thePoints[i].getId()
							<< " was wrongly found to be outside the box."
							<< std::endl;

				std::cerr << "Point " <<std::endl;
				thePoints[i].print();
				std::cerr << "minPoint " <<std::endl;
				minPoint.print();
				std::cerr << "maxPoint " <<std::endl;
				maxPoint.print();
				testGood &= false;

			}
		}

		if (testGood)
			std::cout << "Search correctness test completed successfully."
					<< std::endl;
		return testGood;
	}

	std::vector<TYPE> getDimensionVector(const int dimension) const
	{
		if (dimension < numberOfDimensions)
			return theDimensions[dimension];
	}

	std::vector<unsigned int> getIdVector() const
	{
		return theIds;
	}
	void build()
	{
		theDepth = FLOOR_LOG2(theNumberOfPoints);
		for (auto& x : theDimensions)
			x.resize(theNumberOfPoints);
		theIntervalLength.resize(theNumberOfPoints, 0);
		theIntervalMin.resize(theNumberOfPoints, 0);
		theIds.resize(theNumberOfPoints);
		thePoints.reserve(theNumberOfPoints);

		//gather kdtree building
		int dimension;
		theIntervalMin[0] = 0;
		theIntervalLength[0] = theNumberOfPoints;

		for (unsigned int depth = 0; depth < theDepth; ++depth)
		{

			dimension = depth % numberOfDimensions;
			unsigned int firstIndexInDepth = (1 << depth) - 1;
			for (unsigned int indexInDepth = 0; indexInDepth < (unsigned int)(1 << depth);
					++indexInDepth)
			{
				unsigned int indexInArray = firstIndexInDepth + indexInDepth;
				unsigned int leftSonIndexInArray = 2 * indexInArray + 1;
				unsigned int rightSonIndexInArray = leftSonIndexInArray + 1;

				unsigned int whichElementInInterval = partition_complete_kdtree(
						theIntervalLength[indexInArray]);
				std::nth_element(
						thePoints.begin() + theIntervalMin[indexInArray],
						thePoints.begin() + theIntervalMin[indexInArray]
								+ whichElementInInterval,
						thePoints.begin() + theIntervalMin[indexInArray]
								+ theIntervalLength[indexInArray],
						[dimension](const FKDPoint<TYPE,numberOfDimensions> & a, const FKDPoint<TYPE,numberOfDimensions> & b) -> bool
						{
							if(a[dimension] == b[dimension])
							return a.getId() < b.getId();
							else
							return a[dimension] < b[dimension];
						});
				add_at_position(
						thePoints[theIntervalMin[indexInArray]
								+ whichElementInInterval], indexInArray);

				if (leftSonIndexInArray < theNumberOfPoints)
				{
					theIntervalMin[leftSonIndexInArray] =
							theIntervalMin[indexInArray];
					theIntervalLength[leftSonIndexInArray] =
							whichElementInInterval;
				}

				if (rightSonIndexInArray < theNumberOfPoints)
				{
					theIntervalMin[rightSonIndexInArray] =
							theIntervalMin[indexInArray]
									+ whichElementInInterval + 1;
					theIntervalLength[rightSonIndexInArray] =
							(theIntervalLength[indexInArray] - 1
									- whichElementInInterval);
				}
			}
		}

		dimension = theDepth % numberOfDimensions;
		unsigned int firstIndexInDepth = (1 << theDepth) - 1;
		for (unsigned int indexInArray = firstIndexInDepth;
				indexInArray < theNumberOfPoints; ++indexInArray)
		{
			add_at_position(thePoints[theIntervalMin[indexInArray]],
					indexInArray);

		}

	}

private:

	unsigned int partition_complete_kdtree(unsigned int length)
	{
		if (length == 1)
			return 0;
		unsigned int index = 1 << (FLOOR_LOG2(length));

		if ((index / 2) - 1 <= length - index)
			return index - 1;
		else
			return length - index / 2;

	}

	unsigned int leftSonIndex(unsigned int index) const
	{
		return 2 * index + 1;
	}

	unsigned int rightSonIndex(unsigned int index) const
	{
		return 2 * index + 2;
	}

	bool intersects(unsigned int index,
			const FKDPoint<TYPE, numberOfDimensions>& minPoint,
			const FKDPoint<TYPE, numberOfDimensions>& maxPoint,
			int dimension) const
	{
		return (theDimensions[dimension][index] <= maxPoint[dimension]
				&& theDimensions[dimension][index] >= minPoint[dimension]);
	}

	bool is_in_the_box(unsigned int index,
			const FKDPoint<TYPE, numberOfDimensions>& minPoint,
			const FKDPoint<TYPE, numberOfDimensions>& maxPoint) const
	{

		for (int i = 0; i < numberOfDimensions; ++i)
		{
			if (!(theDimensions[i][index] <= maxPoint[i]
					&& theDimensions[i][index] >= minPoint[i]))
				return false;
		}

		return true;
	}

	unsigned int theNumberOfPoints;
	unsigned int theDepth;
	std::vector<FKDPoint<TYPE, numberOfDimensions> > thePoints;
	std::array<std::vector<TYPE>, numberOfDimensions> theDimensions;
	FQueue<unsigned int> theIndecesToVisit;
	std::vector<unsigned int> theIntervalLength;
	std::vector<unsigned int> theIntervalMin;
	std::vector<unsigned int> theIds;

};

#endif /* FKDTREE_FKDTREE_H_ */
