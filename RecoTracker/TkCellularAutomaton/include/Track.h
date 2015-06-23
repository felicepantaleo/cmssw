/*
 * Track.h
 *
 *  Created on: Jun 19, 2015
 *      Author: fpantale
 */

#ifndef INCLUDE_TRACK_H_
#define INCLUDE_TRACK_H_


#include "CUDAQueue.h"


constexpr int c_minHitsPerTrack = 4;

template<int maxHitsNum>
struct Track {

// track constructor should be passed the pointer of the root Cell
	//
	CUDAQueue<maxHitsNum-1, int> Cells;


};



#endif /* INCLUDE_TRACK_H_ */
