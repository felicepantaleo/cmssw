#pragma once

// no need to use SoAs as I am not parallelizing the adc evaluation
///////////////////////////////////////////////////////////////////////////////
struct GPUPixelSoA
{
	uint16_t x[64];
	uint16_t y[64];
	uint16_t adc[64];
}

struct PixelClusterUtils
{
	int size_y;
	int binjetZOverRho;
	int xmin;
	int xmax;
	int ymin;
	int ymax;
	int BinsXposition;
	int BinsDirections;
	int BinsX;
	int BinsY;

	int BinsJetOverRho ;
	float jetZOverRhoWidth ;
	float expectedADC;

};
