#pragma once

// no need to use SoAs as I am not parallelizing the adc evaluation
///////////////////////////////////////////////////////////////////////////////
//typedef struct
//{
//	int x[16];
//	int y[16];
//	int adc[16];
//	int n;
//} GPUPixelSoA;

typedef struct
{
	int x;
	int y;
	int adc;
} GPUPixelSoA;

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

struct __attribute__((__packed__)) Chi2Comb {
	unsigned short int chi2;
	int8_t comb[6];
};
