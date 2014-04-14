#pragma once

// no need to use SoAs as I am not parallelizing the adc evaluation
///////////////////////////////////////////////////////////////////////////////
struct GPUPixelSoA
{
	int x[16];
	int y[16];
	int adc[16];
	int n;
};

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
