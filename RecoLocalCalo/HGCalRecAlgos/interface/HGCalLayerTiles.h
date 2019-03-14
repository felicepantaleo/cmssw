// Authors: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 03/2019
// Copyright CERN

#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalLayerTiles
#define RecoLocalCalo_HGCalRecAlgos_HGCalLayerTiles


#include <vector>
#include <array>
#include "HGCalTilesConstants.h"


template<hgcalTilesConstants::subdet subdet>
class HGCalLayerTiles {
    public:

        int getXBin(float x) const {
        constexpr float xRange = hgcalTilesConstants::maxX[subdet] - hgcalTilesConstants::minX[subdet];
        static_assert(xRange>=0.);
        constexpr float r = hgcalTilesConstants::nColumns[subdet]/xRange;
        int xBin = (x - hgcalTilesConstants::minX[subdet])*r;
        xBin = std::min(xBin,hgcalTilesConstants::nColumns[subdet]);
        xBin = std::max(xBin,0);
        return xBin;
    }
        int getYBin(float y) const {
        constexpr float yRange = hgcalTilesConstants::maxY[subdet] - hgcalTilesConstants::minY[subdet];
        static_assert(yRange>=0.);
        constexpr float r = hgcalTilesConstants::nRows[subdet]/yRange;
        int yBin = (y - hgcalTilesConstants::minY[subdet])*r;
        yBin = std::min(yBin,hgcalTilesConstants::nRows[subdet]);
        yBin = std::max(yBin,0);
        return yBin;
    }
    //        
    // int globalBin(int etaBin, int phiBin) const {
    //     return phiBin + etaBin*nPhiBins_;
    // }

    // void clearHistogram()
    // {
    //     auto nBins = nEtaBins_*nPhiBins_;
    //     for(int i = 0; i < nLayers_; ++i)
    //     {
    //         for(int j = 0; j< nBins; ++j)  histogram_[i][j].clear();
    //     }
    // }

    private:
        std::vector< std::vector<int> > tiles_;
        
};






#endif