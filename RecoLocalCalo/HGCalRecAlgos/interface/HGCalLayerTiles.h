// Authors: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 03/2019
// Copyright CERN

#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalLayerTiles
#define RecoLocalCalo_HGCalRecAlgos_HGCalLayerTiles


#include <vector>
#include <array>
#include "HGCalTilesConstants.h"

class HGCalLayerTiles {
    public:
       




    //         int getEtaBin(float eta) const {
    //     constexpr float etaRange = ticlConstants::maxEta - ticlConstants::minEta;
    //     static_assert(etaRange>=0.f);
    //     float r = nEtaBins_/etaRange;
    //     int etaBin = (eta - ticlConstants::minEta)*r;
    //     etaBin = std::min(etaBin,nEtaBins_);
    //     etaBin = std::max(etaBin,0);
    //     return etaBin;
    // }

    // int getPhiBin(float phi) const {
    //     auto normPhi = normalizedPhi(phi);
    //     float r = nPhiBins_*M_1_PI*0.5f;
    //     int phiBin = (normPhi + M_PI)*r;
        
    //     return phiBin;
    // }

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