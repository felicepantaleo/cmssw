#ifndef RecoLocalCalo_HGCalRecAlgos_interface_HGCalTilesConstants_h
#define RecoLocalCalo_HGCalRecAlgos_interface_HGCalTilesConstants_h

#include <cstdint>
#include <array>


namespace hgcalTilesConstants {

  enum subdet { CEE=0, CEH=1};


constexpr int32_t ceil(float num)
{
    return (static_cast<float>(static_cast<int32_t>(num)) == num)
        ? static_cast<int32_t>(num)
        : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
}

  // first is for CE-E, second for CE-H in cm
  constexpr std::array<float, 2> minX = {{-165.f, -265.f}}; 
  constexpr std::array<float, 2> maxX = {{ 165.f,  265.f}}; 
  constexpr std::array<float, 2> minY = {{-165.f, -265.f}};
  constexpr std::array<float, 2> maxY = {{ 165.f,  265.f}}; 
  constexpr float tileSize = 5.f;
  constexpr std::array<int, 2> nColumns = {{hgcalTilesConstants::ceil(maxX[0]-minX[0]/tileSize),
                                              hgcalTilesConstants::ceil(maxX[1]-minX[1]/tileSize)}};
  constexpr std::array<int, 2> nRows = {{   hgcalTilesConstants::ceil(maxY[0]-minY[0]/tileSize),
                                              hgcalTilesConstants::ceil(maxY[1]-minY[1]/tileSize)}};

}

#endif // RecoLocalCalo_HGCalRecAlgos_interface_HGCalTilesConstants_h