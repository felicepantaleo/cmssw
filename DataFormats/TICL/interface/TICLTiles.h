// Authors: Felice Pantaleo - felice.pantaleo@cern.ch, Aurora Perego aurora.perego@cern.ch
// Date: 03/2024

#ifndef DataFormats_TICL_TICLTiles_h
#define DataFormats_TICL_TICLTiles_h

#include "DataFormats/Math/interface/normalizedPhi.h"
#include "DataFormats/Math/interface/constexpr_cmath.h"
#include "DataFormats/TICL/interface/Common.h"
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cassert>


// TODO FP: As soon as C++20 is available, we can use floats as non-type template parameters and write something like
// #include <cmath>
// namespace ticl {
// template <float minDim1, float maxDim1, float minDim2, float maxDim2, float tileSize=0.15f>
// struct TileConstantsEtaPhi {
//   static constexpr int nColumns = std::ceil((maxDim1 - minDim1) / tileSize);
//   static constexpr int nRows = std::ceil(2. * 3.14 / tileSize);
//   static constexpr int nTiles = nColumns * nRows;
// };
// }
//int main() { ticl::TileConstantsEtaPhi<-3.f, 21.f/7.f, -3.14f, 3.14f, 0.2f> tile; }
namespace ticl {
template <typename T, typename objType = int>
class Tiles {
public:
  typedef T type;
  /**
     * @brief fill the tile 
     * 
     * @param[in] dim1 represents x or eta
     * @param[in] dim2 represents y or phi
     * 
    */
  void fill(const float dim1, const float dim2, const objType& obj) {
    auto idx = getGlobalBin(dim1, dim2);
    tiles_[idx].push_back(obj);
  }

  /** 
    * @brief compute bin for dim1 (x or eta)
    * 
    * @param[in] dim for binning
    * @return computed bin
    */
  int getDim1Bin(float dim) const {
    constexpr float dimRange = T::maxDim1 - T::minDim1;
    static_assert(dimRange >= 0.);
    constexpr float r = nColumns / dimRange;
    int dimBin = (dim - T::minDim1) * r;
    dimBin = std::clamp(dimBin, 0, nColumns - 1);
    return dimBin;
  }

  /** 
    * @brief compute bin for dim2 (y or phi)
    * 
    * @param[in] dim for bining
    * @return computed bin
    */
  int getDim2Bin(float dim2) const {
    if constexpr (not T::wrapped) {
      constexpr float dimRange = T::maxDim2 - T::minDim2;
      static_assert(dimRange >= 0.);
      constexpr float r = nRows / dimRange;
      int dimBin = (dim2 - T::minDim2) * r;
      dimBin = std::clamp(dimBin, 0, nRows - 1);
      return dimBin;
    } else {
      auto normPhi = normalizedPhi(dim2);
      constexpr float r = nRows * M_1_PI * 0.5f;
      int phiBin = (normPhi + M_PI) * r;
      return phiBin;
    }
  }

  inline float distance2(float dim1Cell1, float dim2Cell1, float dim1Cell2, float dim2Cell2) const {  // distance squared
    float d1 = dim1Cell1 - dim1Cell2;
    float d2 = dim2Cell1 - dim2Cell2;
    if constexpr (T::wrapped) {
      d2 = reco::deltaPhi(dim2Cell1, dim2Cell2);
    }
    return (d1 * d1 + d2 * d2);
  }
  int getGlobalBin(float dim1, float dim2) const { return getDim1Bin(dim1) + getDim2Bin(dim2) * nColumns; }

  int getGlobalBinByBin(int dim1Bin, int dim2Bin) const { return dim1Bin + dim2Bin * nColumns; }

  std::array<int, 4> getSearchBox(float dim1Min, float dim1Max, float dim2Min, float dim2Max) const {
    if constexpr (T::wrapped) {
      if (dim1Max - dim1Min < 0) {
        return std::array<int, 4>({{0, 0, 0, 0}});
      }
    }
    int dim1BinMin = getDim1Bin(dim1Min); assert(dim1BinMin >= 0);
    int dim1BinMax = getDim1Bin(dim1Max); assert(dim1BinMax >= 0);
    int dim2BinMin = getDim2Bin(dim2Min); assert(dim2BinMin >= 0);
    int dim2BinMax = getDim2Bin(dim2Max); assert(dim2BinMax >= 0);
    if constexpr (T::wrapped) {
      if (dim2BinMax < dim2BinMin) {
        dim2BinMax += nRows;
      }
    }
    return std::array<int, 4>({{dim1BinMin, dim1BinMax, dim2BinMin, dim2BinMax}});
  }


  void clear() {
    for (auto& t : tiles_)
      t.clear();
  }

  const std::vector<objType>& operator[](int globalBinId) const { return tiles_[globalBinId]; }
  static constexpr int nColumns = reco::ceil((T::maxDim1 - T::minDim1) / T::tileSize);
  static constexpr int nRows = reco::ceil((T::maxDim2 - T::minDim2) / T::tileSize);
  static constexpr int nTiles = nColumns * nRows;
private:

  std::array<std::vector<objType>, nTiles> tiles_;


};

}  // namespace ticl

#endif
