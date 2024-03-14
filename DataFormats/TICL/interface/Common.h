#ifndef DataFormats_TICL_Common_h
#define DataFormats_TICL_Common_h

#include <vector>
#include <array>
#include <cstdint>

namespace ticl {
  struct TileConstantsGlobal_EtaPhi {
  static constexpr float tileSize = 0.15f;
  static constexpr float minDim1 = -3.f;
  static constexpr float maxDim1 = 3.f;
  static constexpr float minDim2 = -M_PI;
  static constexpr float maxDim2 = M_PI;
  static constexpr bool wrapped = true;
  };

  struct TileConstantsEndcapNeg_EtaPhi {
    static constexpr float tileSize = 0.15f;
    static constexpr float minDim1 = -3.f;
    static constexpr float maxDim1 = -1.5f;
    static constexpr float minDim2 = -M_PI;
    static constexpr float maxDim2 = M_PI;
    static constexpr bool wrapped = true;

  };

    struct TileConstantsEndcapPos_EtaPhi {
    static constexpr float tileSize = 0.15f;
    static constexpr float minDim1 = 1.5f;
    static constexpr float maxDim1 = 3.f;
    static constexpr float minDim2 = -M_PI;
    static constexpr float maxDim2 = M_PI;
    static constexpr bool wrapped = true;

  };

  struct TileConstantsBarrel_EtaPhi {
    static constexpr float tileSize = 3*0.087f;
    static constexpr float minDim1 = -1.5f;
    static constexpr float maxDim1 = 1.5f;
    static constexpr float minDim2 = -M_PI;
    static constexpr float maxDim2 = M_PI;
             static constexpr bool wrapped = true;
  };


  struct TileConstantsEndcap_XY {
  static constexpr float tileSize = 5.f;
    static constexpr float minDim1 = -285.f;
    static constexpr float maxDim1 = 285.f;
    static constexpr float minDim2 = -285.f;
    static constexpr float maxDim2 = 285.f;
    static constexpr bool wrapped = false;

  };


}  // namespace ticl


#endif 
