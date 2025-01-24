#ifndef DataFormats_ClusterSoA_CaloClusterSoA_h
#define DataFormats_ClusterSoA_CaloClusterSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"


enum AlgoId {
      island = 0,
      hybrid = 1,
      fixedMatrix = 2,
      dynamicHybrid = 3,
      multi5x5 = 4,
      particleFlow = 5,
      hgcal_em = 6,
      hgcal_had = 7,
      hgcal_scintillator = 8,
      hfnose = 9,
      undefined = 1000
    };



GENERATE_SOA_LAYOUT(Position4D_Energy_SoALayout,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_COLUMN(float, raw_energy),
                    SOA_COLUMN(float, corrected_energy),
                    SOA_COLUMN(float, time)
)

using Position4D_Energy_SoA = Position4D_Energy_SoALayout<>;
using Position4D_Energy_SoAView = Position4D_Energy_SoA::View;
using Position4D_Energy_SoAConstView = Position4D_Energy_SoA::ConstView;


// This layout holds the minimal set of "common" fields
GENERATE_SOA_LAYOUT(Position4D_Energy_Errors_SoALayout,
                    SOA_COLUMN(float, xErr),  // error on x, if relevant
                    SOA_COLUMN(float, yErr),
                    SOA_COLUMN(float, zErr),
                    SOA_COLUMN(float, timeErr),
                    SOA_COLUMN(float, energyErr)
)



using Position4D_Energy_Errors_SoA = Position4D_Energy_Errors_SoALayout<>;
using Position4D_Energy_Errors_SoAView = Position4D_Energy_Errors_SoA::View;
using Position4D_Energy_Errors_SoAConstView = Position4D_Energy_Errors_SoA::ConstView;



GENERATE_SOA_LAYOUT(CaloClusterExtra_SoALayout,
                    SOA_COLUMN(AlgoId, algoId),
                    SOA_COLUMN(uint32_t, CaloID),
                    SOA_COLUMN(uint32_t, flags),
                    SOA_COLUMN(uint32_t, seedId)
)

using CaloClusterExtra_SoA = CaloClusterExtra_SoALayout<>;
using CaloClusterExtra_SoAView = CaloClusterExtra_SoA::View;
using CaloClusterExtra_SoAConstView = CaloClusterExtra_SoA::ConstView;




#endif // DataFormats_ClusterSoA_CaloClusterSoA_h
