#ifndef CondFormats_HGCalReco_HGCalGeomSoA_h
#define CondFormats_HGCalReco_HGCalGeomSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(HGCalGeomSoALayout,
                    SOA_COLUMN(uint32_t, rawDetId),
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_COLUMN(float, eta),
                    SOA_COLUMN(float, phi)
                    // SOA_COLUMN(float, cellSize),
                    // SOA_COLUMN(int, layerId),
                    // SOA_COLUMN(bool, isSilicon)

)

using HGCalGeomSoA = HGCalGeomSoALayout<>;
using HGCalGeomSoAView = HGCalGeomSoA::View;
using HGCalGeomSoAConstView = HGCalGeomSoA::ConstView;

#endif