#ifndef DataFormats_HGCalReco_interface_HGCalSoALayerClustersExtra_h
#define DataFormats_HGCalReco_interface_HGCalSoALayerClustersExtra_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

// Clustering of the layer clusters into tracksters: the trackster of each layer cluster (-1 for none), the number of
// tracksters and the number of layer clusters in them
GENERATE_SOA_LAYOUT(HGCalSoALayerClustersExtraLayout,
                    SOA_COLUMN(int, tracksterIndex),
                    SOA_SCALAR(unsigned int, numberOfTrackstersScalar),
                    SOA_SCALAR(unsigned int, numberOfVerticesScalar))

using HGCalSoALayerClustersExtra = HGCalSoALayerClustersExtraLayout<>;

#endif
