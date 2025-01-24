#ifndef DataFormats_ClusterSoA_interface_PFSoA_h
#define DataFormats_ClusterSoA_interface_PFSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

// Example: Some fields that might only apply to PF clusters
GENERATE_SOA_LAYOUT(PFSoALayout,
                    SOA_COLUMN(float, layer),        // e.g. PF layer index
                    SOA_COLUMN(float, depth),        // e.g. depth
                    SOA_COLUMN(int, nRecHits),       // number of rec hits
                    SOA_COLUMN(float, mvaScore)      // MVA-based quality score, if used
)

using PFSoA = PFSoALayout<>;
using PFSoAView = PFSoA::View;
using PFSoAConstView = PFSoA::ConstView;

#endif  // DataFormats_ClusterSoA_interface_PFSoA_h
