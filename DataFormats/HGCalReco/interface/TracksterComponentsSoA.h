#ifndef DataFormats_HGCalReco_interface_TracksterComponentsSoA_h
#define DataFormats_HGCalReco_interface_TracksterComponentsSoA_h

// Output of any TICL linking step that reduces to connected components: one
// label per input trackster, parallel to the TracksterSoA it was computed from.
//
// The label of a component is the SMALLEST input index belonging to it. That is
// not an arbitrary choice: it is the fixed point of both the host union-find
// (parent[max] = min) and of device label propagation by atomicMin, so the two
// backends produce identical labels element by element and can be compared with
// a bitwise diff instead of a physics argument.

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(TracksterComponentsSoALayout, SOA_COLUMN(int32_t, label))

using TracksterComponentsSoA = TracksterComponentsSoALayout<>;

#endif  // DataFormats_HGCalReco_interface_TracksterComponentsSoA_h
