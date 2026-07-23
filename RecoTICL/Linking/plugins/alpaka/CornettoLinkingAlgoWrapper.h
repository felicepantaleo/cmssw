#ifndef RecoTICL_Linking_plugins_alpaka_CornettoLinkingAlgoWrapper_h
#define RecoTICL_Linking_plugins_alpaka_CornettoLinkingAlgoWrapper_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/HGCalReco/interface/alpaka/TracksterComponentsDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The Cornetto cone parameters, identical in meaning to the CPU plugin's
  // ParameterSet (RecoHGCal/TICL/plugins/TracksterLinkingByCornetto.h).
  struct CornettoParameters {
    float etaWindow;
    float maxLongitudinalDistance;
    float transverseRadius0;
    float transverseSlope;
    float timeCompatibilityNSigma;
    float maxLongitudinalSlope;
    float longitudinalZRef;
  };

  class CornettoLinkingAlgoWrapper {
  public:
    // Fills components.label() with the connected-component label of each input
    // trackster, where the label is the SMALLEST input index in the component,
    // matching the host union-find convention exactly.
    //
    // Synchronous with respect to the queue: it reads two counters back to host
    // (the total number of edges, and the label-propagation convergence flag), so
    // the caller must be a SynchronizingEDProducer or otherwise tolerate the wait.
    void run(Queue& queue,
             const CornettoParameters& params,
             const TracksterDeviceCollection::ConstView tracksters,
             TracksterComponentsDeviceCollection::View components) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Linking_plugins_alpaka_CornettoLinkingAlgoWrapper_h
