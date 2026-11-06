#ifndef RecoTICL_Common_interface_alpaka_TracksterCollections_h
#define RecoTICL_Common_interface_alpaka_TracksterCollections_h

// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"

#include <string_view>
#include <vector>

#include "FWCore/Utilities/interface/EDMException.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  // The trackster collections of tokens, in the order of refs. Each collection of tokens must be in refs exactly once,
  // and refs must have no other collection.
  inline std::vector<TracksterDeviceCollection const*> orderedTracksterCollections(
      device::Event const& event,
      TracksterCollectionRefs const& refs,
      std::vector<device::EDGetToken<TracksterDeviceCollection>> const& tokens,
      std::string_view module) {
    if (refs.size() != tokens.size()) {
      throw edm::Exception(edm::errors::LogicError)
          << module << ": the ordered list has " << refs.size() << " trackster collections, the module reads "
          << tokens.size() << ".";
    }
    std::vector<TracksterDeviceCollection const*> ordered(refs.size(), nullptr);
    for (auto const& token : tokens) {
      const auto handle = event.getHandle(token);
      std::size_t matches = 0;
      for (std::size_t i = 0; i < refs.size(); ++i) {
        if (refs[i].id() == handle.id()) {
          ordered[i] = handle.product();
          ++matches;
        }
      }
      if (matches != 1) {
        throw edm::Exception(edm::errors::LogicError)
            << module << ": the trackster collection " << handle.provenance()->moduleLabel() << " is " << matches
            << " times in the ordered list, it must be there once.";
      }
    }
    return ordered;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif  // RecoTICL_Common_interface_alpaka_TracksterCollections_h
