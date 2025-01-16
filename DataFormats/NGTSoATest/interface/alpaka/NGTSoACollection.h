#ifndef DataFormats_NGTSoATest_interface_alpaka_NGTSoACollection_h
#define DataFormats_NGTSoATest_interface_alpaka_NGTSoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

#include "DataFormats/NGTSoATest/interface/SoALayoutTest.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using NGTSoACollection = std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>,
                                              PortableHostCollection<NGTSoA>,
                                              PortableDeviceCollection<NGTSoA, Device>>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TDevice>
  struct CopyToHost<PortableDeviceCollection<NGTSoA, TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableDeviceCollection<NGTSoA, TDevice> const& deviceData) {
      PortableHostCollection<NGTSoA> hostData(deviceData.view().metadata().size() - 1, queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
      printf("TracksSoACollection: I'm copying to host.\n");
      return hostData;
    }
  };
}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(NGTSoACollection, PortableHostCollection<NGTSoA>);
#endif  // DataFormats_NGTSoATest_interface_alpaka_NGTSoACollection_h