#ifndef DataFormats_NGTSoATest_interface_alpaka_NGTSoACollection_h
#define DataFormats_NGTSoATest_interface_alpaka_NGTSoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/NGTSoATest/interface/PortableHostCollectionSoATest.h"
#include "DataFormats/NGTSoATest/interface/PortableDeviceCollectionSoATest.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using NGTSoACollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, PortableHostCollectionSoATest, PortableDeviceCollectionSoATest<Device>>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TDevice>
  struct CopyToHost<PortableDeviceCollectionSoATest<TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableDeviceCollectionSoATest<TDevice> const& deviceData) {
      PortableHostCollectionSoATest hostData(deviceData.view().metadata().size() - 1, queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
#ifdef GPU_DEBUG
      printf("TracksSoACollection: I'm copying to host.\n");
#endif
      return hostData;
    }
  };
}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(NGTSoACollection, PortableHostCollectionSoATest);
#endif  // DataFormats_NGTSoATest_interface_alpaka_NGTSoACollection_h