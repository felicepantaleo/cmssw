#ifndef DataFormats_NGTSoATest_interface_PortableDeviceCollectionSoATest_h
#define DataFormats_NGTSoATest_interface_PortableDeviceCollectionSoATest_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/NGTSoATest/interface/SoALayoutTest.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TDev>
class PortableDeviceCollectionSoATest : public PortableDeviceCollection<NGTSoA, TDev> {
public:
    PortableDeviceCollectionSoATest(edm::Uninitialized) : PortableDeviceCollection<NGTSoA, TDev>{edm::kUninitialized} {}

    template <typename TQueue>
    explicit PortableDeviceCollectionSoATest(size_t maxModules, TQueue queue)
        : PortableDeviceCollection<NGTSoA, TDev>(maxModules + 1, queue) {}    

    // Constructor which specifies the SoA size
    explicit PortableDeviceCollectionSoATest(size_t maxModules, TDev const &device)
        : PortableDeviceCollection<NGTSoA, TDev>(maxModules + 1, device) {}
};

#endif  // DataFormats_NGTSoATest_interface_PortableDeviceCollectionSoATest_h