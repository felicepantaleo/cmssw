#ifndef DataFormats_NGTSoATest_interface_PortableHostCollectionSoATest_h
#define DataFormats_NGTSoATest_interface_PortableHostCollectionSoATest_h

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/NGTSoATest/interface/SoALayoutTest.h"

class PortableHostCollectionSoATest : public PortableHostCollection<NGTSoA> {
public:
    PortableHostCollectionSoATest(edm::Uninitialized) : PortableHostCollection<NGTSoA>{edm::kUninitialized} {}

    template <typename TQueue>
    explicit PortableHostCollectionSoATest(size_t maxModules, TQueue queue)
      : PortableHostCollection<NGTSoA>(maxModules + 1, queue) {}
};

#endif  // DataFormats_NGTSoATest_interface_PortableHostCollectionSoATest_h
