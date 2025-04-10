#ifndef DATA_FORMATS__PYTORCH_TEST__INTERFACE__ALPAKA__COLLECTIONS_H_
#define DATA_FORMATS__PYTORCH_TEST__INTERFACE__ALPAKA__COLLECTIONS_H_

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/PyTorchTest/interface/Device.h"
#include "DataFormats/PyTorchTest/interface/Host.h"
#include "DataFormats/PyTorchTest/interface/Layout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
namespace torchportable {

using namespace ::torchportable;
using ::torchportable::ParticleCollectionHost;
using ::torchportable::ClassificationCollectionHost;
using ::torchportable::RegressionCollectionHost;
using ::torchportable::ParticleCollectionDevice;
using ::torchportable::ClassificationCollectionDevice;
using ::torchportable::RegressionCollectionDevice;

using ParticleCollection =
    std::conditional_t<
        std::is_same_v<Device, alpaka::DevCpu>, 
        ParticleCollectionHost, 
        ParticleCollectionDevice<Device>>;

using ClassificationCollection =
    std::conditional_t<
        std::is_same_v<Device, alpaka::DevCpu>, 
        ClassificationCollectionHost, 
        ClassificationCollectionDevice<Device>>;

using RegressionCollection =
    std::conditional_t<
        std::is_same_v<Device, alpaka::DevCpu>, 
        RegressionCollectionHost, 
        RegressionCollectionDevice<Device>>;

}  // namespace torchportable
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(torchportable::ParticleCollection, torchportable::ParticleCollectionHost);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(torchportable::ClassificationCollection, torchportable::ClassificationCollectionHost);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(torchportable::RegressionCollection, torchportable::RegressionCollectionHost);

#endif  // DATA_FORMATS__PYTORCH_TEST__INTERFACE__ALPAKA__COLLECTIONS_H_