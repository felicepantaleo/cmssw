#include <cmath>
#include <random>

#include <cppunit/extensions/HelperMacros.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <torch/script.h>
#include <torch/torch.h>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorch/interface/Converter.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE::torch_alpaka {
  
using namespace ::torch_alpaka;

class testSOADataTypes : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSOADataTypes);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST(testSingleElement);
  CPPUNIT_TEST(testNoElement);
  CPPUNIT_TEST(testEmptyMetadata);
  CPPUNIT_TEST_SUITE_END();

 public:
  void test();
  void testSingleElement();
  void testNoElement();
  void testEmptyMetadata();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSOADataTypes);

GENERATE_SOA_LAYOUT(SoATemplate,
  SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
  SOA_EIGEN_COLUMN(Eigen::Vector3d, b),

  SOA_EIGEN_COLUMN(Eigen::Matrix2f, c),

  SOA_COLUMN(double, x),
  SOA_COLUMN(double, y),
  SOA_COLUMN(double, z),

  SOA_SCALAR(float, type),
  SOA_SCALAR(int, someNumber)
);

using SoA = SoATemplate<>;
using SoAView = SoA::View;

class FillKernel {
 public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, PortableCollection<SoA, Device>::View view) const {
    if (cms::alpakatools::once_per_grid(acc)) {
      view.type() = 4;
      view.someNumber() = 5;
    }

    for (int32_t i : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
      view[i].a()(0) = 1 + i;
      view[i].a()(1) = 2 + i;
      view[i].a()(2) = 3 + i;

      view[i].b()(0) = 4 + i;
      view[i].b()(1) = 5 + i;
      view[i].b()(2) = 6 + i;

      view[i].c()(0, 0) = 4 + i;
      view[i].c()(0, 1) = 6 + i;
      view[i].c()(1, 0) = 8 + i;
      view[i].c()(1, 1) = 10 + i;

      view.x()[i] = 12 + i;
      view.y()[i] = 2.5 * i;
      view.z()[i] = 36 * i;
    }
  }
};

class InputVerifyKernel {
 public:
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, PortableCollection<SoA, Device>::View view) const {
    if (cms::alpakatools::once_per_grid(acc)) {
      ALPAKA_ASSERT_ACC(view.type() == 4);
      ALPAKA_ASSERT_ACC(view.someNumber() == 5);
    }

    for (uint32_t i : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
      ALPAKA_ASSERT_ACC(view[i].a()(0) == 1 + i);
      ALPAKA_ASSERT_ACC(view[i].a()(1) == 2 + i);
      ALPAKA_ASSERT_ACC(view[i].a()(2) == 3 + i);

      ALPAKA_ASSERT_ACC(view[i].b()(0) == 4 + i);
      ALPAKA_ASSERT_ACC(view[i].b()(1) == 5 + i);
      ALPAKA_ASSERT_ACC(view[i].b()(2) == 6 + i);

      ALPAKA_ASSERT_ACC(view[i].c()(0, 0) == 4 + i);
      ALPAKA_ASSERT_ACC(view[i].c()(0, 1) == 6 + i);
      ALPAKA_ASSERT_ACC(view[i].c()(1, 0) == 8 + i);
      ALPAKA_ASSERT_ACC(view[i].c()(1, 1) == 10 + i);

      ALPAKA_ASSERT_ACC(view.x()[i] == 12 + i);
      ALPAKA_ASSERT_ACC(view.y()[i] == 2.5 * i);
      ALPAKA_ASSERT_ACC(view.z()[i] == 36 * i);
    }
  }
};

class TestVerifyKernel {
 public:
  ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                PortableCollection<SoA, Device>::View view,
                                torch::PackedTensorAccessor64<double, 3> tensor_vector,
                                torch::PackedTensorAccessor64<float, 4> tensor_matrix,
                                torch::PackedTensorAccessor64<double, 2> tensor_column,
                                torch::PackedTensorAccessor64<float, 2> tensor_scalar) const {
    for (uint32_t i : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
      ALPAKA_ASSERT_ACC(view[i].a()(0) - tensor_vector[i][0][0] < 1.0e-05);
      ALPAKA_ASSERT_ACC(view[i].a()(1) - tensor_vector[i][0][1] < 1.0e-05);
      ALPAKA_ASSERT_ACC(view[i].a()(2) - tensor_vector[i][0][2] < 1.0e-05);
      ALPAKA_ASSERT_ACC(view[i].a()(0) - tensor_vector[i][0][0] > -1.0e-05);
      ALPAKA_ASSERT_ACC(view[i].a()(1) - tensor_vector[i][0][1] > -1.0e-05);
      ALPAKA_ASSERT_ACC(view[i].a()(2) - tensor_vector[i][0][2] > -1.0e-05);

      ALPAKA_ASSERT_ACC(view[i].b()(0) - tensor_vector[i][1][0] < 1.0e-05);
      ALPAKA_ASSERT_ACC(view[i].b()(1) - tensor_vector[i][1][1] < 1.0e-05);
      ALPAKA_ASSERT_ACC(view[i].b()(2) - tensor_vector[i][1][2] < 1.0e-05);
      ALPAKA_ASSERT_ACC(view[i].b()(0) - tensor_vector[i][1][0] > -1.0e-05);
      ALPAKA_ASSERT_ACC(view[i].b()(1) - tensor_vector[i][1][1] > -1.0e-05);
      ALPAKA_ASSERT_ACC(view[i].b()(2) - tensor_vector[i][1][2] > -1.0e-05);

      ALPAKA_ASSERT_ACC(view[i].c()(0, 0) - tensor_matrix[i][0][0][0] < 1.0e-05);
      ALPAKA_ASSERT_ACC(view[i].c()(0, 0) - tensor_matrix[i][0][0][0] > -1.0e-05);
      ALPAKA_ASSERT_ACC(view[i].c()(0, 1) - tensor_matrix[i][0][0][1] < 1.0e-05);
      ALPAKA_ASSERT_ACC(view[i].c()(0, 1) - tensor_matrix[i][0][0][1] > -1.0e-05);
      ALPAKA_ASSERT_ACC(view[i].c()(1, 0) - tensor_matrix[i][0][1][0] < 1.0e-05);
      ALPAKA_ASSERT_ACC(view[i].c()(1, 0) - tensor_matrix[i][0][1][0] > -1.0e-05);
      ALPAKA_ASSERT_ACC(view[i].c()(1, 1) - tensor_matrix[i][0][1][1] < 1.0e-05);
      ALPAKA_ASSERT_ACC(view[i].c()(1, 1) - tensor_matrix[i][0][1][1] > -1.0e-05);

      ALPAKA_ASSERT_ACC(view.x()[i] - tensor_column[i][0] < 1.0e-05);
      ALPAKA_ASSERT_ACC(view.x()[i] - tensor_column[i][0] > -1.0e-05);

      ALPAKA_ASSERT_ACC(view.y()[i] - tensor_column[i][1] < 1.0e-05);
      ALPAKA_ASSERT_ACC(view.y()[i] - tensor_column[i][1] > -1.0e-05);

      ALPAKA_ASSERT_ACC(view.z()[i] - tensor_column[i][2] < 1.0e-05);
      ALPAKA_ASSERT_ACC(view.z()[i] - tensor_column[i][2] > -1.0e-05);

      ALPAKA_ASSERT_ACC(view.type() - tensor_scalar[i][0] < 1.0e-05);
      ALPAKA_ASSERT_ACC(view.type() - tensor_scalar[i][0] > -1.0e-05);
    }
  }
};

void fill(Queue& queue, PortableCollection<SoA, Device>& collection) {
  uint32_t items = 64;
  uint32_t groups = cms::alpakatools::divide_up_by(collection->metadata().size(), items);
  auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
  alpaka::exec<Acc1D>(queue, workDiv, FillKernel{}, collection.view());
  alpaka::exec<Acc1D>(queue, workDiv, InputVerifyKernel{}, collection.view());
}

void check(Queue& queue, PortableCollection<SoA, Device>& collection, std::vector<torch::IValue> tensors) {
  uint32_t items = 64;
  uint32_t groups = cms::alpakatools::divide_up_by(collection->metadata().size(), items);
  auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
  alpaka::exec<Acc1D>(queue,
                      workDiv,
                      TestVerifyKernel{},
                      collection.view(),
                      tensors[3].toTensor().packed_accessor64<double, 3>(),
                      tensors[2].toTensor().packed_accessor64<float, 4>(),
                      tensors[0].toTensor().packed_accessor64<double, 2>(),
                      tensors[1].toTensor().packed_accessor64<float, 2>());
}

void testSOADataTypes::test() {
  Platform platform;
  std::vector<Device> alpakaDevices = alpaka::getDevs(platform);
  CPPUNIT_ASSERT(alpakaDevices.size());
  const auto& alpakaDevice = alpakaDevices[0];
  Queue queue(alpakaDevice);
  torch::Device torchDevice(kTorchDeviceType);

  // Large batch size, so multiple bunches needed
  const std::size_t batch_size = 325;

  // Create and fill portable collections
  PortableCollection<SoA, Device> deviceCollection(batch_size, queue);
  fill(queue, deviceCollection);

  // Run Converter for multiple tensors
  InputMetadata input({Double, Float, Double, Float, Int}, {{{2, 3}}, {{1, 2, 2}}, 3, 0, 0}, {3, 2, 0, 1, -1});
  OutputMetadata output(Double, 3);
  ModelMetadata metadata(batch_size, input, output);

  alpaka::wait(queue);
  std::vector<torch::IValue> tensors =
      Converter<SoA>::convert_input(metadata, torchDevice, deviceCollection.buffer().data());

  // Check if tensor list built correctly
  check(queue, deviceCollection, tensors);
};

void testSOADataTypes::testSingleElement() {
  Platform platform;
  std::vector<Device> alpakaDevices = alpaka::getDevs(platform);
  CPPUNIT_ASSERT(alpakaDevices.size());
  const auto& alpakaDevice = alpakaDevices[0];
  Queue queue(alpakaDevice);
  torch::Device torchDevice(kTorchDeviceType);


  // Create and fill portable collections
  const std::size_t batch_size = 1;
  PortableCollection<SoA, Device> deviceCollection(batch_size, queue);
  fill(queue, deviceCollection);

  // Run Converter for single tensor
  InputMetadata input({Double, Float, Double, Float, Int}, {{{2, 3}}, {{1, 2, 2}}, 3, 0, 0}, {3, 2, 0, 1, -1});
  OutputMetadata output(Double, 3);
  ModelMetadata metadata(batch_size, input, output);

  alpaka::wait(queue);
  std::vector<torch::IValue> tensors =
      Converter<SoA>::convert_input(metadata, torchDevice, deviceCollection.buffer().data());

  // Check if tensor list built correctly
  check(queue, deviceCollection, tensors);
};

void testSOADataTypes::testNoElement() {
  Platform platform;
  std::vector<Device> alpakaDevices = alpaka::getDevs(platform);
  CPPUNIT_ASSERT(alpakaDevices.size());
  const auto& alpakaDevice = alpakaDevices[0];
  Queue queue(alpakaDevice);
  torch::Device torchDevice(kTorchDeviceType);

  //Create empty portable collection
  const std::size_t batch_size = 0;
  PortableCollection<SoA, Device> deviceCollection(batch_size, queue);

  // Run Converter
  InputMetadata input({Double, Float, Double, Float, Int}, {{{2, 3}}, {{1, 2, 2}}, 3, 0, 0}, {3, 2, 0, 1, -1});
  OutputMetadata output(Double, 3);
  ModelMetadata metadata(batch_size, input, output);

  alpaka::wait(queue);
  std::vector<torch::IValue> tensors =
      Converter<SoA>::convert_input(metadata, torchDevice, deviceCollection.buffer().data());

  // Check if tensor list has empty tensors
  CPPUNIT_ASSERT(tensors[0].toTensor().size(0) == 0);
  CPPUNIT_ASSERT(tensors[1].toTensor().size(0) == 0);
  CPPUNIT_ASSERT(tensors[2].toTensor().size(0) == 0);
  CPPUNIT_ASSERT(tensors[3].toTensor().size(0) == 0);
};

void testSOADataTypes::testEmptyMetadata() {
  Platform platform;
  std::vector<Device> alpakaDevices = alpaka::getDevs(platform);
  CPPUNIT_ASSERT(alpakaDevices.size());
  const auto& alpakaDevice = alpakaDevices[0];
  Queue queue(alpakaDevice);
  torch::Device torchDevice(kTorchDeviceType);


  // Create and fill portable collections
  const std::size_t batch_size = 12;
  PortableCollection<SoA, Device> deviceCollection(batch_size, queue);
  fill(queue, deviceCollection);

  // Run Converter for empty metadata
  InputMetadata input;
  OutputMetadata output;
  ModelMetadata metadata(batch_size, input, output);

  alpaka::wait(queue);
  std::vector<torch::IValue> tensors =
      Converter<SoA>::convert_input(metadata, torchDevice, deviceCollection.buffer().data());

  // Check if tensor list is empty
  CPPUNIT_ASSERT(tensors.size() == 0);
};

}// namespace ALPAKA_ACCELERATOR_NAMESPACE::torch_alpaka
