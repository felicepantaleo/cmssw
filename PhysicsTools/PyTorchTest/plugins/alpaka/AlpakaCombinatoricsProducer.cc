
#include <alpaka/alpaka.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include "DataFormats/PyTorchTest/interface/alpaka/Collections.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorchTest/plugins/alpaka/Kernels.h"
#include "PhysicsTools/PyTorchTest/interface/nvtx.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

class AlpakaCombinatoricsProducer : public stream::EDProducer<> {
 public:
  AlpakaCombinatoricsProducer(const edm::ParameterSet &params);

  void produce(device::Event &event, const device::EventSetup &event_setup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:  
  const device::EDGetToken<torchportable::ParticleCollection> inputs_token_;
  const device::EDPutToken<torchportable::ParticleCollection> outputs_token_;
  std::unique_ptr<Kernels> kernels_ = nullptr;
};

///////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION
///////////////////////////////////////////////////////////////////////////////

AlpakaCombinatoricsProducer::AlpakaCombinatoricsProducer(edm::ParameterSet const& params)
  : EDProducer<>(params),
    inputs_token_{consumes(params.getParameter<edm::InputTag>("inputs"))},
    outputs_token_{produces()},
    kernels_(std::make_unique<Kernels>()) {}

void AlpakaCombinatoricsProducer::produce(device::Event &event, const device::EventSetup &event_setup) {
  // debug stream usage in concurrently scheduled modules
  torch_alpaka::NVTXScopedRange produceRange("Combinatorics::produce");
  std::cout << "(Combinatorics) hash=" << torch_alpaka::tools::queue_hash(event.queue()) << std::endl;

  // get data
  const auto& inputs = event.get(inputs_token_);
  const size_t batch_size = inputs.const_view().metadata().size();
  auto outputs = torchportable::ParticleCollection(batch_size, event.queue());

  // dummy kernel emulation
  torch_alpaka::NVTXScopedRange kernelRange("Combinatorics::kernel");
  kernels_->FillParticleCollection(event.queue(), outputs, 0.32f);
  kernelRange.end();

  // assert output match expected  
  torch_alpaka::NVTXScopedRange assertRange("Combinatorics::assert");
  kernels_->AssertCombinatorics(event.queue(), outputs, 0.32f);
  assertRange.end();
  event.emplace(outputs_token_, std::move(outputs));
  std::cout << "(Combinatorics) OK" << std::endl; 
  produceRange.end();
}

void AlpakaCombinatoricsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputs");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(AlpakaCombinatoricsProducer);
