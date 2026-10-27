#ifndef RecoTICL_Inference_PyTorchInferenceConfiguration_h
#define RecoTICL_Inference_PyTorchInferenceConfiguration_h

#include <string>
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace ticl {
  inline std::string inferencePluginName(std::string const& backend, std::string const& name) {
    if (backend != "serial_sync" && backend != "cuda_async" && backend != "rocm_async")
      throw cms::Exception("Configuration") << "Unsupported TICL inference backend '" << backend << "'";
    return "alpaka_" + backend + "::" + name;
  }

  inline edm::ParameterSetDescription pytorchInferenceDescription() {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("backend", "serial_sync");
    desc.add<unsigned int>("batchSize", 64);
    desc.add<bool>("convertToFP16", false);
    desc.add<unsigned int>("warmupIterations", 0);
    desc.add<int>("algo_verbosity", 0);
    desc.add<std::string>("modelFamily", "PFN");
    desc.add<bool>("doPID", true);
    desc.add<bool>("doRegression", true);
    desc.add<std::string>("pidModel", "RecoTICL/Inference/data/candidate_pid_pfn.pt");
    desc.add<std::string>("regressionModel", "RecoTICL/Inference/data/candidate_energy_pfn.pt");
    desc.add<double>("minClusterEnergy", 2.5);
    return desc;
  }
}  // namespace ticl
#endif
