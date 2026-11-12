#ifndef RecoTICL_Interpretation_test_onnxServices_h
#define RecoTICL_Interpretation_test_onnxServices_h

#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

// The services that cms::Ort::ONNXRuntime needs on the CPU. They are available while the returned token is alive.
inline edm::OwningServiceToken makeONNXServices() {
  if (not edmplugin::PluginManager::isAvailable())
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  std::vector<edm::ParameterSet> psets;
  for (std::string const& service : {std::string("ResourceInformationService"), std::string("ONNXService")}) {
    edm::ParameterSet pset;
    pset.addParameter<std::string>("@service_type", service);
    psets.push_back(pset);
  }
  return edm::ServiceRegistry::createSet(psets);
}

#endif
