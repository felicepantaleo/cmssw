#ifndef RecoHGCal_TICL_SeedingRegionAlgoFactory_h
#define RecoHGCal_TICL_SeedingRegionAlgoFactory_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoTICL/SeedingRegions/interface/SeedingRegionAlgoBase.h"

using SeedingRegionAlgoFactory =
    edmplugin::PluginFactory<ticl::SeedingRegionAlgoBase*(const edm::ParameterSet&, edm::ConsumesCollector&)>;

#endif
