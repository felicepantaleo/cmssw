#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "RecoTICL/SeedingRegions/interface/SeedingRegionAlgoFactory.h"
#include "RecoTICL/SeedingRegions/interface/SeedingRegionByTracks.h"
#include "RecoTICL/SeedingRegions/interface/SeedingRegionGlobal.h"
#include "RecoTICL/SeedingRegions/interface/SeedingRegionByL1.h"
#include "RecoTICL/SeedingRegions/interface/SeedingRegionByHF.h"

EDM_REGISTER_VALIDATED_PLUGINFACTORY(SeedingRegionAlgoFactory, "SeedingRegionAlgoFactory");

DEFINE_EDM_VALIDATED_PLUGIN(SeedingRegionAlgoFactory, ticl::SeedingRegionByTracks, "SeedingRegionByTracks");
DEFINE_EDM_VALIDATED_PLUGIN(SeedingRegionAlgoFactory, ticl::SeedingRegionGlobal, "SeedingRegionGlobal");
DEFINE_EDM_VALIDATED_PLUGIN(SeedingRegionAlgoFactory, ticl::SeedingRegionByL1, "SeedingRegionByL1");
DEFINE_EDM_VALIDATED_PLUGIN(SeedingRegionAlgoFactory, ticl::SeedingRegionByHF, "SeedingRegionByHF");
