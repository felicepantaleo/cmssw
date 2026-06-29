#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "RecoTICL/ClusterFilters/interface/ClusterFilterFactory.h"

#include "RecoTICL/ClusterFilters/interface/ClusterFilterByAlgo.h"
#include "RecoTICL/ClusterFilters/interface/ClusterFilterByAlgoAndSize.h"
#include "RecoTICL/ClusterFilters/interface/ClusterFilterBySize.h"
#include "RecoTICL/ClusterFilters/interface/ClusterFilterByAlgoAndSizeAndLayerRange.h"

using namespace ticl;

DEFINE_EDM_PLUGIN(ClusterFilterFactory, ClusterFilterByAlgo, "ClusterFilterByAlgo");
DEFINE_EDM_PLUGIN(ClusterFilterFactory, ClusterFilterByAlgoAndSize, "ClusterFilterByAlgoAndSize");
DEFINE_EDM_PLUGIN(ClusterFilterFactory, ClusterFilterBySize, "ClusterFilterBySize");
DEFINE_EDM_PLUGIN(ClusterFilterFactory,
                  ClusterFilterByAlgoAndSizeAndLayerRange,
                  "ClusterFilterByAlgoAndSizeAndLayerRange");
