#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "ClusterFilterFactory.h"

#include "ClusterFilterByAlgo.h"
#include "ClusterFilterByAlgoAndSize.h"
#include "ClusterFilterBySize.h"
#include "ClusterFilterByAlgoAndSizeAndLayerRange.h"

#include "TracksterFilterFactory.h"

#include "TracksterFilterByAlgo.h"
#include "TracksterFilterBySize.h"


using namespace ticl;

DEFINE_EDM_PLUGIN(ClusterFilterFactory, ClusterFilterByAlgo, "ClusterFilterByAlgo");
DEFINE_EDM_PLUGIN(ClusterFilterFactory, ClusterFilterByAlgoAndSize, "ClusterFilterByAlgoAndSize");
DEFINE_EDM_PLUGIN(ClusterFilterFactory, ClusterFilterBySize, "ClusterFilterBySize");
DEFINE_EDM_PLUGIN(ClusterFilterFactory,
                  ClusterFilterByAlgoAndSizeAndLayerRange,
                  "ClusterFilterByAlgoAndSizeAndLayerRange");


DEFINE_EDM_PLUGIN(TracksterFilterFactory, TracksterFilterByAlgo, "TracksterFilterByAlgo");
DEFINE_EDM_PLUGIN(TracksterFilterFactory, TracksterFilterBySize, "TracksterFilterBySize");

