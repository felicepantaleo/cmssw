#ifndef RecoHGCal_TICL_TracksterFilterFactory_H
#define RecoHGCal_TICL_TracksterFilterFactory_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "TracksterFilterBase.h"

typedef edmplugin::PluginFactory<ticl::TracksterFilterBase*(const edm::ParameterSet&)> TracksterFilterFactory;

#endif
