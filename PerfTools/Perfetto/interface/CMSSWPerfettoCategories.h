#ifndef CMSSW_PERFETTO_CATEGORIES_H
#define CMSSW_PERFETTO_CATEGORIES_H
#include "PerfTools/Perfetto/interface/perfetto.h"

PERFETTO_DEFINE_CATEGORIES(
  perfetto::Category("cmssw.event"),
  perfetto::Category("cmssw.source"),
  perfetto::Category("cmssw.module"),
  perfetto::Category("cmssw.acquire"),
  perfetto::Category("cmssw.cleanup"),
  perfetto::Category("cmssw.es"),
  perfetto::Category("cmssw.func")
);

#endif