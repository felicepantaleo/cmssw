#pragma once

#include "PerfTools/Perfetto/interface/CMSSWPerfettoCategories.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoTLS.h"
#include "PerfTools/Perfetto/interface/perfetto.h"

namespace cms::perfetto_trace {

  struct SliceScope {
    explicit SliceScope(const char* name) noexcept {
      if (!perfetto::TrackEvent::IsEnabled() || !cms::perfetto_tls::enabled())
        return;
      auto t = cms::perfetto_tls::track();
      TRACE_EVENT_BEGIN("cmssw.func", perfetto::DynamicString(name), t);
      active_ = true;
    }

    ~SliceScope() noexcept {
      if (!active_)
        return;
      auto t = cms::perfetto_tls::track();
      TRACE_EVENT_END("cmssw.func", t);
    }

    SliceScope(SliceScope const&) = delete;
    SliceScope& operator=(SliceScope const&) = delete;

  private:
    bool active_ = false;
  };

}  // namespace cms::perfetto_trace

#define CMS_PERFETTO_FUNC() \
  cms::perfetto_trace::SliceScope PERFETTO_UID(_cms_perfetto_func_){__func__}

#define CMS_PERFETTO_SCOPE(name_literal) \
  cms::perfetto_trace::SliceScope PERFETTO_UID(_cms_perfetto_scope_){name_literal}