#include "PerfTools/Perfetto/interface/CMSSWPerfettoTLS.h"

namespace cms::perfetto_tls {
  static thread_local bool g_enabled = false;
  static thread_local uint64_t g_event_uuid = 0;
  static thread_local uint64_t g_stream_child = 0;
  static thread_local uint64_t g_phase_child = 0;

  void set(uint64_t event_uuid, uint64_t stream_child, uint64_t phase_child) noexcept {
    g_event_uuid = event_uuid;
    g_stream_child = stream_child;
    g_phase_child = phase_child;
    g_enabled = true;
  }

  void clear() noexcept { g_enabled = false; }

  bool enabled() noexcept { return g_enabled; }

  perfetto::Track track() noexcept {
    // Process -> Event -> Stream -> Phase
    auto proc = perfetto::ProcessTrack::Current();
    perfetto::Track ev(g_event_uuid, proc);
    perfetto::Track st(g_stream_child, ev);
    return perfetto::Track(g_phase_child, st);
  }
}  // namespace cms::perfetto_tls