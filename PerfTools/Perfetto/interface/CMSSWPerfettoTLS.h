#pragma once
#include "PerfTools/Perfetto/interface/perfetto.h"
#include <cstdint>

namespace cms::perfetto_tls {
  // Set current lane for Tier-B slices (per thread).
  void set(uint64_t event_uuid, uint64_t stream_child, uint64_t phase_child) noexcept;
  void clear() noexcept;

  bool enabled() noexcept;
  perfetto::Track track() noexcept;
}  // namespace cms::perfetto_tls