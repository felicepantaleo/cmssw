#include "PerfTools/Perfetto/interface/CMSSWPerfettoCategories.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoTrace.h"
#include "PerfTools/Perfetto/interface/perfetto.h"

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/PlaceInPathContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/Framework/interface/ComponentDescription.h"


namespace {
  inline uint64_t makeEventTrackId(unsigned sid, uint64_t seq) noexcept {
    return (uint64_t{sid} << 56) | (seq & ((uint64_t{1} << 56) - 1));
  }

  inline std::string formatEventName(edm::EventID const& id, unsigned sid) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "Event %u:%u:%llu (stream %u)",
                  id.run(), id.luminosityBlock(),
                  static_cast<unsigned long long>(id.event()), sid);
    return std::string(buf);
  }

  constexpr uint64_t kStreamTrackIdBase = 0x100;
  constexpr uint64_t kPhaseSourceId     = 0x200;
  constexpr uint64_t kPhaseModuleId     = 0x201;
  constexpr uint64_t kPhaseAcquireId    = 0x202;
  constexpr uint64_t kPhaseCleanupId    = 0x203;
  constexpr uint64_t kPhaseESId         = 0x204;

  struct TrackIds {
    uint64_t event_uuid = 0;
    uint64_t stream_child = 0;  // kStreamTrackIdBase + sid
  };
}  // namespace

class PerfettoTraceService {
public:
  PerfettoTraceService(edm::ParameterSet const& pset, edm::ActivityRegistry& ar)
      : enabled_(pset.getUntrackedParameter<bool>("enabled", true)),
        fileName_(pset.getUntrackedParameter<std::string>("fileName", "cmsrun.pftrace")),
        bufferSizeKB_(pset.getUntrackedParameter<unsigned>("bufferSizeKB", 256 * 1024)),
        maxEvents_(pset.getUntrackedParameter<unsigned>("maxEvents", 200)) {
    if (!enabled_)
      return;

    perfetto::TracingInitArgs args;
    args.backends = perfetto::kInProcessBackend;
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();

    perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(bufferSizeKB_);
    cfg.add_data_sources()->mutable_config()->set_name("track_event");

    session_ = perfetto::Tracing::NewTrace();
    traceFd_ = ::open(fileName_.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
    if (traceFd_ >= 0) session_->Setup(cfg, traceFd_);
    else session_->Setup(cfg);
    session_->StartBlocking();

    {
      auto proc = perfetto::ProcessTrack::Current();
      auto desc = proc.Serialize();
      desc.mutable_process()->set_process_name("cmsRun");
      perfetto::TrackEvent::SetTrackDescriptor(proc, desc);
    }

    ar.watchPreallocate(this, &PerfettoTraceService::preallocate);

    ar.watchPreSourceEvent(this, &PerfettoTraceService::preSourceEvent);
    ar.watchPreEvent(this, &PerfettoTraceService::preEvent);
    ar.watchPostSourceEvent(this, &PerfettoTraceService::postSourceEvent);
    ar.watchPreClearEvent(this, &PerfettoTraceService::preClearEvent);
    ar.watchPostClearEvent(this, &PerfettoTraceService::postClearEvent);

    ar.watchPreModuleEvent(this, &PerfettoTraceService::preModuleEvent);
    ar.watchPostModuleEvent(this, &PerfettoTraceService::postModuleEvent);
    ar.watchPreModuleEventAcquire(this, &PerfettoTraceService::preModuleEventAcquire);
    ar.watchPostModuleEventAcquire(this, &PerfettoTraceService::postModuleEventAcquire);

    ar.watchPreESModule(this, &PerfettoTraceService::preESModule);
    ar.watchPostESModule(this, &PerfettoTraceService::postESModule);

    ar.watchPostEndJob(this, &PerfettoTraceService::postEndJob);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<bool>("enabled", true);
    desc.addUntracked<std::string>("fileName", "cmsrun.pftrace");
    desc.addUntracked<unsigned>("bufferSizeKB", 256 * 1024);
    desc.addUntracked<unsigned>("maxEvents", 200);
    descriptions.add("PerfettoTraceService", desc);
  }

private:
  struct PerStream {
    uint64_t seq = 0;
    bool active = false;
    unsigned sid = 0;
    TrackIds ids;
    bool event_named = false;
  };

  perfetto::ProcessTrack procTrack() const noexcept { return perfetto::ProcessTrack::Current(); }

  perfetto::Track eventTrack(PerStream const& st) const noexcept {
    return perfetto::Track(st.ids.event_uuid, procTrack());
  }
  perfetto::Track streamTrack(PerStream const& st) const noexcept {
    return perfetto::Track(st.ids.stream_child, eventTrack(st));
  }
  perfetto::Track phaseTrack(PerStream const& st, uint64_t phase_child) const noexcept {
    return perfetto::Track(phase_child, streamTrack(st));
  }

  void preallocate(edm::service::SystemBounds const& bounds) {
    states_.assign(bounds.maxNumberOfStreams(), PerStream{});
    for (unsigned i = 0; i < states_.size(); ++i) {
      states_[i].sid = i;
      states_[i].ids.stream_child = kStreamTrackIdBase + i;
    }
  }

  void preSourceEvent(edm::StreamID sid) {
    if (!enabled_ || !perfetto::TrackEvent::IsEnabled()) return;

    auto& st = states_[sid.value()];
    if (++seenEvents_ > maxEvents_) { st.active = false; return; }
    st.event_named = false;
    st.active = true;
    ++st.seq;
    st.ids.event_uuid = makeEventTrackId(st.sid, st.seq);

    auto eT = eventTrack(st);
    auto sT = streamTrack(st);

    auto srcT = phaseTrack(st, kPhaseSourceId);
    auto modT = phaseTrack(st, kPhaseModuleId);
    auto acqT = phaseTrack(st, kPhaseAcquireId);
    auto clnT = phaseTrack(st, kPhaseCleanupId);
    auto esT  = phaseTrack(st, kPhaseESId);

    {
      auto d = eT.Serialize();
      d.set_name("Event (pending EventID)");
      perfetto::TrackEvent::SetTrackDescriptor(eT, d);
    }
    {
      auto d = sT.Serialize();
      d.set_name("edm::stream " + std::to_string(st.sid));
      perfetto::TrackEvent::SetTrackDescriptor(sT, d);
    }
    auto namePhase = [](perfetto::Track t, const char* n) {
      auto d = t.Serialize();
      d.set_name(n);
      perfetto::TrackEvent::SetTrackDescriptor(t, d);
    };
    namePhase(srcT, "source");
    namePhase(modT, "modules");
    namePhase(acqT, "acquire");
    namePhase(clnT, "cleanup");
    namePhase(esT,  "eventsetup");

    TRACE_EVENT_BEGIN("cmssw.event", "Event", eT);
    TRACE_EVENT_BEGIN("cmssw.source", "Source", srcT);
  }

  void preEvent(edm::StreamContext const& sc) {
    if (!enabled_ || !perfetto::TrackEvent::IsEnabled()) return;
    auto& st = states_[sc.streamID().value()];
    if (!st.active) return;

    auto eT = eventTrack(st);
    auto d = eT.Serialize();
    d.set_name(formatEventName(sc.eventID(), st.sid));
    perfetto::TrackEvent::SetTrackDescriptor(eT, d);
  }

  void postSourceEvent(edm::StreamID sid) {
    if (!enabled_ || !perfetto::TrackEvent::IsEnabled()) return;
    auto& st = states_[sid.value()];
    if (!st.active) return;
    TRACE_EVENT_END("cmssw.source", phaseTrack(st, kPhaseSourceId));
  }

  void preModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
    if (!enabled_ || !perfetto::TrackEvent::IsEnabled()) return;
    auto& st = states_[sc.streamID().value()];
    if (!st.active) return;
    if (!st.event_named) {
  auto eT = eventTrack(st);
  auto d = eT.Serialize();
  d.set_name(formatEventName(sc.eventID(), st.sid));
  perfetto::TrackEvent::SetTrackDescriptor(eT, d);
  st.event_named = true;
}
    cms::perfetto_tls::set(st.ids.event_uuid, st.ids.stream_child, kPhaseModuleId);

    auto const& md = *mcc.moduleDescription();
    TRACE_EVENT_BEGIN("cmssw.module",
                      perfetto::DynamicString(md.moduleLabel()),
                      phaseTrack(st, kPhaseModuleId),
                      "id", md.id(),
                      "type", perfetto::DynamicString(md.moduleName()));
  }

  void postModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const&) {
    if (!enabled_ || !perfetto::TrackEvent::IsEnabled()) return;
    auto& st = states_[sc.streamID().value()];
    if (!st.active) return;

    TRACE_EVENT_END("cmssw.module", phaseTrack(st, kPhaseModuleId));
    cms::perfetto_tls::clear();
  }

  void preModuleEventAcquire(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
    if (!enabled_ || !perfetto::TrackEvent::IsEnabled()) return;
    auto& st = states_[sc.streamID().value()];
    if (!st.active) return;

    cms::perfetto_tls::set(st.ids.event_uuid, st.ids.stream_child, kPhaseAcquireId);

    auto const& md = *mcc.moduleDescription();
    TRACE_EVENT_BEGIN("cmssw.acquire",
                      perfetto::DynamicString(md.moduleLabel()),
                      phaseTrack(st, kPhaseAcquireId));
  }

  void postModuleEventAcquire(edm::StreamContext const& sc, edm::ModuleCallingContext const&) {
    if (!enabled_ || !perfetto::TrackEvent::IsEnabled()) return;
    auto& st = states_[sc.streamID().value()];
    if (!st.active) return;

    TRACE_EVENT_END("cmssw.acquire", phaseTrack(st, kPhaseAcquireId));
    cms::perfetto_tls::clear();
  }

  void preESModule(edm::eventsetup::EventSetupRecordKey const&, edm::ESModuleCallingContext const& cc) {
    if (!enabled_ || !perfetto::TrackEvent::IsEnabled()) return;

    auto top = cc.getTopModuleCallingContext();
    if (!top) return;
    if (top->type() != edm::ParentContext::Type::kPlaceInPath) return;

    auto const* pip = top->parent().placeInPathContext();
    if (!pip) return;
    auto const* pc = pip->pathContext();
    if (!pc) return;
    auto const* sc = pc->streamContext();
    if (!sc) return;

    unsigned sid = sc->streamID().value();
    auto& st = states_[sid];
    if (!st.active) return;

    cms::perfetto_tls::set(st.ids.event_uuid, st.ids.stream_child, kPhaseESId);

    auto const* cd = cc.componentDescription();  // pointer in your release
    const char* name = (cd && !cd->label_.empty()) ? cd->label_.c_str()
                                                   : (cd ? cd->type_.c_str() : "ESModule");

    TRACE_EVENT_BEGIN("cmssw.es", perfetto::DynamicString(name), phaseTrack(st, kPhaseESId));
  }

  void postESModule(edm::eventsetup::EventSetupRecordKey const&, edm::ESModuleCallingContext const& cc) {
    if (!enabled_ || !perfetto::TrackEvent::IsEnabled()) return;

    auto top = cc.getTopModuleCallingContext();
    if (!top) return;
    if (top->type() != edm::ParentContext::Type::kPlaceInPath) return;

    auto const* sc = top->parent().placeInPathContext()->pathContext()->streamContext();
    if (!sc) return;

    unsigned sid = sc->streamID().value();
    auto& st = states_[sid];
    if (!st.active) return;

    TRACE_EVENT_END("cmssw.es", phaseTrack(st, kPhaseESId));
    cms::perfetto_tls::clear();
  }

  void preClearEvent(edm::StreamContext const& sc) {
    if (!enabled_ || !perfetto::TrackEvent::IsEnabled()) return;
    auto& st = states_[sc.streamID().value()];
    if (!st.active) return;
    TRACE_EVENT_BEGIN("cmssw.cleanup", "Cleanup", phaseTrack(st, kPhaseCleanupId));
  }

  void postClearEvent(edm::StreamContext const& sc) {
    if (!enabled_ || !perfetto::TrackEvent::IsEnabled()) return;
    auto& st = states_[sc.streamID().value()];
    if (!st.active) return;

    TRACE_EVENT_END("cmssw.cleanup", phaseTrack(st, kPhaseCleanupId));
    TRACE_EVENT_END("cmssw.event", eventTrack(st));
    st.active = false;
    cms::perfetto_tls::clear();
  }

  void postEndJob() {
    if (!enabled_ || !session_) return;

    perfetto::TrackEvent::Flush();
    session_->StopBlocking();

    if (traceFd_ >= 0) {
      ::close(traceFd_);
      traceFd_ = -1;
    } else {
      auto trace_data = session_->ReadTraceBlocking();
      int fd = ::open(fileName_.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
      if (fd >= 0) {
        ::write(fd, trace_data.data(), trace_data.size());
        ::close(fd);
      }
    }
  }

  bool enabled_;
  std::string fileName_;
  unsigned bufferSizeKB_;
  unsigned maxEvents_;

  std::unique_ptr<perfetto::TracingSession> session_;
  int traceFd_ = -1;

  std::vector<PerStream> states_;
  std::atomic<unsigned> seenEvents_{0};
};

DEFINE_FWK_SERVICE(PerfettoTraceService);