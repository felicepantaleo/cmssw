// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ElasticGateTestAcquirer
//
// An ExternalWork stream producer standing in for a module that offloads: acquire
// starts work that finishes on another thread, and produce reads the result. The
// per-stream buffer written by that work is checked for exclusive use, so the test
// fails if the gate hands the module's slot to another event across the gap.

#include <atomic>
#include <chrono>
#include <thread>

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edmtest {

  class ElasticGateTestAcquirer : public edm::stream::EDProducer<edm::ExternalWork> {
  public:
    explicit ElasticGateTestAcquirer(edm::ParameterSet const& pset)
        : asyncMicros_{pset.getParameter<unsigned int>("asyncMicros")} {
      produces<unsigned int>();
    }

    void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder holder) override {
      if (inUse_.exchange(true)) {
        throw cms::Exception("ElasticGateOverlap")
            << "two events entered the same instance of ElasticGateTestAcquirer at once";
      }
      buffer_ = 0;
      std::thread([this, holder]() mutable {
        std::this_thread::sleep_for(std::chrono::microseconds(asyncMicros_));
        buffer_ = 1;
        holder.doneWaiting(std::exception_ptr{});
      }).detach();
    }

    void produce(edm::Event& event, edm::EventSetup const&) override {
      if (buffer_ != 1) {
        throw cms::Exception("ElasticGateCorruption") << "produce saw a buffer the asynchronous work did not fill";
      }
      event.put(std::make_unique<unsigned int>(buffer_));
      inUse_ = false;
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<unsigned int>("asyncMicros", 2000);
      descriptions.addDefault(desc);
    }

  private:
    const unsigned int asyncMicros_;
    std::atomic<bool> inUse_{false};
    unsigned int buffer_ = 0;
  };

}  // namespace edmtest

using edmtest::ElasticGateTestAcquirer;
DEFINE_FWK_MODULE(ElasticGateTestAcquirer);
