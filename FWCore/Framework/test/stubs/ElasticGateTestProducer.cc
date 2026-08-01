// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ElasticGateTestProducer
//
// A stream producer that spends a configurable wall time in produce, used to
// check that the elastic gate converges on the concurrency a known load needs.
// busyMicros sets the time spent per event; with a known event rate that fixes
// the module's occupancy, and so the number of slots the gate should settle on.

#include <chrono>
#include <thread>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edmtest {

  class ElasticGateTestProducer : public edm::stream::EDProducer<> {
  public:
    explicit ElasticGateTestProducer(edm::ParameterSet const& pset)
        : busy_{pset.getParameter<unsigned int>("busyMicros")}, spin_{pset.getParameter<bool>("spin")} {
      produces<unsigned int>();
    }

    void produce(edm::Event& event, edm::EventSetup const&) override {
      const auto work = std::chrono::microseconds(busy_);
      if (spin_) {
        // Occupies a core, the way a real reconstruction module does.
        const auto until = std::chrono::steady_clock::now() + work;
        while (std::chrono::steady_clock::now() < until) {
        }
      } else {
        // Occupies no core, the way an module waiting on a device does.
        std::this_thread::sleep_for(work);
      }
      event.put(std::make_unique<unsigned int>(busy_));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<unsigned int>("busyMicros", 1000);
      desc.add<bool>("spin", true);
      descriptions.addDefault(desc);
    }

  private:
    const unsigned int busy_;
    const bool spin_;
  };

}  // namespace edmtest

using edmtest::ElasticGateTestProducer;
DEFINE_FWK_MODULE(ElasticGateTestProducer);
