// Necessary includes for an effective Producer file
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "DataFormats/NGTSoATest/interface/PortableCollectionSoATest.h"
#include <iostream>

class SoAProducer : public edm::global::EDProducer<> {

    public :
        explicit SoAProducer(const edm::ParameterSet&);
        ~SoAProducer();

        void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

    private:    
};

// Constructor
SoAProducer::SoAProducer(const edm::ParameterSet& iConfig) {
    // soaParameter_ = iConfig.getParameter<int>("soaParameter");
    // Product produced by the Producer
    produces<PhysicsObjCollection>("SoAProduct");
}

// Destructor
SoAProducer::~SoAProducer() {}

// Macro to label the label the module
DEFINE_FWK_MODULE(SoAProducer);

// Method to produce
void SoAProducer::produce(edm::StreamID iID, edm::Event& event, const edm::EventSetup& iSetup) const {
    std::size_t elems = 16;
    // std::array<int32_t, 2> sizes = {{16,0}};
    // SoA producer input
    auto SoAProduct = std::make_unique<PhysicsObjCollection>(elems, cms::alpakatools::host());

    // auto& view0 = SoAProduct -> view<0>();
    auto& view = SoAProduct -> view();

    for (int i = 0; i < view.metadata().size(); i++) {
          view.x()[i] = static_cast<double>(i);
          view.y()[i] = static_cast<double>(i) * 2.0;
          view.z()[i] = static_cast<double>(i) * 3.0;
    }
    view.detectorType() = 42;

    // Put the product in the event
    event.put(std::move(SoAProduct), "SoAProduct");
    std::cout << "Hey I work with SoA" << std::endl;
    printSoAView(view);
} 
