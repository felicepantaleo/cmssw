// Necessary includes
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "DataFormats/NGTSoATest/interface/PortableCollectionSoATest.h"
#include <iostream>

#define TIME 0.01

class SoAProducer2 : public edm::global::EDProducer<> {
public:
  explicit SoAProducer2(const edm::ParameterSet&);
  ~SoAProducer2();

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  edm::EDGetTokenT<PhysicsObjCollection> inputToken_;
};

// Constructor
SoAProducer2::SoAProducer2(const edm::ParameterSet& iConfig)
    : inputToken_(consumes<PhysicsObjCollection>(iConfig.getParameter<edm::InputTag>("soaInput"))) {
  produces<PhysicsObjExtraCollection>("SoAProduct2");
}

// Destructor
SoAProducer2::~SoAProducer2() {}

// Macro to label the module
DEFINE_FWK_MODULE(SoAProducer2);

//SOA_COLUMN(double, k)

// Method to produce in SoAProducer2
void SoAProducer2::produce(edm::StreamID iID, edm::Event& event, const edm::EventSetup& iSetup) const {
  auto const& soaInputHandle = event.getHandle(inputToken_);
  const auto& view = soaInputHandle->view();

  // // Copy input and modify it
  auto elems = view.metadata().size();
  auto modifiedSoA = std::make_unique<PhysicsObjExtraCollection>(elems, cms::alpakatools::host());
  auto& modifiedView = modifiedSoA->view();

  for (int i = 0; i < modifiedView.metadata().size(); i++) {
    modifiedView.eigenvector_1()[i] = view.x()[i] / TIME;
    modifiedView.eigenvector_2()[i] = view.y()[i] / TIME;
    modifiedView.eigenvector_3()[i] = view.z()[i] / TIME;
    modifiedView.eigenvalues()[i] = view.x()[i] * TIME;
    for (int j = 0; j < 3; j++)
      modifiedView[i].candidateDirection()(j) = view.y()[i] / (TIME * (j + 1));
  }

  // Put the modified product in the event
  event.put(std::move(modifiedSoA), "SoAProduct2");

  std::cout << "Hey, I modified SoA!" << std::endl;
  //printSoAView(view);
  printSoAView(modifiedView);
}