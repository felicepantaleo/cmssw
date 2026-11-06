#include <string>
#include <utility>
#include <vector>

#include "DataFormats/HGCalReco/interface/LayerClusterMaskHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"

// Converts a layer cluster mask SoA to the std::vector<float> mask of TICL. The SoA covers the first layer clusters of
// the host collection, which can have more: the values of the other layer clusters come from the reference mask.
class LayerClusterMaskFromSoAProducer : public edm::global::EDProducer<> {
public:
  LayerClusterMaskFromSoAProducer(const edm::ParameterSet& config)
      : maskSoAToken_(consumes(config.getParameter<edm::InputTag>("src"))),
        referenceToken_(consumes(config.getParameter<edm::InputTag>("referenceMask"))),
        maskToken_(produces(config.getParameter<std::string>("instance"))) {}

  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const override {
    const auto maskSoA = iEvent.get(maskSoAToken_).const_view();
    std::vector<float> mask = iEvent.get(referenceToken_);
    const auto n = static_cast<std::size_t>(maskSoA.metadata().size());
    if (n > mask.size()) {
      throw edm::Exception(edm::errors::LogicError) << "LayerClusterMaskFromSoAProducer: the mask SoA has " << n
                                                    << " values, the reference mask " << mask.size() << ".";
    }
    for (std::size_t i = 0; i < n; ++i) {
      mask[i] = maskSoA[i].mask();
    }
    iEvent.emplace(maskToken_, std::move(mask));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag());
    desc.add<edm::InputTag>("referenceMask", edm::InputTag("hgcalMergeLayerClusters", "InitialLayerClustersMask"))
        ->setComment("The mask of the host layer clusters that gives the size and the values after the SoA.");
    desc.add<std::string>("instance", "")->setComment("The instance name of the produced mask.");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const edm::EDGetTokenT<ticl::LayerClusterMaskHostCollection> maskSoAToken_;
  const edm::EDGetTokenT<std::vector<float>> referenceToken_;
  const edm::EDPutTokenT<std::vector<float>> maskToken_;
};

DEFINE_FWK_MODULE(LayerClusterMaskFromSoAProducer);
