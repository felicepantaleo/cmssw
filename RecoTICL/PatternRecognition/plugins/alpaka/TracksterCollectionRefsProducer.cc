// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"

#include <vector>

#include "DataFormats/Common/interface/RefProd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The ordered list of trackster collections that defines the global index of a trackster, in the order of
  // 'tracksters'.
  class TracksterCollectionRefsProducer : public global::EDProducer<> {
  public:
    TracksterCollectionRefsProducer(const edm::ParameterSet& config) : EDProducer(config), refsToken_{produces()} {
      for (const auto& tag : config.getParameter<std::vector<edm::InputTag>>("tracksters")) {
        trackstersTokens_.emplace_back(consumes(tag));
      }
    }

    void produce(edm::StreamID, device::Event& iEvent, const device::EventSetup&) const override {
      ticl::TracksterCollectionRefs refs;
      refs.reserve(trackstersTokens_.size());
      for (const auto& token : trackstersTokens_) {
        refs.emplace_back(iEvent.getHandle(token));
      }
      iEvent.emplace(refsToken_, std::move(refs));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::vector<edm::InputTag>>("tracksters", {})->setComment("The trackster collections, in order.");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    std::vector<device::EDGetToken<ticl::TracksterDeviceCollection>> trackstersTokens_;
    const edm::EDPutTokenT<ticl::TracksterCollectionRefs> refsToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterCollectionRefsProducer);
