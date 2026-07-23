// Device Cornetto linking: consumes the portable trackster SoA and produces one
// connected-component label per trackster. It does not build merged tracksters;
// that stays on host in TracksterLinksFromComponentsProducer, which reuses the
// existing Trackster::mergeTracksters.
//
// A plain stream::EDProducer: the algorithm blocks on the queue twice internally
// (for the exact edge count, and once per label-propagation iteration). That is
// correct but not asynchronous; removing the second wait needs a fixed iteration
// count, and the first needs a two-pass-free edge allocation. Both are worth doing
// only once the profiling in the design doc says this stage matters.

#include "DataFormats/HGCalReco/interface/alpaka/TracksterComponentsDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterHostCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "CornettoLinkingAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TracksterLinkingByCornettoAlpakaProducer : public stream::EDProducer<> {
  public:
    TracksterLinkingByCornettoAlpakaProducer(edm::ParameterSet const& config)
        : EDProducer(config),
          trackstersToken_{consumes<TracksterHostCollection>(config.getParameter<edm::InputTag>("tracksterSoA"))},
          componentsToken_{produces()} {
      params_.etaWindow = config.getParameter<double>("etaWindow");
      params_.maxLongitudinalDistance = config.getParameter<double>("maxLongitudinalDistance");
      params_.transverseRadius0 = config.getParameter<double>("transverseRadius0");
      params_.transverseSlope = config.getParameter<double>("transverseSlope");
      params_.timeCompatibilityNSigma = config.getParameter<double>("timeCompatibilityNSigma");
    }

    void produce(device::Event& event, device::EventSetup const&) override {
      // Explicit host to device copy. The SoA is produced by a plain host
      // EDProducer, so there is no device product for the framework to hand over
      // and no implicit transfer: a device::EDGetToken on the device collection
      // finds nothing. Note this is invisible on the serial_sync backend, where
      // the device collection IS the host collection, so only a real GPU backend
      // exercises this path.
      auto const& hostTracksters = event.get(trackstersToken_);
      const int32_t n = hostTracksters.view().metadata().size();
      TracksterDeviceCollection deviceTracksters(event.queue(), n);
      alpaka::memcpy(event.queue(), deviceTracksters.buffer(), hostTracksters.buffer());

      TracksterComponentsDeviceCollection components(event.queue(), n);
      algo_.run(event.queue(), params_, deviceTracksters.const_view(), components.view());
      event.emplace(componentsToken_, std::move(components));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("tracksterSoA", edm::InputTag("tracksterSoACLUE3D"))
          ->setComment("Portable trackster SoA, from LegacyTracksterToSoAProducer.");
      desc.add<double>("etaWindow", 0.3)
          ->setComment("Barycenter |deta| candidate window; pairs farther apart are never tested.");
      desc.add<double>("maxLongitudinalDistance", 60.0)->setComment("Max |separation along the anchor axis| [cm].");
      desc.add<double>("transverseRadius0", 5.0)->setComment("Cone transverse radius at zero separation [cm].");
      desc.add<double>("transverseSlope", 0.05)->setComment("Cone opening: radius growth per cm of separation.");
      desc.add<double>("timeCompatibilityNSigma", 3.0)
          ->setComment("Max |time difference| in combined sigmas when both tracksters have valid time.");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    edm::EDGetTokenT<TracksterHostCollection> const trackstersToken_;
    device::EDPutToken<TracksterComponentsDeviceCollection> const componentsToken_;
    CornettoParameters params_;
    CornettoLinkingAlgoWrapper algo_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterLinkingByCornettoAlpakaProducer);
