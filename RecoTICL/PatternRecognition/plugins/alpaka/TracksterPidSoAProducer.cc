// TracksterHostCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/HGCalReco/interface/TracksterHostCollection.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TracksterPidHostCollection.h"
#include "DataFormats/Portable/interface/PortableCollection.h"  // registers the implicit copy to the device
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The PID probabilities of the tracksters of a collection as a portable SoA in the host memory, in the order of the
  // collection, for the device modules that read them: the module registers the implicit copy to the device. The
  // tracksters must be the host conversion of 'reference', so the two have the same size.
  class TracksterPidSoAProducer : public global::EDProducer<> {
  public:
    TracksterPidSoAProducer(const edm::ParameterSet& config)
        : EDProducer(config),
          srcToken_(consumes(config.getParameter<edm::InputTag>("src"))),
          referenceToken_(consumes(config.getParameter<edm::InputTag>("reference"))),
          pidToken_(produces()) {}

    void produce(edm::StreamID, device::Event& iEvent, const device::EventSetup&) const override {
      const auto& tracksters = iEvent.get(srcToken_);
      const auto& reference = iEvent.get(referenceToken_);
      const auto n = tracksters.size();
      if (n != static_cast<std::size_t>(reference.const_view().energy().metadata().size())) {
        throw edm::Exception(edm::errors::LogicError)
            << "TracksterPidSoAProducer: " << n << " tracksters, the reference SoA has "
            << reference.const_view().energy().metadata().size() << ".";
      }
      ::ticl::TracksterPidHostCollection pid(iEvent.queue(), static_cast<int32_t>(n));
      auto view = pid.view();
      for (std::size_t i = 0; i < n; ++i) {
        for (int k = 0; k < ::ticl::TracksterIdProbabilities::RowsAtCompileTime; ++k) {
          view[i].idProbabilities()(k) = tracksters[i].id_probabilities()[k];
        }
      }
      iEvent.emplace(pidToken_, std::move(pid));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src", edm::InputTag("ticlTrackstersCLUEstering"))
          ->setComment("The tracksters whose PID probabilities the SoA holds.");
      desc.add<edm::InputTag>("reference", edm::InputTag("ticlSoATracksters"))
          ->setComment("The trackster SoA that the SoA of probabilities describes. Its size must equal that of src.");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const edm::EDGetTokenT<std::vector<::ticl::Trackster>> srcToken_;
    const edm::EDGetTokenT<::ticl::TracksterHostCollection> referenceToken_;
    const edm::EDPutTokenT<::ticl::TracksterPidHostCollection> pidToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterPidSoAProducer);
