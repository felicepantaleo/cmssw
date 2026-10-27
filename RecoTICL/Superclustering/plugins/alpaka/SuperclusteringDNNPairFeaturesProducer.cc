#include <utility>
#include <vector>

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTICL/Superclustering/interface/SuperclusteringDNNInputs.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNNPairBuilder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  /*
   Builds the (seed, candidate) pairs of the superclustering DNN and their input features.

   The output size depends on the number of pairs found on the device: acquire() counts them and copies the count
   to the host, produce() allocates and fills the collections.

   Products:
     ""         SuperclusteringDNNPairsDevice         seed and candidate trackster indices of each pair
     "features" SuperclusteringDNNPairFeaturesDevice  DNN inputs of each pair, same row order
  */
  class SuperclusteringDNNPairFeaturesProducer : public stream::SynchronizingEDProducer<> {
  public:
    explicit SuperclusteringDNNPairFeaturesProducer(edm::ParameterSet const& config)
        : SynchronizingEDProducer<>(config),
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("tracksters"))},
          builder_{makeSelection(config)},
          pairsToken_{produces()},
          featuresToken_{produces("features")} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlSoATracksters"));
      // Same preselection parameters and defaults as TracksterLinkingbySuperClusteringDNN.
      desc.ifValue(edm::ParameterDescription<std::string>("dnnInputsVersion", "v3", true),
                   edm::allowedValues<std::string>("v1", "v2", "v3"));
      desc.add<double>("deltaEtaWindow", 0.1);
      desc.add<double>("deltaPhiWindow", 0.5);
      desc.add<double>("seedPtThreshold", 4.);
      desc.add<double>("candidateEnergyThreshold", 2.);
      desc.add<double>("explVarRatioCut_energyBoundary", 50.);
      desc.add<double>("explVarRatioMinimum_lowEnergy", 0.92);
      desc.add<double>("explVarRatioMinimum_highEnergy", 0.95);
      desc.add<bool>("filterByTracksterPID", true);
      desc.add<std::vector<int>>(
          "tracksterPIDCategoriesToFilter",
          {static_cast<int>(::ticl::Trackster::ParticleType::photon),
           static_cast<int>(::ticl::Trackster::ParticleType::electron)});
      desc.add<double>("PIDThreshold", 0.8);
      descriptions.addWithDefaultLabel(desc);
    }

    void acquire(device::Event const& event, device::EventSetup const&) override {
      builder_.prepare(event.queue(), event.get(trackstersToken_).const_view());
    }

    void produce(device::Event& event, device::EventSetup const&) override {
      auto& queue = event.queue();
      const auto nPairs = static_cast<int32_t>(builder_.pairCount());
      ticl::SuperclusteringDNNPairsDevice pairs(queue, nPairs);
      ticl::SuperclusteringDNNPairFeaturesDevice features(queue, nPairs);
      builder_.fill(queue, event.get(trackstersToken_).const_view(), pairs, features);
      event.emplace(pairsToken_, std::move(pairs));
      event.emplace(featuresToken_, std::move(features));
    }

  private:
    static SuperclusteringDNNPairSelection makeSelection(edm::ParameterSet const& config) {
      // TODO: fill from the ParameterSet; the feature count comes from the DNN inputs version
      SuperclusteringDNNPairSelection selection{};
      selection.featureCount =
          ::ticl::makeSuperclusteringDNNInputFromString(config.getParameter<std::string>("dnnInputsVersion"))
              ->featureCount();
      return selection;
    }

    device::EDGetToken<ticl::TracksterDeviceCollection> const trackstersToken_;
    SuperclusteringDNNPairBuilder builder_;
    device::EDPutToken<ticl::SuperclusteringDNNPairsDevice> const pairsToken_;
    device::EDPutToken<ticl::SuperclusteringDNNPairFeaturesDevice> const featuresToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SuperclusteringDNNPairFeaturesProducer);
