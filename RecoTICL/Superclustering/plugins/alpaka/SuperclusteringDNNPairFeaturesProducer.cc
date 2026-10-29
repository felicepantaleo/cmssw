#include <string>
#include <utility>
#include <vector>

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTICL/Superclustering/interface/SuperclusteringDNNInputs.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNNPairBuilder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  /*
   Builds the (seed, candidate) pairs of the superclustering DNN and their input features, from the device
   tracksters.

   The output size depends on the number of pairs found on the device: acquire() counts them and copies the count
   to the host, produce() allocates and fills the collections. A SynchronizingEDProducer is what lets acquire()
   enqueue device work and have the framework wait for it, without blocking a thread, before produce().

   Products, with the same rows:
     ""         SuperclusteringDNNPairsDevice         the seed and candidate trackster indices of each pair
     "features" SuperclusteringDNNPairFeaturesDevice  the DNN inputs of each pair
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
      desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlSoATracksters"))
          ->setComment("Device tracksters, in the order of the legacy tracksters that the linking reads");
      // The preselection of TracksterLinkingbySuperClusteringDNN, with its types and defaults.
      desc.ifValue(edm::ParameterDescription<std::string>("dnnInputsVersion", "v3", true),
                   edm::allowedValues<std::string>("v1", "v2", "v3"));
      desc.add<float>("deltaEtaWindow", 0.1);
      desc.add<float>("deltaPhiWindow", 0.5);
      desc.add<float>("seedPtThreshold", 4.);
      desc.add<float>("candidateEnergyThreshold", 2.);
      desc.add<float>("explVarRatioCut_energyBoundary", 50.);
      desc.add<float>("explVarRatioMinimum_lowEnergy", 0.92);
      desc.add<float>("explVarRatioMinimum_highEnergy", 0.95);
      desc.add<bool>("filterByTracksterPID", true);
      desc.add<std::vector<int>>("tracksterPIDCategoriesToFilter",
                                 {static_cast<int>(::ticl::Trackster::ParticleType::photon),
                                  static_cast<int>(::ticl::Trackster::ParticleType::electron)});
      desc.add<float>("PIDThreshold", 0.8f);
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
      builder_.fill(queue, pairs, features);
      event.emplace(pairsToken_, std::move(pairs));
      event.emplace(featuresToken_, std::move(features));
    }

  private:
    static SuperclusteringDNNPairSelection makeSelection(edm::ParameterSet const& config) {
      SuperclusteringDNNPairSelection selection{};
      selection.featureCount =
          ::ticl::makeSuperclusteringDNNInputFromString(config.getParameter<std::string>("dnnInputsVersion"))
              ->featureCount();
      selection.deltaEtaWindow = config.getParameter<float>("deltaEtaWindow");
      selection.deltaPhiWindow = config.getParameter<float>("deltaPhiWindow");
      selection.seedPtThreshold = config.getParameter<float>("seedPtThreshold");
      selection.candidateEnergyThreshold = config.getParameter<float>("candidateEnergyThreshold");
      selection.explVarRatioCut_energyBoundary = config.getParameter<float>("explVarRatioCut_energyBoundary");
      selection.explVarRatioMinimum_lowEnergy = config.getParameter<float>("explVarRatioMinimum_lowEnergy");
      selection.explVarRatioMinimum_highEnergy = config.getParameter<float>("explVarRatioMinimum_highEnergy");
      selection.filterByTracksterPID = config.getParameter<bool>("filterByTracksterPID");
      selection.PIDThreshold = config.getParameter<float>("PIDThreshold");

      const auto categories = config.getParameter<std::vector<int>>("tracksterPIDCategoriesToFilter");
      if (categories.size() > SuperclusteringDNNPairSelection::kMaxPIDCategories) {
        throw cms::Exception("Configuration")
            << "SuperclusteringDNNPairFeaturesProducer: at most " << SuperclusteringDNNPairSelection::kMaxPIDCategories
            << " tracksterPIDCategoriesToFilter";
      }
      selection.nPIDCategories = categories.size();
      for (unsigned int i = 0; i < categories.size(); ++i) {
        if (categories[i] < 0 || categories[i] >= static_cast<int>(::ticl::Trackster::ParticleType::unknown) + 1) {
          throw cms::Exception("Configuration")
              << "SuperclusteringDNNPairFeaturesProducer: no PID category " << categories[i];
        }
        selection.pidCategories[i] = categories[i];
      }
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
