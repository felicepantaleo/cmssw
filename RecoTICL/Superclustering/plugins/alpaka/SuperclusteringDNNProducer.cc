#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingFixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNNInferenceRunner.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNNPairBuilder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  /*
   Builds the (seed, candidate) pairs of the superclustering DNN from the device tracksters and scores them with the
   DNN. acquire() counts the pairs, and produce(), with their number known on the host, fills them and their features
   and runs the model in batches of batchSize.

   Products, with the same rows:
     ""       SuperclusteringDNNPairsDevice       the seed and candidate trackster indices of each pair
     "scores" SuperclusteringDNNPairScoresDevice  the DNN score of each pair
  */
  class SuperclusteringDNNProducer : public stream::SynchronizingFixedQueueEDProducer<> {
  public:
    explicit SuperclusteringDNNProducer(edm::ParameterSet const& config)
        : SynchronizingFixedQueueEDProducer<>(config),
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("tracksters"))},
          builder_{makeSelection(config)},
          runner_{edm::FileInPath(config.getParameter<std::string>("modelPath")).fullPath(),
                  config.getParameter<uint32_t>("batchSize"),
                  config.getParameter<bool>("convertToFP16"),
                  config.getParameter<uint32_t>("warmupIterations")},
          pairsToken_{produces()},
          scoresToken_{produces("scores")} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlSoATracksters"))
          ->setComment("Device tracksters, in the order of the legacy tracksters that the linking reads");
      desc.add<std::string>("modelPath", "RecoTICL/Superclustering/data/supercls_v3.pt")
          ->setComment("TorchScript export of the superclustering DNN");
      desc.add<uint32_t>("batchSize", 256)->setComment("Pairs per forward pass");
      desc.add<bool>("convertToFP16", false);
      desc.add<uint32_t>("warmupIterations", 3);
      // The preselection of TracksterLinkingbySuperClusteringDNN, with its types and defaults.
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

    void beginStream(edm::StreamID, Queue queue) override { runner_.warmup(queue); }

    void acquire(device::Event const& event, device::EventSetup const&) override {
      builder_.prepare(event.queue(), event.get(trackstersToken_).const_view());
    }

    void produce(device::Event& event, device::EventSetup const&) override {
      auto& queue = event.queue();
      const auto nPairs = static_cast<int32_t>(builder_.pairCount());
      ticl::SuperclusteringDNNPairsDevice pairs(queue, nPairs);
      ticl::SuperclusteringDNNPairFeaturesDevice features(queue, nPairs);
      ticl::SuperclusteringDNNPairScoresDevice scores(queue, nPairs);
      builder_.fill(queue, pairs, features);
      runner_.runInto(queue, features, scores);
      event.emplace(pairsToken_, std::move(pairs));
      event.emplace(scoresToken_, std::move(scores));
    }

  private:
    static SuperclusteringDNNPairSelection makeSelection(edm::ParameterSet const& config) {
      SuperclusteringDNNPairSelection selection{};
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
            << "SuperclusteringDNNProducer: at most " << SuperclusteringDNNPairSelection::kMaxPIDCategories
            << " tracksterPIDCategoriesToFilter";
      }
      selection.nPIDCategories = categories.size();
      for (unsigned int i = 0; i < categories.size(); ++i) {
        if (categories[i] < 0 || categories[i] >= static_cast<int>(::ticl::Trackster::ParticleType::unknown) + 1) {
          throw cms::Exception("Configuration") << "SuperclusteringDNNProducer: no PID category " << categories[i];
        }
        selection.pidCategories[i] = categories[i];
      }
      return selection;
    }

    device::EDGetToken<ticl::TracksterDeviceCollection> const trackstersToken_;
    SuperclusteringDNNPairBuilder builder_;
    SuperclusteringDNNInferenceRunner runner_;
    device::EDPutToken<ticl::SuperclusteringDNNPairsDevice> const pairsToken_;
    device::EDPutToken<ticl::SuperclusteringDNNPairScoresDevice> const scoresToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SuperclusteringDNNProducer);
