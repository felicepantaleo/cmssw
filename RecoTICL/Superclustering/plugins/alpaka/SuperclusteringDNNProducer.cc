#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>

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
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "PhysicsTools/PyTorchAlpaka/interface/BatchedTensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNN.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNNDeviceCollections.h"

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
    using Tensors = cms::torch::alpakatools::BatchedTensorCollection<Queue>;

  public:
    explicit SuperclusteringDNNProducer(const edm::ParameterSet& config)
        : SynchronizingFixedQueueEDProducer<>(config),
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("tracksters"))},
          selection_{makeSelection(config)},
          batchSize_{config.getParameter<uint32_t>("batchSize")},
          warmupIterations_{config.getParameter<uint32_t>("warmupIterations")},
          dtype_{config.getParameter<bool>("convertToFP16") ? ::torch::kHalf : ::torch::kFloat},
          model_{edm::FileInPath(config.getParameter<std::string>("modelPath")).fullPath()},
          pairsToken_{produces()},
          scoresToken_{produces("scores")} {
      // TensorCollection takes int sizes
      if (batchSize_ == 0 || batchSize_ > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        throw cms::Exception("Configuration") << "SuperclusteringDNNProducer: batchSize must be in [1, 2^31)";
      }
      // Set the precision on the CPU; the first forward pass moves the model to the device of the queue.
      model_.to(dtype_);
    }

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

    void beginStream(edm::StreamID, Queue queue) override { warmup(queue); }

    void acquire(const device::Event& event, const device::EventSetup&) override {
      auto& queue = event.queue();
      const auto tracksters = event.get(trackstersToken_).const_view();
      const uint32_t n = tracksters.energy().metadata().size();
      nPairs_ = cms::alpakatools::make_host_buffer<uint32_t>(queue);
      *nPairs_->data() = 0;
      if (n == 0)
        return;
      search_.emplace(queue, n);
      ticl::SuperclusteringDNNAlgo::findPairs(queue, tracksters, selection_, *search_);
      alpaka::memcpy(queue, *nPairs_, search_->nPairs);
    }

    void produce(device::Event& event, const device::EventSetup&) override {
      auto& queue = event.queue();
      const auto nPairs = static_cast<int32_t>(*nPairs_->data());
      ticl::SuperclusteringDNNPairsDevice pairs(queue, nPairs);
      ticl::SuperclusteringDNNPairFeaturesDevice features(queue, nPairs);
      ticl::SuperclusteringDNNPairScoresDevice scores(queue, nPairs);
      if (nPairs > 0) {
        ticl::SuperclusteringDNNAlgo::fillPairs(queue, selection_, *search_, pairs.view(), features.view());
        run(queue, features, scores);
      }
      event.emplace(pairsToken_, std::move(pairs));
      event.emplace(scoresToken_, std::move(scores));
      search_.reset();
      nPairs_.reset();
    }

  private:
    // The forward pass in batches of batchSize pairs: input [pairs, 18], output [pairs, 1]
    void run(Queue& queue,
             const ticl::SuperclusteringDNNPairFeaturesDevice& features,
             ticl::SuperclusteringDNNPairScoresDevice& scores) {
      const auto f = features.const_view().records();
      Tensors in, out;
      in.addBatched<::ticl::SuperclusteringDNNPairFeaturesSoA>("input",
                                                               batchSize_,
                                                               f.deltaEtaBaryc(),
                                                               f.deltaPhiBaryc(),
                                                               f.multiEn(),
                                                               f.multiEta(),
                                                               f.multiPt(),
                                                               f.seedEta(),
                                                               f.seedPhi(),
                                                               f.seedEn(),
                                                               f.seedPt(),
                                                               f.theta(),
                                                               f.thetaXZSeedFrame(),
                                                               f.thetaYZSeedFrame(),
                                                               f.thetaXYCmsFrame(),
                                                               f.thetaYZCmsFrame(),
                                                               f.thetaXZCmsFrame(),
                                                               f.explVar(),
                                                               f.explVarRatio(),
                                                               f.modDeltaTime());
      out.addBatched<::ticl::SuperclusteringDNNPairScoresSoA>("output", batchSize_, scores.view().records().score());
      model_.forward(queue, in, out, dtype_);
    }

    // A full batch of zeros: the first passes on a device load the model and initialise PyTorch's kernels, which would
    // otherwise land in the first events.
    void warmup(Queue& queue) {
      if (warmupIterations_ == 0)
        return;
      ticl::SuperclusteringDNNPairFeaturesDevice features(queue, batchSize_);
      features.zeroInitialise(queue);
      ticl::SuperclusteringDNNPairScoresDevice scores(queue, batchSize_);
      for (uint32_t i = 0; i < warmupIterations_; ++i)
        run(queue, features, scores);
    }

    static ticl::SuperclusteringDNNPairSelection makeSelection(const edm::ParameterSet& config) {
      ticl::SuperclusteringDNNPairSelection selection{};
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
      if (categories.size() > ticl::SuperclusteringDNNPairSelection::kMaxPIDCategories) {
        throw cms::Exception("Configuration")
            << "SuperclusteringDNNProducer: at most " << ticl::SuperclusteringDNNPairSelection::kMaxPIDCategories
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

    const device::EDGetToken<ticl::TracksterDeviceCollection> trackstersToken_;
    const ticl::SuperclusteringDNNPairSelection selection_;
    const uint32_t batchSize_;
    const uint32_t warmupIterations_;
    const ::torch::Dtype dtype_;
    torch::AlpakaModel model_;
    const device::EDPutToken<ticl::SuperclusteringDNNPairsDevice> pairsToken_;
    const device::EDPutToken<ticl::SuperclusteringDNNPairScoresDevice> scoresToken_;
    // from acquire() to produce(): the device buffers of the pairs, and their number (pinned)
    std::optional<ticl::SuperclusteringDNNPairSearch> search_;
    std::optional<cms::alpakatools::host_buffer<uint32_t>> nPairs_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SuperclusteringDNNProducer);
