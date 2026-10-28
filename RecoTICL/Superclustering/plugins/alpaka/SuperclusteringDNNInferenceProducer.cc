#include <memory>
#include <string>
#include <utility>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "RecoTICL/Superclustering/interface/SuperclusteringDNNInputs.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNNInferenceRunner.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Runs the superclustering DNN on the pair features: one score per pair, same row order.
  class SuperclusteringDNNInferenceProducer : public stream::FixedQueueEDProducer<> {
  public:
    explicit SuperclusteringDNNInferenceProducer(edm::ParameterSet const& config)
        : FixedQueueEDProducer<>(config),
          featuresToken_{consumes(config.getParameter<edm::InputTag>("features"))},
          runner_{edm::FileInPath(config.getParameter<std::string>("modelPath")).fullPath(),
                  ::ticl::makeSuperclusteringDNNInputFromString(config.getParameter<std::string>("dnnInputsVersion"))
                      ->featureCount(),
                  config.getParameter<uint32_t>("batchSize"),
                  config.getParameter<bool>("convertToFP16"),
                  config.getParameter<uint32_t>("warmupIterations")},
          scoresToken_{produces()} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("features", edm::InputTag("ticlSuperclusteringDNNPairFeatures", "features"));
      desc.add<std::string>("modelPath", "RecoTICL/Superclustering/data/supercls_v3.pt")
          ->setComment("TorchScript export of the superclustering DNN");
      desc.ifValue(edm::ParameterDescription<std::string>("dnnInputsVersion", "v3", true),
                   edm::allowedValues<std::string>("v1", "v2", "v3"));
      desc.add<uint32_t>("batchSize", 256)->setComment("Pairs per forward pass");
      desc.add<bool>("convertToFP16", false);
      desc.add<uint32_t>("warmupIterations", 3);
      descriptions.addWithDefaultLabel(desc);
    }

    void beginStream(edm::StreamID, Queue queue) override { runner_.warmup(queue); }

    void produce(device::Event& event, device::EventSetup const&) override {
      auto& queue = event.queue();
      auto const& features = event.get(featuresToken_);
      ticl::SuperclusteringDNNPairScoresDevice scores(queue, features.const_view().metadata().size());
      runner_.runInto(queue, features, scores);
      event.emplace(scoresToken_, std::move(scores));
    }

  private:
    device::EDGetToken<ticl::SuperclusteringDNNPairFeaturesDevice> const featuresToken_;
    SuperclusteringDNNInferenceRunner runner_;
    device::EDPutToken<ticl::SuperclusteringDNNPairScoresDevice> const scoresToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SuperclusteringDNNInferenceProducer);
