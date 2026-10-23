#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <Eigen/Core>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceRunner.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename PIDAdapter, typename RegressionAdapter>
  class TracksterMultiTaskInferenceProducerT : public stream::FixedQueueEDProducer<> {
  public:
    using InputDevice = typename PIDAdapter::InputDevice;
    explicit TracksterMultiTaskInferenceProducerT(edm::ParameterSet const& config)
        : FixedQueueEDProducer<>(config),
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("tracksters"))},
          featuresToken_{consumes(config.getParameter<edm::InputTag>("features"))},
          minEnergy_{static_cast<float>(config.getParameter<double>("minClusterEnergy"))},
          outputToken_{produces()} {
      if (!std::isfinite(minEnergy_))
        throw cms::Exception("Configuration") << "minClusterEnergy must be finite";
      auto const batchSize = config.getParameter<unsigned int>("batchSize");
      if (batchSize == 0 || batchSize > static_cast<unsigned int>(std::numeric_limits<int32_t>::max()))
        throw cms::Exception("Configuration") << "Require 0 < batchSize <= INT32_MAX";
      if constexpr (!PIDAdapter::supportsRegression) {
        if (config.getParameter<bool>("doRegression"))
          throw cms::Exception("Configuration") << "CNN supports PID only";
      }
      if (config.getParameter<bool>("doPID"))
        pid_ = std::make_unique<TracksterInferenceRunner<PIDAdapter>>(
            edm::FileInPath(config.getParameter<std::string>("pidModel")).fullPath(),
            batchSize,
            config.getParameter<bool>("convertToFP16"),
            config.getParameter<uint32_t>("warmupIterations"));
      if (config.getParameter<bool>("doRegression"))
        regression_ = std::make_unique<TracksterInferenceRunner<RegressionAdapter>>(
            edm::FileInPath(config.getParameter<std::string>("regressionModel")).fullPath(),
            batchSize,
            config.getParameter<bool>("convertToFP16"),
            config.getParameter<uint32_t>("warmupIterations"));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription description;
      description.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersToSoAProducer"));
      description.add<edm::InputTag>("features");
      // Strings deliberately defer FileInPath resolution until an enabled task is constructed.
      description.add<std::string>("pidModel", "");
      description.add<std::string>("regressionModel", "");
      description.add<bool>("doPID", true);
      description.add<bool>("doRegression", PIDAdapter::supportsRegression);
      description.add<uint32_t>("batchSize", 64);
      description.add<bool>("convertToFP16", false);
      description.add<uint32_t>("warmupIterations", 3);
      description.add<double>("minClusterEnergy", 1.0);
      descriptions.addWithDefaultLabel(description);
    }

    void beginStream(edm::StreamID, Queue queue) override {
      if (pid_)
        pid_->warmup(queue);
      if (regression_)
        regression_->warmup(queue);
    }

    void produce(device::Event& event, device::EventSetup const&) override {
      auto const& tracksters = event.get(trackstersToken_);
      auto& queue = event.queue();
      auto output = PIDAdapter::makePublishedOutput(queue, tracksters);
      if (pid_ || regression_) {
        auto const& features = event.get(featuresToken_);
        auto total = static_cast<int32_t>(features.const_view().metadata().size());

        if constexpr (PIDAdapter::resetSelectedPredictions)
          resetTracksterPredictions(
              queue, std::get<0>(features.const_view().records().selectedTracksterIndex()).data(), output, total);
        if (regression_)
          regression_->runInto(queue, features, output, minEnergy_);
        if (pid_)
          pid_->runInto(queue, features, output);
      }
      event.emplace(outputToken_, std::move(output));
    }

  private:
    device::EDGetToken<ticl::TracksterDeviceCollection> const trackstersToken_;
    device::EDGetToken<InputDevice> const featuresToken_;
    float const minEnergy_;
    std::unique_ptr<TracksterInferenceRunner<PIDAdapter>> pid_;
    std::unique_ptr<TracksterInferenceRunner<RegressionAdapter>> regression_;
    device::EDPutToken<ticl::TracksterDeviceCollection> const outputToken_;
  };

  using TracksterCNNInferenceProducer = TracksterMultiTaskInferenceProducerT<CNNPIDAdapter, DNNEnergyAdapter>;
  using TracksterDNNInferenceProducer = TracksterMultiTaskInferenceProducerT<DNNPIDAdapter, DNNEnergyAdapter>;
  using TracksterPFNInferenceProducer = TracksterMultiTaskInferenceProducerT<PFNPIDAdapter, PFNEnergyAdapter>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterCNNInferenceProducer);
DEFINE_FWK_ALPAKA_MODULE(TracksterDNNInferenceProducer);
DEFINE_FWK_ALPAKA_MODULE(TracksterPFNInferenceProducer);
