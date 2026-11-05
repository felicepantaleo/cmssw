// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "CondFormats/HGCalObjects/interface/TICLGeomHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLayersHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLookupHost.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInference.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ticl;

  // Identifies the device tracksters with model M, and with a regression gives their energy: a copy of the tracksters
  // with the PID and the regressed energy of the tracksters that M selects. acquire() selects them, and produce(), with
  // their number known on the host, fills their inputs and runs the models in batches of batchSize.
  template <TracksterModel M>
  class TracksterInferenceProducer : public stream::SynchronizingEDProducer<edm::stream::WatchRuns> {
    using InputsDevice = TracksterModelInputsDevice<M>;

  public:
    explicit TracksterInferenceProducer(edm::ParameterSet const& config)
        : SynchronizingEDProducer(config),
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("tracksters"))},
          outputToken_{produces()},
          minEnergy_{static_cast<float>(config.getParameter<double>("minClusterEnergy"))},
          batchSize_{config.getParameter<uint32_t>("batchSize")},
          warmupIterations_{config.getParameter<uint32_t>("warmupIterations")},
          dtype_{config.getParameter<bool>("convertToFP16") ? ::torch::kHalf : ::torch::kFloat} {
      if (not std::isfinite(minEnergy_))
        throw cms::Exception("Configuration") << "minClusterEnergy must be finite";
      if (batchSize_ == 0 or batchSize_ > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
        throw cms::Exception("Configuration") << "Require 0 < batchSize <= INT32_MAX";
      if constexpr (M::layerClusters) {
        const edm::ESInputTag geometry("",
                                       config.getParameter<std::string>("detector") == "Barrel" ? "withBarrel" : "");
        layerClustersToken_ = consumes(config.getParameter<edm::InputTag>("layerClusters"));
        ticlGeomToken_ = esConsumes<TICLGeomHost, CaloGeometryRecord, edm::Transition::BeginRun>(geometry);
        ticlGeomLookupToken_ = esConsumes<TICLGeomLookupHost, CaloGeometryRecord, edm::Transition::BeginRun>(geometry);
        ticlGeomLayersToken_ =
            esConsumes<TICLGeomLayersHost, CaloGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag("", ""));
      }
      if (config.getParameter<bool>("doRegression") and not M::regression)
        throw cms::Exception("Configuration") << "this model has no energy regression";
      if (config.getParameter<bool>("doPID"))
        pid_.emplace(edm::FileInPath(config.getParameter<std::string>("pidModel")).fullPath());
      if (config.getParameter<bool>("doRegression"))
        regression_.emplace(edm::FileInPath(config.getParameter<std::string>("regressionModel")).fullPath());
      if (pid_)
        pid_->to(dtype_);
      if (regression_)
        regression_->to(dtype_);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlSoATracksters"));
      if constexpr (M::layerClusters) {
        desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalSoAMergeLayerClusters"));
        desc.add<std::string>("detector", "HGCAL");
      }
      // Strings, so that a disabled model needs no file
      desc.add<std::string>("pidModel", "");
      desc.add<std::string>("regressionModel", "");
      desc.add<bool>("doPID", true);
      desc.add<bool>("doRegression", M::regression);
      desc.add<uint32_t>("batchSize", 64);
      desc.add<bool>("convertToFP16", false);
      desc.add<uint32_t>("warmupIterations", 3)
          ->setComment("Inferences on empty inputs at the first event of a stream");
      desc.add<double>("minClusterEnergy", 1.0)->setComment(M::minEnergyComment);
      descriptions.addWithDefaultLabel(desc);
    }

    // The image models place the layer clusters by their layer in an endcap
    void beginRun(edm::Run const&, edm::EventSetup const& setup) override {
      if constexpr (M::layerClusters) {
        ticlgeom::Tools tools;
        tools.setGeometry(
            setup.getData(ticlGeomToken_), setup.getData(ticlGeomLookupToken_), setup.getData(ticlGeomLayersToken_));
        layersPerEndcap_ = tools.lastLayer(false);
        if (layersPerEndcap_ <= 0)
          throw cms::Exception("TracksterInferenceProducer") << "Invalid number of layers per endcap";
      }
    }

    void acquire(device::Event const& event, device::EventSetup const&) override {
      auto& queue = event.queue();
      if (not warmedUp_) {
        warmup(queue);
        warmedUp_ = true;
      }
      const auto tracksters = event.get(trackstersToken_).const_view();
      const uint32_t n = tracksters.energy().metadata().size();
      nSelected_ = cms::alpakatools::make_host_buffer<uint32_t>(queue);
      *nSelected_->data() = 0;
      if (n == 0)
        return;
      selected_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, n));
      rows_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, n));
      TracksterInferenceAlgo<M>::select(queue, context(event, tracksters), selected_->data(), rows_->data());
      alpaka::memcpy(queue,
                     cms::alpakatools::make_host_view(nSelected_->data(), 1u),
                     cms::alpakatools::make_device_view(queue, rows_->data() + n - 1, 1u));
    }

    void produce(device::Event& event, device::EventSetup const&) override {
      auto& queue = event.queue();
      const auto& input = event.get(trackstersToken_);
      const auto tracksters = input.const_view();
      ticl::TracksterDeviceCollection output(queue,
                                             tracksters.energy().metadata().size(),
                                             tracksters.barycenter().metadata().size(),
                                             tracksters.timing().metadata().size(),
                                             tracksters.pca().metadata().size(),
                                             tracksters.id().metadata().size(),
                                             tracksters.profile().metadata().size(),
                                             tracksters.vertices().offsets().metadata().size() - 1,
                                             tracksters.vertices().content().metadata().size());
      alpaka::memcpy(queue, output.buffer(), input.buffer());

      const uint32_t nRows = *nSelected_->data();
      if (nRows > 0 and (pid_ or regression_)) {
        InputsDevice inputs(queue, nRows);
        inputs.zeroInitialise(queue);  // the empty slots of the images
        auto trackster = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nRows);
        TracksterInferenceAlgo<M>::fill(
            queue, context(event, tracksters), selected_->data(), rows_->data(), trackster.data(), inputs.view());
        std::optional<TracksterPIDScoresDevice> pid;
        std::optional<TracksterEnergyDevice> energy;
        if (pid_) {
          pid.emplace(queue, nRows);
          run(queue, *pid_, inputs, *pid);
        }
        if (regression_) {
          energy.emplace(queue, nRows);
          run(queue, *regression_, inputs, *energy);
        }
        TracksterInferenceAlgo<M>::write(queue,
                                         trackster.data(),
                                         nRows,
                                         pid ? &*pid : nullptr,
                                         energy ? &*energy : nullptr,
                                         minEnergy_,
                                         output.view());
      }
      event.emplace(outputToken_, std::move(output));
      selected_.reset();
      rows_.reset();
    }

  private:
    reco::CaloClusterDeviceCollection::ConstView layerClusters(device::Event const& event) const {
      if constexpr (M::layerClusters)
        return event.get(layerClustersToken_).const_view();
      else
        return {};
    }

    TracksterModelContext context(device::Event const& event, ::ticl::TracksterSoA::ConstView tracksters) const {
      return {tracksters, layerClusters(event), static_cast<uint32_t>(layersPerEndcap_), minEnergy_};
    }

    // The outputs: the PID [batch, 8] or the energy [batch, 1]
    void addOutput(InferenceTensors& out, TracksterPIDScoresDevice& pid) const {
      auto r = pid.view().records();
      out.addBatched<::ticl::TracksterPIDScoresLayout<>>("pid_output",
                                                         batchSize_,
                                                         r.probability0(),
                                                         r.probability1(),
                                                         r.probability2(),
                                                         r.probability3(),
                                                         r.probability4(),
                                                         r.probability5(),
                                                         r.probability6(),
                                                         r.probability7());
    }
    void addOutput(InferenceTensors& out, TracksterEnergyDevice& energy) const {
      out.addBatched<::ticl::TracksterEnergyLayout<>>("enreg_output", batchSize_, energy.view().records().energy());
    }

    template <typename Output>
    void run(Queue& queue, torch::AlpakaModel& model, InputsDevice const& inputs, Output& output) const {
      InferenceTensors in, out;
      M::add(in, inputs.const_view().records(), batchSize_);
      addOutput(out, output);
      model.forward(queue, in, out, dtype_);
    }

    void warmup(Queue& queue) {
      if (warmupIterations_ == 0 or not(pid_ or regression_))
        return;
      InputsDevice inputs(queue, batchSize_);
      inputs.zeroInitialise(queue);
      TracksterPIDScoresDevice pid(queue, batchSize_);
      TracksterEnergyDevice energy(queue, batchSize_);
      for (uint32_t i = 0; i < warmupIterations_; ++i) {
        if (pid_)
          run(queue, *pid_, inputs, pid);
        if (regression_)
          run(queue, *regression_, inputs, energy);
      }
    }

    const device::EDGetToken<ticl::TracksterDeviceCollection> trackstersToken_;
    device::EDGetToken<reco::CaloClusterDeviceCollection> layerClustersToken_;
    edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
    edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
    edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;
    const device::EDPutToken<ticl::TracksterDeviceCollection> outputToken_;
    const float minEnergy_;
    const uint32_t batchSize_;
    const uint32_t warmupIterations_;
    const ::torch::Dtype dtype_;
    std::optional<torch::AlpakaModel> pid_;
    std::optional<torch::AlpakaModel> regression_;
    int32_t layersPerEndcap_ = 0;
    bool warmedUp_ = false;
    // from acquire() to produce(): the selected tracksters, their rows, and their number (pinned)
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> selected_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> rows_;
    std::optional<cms::alpakatools::host_buffer<uint32_t>> nSelected_;
  };

  using TracksterCNNInferenceProducer = TracksterInferenceProducer<TracksterCNN>;
  using TracksterDNNInferenceProducer = TracksterInferenceProducer<TracksterDNN>;
  using TracksterPFNInferenceProducer = TracksterInferenceProducer<TracksterPFN>;
  using TracksterShowerShapeInferenceProducer = TracksterInferenceProducer<TracksterShowerShape>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterCNNInferenceProducer);
DEFINE_FWK_ALPAKA_MODULE(TracksterDNNInferenceProducer);
DEFINE_FWK_ALPAKA_MODULE(TracksterPFNInferenceProducer);
DEFINE_FWK_ALPAKA_MODULE(TracksterShowerShapeInferenceProducer);
