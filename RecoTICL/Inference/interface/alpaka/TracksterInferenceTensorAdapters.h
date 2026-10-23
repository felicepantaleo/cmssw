#ifndef RecoTICL_Inference_alpaka_TracksterInferenceTensorAdapters_h
#define RecoTICL_Inference_alpaka_TracksterInferenceTensorAdapters_h

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <string>

#include <alpaka/alpaka.hpp>

#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/PyTorchAlpaka/interface/BatchedTensorCollection.h"
#include "RecoTICL/Inference/interface/TracksterInferenceSoA.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using InferenceTensors = cms::torch::alpakatools::BatchedTensorCollection<Queue>;

  // ---------------------------------------------------------------------------
  // Utilities
  // ---------------------------------------------------------------------------

  namespace detail {

    template <typename View>
    std::array<int32_t, ::ticl::TracksterSoA::blocksNumber> tracksterBlockSizes(const View& view) {
      return {static_cast<int32_t>(view.tracksters().metadata().size()),
              static_cast<int32_t>(view.vertices().keys()),
              static_cast<int32_t>(view.vertices().content().metadata().size()),
              static_cast<int32_t>(view.multiplicity().keys()),
              static_cast<int32_t>(view.multiplicity().content().metadata().size()),
              static_cast<int32_t>(view.edges().keys()),
              static_cast<int32_t>(view.edges().content().metadata().size()),
              static_cast<int32_t>(view.tracks().keys()),
              static_cast<int32_t>(view.tracks().content().metadata().size()),
              static_cast<int32_t>(view.tracksterGsfTrack().keys()),
              static_cast<int32_t>(view.tracksterGsfTrack().content().metadata().size()),
              static_cast<int32_t>(view.globalSeedingTracks().keys()),
              static_cast<int32_t>(view.globalSeedingTracks().content().metadata().size())};
    }

    template <typename SoA, typename... Columns>
    void addTensor(InferenceTensors& tensors,
                   const std::string& name,
                   std::optional<uint32_t> batchSize,
                   const Columns&... columns) {
      if (batchSize)
        tensors.addBatched<SoA>(name, *batchSize, columns...);
      else
        tensors.add<SoA>(name, columns...);
    }
  }  // namespace detail

  // ---------------------------------------------------------------------------
  // Input adapters
  // ---------------------------------------------------------------------------

  // Base class for the Input Adapters
  template <typename Device, typename SoA>
  struct InputAdapterBase {
    using InputDevice = Device;
    using InputSoA = SoA;

    template <typename Records>
    static uint32_t const* selectedIndices(const Records& records) {
      return std::get<0>(records.selectedTracksterIndex()).data();
    }
  };

  // CNN and DNN expose the same channels-first tensor layout.
  // Parameterize the common adapter to keep their model-specific types explicit.
  template <typename FeaturesDevice, typename FeaturesSoA, bool SupportsRegression>
  struct ImageInputAdapter : InputAdapterBase<FeaturesDevice, FeaturesSoA> {
    static constexpr bool supportsRegression = SupportsRegression;
    static constexpr bool resetSelectedPredictions = SupportsRegression;

    template <typename Records>
    static void registerTensors(InferenceTensors& inputs, Records const& records, std::optional<uint32_t> batchSize) {
      detail::addTensor<FeaturesSoA>(inputs, "input", batchSize, records.energy(), records.absEta(), records.phi());
    }
  };

  // CNN input: [batch, features, layers, clusters]
  using CNNInputAdapter = ImageInputAdapter<ticl::TracksterCNNFeaturesDevice, ::ticl::TracksterCNNFeaturesSoA, false>;
  // DNN input: [batch, features, layers, clusters]; any permutation belongs in the model.
  using DNNInputAdapter = ImageInputAdapter<ticl::TracksterDNNFeaturesDevice, ::ticl::TracksterDNNFeaturesSoA, true>;

  // PFN inputs:
  //   "input":            [batch, features, layers, clusters]
  //   "input_trackster_features": [batch, trackster_features]
  struct PFNInputAdapter : InputAdapterBase<ticl::TracksterPFNFeaturesDevice, ::ticl::TracksterPFNFeaturesSoA> {
    static constexpr bool supportsRegression = true;
    static constexpr bool resetSelectedPredictions = true;
    using FeaturesSoA = ::ticl::TracksterPFNFeaturesSoA;

    template <typename Records>
    static void registerTensors(InferenceTensors& inputs, Records const& records, std::optional<uint32_t> batchSize) {
      detail::addTensor<FeaturesSoA>(inputs,
                                     "input",
                                     batchSize,
                                     records.energy(),
                                     records.absEta(),
                                     records.phi(),
                                     records.x(),
                                     records.y(),
                                     records.absZ(),
                                     records.cells());

      detail::addTensor<FeaturesSoA>(inputs,
                                     "input_tr_features",
                                     batchSize,
                                     records.rawEnergy(),
                                     records.rawEmEnergy(),
                                     records.barycenterX(),
                                     records.barycenterY(),
                                     records.barycenterAbsZ(),
                                     records.barycenterAbsEta(),
                                     records.barycenterPhi());
    }
  };

  // ---------------------------------------------------------------------------
  // Output adapters
  // ---------------------------------------------------------------------------

  template <typename Device, typename SoA>
  struct OutputAdapterBase {
    using ModelOutputDevice = Device;
    using ModelOutputSoA = SoA;
    using PublishedSourceDevice = ticl::TracksterDeviceCollection;
    using PublishedOutputDevice = ticl::TracksterDeviceCollection;

    static PublishedOutputDevice makePublishedOutput(Queue& queue, const PublishedSourceDevice& original) {
      PublishedOutputDevice output(queue, detail::tracksterBlockSizes(original.const_view()));
      alpaka::memcpy(queue, output.buffer(), original.buffer());
      return output;
    }
  };

  // Model output: PID tensor [B, 8].
  // Published output: a copy of the original TracksterDevice with updated PID scores.
  // PID predictions: [batch, n_classes].
  struct PIDOutputAdapter
      : OutputAdapterBase<ticl::TracksterInferencePIDScoresDevice, ::ticl::TracksterInferencePIDScoresSoA> {
    using Base = OutputAdapterBase<ticl::TracksterInferencePIDScoresDevice, ::ticl::TracksterInferencePIDScoresSoA>;

    template <typename Records>
    static void registerTensors(InferenceTensors& outputs, const Records& records, std::optional<uint32_t> batchSize) {
      detail::addTensor<typename Base::ModelOutputSoA>(outputs,
                                                       "pid_output",
                                                       batchSize,
                                                       records.id_probabilities0(),
                                                       records.id_probabilities1(),
                                                       records.id_probabilities2(),
                                                       records.id_probabilities3(),
                                                       records.id_probabilities4(),
                                                       records.id_probabilities5(),
                                                       records.id_probabilities6(),
                                                       records.id_probabilities7());
    }

    static void applyResults(Queue& queue,
                             const uint32_t* selectedTracksters,
                             const ModelOutputDevice& predictions,
                             PublishedOutputDevice& published,
                             uint32_t total,
                             float minEnergy = 1.f);
  };

  // Energy regression predictions: [batch, 1].
  // The template parameter selects the PFN-specific low-energy fallback.
  template <bool PFN>
  struct EnergyOutputAdapter
      : OutputAdapterBase<ticl::TracksterInferenceEnergyDevice, ::ticl::TracksterInferenceEnergySoA> {
    using Base = OutputAdapterBase<ticl::TracksterInferenceEnergyDevice, ::ticl::TracksterInferenceEnergySoA>;

    template <typename Records>
    static void registerTensors(InferenceTensors& outputs, const Records& records, std::optional<uint32_t> batchSize) {
      detail::addTensor<typename Base::ModelOutputSoA>(outputs, "enreg_output", batchSize, records.energy());
    }

    static void applyResults(Queue& queue,
                             const uint32_t* selectedTracksters,
                             const ModelOutputDevice& predictions,
                             PublishedOutputDevice& published,
                             uint32_t total,
                             float minEnergy);
  };

  // ---------------------------------------------------------------------------
  // Prediction initialization
  // ---------------------------------------------------------------------------

  // CPU DNN/PFN reset both outputs for selected Tracksters before enabled tasks.
  void resetTracksterPredictions(Queue& queue,
                                 uint32_t const* selectedTracksters,
                                 ticl::TracksterDevice& published,
                                 uint32_t total);

  // ---------------------------------------------------------------------------
  // Combined inference adapter
  // ---------------------------------------------------------------------------

  // The inference producer deals only with this interface, independent of
  // which concrete SoA columns the adapters register.
  template <typename InputAdapter, typename OutputAdapter>
  struct TracksterInferenceTensorAdapter {
    static constexpr bool supportsRegression = InputAdapter::supportsRegression;
    static constexpr bool resetSelectedPredictions = InputAdapter::resetSelectedPredictions;
    using InputDevice = typename InputAdapter::InputDevice;
    using InputSoA = typename InputAdapter::InputSoA;
    using ModelOutputDevice = typename OutputAdapter::ModelOutputDevice;
    using ModelOutputSoA = typename OutputAdapter::ModelOutputSoA;
    using PublishedSourceDevice = typename OutputAdapter::PublishedSourceDevice;
    using PublishedOutputDevice = typename OutputAdapter::PublishedOutputDevice;

    static PublishedOutputDevice makePublishedOutput(Queue& queue, PublishedSourceDevice const& original) {
      return OutputAdapter::makePublishedOutput(queue, original);
    }

    template <typename Records>
    static void registerInputs(InferenceTensors& tensors, Records const& records, std::optional<uint32_t> batchSize) {
      InputAdapter::registerTensors(tensors, records, batchSize);
    }

    template <typename Records>
    static void registerOutputs(InferenceTensors& tensors, Records const& records, std::optional<uint32_t> batchSize) {
      OutputAdapter::registerTensors(tensors, records, batchSize);
    }

    template <typename FeatureRecords>
    static void applyResults(Queue& queue,
                             FeatureRecords const& featureRecords,
                             ModelOutputDevice const& predictions,
                             PublishedOutputDevice& published,
                             uint32_t total,
                             float minEnergy = 1.f) {
      OutputAdapter::applyResults(
          queue, InputAdapter::selectedIndices(featureRecords), predictions, published, total, minEnergy);
    }
  };

  // ---------------------------------------------------------------------------
  // Model-specific adapters
  // ---------------------------------------------------------------------------

  using CNNPIDAdapter = TracksterInferenceTensorAdapter<CNNInputAdapter, PIDOutputAdapter>;
  using DNNPIDAdapter = TracksterInferenceTensorAdapter<DNNInputAdapter, PIDOutputAdapter>;
  using PFNPIDAdapter = TracksterInferenceTensorAdapter<PFNInputAdapter, PIDOutputAdapter>;

  using DNNEnergyAdapter = TracksterInferenceTensorAdapter<DNNInputAdapter, EnergyOutputAdapter<false>>;
  using PFNEnergyAdapter = TracksterInferenceTensorAdapter<PFNInputAdapter, EnergyOutputAdapter<true>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Inference_alpaka_TracksterInferenceTensorAdapters_h
