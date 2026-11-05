#ifndef RecoTICL_Superclustering_alpaka_SuperclusteringDNNTensorAdapter_h
#define RecoTICL_Superclustering_alpaka_SuperclusteringDNNTensorAdapter_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpaka/interface/BatchedTensorCollection.h"
#include "RecoTICL/Superclustering/interface/SuperclusteringDNNSoA.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNNDeviceCollections.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using SuperclusteringTensors = cms::torch::alpakatools::BatchedTensorCollection<Queue>;

  // Maps the pair-feature and pair-score SoA columns onto the model tensors.
  //   input:  [pairs, 18]
  //   output: [pairs, 1]
  struct SuperclusteringDNNTensorAdapter {
    using InputDevice = ticl::SuperclusteringDNNPairFeaturesDevice;
    using InputSoA = ::ticl::SuperclusteringDNNPairFeaturesSoA;
    using ModelOutputDevice = ticl::SuperclusteringDNNPairScoresDevice;
    using ModelOutputSoA = ::ticl::SuperclusteringDNNPairScoresSoA;

    template <typename Records>
    static void registerInputs(SuperclusteringTensors& tensors, Records const& records, uint32_t batchSize) {
      tensors.addBatched<InputSoA>("input",
                                   batchSize,
                                   records.deltaEtaBaryc(),
                                   records.deltaPhiBaryc(),
                                   records.multiEn(),
                                   records.multiEta(),
                                   records.multiPt(),
                                   records.seedEta(),
                                   records.seedPhi(),
                                   records.seedEn(),
                                   records.seedPt(),
                                   records.theta(),
                                   records.thetaXZSeedFrame(),
                                   records.thetaYZSeedFrame(),
                                   records.thetaXYCmsFrame(),
                                   records.thetaYZCmsFrame(),
                                   records.thetaXZCmsFrame(),
                                   records.explVar(),
                                   records.explVarRatio(),
                                   records.modDeltaTime());
    }

    template <typename Records>
    static void registerOutputs(SuperclusteringTensors& tensors, Records const& records, uint32_t batchSize) {
      tensors.addBatched<ModelOutputSoA>("output", batchSize, records.score());
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Superclustering_alpaka_SuperclusteringDNNTensorAdapter_h
