#ifndef RecoTICL_Superclustering_alpaka_SuperclusteringDNNTensorAdapter_h
#define RecoTICL_Superclustering_alpaka_SuperclusteringDNNTensorAdapter_h

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpaka/interface/BatchedTensorCollection.h"
#include "RecoTICL/Superclustering/interface/SuperclusteringDNNSoA.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNNDeviceCollections.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using SuperclusteringTensors = cms::torch::alpakatools::BatchedTensorCollection<Queue>;

  // Maps the pair-feature and pair-score SoA columns onto the model tensors.
  // Same role as TracksterInferenceTensorAdapter in the CNN PID port; the inference runner deals only with this.
  //   input:  [pairs, featureCount]   (featureCount = 9, 17 or 18 for v1, v2, v3)
  //   output: [pairs, 1]
  struct SuperclusteringDNNTensorAdapter {
    using InputDevice = ticl::SuperclusteringDNNPairFeaturesDevice;
    using InputSoA = ::ticl::SuperclusteringDNNPairFeaturesSoA;
    using ModelOutputDevice = ticl::SuperclusteringDNNPairScoresDevice;
    using ModelOutputSoA = ::ticl::SuperclusteringDNNPairScoresSoA;

    template <typename Records>
    static void registerInputs(SuperclusteringTensors& tensors,
                               Records const& records,
                               uint32_t batchSize,
                               unsigned int featureCount) {
      switch (featureCount) {
        case 18:  // v3
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
          break;
        case 17:  // v2: the first 17 columns
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
                                records.explVarRatio());
          break;
        case 9:  // v1: the first 9 columns
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
                                records.seedPt());
          break;
        default:
          throw cms::Exception("SuperclusteringDNNTensorAdapter")
              << "No tensor registration for " << featureCount << " features";
      }
    }

    template <typename Records>
    static void registerOutputs(SuperclusteringTensors& tensors, Records const& records, uint32_t batchSize) {
      tensors.addBatched<ModelOutputSoA>("output", batchSize, records.score());
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Superclustering_alpaka_SuperclusteringDNNTensorAdapter_h
