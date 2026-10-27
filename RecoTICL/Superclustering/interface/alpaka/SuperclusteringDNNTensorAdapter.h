#ifndef RecoTICL_Superclustering_alpaka_SuperclusteringDNNTensorAdapter_h
#define RecoTICL_Superclustering_alpaka_SuperclusteringDNNTensorAdapter_h

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "RecoTICL/Superclustering/interface/SuperclusteringDNNSoA.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNNDeviceCollections.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using SuperclusteringTensors = cms::torch::alpakatools::TensorCollection<Queue>;

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
                               int batchId,
                               unsigned int featureCount) {
      switch (featureCount) {
        case 18:  // v3
          tensors.add<InputSoA>("input",
                                batchId,
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
        // TODO: v1 (first 9 columns) and v2 (first 17 columns)
        default:
          throw cms::Exception("SuperclusteringDNNTensorAdapter")
              << "No tensor registration for " << featureCount << " features";
      }
    }

    template <typename Records>
    static void registerOutputs(SuperclusteringTensors& tensors, Records const& records, int batchId) {
      tensors.add<ModelOutputSoA>("output", batchId, records.score());
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Superclustering_alpaka_SuperclusteringDNNTensorAdapter_h
