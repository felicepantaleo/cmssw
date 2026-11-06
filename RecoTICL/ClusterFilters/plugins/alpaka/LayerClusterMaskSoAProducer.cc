// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"

#include <utility>
#include <vector>

#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/LayerClusterMaskDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "LayerClusterMaskAlgos.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The mask of the layer clusters after one or more iterations: the input mask (all 1 when inputMask is empty), with
  // 0 for each layer cluster of the tracksters.
  class LayerClusterMaskSoAProducer : public global::EDProducer<> {
  public:
    LayerClusterMaskSoAProducer(const edm::ParameterSet& config)
        : EDProducer(config),
          layerClustersToken_{consumes(config.getParameter<edm::InputTag>("layerClusters"))},
          maskToken_{produces()},
          useInputMask_{!config.getParameter<edm::InputTag>("inputMask").label().empty()} {
      if (useInputMask_) {
        inputMaskToken_ = consumes(config.getParameter<edm::InputTag>("inputMask"));
      }
      for (const auto& tag : config.getParameter<std::vector<edm::InputTag>>("tracksters")) {
        trackstersTokens_.emplace_back(consumes(tag));
      }
    }

    void produce(edm::StreamID, device::Event& iEvent, const device::EventSetup&) const override {
      const int nLayerClusters = iEvent.get(layerClustersToken_).const_view().position().metadata().size();
      ticl::LayerClusterMaskDeviceCollection mask(iEvent.queue(), nLayerClusters);
      if (useInputMask_) {
        const auto& inputMask = iEvent.get(inputMaskToken_);
        if (inputMask.const_view().metadata().size() != nLayerClusters) {
          throw edm::Exception(edm::errors::LogicError)
              << "LayerClusterMaskSoAProducer: the input mask has " << inputMask.const_view().metadata().size()
              << " values for " << nLayerClusters << " layer clusters.";
        }
        ticl::copyMask(iEvent.queue(), inputMask.const_view(), mask.view());
      } else {
        ticl::fillMaskWithOnes(iEvent.queue(), mask.view());
      }
      for (const auto& token : trackstersTokens_) {
        ticl::maskLayerClustersOfTracksters(iEvent.queue(), iEvent.get(token).const_view(), mask.view());
      }
      iEvent.emplace(maskToken_, std::move(mask));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalSoAMergeLayerClusters"))
          ->setComment("The layer clusters of the mask, which the vertices of the tracksters index.");
      desc.add<edm::InputTag>("inputMask", edm::InputTag())->setComment("The input mask. Empty: all 1.");
      desc.add<std::vector<edm::InputTag>>("tracksters", {})
          ->setComment("The tracksters whose layer clusters get 0 in the mask.");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDGetToken<reco::CaloClusterDeviceCollection> layerClustersToken_;
    device::EDGetToken<ticl::LayerClusterMaskDeviceCollection> inputMaskToken_;
    std::vector<device::EDGetToken<ticl::TracksterDeviceCollection>> trackstersTokens_;
    const device::EDPutToken<ticl::LayerClusterMaskDeviceCollection> maskToken_;
    const bool useInputMask_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(LayerClusterMaskSoAProducer);
