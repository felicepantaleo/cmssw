// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/LayerClusterMaskDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTICL/Common/interface/alpaka/TracksterCollections.h"

#include "LayerClusterMaskAlgos.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The mask of the layer clusters after one or more iterations: the input mask (all 1 when inputMask is empty), with
  // 0 for each layer cluster of the tracksters. With the mask, the module puts the ordered list of the trackster
  // collections whose layer clusters the mask sets to 0: those of the input mask, then those of trackstersRefs.
  class LayerClusterMaskSoAProducer : public global::EDProducer<> {
  public:
    LayerClusterMaskSoAProducer(const edm::ParameterSet& config)
        : EDProducer(config),
          layerClustersToken_{consumes(config.getParameter<edm::InputTag>("layerClusters"))},
          maskToken_{produces()},
          refsToken_{produces()},
          useInputMask_{!config.getParameter<edm::InputTag>("inputMask").label().empty()} {
      if (useInputMask_) {
        inputMaskToken_ = consumes(config.getParameter<edm::InputTag>("inputMask"));
        inputRefsToken_ = consumes(config.getParameter<edm::InputTag>("inputMask"));
      }
      const auto& tracksters = config.getParameter<std::vector<edm::InputTag>>("tracksters");
      for (const auto& tag : tracksters) {
        trackstersTokens_.emplace_back(consumes(tag));
      }
      if (!tracksters.empty()) {
        trackstersRefsToken_ = consumes(config.getParameter<edm::InputTag>("trackstersRefs"));
      }
    }

    void produce(edm::StreamID, device::Event& iEvent, const device::EventSetup&) const override {
      const int nLayerClusters = iEvent.get(layerClustersToken_).const_view().position().metadata().size();
      ticl::LayerClusterMaskDeviceCollection mask(iEvent.queue(), nLayerClusters);
      ticl::TracksterCollectionRefs refs;
      if (useInputMask_) {
        const auto& inputMask = iEvent.get(inputMaskToken_);
        if (inputMask.const_view().metadata().size() != nLayerClusters) {
          throw edm::Exception(edm::errors::LogicError)
              << "LayerClusterMaskSoAProducer: the input mask has " << inputMask.const_view().metadata().size()
              << " values for " << nLayerClusters << " layer clusters.";
        }
        ticl::copyMask(iEvent.queue(), inputMask.const_view(), mask.view());
        refs = iEvent.get(inputRefsToken_);
      } else {
        ticl::fillMaskWithOnes(iEvent.queue(), mask.view());
      }
      if (!trackstersTokens_.empty()) {
        const auto& trackstersRefs = iEvent.get(trackstersRefsToken_);
        const auto tracksters =
            ticl::orderedTracksterCollections(iEvent, trackstersRefs, trackstersTokens_, "LayerClusterMaskSoAProducer");
        for (std::size_t i = 0; i < tracksters.size(); ++i) {
          if (std::ranges::any_of(refs, [&](const auto& ref) { return ref.id() == trackstersRefs[i].id(); })) {
            throw edm::Exception(edm::errors::LogicError)
                << "LayerClusterMaskSoAProducer: the input mask already sets to 0 the layer clusters of trackster "
                   "collection "
                << i << " of trackstersRefs.";
          }
          ticl::maskLayerClustersOfTracksters(iEvent.queue(), tracksters[i]->const_view(), mask.view());
          refs.push_back(trackstersRefs[i]);
        }
      }
      iEvent.emplace(maskToken_, std::move(mask));
      iEvent.emplace(refsToken_, std::move(refs));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalSoAMergeLayerClusters"))
          ->setComment("The layer clusters of the mask, which the vertices of the tracksters index.");
      desc.add<edm::InputTag>("inputMask", edm::InputTag())
          ->setComment("The input mask and its ordered list of trackster collections. Empty: all 1, no collection.");
      desc.add<std::vector<edm::InputTag>>("tracksters", {})
          ->setComment("The tracksters whose layer clusters get 0 in the mask.");
      desc.add<edm::InputTag>("trackstersRefs", edm::InputTag())
          ->setComment("The ordered list of the collections of 'tracksters'.");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDGetToken<reco::CaloClusterDeviceCollection> layerClustersToken_;
    device::EDGetToken<ticl::LayerClusterMaskDeviceCollection> inputMaskToken_;
    edm::EDGetTokenT<ticl::TracksterCollectionRefs> inputRefsToken_;
    std::vector<device::EDGetToken<ticl::TracksterDeviceCollection>> trackstersTokens_;
    edm::EDGetTokenT<ticl::TracksterCollectionRefs> trackstersRefsToken_;
    const device::EDPutToken<ticl::LayerClusterMaskDeviceCollection> maskToken_;
    const edm::EDPutTokenT<ticl::TracksterCollectionRefs> refsToken_;
    const bool useInputMask_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(LayerClusterMaskSoAProducer);
