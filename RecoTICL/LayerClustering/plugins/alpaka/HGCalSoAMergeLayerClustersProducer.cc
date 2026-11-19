#include <cstdint>
#include <utility>
#include <vector>

#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "HGCalMergeLayerClustersSoAAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Merges the layer clusters of the HGCAL components into one collection, one component after the other, in the order
  // of hgcalLayerClustersSoA. With the order of the layerClustersSoA of hgcalMergeLayerClusters, row i of this
  // collection and element i of the legacy merged collection are the same layer cluster.
  class HGCalSoAMergeLayerClustersProducer : public stream::EDProducer<> {
  public:
    HGCalSoAMergeLayerClustersProducer(const edm::ParameterSet& config)
        : EDProducer(config), mergedLayerClustersToken_{produces()} {
      for (const auto& tag : config.getParameter<std::vector<edm::InputTag>>("hgcalLayerClustersSoA")) {
        layerClustersTokens_.emplace_back(consumes(tag));
      }
    }

    void produce(device::Event& iEvent, const device::EventSetup&) override {
      // The sizes of the collections are known on the host
      uint32_t nLayerClusters = 0;
      for (const auto& token : layerClustersTokens_) {
        nLayerClusters += iEvent.get(token).const_view().position().metadata().size();
      }

      reco::CaloClusterDeviceCollection mergedLayerClusters(
          iEvent.queue(), nLayerClusters, nLayerClusters, nLayerClusters, nLayerClusters);
      uint32_t offset = 0;
      for (const auto& token : layerClustersTokens_) {
        const auto layerClusters = iEvent.get(token).const_view();
        algo_.run(iEvent.queue(), layerClusters, offset, mergedLayerClusters.view());
        offset += layerClusters.position().metadata().size();
      }
      iEvent.emplace(mergedLayerClustersToken_, std::move(mergedLayerClusters));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::vector<edm::InputTag>>("hgcalLayerClustersSoA",
                                           {edm::InputTag("hgcalSoALayerClustersEE"),
                                            edm::InputTag("hgcalSoALayerClustersHSi"),
                                            edm::InputTag("hgcalSoALayerClustersHSci")})
          ->setComment("the SoA layer clusters of each HGCAL component, in the order of the merged collection");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    std::vector<device::EDGetToken<reco::CaloClusterDeviceCollection>> layerClustersTokens_;
    const device::EDPutToken<reco::CaloClusterDeviceCollection> mergedLayerClustersToken_;
    const HGCalMergeLayerClustersSoAAlgoWrapper algo_{};
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalSoAMergeLayerClustersProducer);
