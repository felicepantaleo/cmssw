#include <algorithm>
#include <utility>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
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

  // The mask of the layer clusters that an iteration can use: the input mask, with 0 for each layer cluster that
  // ClusterFilterByAlgoAndSize rejects.
  class FilteredLayerClustersSoAProducer : public global::EDProducer<> {
  public:
    FilteredLayerClustersSoAProducer(const edm::ParameterSet& config)
        : EDProducer(config),
          layerClustersToken_{consumes(config.getParameter<edm::InputTag>("LayerClusters"))},
          inputMaskToken_{consumes(config.getParameter<edm::InputTag>("LayerClustersInputMask"))},
          maskToken_{produces()},
          filter_{makeFilter(config)} {}

    void produce(edm::StreamID, device::Event& iEvent, const device::EventSetup&) const override {
      const auto layerClusters = iEvent.get(layerClustersToken_).const_view();
      const auto inputMask = iEvent.get(inputMaskToken_).const_view();
      const int nLayerClusters = layerClusters.position().metadata().size();
      if (inputMask.metadata().size() != nLayerClusters) {
        throw edm::Exception(edm::errors::LogicError)
            << "FilteredLayerClustersSoAProducer: the input mask has " << inputMask.metadata().size() << " values for "
            << nLayerClusters << " layer clusters.";
      }
      ticl::LayerClusterMaskDeviceCollection mask(iEvent.queue(), nLayerClusters);
      ticl::filterByAlgoAndSize(iEvent.queue(), filter_, layerClusters, inputMask, mask.view());
      iEvent.emplace(maskToken_, std::move(mask));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("LayerClusters", edm::InputTag("hgcalSoAMergeLayerClusters"));
      desc.add<edm::InputTag>("LayerClustersInputMask", edm::InputTag());
      desc.add<std::vector<int>>(
              "algo_number",
              {::reco::CaloCluster::hgcal_em, ::reco::CaloCluster::hgcal_had, ::reco::CaloCluster::hgcal_scintillator})
          ->setComment("The accepted algorithms of the layer clusters.");
      desc.add<int>("min_cluster_size", 0)->setComment("Minimum number of cells of a silicon layer cluster.");
      desc.add<int>("max_cluster_size", 9999)->setComment("Maximum number of cells of a layer cluster.");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    static ticl::AlgoAndSizeFilter makeFilter(const edm::ParameterSet& config) {
      ticl::AlgoAndSizeFilter filter;
      const auto& algos = config.getParameter<std::vector<int>>("algo_number");
      if (algos.size() > ticl::AlgoAndSizeFilter::kMaxAlgos) {
        throw edm::Exception(edm::errors::Configuration)
            << "FilteredLayerClustersSoAProducer: 'algo_number' has " << algos.size() << " values, at most "
            << ticl::AlgoAndSizeFilter::kMaxAlgos << " are allowed.";
      }
      std::ranges::copy(algos, filter.algos.begin());
      filter.nAlgos = algos.size();
      filter.minSize = config.getParameter<int>("min_cluster_size");
      filter.maxSize = config.getParameter<int>("max_cluster_size");
      return filter;
    }

    const device::EDGetToken<reco::CaloClusterDeviceCollection> layerClustersToken_;
    const device::EDGetToken<ticl::LayerClusterMaskDeviceCollection> inputMaskToken_;
    const device::EDPutToken<ticl::LayerClusterMaskDeviceCollection> maskToken_;
    const ticl::AlgoAndSizeFilter filter_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(FilteredLayerClustersSoAProducer);
