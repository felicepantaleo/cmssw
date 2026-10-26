#include <algorithm>
#include <utility>
#include <vector>

#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoALayerClustersExtraDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "HGCalCLUEsteringTrackstersAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Clusters the layer clusters of all the HGCAL components together into tracksters with CLUEstering: produces the
  // trackster of each layer cluster, the number of tracksters and the number of layer clusters in them. The
  // clustering coordinates are (x / |z| / sigmaT, y / |z| / sigmaT, layer / layerScale), with the distance
  // max(transverse distance / mean sigmaT of the pair, layer distance).
  class HGCalCLUEsteringTrackstersProducer : public stream::EDProducer<> {
  public:
    HGCalCLUEsteringTrackstersProducer(const edm::ParameterSet& config)
        : EDProducer(config),
          layerClustersToken_{consumes(config.getParameter<edm::InputTag>("hgcalLayerClustersSoA"))},
          layerClustersClusteringToken_{produces()},
          parameters_{makeParameters(config)} {}


    void produce(device::Event& iEvent, const device::EventSetup&) override {
      const auto layerClusters = iEvent.get(layerClustersToken_).const_view();

      HGCalSoALayerClustersExtraDeviceCollection layerClustersClustering(iEvent.queue(),
                                                                         layerClusters.position().metadata().size());

      algo_.run(iEvent.queue(), parameters_, layerClusters, layerClustersClustering.view());
      iEvent.emplace(layerClustersClusteringToken_, std::move(layerClustersClustering));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("hgcalLayerClustersSoA", edm::InputTag("hgcalSoAMergeLayerClusters"));
      desc.add<float>("deltac", 1.)->setComment("Density radius, in the clustering coordinates.");
      desc.add<float>("rhoc", 0.255)->setComment("Minimum energy density of a seed [GeV].");
      desc.add<float>("outlierDistance", 2.1155)->setComment("Maximum distance to the nearest higher.");
      desc.add<float>("seedingDistance", 1.6642)
          ->setComment("Maximum distance to the nearest higher of the points above the seed threshold.");
      desc.add<std::vector<double>>("sigmaT", {0.00227, 0.003, 0.01649})
          ->setComment("Transverse angular scale [rad] of CE-E, CE-H silicon and CE-H scintillator.");
      desc.add<float>("layerScale", 6.177)->setComment("Longitudinal scale, in layers.");
      desc.add<float>("rhocEtaExponent", 1.88588)
          ->setComment(
              "The seed threshold of a layer cluster is rhoc * (rhocPivotRadius / r)^rhocEtaExponent, with r = "
              "sqrt(x^2 + y^2) / |z|. 0 keeps it flat.");
      desc.add<float>("rhocPivotRadius", 0.42)->setComment("r at which the seed threshold is rhoc (eta ~ 1.6).");
      desc.add<unsigned int>("minSiliconCells", 2)
          ->setComment("The silicon layer clusters with fewer cells are not clustered, as in the CLUE3D filter.");
      desc.add<unsigned int>("minLayerClusters", 2)->setComment("Minimum number of layer clusters of a trackster.");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    static HGCalCLUEsteringTrackstersAlgoWrapper::Parameters makeParameters(const edm::ParameterSet& config) {
      HGCalCLUEsteringTrackstersAlgoWrapper::Parameters parameters{
          .deltac = config.getParameter<float>("deltac"),
          .rhoc = config.getParameter<float>("rhoc"),
          .outlierDistance = config.getParameter<float>("outlierDistance"),
          .seedingDistance = config.getParameter<float>("seedingDistance"),
          .sigmaT = {},
          .layerScale = config.getParameter<float>("layerScale"),
          .rhocEtaExponent = config.getParameter<float>("rhocEtaExponent"),
          .rhocPivotRadius = config.getParameter<float>("rhocPivotRadius"),
          .minSiliconCells = config.getParameter<unsigned int>("minSiliconCells"),
          .minLayerClusters = config.getParameter<unsigned int>("minLayerClusters")};

      const auto& sigmaT = config.getParameter<std::vector<double>>("sigmaT");
      if (sigmaT.size() != parameters.sigmaT.size()) {
        throw edm::Exception(edm::errors::Configuration)
            << "HGCalCLUEsteringTrackstersProducer: 'sigmaT' needs one value per HGCAL component (CE-E, CE-H "
               "silicon, CE-H scintillator).";
      }
      std::ranges::copy(sigmaT, parameters.sigmaT.begin());
      return parameters;
    }

    const device::EDGetToken<reco::CaloClusterDeviceCollection> layerClustersToken_;
    const device::EDPutToken<HGCalSoALayerClustersExtraDeviceCollection> layerClustersClusteringToken_;
    const HGCalCLUEsteringTrackstersAlgoWrapper::Parameters parameters_;
    const HGCalCLUEsteringTrackstersAlgoWrapper algo_{};
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalCLUEsteringTrackstersProducer);
