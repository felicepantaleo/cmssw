#include <iostream>

// Alpaka-based EDProducer from CMSSW
// Alpaka config (ALPAKA_ACCELERATOR_NAMESPACE, etc.)
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

// Standard CMSSW includes
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

// The input CaloCluster definition
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

// SoA layouts and multi-collection definitions
#include "DataFormats/ClusterSoA/interface/alpaka/ClusterPortableCollection.h"
#include "DataFormats/ClusterSoA/interface/CaloClusterSoA.h"
#include "DataFormats/Portable/interface/PortableCollection.h"



namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Alias for the multi-SoA collection we produce
  // e.g. using CaloClusterSoACollection = PortableDeviceMultiCollection<Device,
  //                                       Position4D_Energy_SoA,
  //                                       Position4D_Energy_Errors_SoA,
  //                                       CaloClusterExtra_SoA>;

  class CaloClusterSoAProducer : public stream::EDProducer<> {
  public:
    explicit CaloClusterSoAProducer(edm::ParameterSet const& iConfig)
        // Standard CPU-based input token for CaloClusters
        : srcToken_{consumes<std::vector<reco::CaloCluster>>(
              iConfig.getParameter<edm::InputTag>("src"))}
        // Alpaka-based output token: we must pass a product instance name (can be empty)
        , putToken_{produces("CaloClustersSoA")} {
    }

    ~CaloClusterSoAProducer() override = default;

    // Configuration validation
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src", edm::InputTag("particleFlowClusterECAL"))
          ->setComment("InputTag for the std::vector<reco::CaloCluster> collection.");
      // The product instance name "CaloClustersSoA" is hard-coded above,
      // but you could add a parameter here if you want a configurable label.

      descriptions.addWithDefaultLabel(desc);
    }

    // The main produce method
    void produce(device::Event& iEvent, device::EventSetup const&) override {
      // 1) Read the input collection from CPU-based input token
      //    getHandle(...) returns a edm::Handle<std::vector<reco::CaloCluster>>
      auto clusterHandle = iEvent.getHandle(srcToken_);
      if (not clusterHandle.isValid()) {
        // If needed, handle the error gracefully
        // (but typically, the framework ensures the handle is valid if the input is present)
        return;
      }
      auto const& clusters = *clusterHandle;
      size_t const nClusters = clusters.size();

      // 2) Create output multi-SoA with the correct size on the given device
      //    iEvent.device() or iEvent.queue() is the Alpaka device/queue for the current accelerator
      CaloClusterSoACollection outCollection(nClusters, iEvent.device());

      // 3) Get references to each SoA
      auto positionEnergyView = outCollection.view<Position4D_Energy_SoA>();
      auto positionErrorView  = outCollection.view<Position4D_Energy_Errors_SoA>();
      auto extraView          = outCollection.view<CaloClusterExtra_SoA>();

      // 4) Fill them
      for (size_t i = 0; i < nClusters; ++i) {
        auto const& cluster = clusters[i];

        auto rowPE = positionEnergyView[i];
        rowPE.x()                = cluster.x();
        rowPE.y()                = cluster.y();
        rowPE.z()                = cluster.z();
        rowPE.raw_energy()       = cluster.energy();
        rowPE.corrected_energy() = cluster.correctedEnergy();
        rowPE.time()             = 0.f;  // no time in reco::CaloCluster

        auto rowErr = positionErrorView[i];
        rowErr.xErr()    = 0.f;
        rowErr.yErr()    = 0.f;
        rowErr.zErr()    = 0.f;
        rowErr.timeErr() = 0.f;
        rowErr.energyErr() = (cluster.correctedEnergyUncertainty() > 0.f)
                                 ? cluster.correctedEnergyUncertainty()
                                 : 0.f;
        std::cout << "cluster " << i << " has energy " << cluster.energy() << " and uncertainty "
                  << rowErr.energyErr() << std::endl;
        // auto rowExtra = extraView[i];
        // cast from reco::CaloCluster::AlgoId => your SoA's AlgoId
        // rowExtra.algoId() = (AlgoId)(cluster.algoID());
        // rowExtra.CaloID() = cluster.caloID().rawId();
        // rowExtra.flags()  = cluster.flags();
        // rowExtra.seedId() = cluster.seed().rawId();
      }

      // (Optional) debugging
      // std::cout << "[CaloClusterSoAProducer] Filled " << nClusters << " cluster(s)\n";

      // 5) Emplace the SoA collection into the event
      //    The putToken_ knows the product + instance label
      iEvent.emplace(putToken_, std::move(outCollection));
    }

  private:
    // The CPU-based input token for the standard vector of CaloCluster
    edm::EDGetTokenT<std::vector<reco::CaloCluster>> srcToken_;

    // The Alpaka-based output token for the multi-SoA product
    device::EDPutToken<CaloClusterSoACollection> putToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// Register as an Alpaka module
DEFINE_FWK_ALPAKA_MODULE(CaloClusterSoAProducer);
