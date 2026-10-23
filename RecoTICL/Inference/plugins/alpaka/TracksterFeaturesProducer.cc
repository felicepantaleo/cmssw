#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "CondFormats/HGCalObjects/interface/TICLGeomHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLayersHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLookupHost.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterFeatureExtractionRunner.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  /*
   Interface for GPU Trackster feature extraction.
  
   The Extractor determines which Tracksters are selected
   and how their input features are represented for the CNN, DNN or PFN model.
  
   The producer uses a two-stage execution because the output size depends on the number of Tracksters selected on the GPU.
   SynchronizingEDProducer allows acquire() to compute and transfer this count 
   before produce() allocates the output collection.  // Interface for GPU Trackster feature extraction.
  
   The Extractor determines which Tracksters are selected
   and how their input features are represented for the CNN, DNN or PFN model.
  
   The producer uses a two-stage execution because the output size depends on the number of Tracksters selected on the GPU.
   SynchronizingEDProducer allows acquire() to compute and transfer this count before produce() allocates the output collection. 
  */
  template <typename Extractor>
  class TracksterFeaturesProducer : public stream::SynchronizingEDProducer<edm::stream::WatchRuns> {
  public:
    using FeaturesDevice = typename Extractor::FeaturesDevice;

    explicit TracksterFeaturesProducer(edm::ParameterSet const& config)
        : SynchronizingEDProducer<edm::stream::WatchRuns>(config),
          doBarrel_{config.getParameter<std::string>("detector") == "Barrel"},
          ticlGeomToken_{esConsumes<TICLGeomHost, CaloGeometryRecord, edm::Transition::BeginRun>(
              edm::ESInputTag("", doBarrel_ ? "withBarrel" : ""))},
          ticlGeomLookupToken_{esConsumes<TICLGeomLookupHost, CaloGeometryRecord, edm::Transition::BeginRun>(
              edm::ESInputTag("", doBarrel_ ? "withBarrel" : ""))},
          ticlGeomLayersToken_{
              esConsumes<TICLGeomLayersHost, CaloGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag("", ""))},
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("tracksters"))},
          runner_{config},
          featuresOutToken_{produces("features")} {
      const auto tags = config.getParameter<std::vector<edm::InputTag>>("layerClusters");
      if (tags.size() != 3)
        throw cms::Exception("TracksterFeaturesProducer")
            << "Expected exactly three LayerCluster collections (EE, HSi, HSci)";

      for (const auto& tag : tags) {
        layerClustersTokens_.emplace_back(consumes(tag));
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription description;
      description.add<std::string>("detector", "HGCAL");
      description.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersToSoAProducer"));
      description.add<std::vector<edm::InputTag>>("layerClusters",
                                                  {edm::InputTag("hgcalSoALayerClustersEE"),
                                                   edm::InputTag("hgcalSoALayerClustersHSi"),
                                                   edm::InputTag("hgcalSoALayerClustersHSci")});
      Extractor::fillDescription(description);
      descriptions.addWithDefaultLabel(description);
    }

    // Initialize the geometry tools once per run.
    // The number of HGCAL layers per endcap is needed by the feature extraction kernels.
    void beginRun(edm::Run const&, edm::EventSetup const& setup) override {
      rhtools_.setGeometry(
          setup.getData(ticlGeomToken_), setup.getData(ticlGeomLookupToken_), setup.getData(ticlGeomLayersToken_));

      layersPerEndcap_ = rhtools_.lastLayer(false);
      if (layersPerEndcap_ <= 0)
        throw cms::Exception("TracksterFeaturesProducer") << "Invalid number of layers per endcap";
    }

    // Determine which Tracksters will be passed to the model.
    // The runner retains the selection information and transfers the selected count to the CPU for output allocation in produce().
    void acquire(device::Event const& event, device::EventSetup const&) override {
      const auto& input = event.get(trackstersToken_);
      const auto tracksters = input.const_view();
      auto& queue = event.queue();

      // No need to retrieve the layer-cluster products if no Tracksters were produced.
      if (tracksters.energy().metadata().size() == 0u) {
        runner_.prepare(queue, tracksters, LayerClusterEnergyMultiView{});
        return;
      }

      const auto& ee = event.get(layerClustersTokens_[0]);
      const auto& hsi = event.get(layerClustersTokens_[1]);
      const auto& hsci = event.get(layerClustersTokens_[2]);
      const auto nEE = ee.const_view().metadata().size()[0];
      const auto nHSi = hsi.const_view().metadata().size()[0];
      const auto nHSci = hsci.const_view().metadata().size()[0];

      // Combine the EE, HSi and HSci LayerCluster columns into multiviews,
      // preserving the collection ordering expected by the Trackster indices.
      LayerClusterIndexMultiView indexes;
      LayerClusterEnergyMultiView energies;

      indexes.addView(ee.const_view().indexes(), nEE);
      indexes.addView(hsi.const_view().indexes(), nHSi);
      indexes.addView(hsci.const_view().indexes(), nHSci);

      energies.addView(ee.const_view().energy(), nEE);
      energies.addView(hsi.const_view().energy(), nHSi);
      energies.addView(hsci.const_view().energy(), nHSci);

      // Enqueue Trackster selection, and the transfer of selected Trackster count to the host.
      runner_.prepare(queue, tracksters, energies, indexes);
    }

    // The selected Trackster count is now available on the host.
    // Allocate the model-specific output SoA and fill its features
    void produce(device::Event& event, device::EventSetup const&) override {
      auto& queue = event.queue();
      const auto& input = event.get(trackstersToken_);
      const auto tracksters = input.const_view();

      // No Tracksters passed selection.
      // Produce an empty feature collection without retrieving the LayerCluster inputs.
      if (runner_.selectedCount() == 0u) {
        auto output = runner_.run(
            queue, tracksters, LayerClusterEnergyMultiView{}, LayerClusterPositionMultiView{}, layersPerEndcap_);
        event.emplace(featuresOutToken_, std::move(output));
        return;
      }

      const auto& ee = event.get(layerClustersTokens_[0]);
      const auto& hsi = event.get(layerClustersTokens_[1]);
      const auto& hsci = event.get(layerClustersTokens_[2]);
      const auto nEE = ee.const_view().metadata().size()[0];
      const auto nHSi = hsi.const_view().metadata().size()[0];
      const auto nHSci = hsci.const_view().metadata().size()[0];

      // Provide the extractor with the LayerCluster indices, energies and positions needed to construct the model-specific input features.
      LayerClusterIndexMultiView indexes;
      LayerClusterEnergyMultiView energies;
      LayerClusterPositionMultiView positions;

      indexes.addView(ee.const_view().indexes(), nEE);
      indexes.addView(hsi.const_view().indexes(), nHSi);
      indexes.addView(hsci.const_view().indexes(), nHSci);

      energies.addView(ee.const_view().energy(), nEE);
      energies.addView(hsi.const_view().energy(), nHSi);
      energies.addView(hsci.const_view().energy(), nHSci);

      positions.addView(ee.const_view().position(), nEE);
      positions.addView(hsi.const_view().position(), nHSi);
      positions.addView(hsci.const_view().position(), nHSci);

      // Fill the feature SoA and put it as a device product for downstream inference modules.
      auto output =
          runner_.run(queue, tracksters, energies, positions, static_cast<uint32_t>(layersPerEndcap_), indexes);
      event.emplace(featuresOutToken_, std::move(output));
    }

  private:
    const bool doBarrel_;
    const edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
    const edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
    const edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;

    ticlgeom::Tools rhtools_;

    const device::EDGetToken<ticl::TracksterDeviceCollection> trackstersToken_;
    std::vector<device::EDGetToken<reco::CaloClusterDeviceCollection>> layerClustersTokens_;

    TracksterFeatureExtractionRunner<Extractor> runner_;
    const device::EDPutToken<FeaturesDevice> featuresOutToken_;

    int32_t layersPerEndcap_ = 0;
  };

  // Register one EDM plugin per model family.
  // Each specialization uses its own feature extraction policy and output SoA type.
  using TracksterDNNFeaturesProducer = TracksterFeaturesProducer<DNNFeatureExtractor>;
  using TracksterPFNFeaturesProducer = TracksterFeaturesProducer<PFNFeatureExtractor>;
  using TracksterCNNFeaturesProducer = TracksterFeaturesProducer<CNNFeatureExtractor>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterCNNFeaturesProducer);
DEFINE_FWK_ALPAKA_MODULE(TracksterDNNFeaturesProducer);
DEFINE_FWK_ALPAKA_MODULE(TracksterPFNFeaturesProducer);
