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
      layerClustersToken_ = consumes(config.getParameter<edm::InputTag>("layerClusters"));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription description;
      description.add<std::string>("detector", "HGCAL");
      description.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersToSoAProducer"));
      description.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalSoAMergeLayerClusters"));
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

      const auto& clusters = event.get(layerClustersToken_);
      const auto nClusters = clusters.const_view().energy().metadata().size();

      // Combine the EE, HSi and HSci LayerCluster columns into multiviews,
      // preserving the collection ordering expected by the Trackster indices.
      LayerClusterIndexMultiView indexes;
      LayerClusterEnergyMultiView energies;

      indexes.addView(clusters.const_view().indexes(), nClusters);
      energies.addView(clusters.const_view().energy(), nClusters);

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

      const auto& clusters = event.get(layerClustersToken_);
      const auto nClusters = clusters.const_view().energy().metadata().size();

      // Provide the extractor with the LayerCluster indices, energies and positions needed to construct the model-specific input features.
      LayerClusterIndexMultiView indexes;
      LayerClusterEnergyMultiView energies;
      LayerClusterPositionMultiView positions;

      indexes.addView(clusters.const_view().indexes(), nClusters);
      energies.addView(clusters.const_view().energy(), nClusters);

      positions.addView(clusters.const_view().position(), nClusters);

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
    device::EDGetToken<reco::CaloClusterDeviceCollection> layerClustersToken_;

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
