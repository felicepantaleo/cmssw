// TracksterHostCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/HGCalReco/interface/TracksterHostCollection.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"
#include "RecoTICL/Inference/interface/TICLONNXGlobalCache.h"
#include "RecoTICL/Inference/interface/TracksterInferenceAlgoBase.h"
#include "RecoTICL/Inference/interface/TracksterInferenceAlgoFactory.h"
#include "RecoTICL/Inference/interface/TracksterInferenceByCNN.h"
#include "RecoTICL/Inference/interface/TracksterInferenceByDNN.h"
#include "RecoTICL/Inference/interface/TracksterInferenceByPFN.h"
#include "RecoTICL/Inference/interface/TracksterInferenceByShowerShape.h"

// Converts the tracksters built on the device to ticl::Trackster, in the same order, and masks their layer clusters
// in a copy of the original layer cluster mask, as TrackstersProducer does. The layer clusters of the tracksters are
// indices in the merged layer clusters, which index the same layer clusters as hgcalMergeLayerClusters.
// The PID runs on the host on the converted tracksters, with the inference plugin of TrackstersProducer, and
// overwrites the probabilities of the device. 
class HGCalTrackstersFromSoAProducer
    : public edm::stream::EDProducer<edm::GlobalCache<ticl::TICLONNXGlobalCache>, edm::stream::WatchRuns> {
public:
  HGCalTrackstersFromSoAProducer(const edm::ParameterSet& config, ticl::TICLONNXGlobalCache const* cache)
      : tracksterSoAToken_(consumes(config.getParameter<edm::InputTag>("src"))),
        originalMaskToken_(consumes(config.getParameter<edm::InputTag>("originalMask"))),
        layerClustersToken_(consumes(config.getParameter<edm::InputTag>("layer_clusters"))),
        ticlGeomToken_(esConsumes<edm::Transition::BeginRun>()),
        ticlGeomLookupToken_(esConsumes<edm::Transition::BeginRun>()),
        ticlGeomLayersToken_(esConsumes<edm::Transition::BeginRun>()),
        trackstersToken_(produces<std::vector<ticl::Trackster>>()),
        maskToken_(produces<std::vector<float>>()),
        doPidCut_(config.getParameter<bool>("doPidCut")),
        cutHadProb_(config.getParameter<double>("cutHadProb")) {
    const std::string inferencePlugin = config.getParameter<std::string>("inferenceAlgo");
    if (!inferencePlugin.empty()) {
      const auto inferencePSet = config.getParameter<edm::ParameterSet>("pluginInferenceAlgo" + inferencePlugin);
      for (const char* key : {"onnxModelPath", "onnxPIDModelPath", "onnxEnergyModelPath"}) {
        if (inferencePSet.existsAs<std::string>(key, true) && !inferencePSet.getParameter<std::string>(key).empty()) {
          inferenceAlgo_ = TracksterInferenceAlgoFactory::get()->create(inferencePlugin, inferencePSet, cache);
          break;
        }
      }
    }
  }

  static std::unique_ptr<ticl::TICLONNXGlobalCache> initializeGlobalCache(const edm::ParameterSet& config) {
    return ticl::TICLONNXGlobalCache::initialize(config);
  }
  static void globalEndJob(ticl::TICLONNXGlobalCache const*) {}

  void beginRun(const edm::Run&, const edm::EventSetup& iSetup) override {
    rhtools_.setGeometry(
        iSetup.getData(ticlGeomToken_), iSetup.getData(ticlGeomLookupToken_), iSetup.getData(ticlGeomLayersToken_));
  }

  void produce(edm::Event& iEvent, const edm::EventSetup&) override {
    const auto tracksterSoA = iEvent.get(tracksterSoAToken_).const_view();
    const auto vertices = tracksterSoA.vertices();
    std::vector<float> mask = iEvent.get(originalMaskToken_);

    std::vector<ticl::Trackster> tracksters(vertices.keys());
    for (unsigned int i = 0; i < tracksters.size(); ++i) {
      auto& trackster = tracksters[i];

      // the device fills the layer clusters of a trackster in no particular order
      const auto layerClusters = vertices[i];
      trackster.vertices().assign(layerClusters.begin(), layerClusters.end());
      std::ranges::sort(trackster.vertices());
      trackster.vertex_multiplicity().assign(layerClusters.size(), 1.f);
      for (const auto layerCluster : trackster.vertices()) {
        if (layerCluster >= mask.size()) {
          throw edm::Exception(edm::errors::LogicError)
              << "HGCalTrackstersFromSoAProducer: a trackster uses layer cluster " << layerCluster << " of "
              << mask.size() << ". The layer clusters must be merged on the device in the order of the legacy ones.";
        }
      }

      const auto energy = tracksterSoA.energy()[i];
      trackster.setRawEnergy(energy.rawEnergy());
      trackster.setRawEmEnergy(energy.rawEmEnergy());
      trackster.setRegressedEnergy(energy.regressedEnergy());
      const auto barycenter = tracksterSoA.barycenter()[i];
      trackster.setBarycenter(ticl::Trackster::Vector(barycenter.x(), barycenter.y(), barycenter.z()));
      trackster.calculateRawPt();
      trackster.calculateRawEmPt();
      trackster.setTimeAndError(tracksterSoA.timing()[i].time(), tracksterSoA.timing()[i].timeError());

      // fillPCAVariables takes the squares of the spreads, and keeps the order and orientation of the device
      const auto pca = tracksterSoA.pca()[i];
      trackster.fillPCAVariables(pca.eigenvalues(),
                                 pca.eigenvectors(),
                                 pca.sigmas().cwiseAbs2(),
                                 pca.sigmasPCA().cwiseAbs2(),
                                 3,
                                 ticl::Trackster::PCAOrdering::descending);
      ticl::TracksterIdProbabilities idProbabilities = tracksterSoA.id()[i].idProbabilities();
      trackster.setProbabilities(idProbabilities.data());
      // The iteration index keeps its default value, which TrackstersProducer also gives the iterations it does not name
    }

    if (inferenceAlgo_) {
      inferenceAlgo_->runInference(iEvent.get(layerClustersToken_), tracksters, rhtools_);
    }

    if (doPidCut_) {
      std::erase_if(tracksters, [this](const ticl::Trackster& trackster) {
        return trackster.id_probability(ticl::Trackster::ParticleType::charged_hadron) +
                   trackster.id_probability(ticl::Trackster::ParticleType::neutral_hadron) >=
               cutHadProb_;
      });
    }
    for (const auto& trackster : tracksters) {
      for (const auto layerCluster : trackster.vertices()) {
        mask[layerCluster] = 0.f;
      }
    }

    iEvent.emplace(trackstersToken_, std::move(tracksters));
    iEvent.emplace(maskToken_, std::move(mask));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("ticlSoATracksters"));
    desc.add<edm::InputTag>("originalMask", edm::InputTag("hgcalMergeLayerClusters", "InitialLayerClustersMask"))
        ->setComment("The mask of the layer clusters before the tracksters: theirs are set to 0 in the output mask.");
    desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"))
        ->setComment("The layer clusters that the vertices of the tracksters index, read by the PID.");
    desc.add<bool>("doPidCut", false);
    desc.add<double>("cutHadProb", 0.5);
    desc.add<std::string>("inferenceAlgo", "");
    desc.add<edm::ParameterSetDescription>(
        "pluginInferenceAlgoTracksterInferenceByDNN",
        ticl::inferencePluginPSetDescription<ticl::TracksterInferenceByDNN>("TracksterInferenceByDNN"));
    desc.add<edm::ParameterSetDescription>(
        "pluginInferenceAlgoTracksterInferenceByCNN",
        ticl::inferencePluginPSetDescription<ticl::TracksterInferenceByCNN>("TracksterInferenceByCNN"));
    desc.add<edm::ParameterSetDescription>(
        "pluginInferenceAlgoTracksterInferenceByPFN",
        ticl::inferencePluginPSetDescription<ticl::TracksterInferenceByPFN>("TracksterInferenceByPFN"));
    desc.add<edm::ParameterSetDescription>(
        "pluginInferenceAlgoTracksterInferenceByShowerShape",
        ticl::inferencePluginPSetDescription<ticl::TracksterInferenceByShowerShape>("TracksterInferenceByShowerShape"));
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const edm::EDGetTokenT<ticl::TracksterHostCollection> tracksterSoAToken_;
  const edm::EDGetTokenT<std::vector<float>> originalMaskToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClustersToken_;
  const edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
  const edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
  const edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;
  const edm::EDPutTokenT<std::vector<ticl::Trackster>> trackstersToken_;
  const edm::EDPutTokenT<std::vector<float>> maskToken_;
  const bool doPidCut_;
  const float cutHadProb_;
  std::unique_ptr<ticl::TracksterInferenceAlgoBase> inferenceAlgo_;
  ticlgeom::Tools rhtools_;
};

DEFINE_FWK_MODULE(HGCalTrackstersFromSoAProducer);
