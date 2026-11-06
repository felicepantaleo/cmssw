// TracksterHostCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/HGCalReco/interface/TracksterHostCollection.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TracksterLinksHostCollection.h"
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
#include "RecoTICL/Common/interface/TrackstersPCA.h"

// Converts the linked tracksters built on the device to the products of TracksterLinksProducer: the tracksters, the
// trivial groups of the result (one trackster each), the input tracksters of each linked trackster as global indices,
// and the layer cluster mask: the product of the original masks with 0 for each layer cluster of a linked trackster.
// A linked trackster with one or two layer clusters gets the geometric frame of ticl::assignGeometricFrame. A layer
// cluster in two input tracksters of a linked trackster is an error.
class TracksterLinksFromSoAProducer : public edm::stream::EDProducer<edm::stream::WatchRuns> {
public:
  TracksterLinksFromSoAProducer(const edm::ParameterSet& config)
      : tracksterSoAToken_(consumes(config.getParameter<edm::InputTag>("src"))),
        linksSoAToken_(consumes(config.getParameter<edm::InputTag>("links"))),
        layerClustersToken_(consumes(config.getParameter<edm::InputTag>("layer_clusters"))),
        ticlGeomToken_(esConsumes<edm::Transition::BeginRun>()),
        ticlGeomLookupToken_(esConsumes<edm::Transition::BeginRun>()),
        ticlGeomLayersToken_(esConsumes<edm::Transition::BeginRun>()),
        trackstersToken_(produces<std::vector<ticl::Trackster>>()),
        linkedTrackstersToken_(produces<std::vector<std::vector<unsigned int>>>()),
        linksToken_(produces<std::vector<std::vector<unsigned int>>>("linkedTracksterIdToInputTracksterId")),
        maskToken_(produces<std::vector<float>>()) {
    for (const auto& tag : config.getParameter<std::vector<edm::InputTag>>("original_masks")) {
      originalMasksTokens_.emplace_back(consumes(tag));
    }
  }

  void beginRun(const edm::Run&, const edm::EventSetup& iSetup) override {
    rhtools_.setGeometry(
        iSetup.getData(ticlGeomToken_), iSetup.getData(ticlGeomLookupToken_), iSetup.getData(ticlGeomLayersToken_));
  }

  void produce(edm::Event& iEvent, const edm::EventSetup&) override {
    const auto tracksterSoA = iEvent.get(tracksterSoAToken_).const_view();
    const auto linksSoA = iEvent.get(linksSoAToken_).const_view();
    const auto& layerClusters = iEvent.get(layerClustersToken_);
    const auto vertices = tracksterSoA.vertices();
    const auto n = static_cast<unsigned int>(vertices.keys());
    if (static_cast<unsigned int>(linksSoA.keys()) != n) {
      throw edm::Exception(edm::errors::LogicError)
          << "TracksterLinksFromSoAProducer: " << n << " linked tracksters, the links have " << linksSoA.keys() << ".";
    }

    std::vector<float> mask(layerClusters.size(), 1.f);
    for (const auto& token : originalMasksTokens_) {
      const auto& original = iEvent.get(token);
      for (unsigned int j = 0; j < std::min(mask.size(), original.size()); ++j) {
        mask[j] *= original[j];
      }
    }

    std::vector<ticl::Trackster> tracksters(n);
    std::vector<std::vector<unsigned int>> linkedTracksters(n);
    std::vector<std::vector<unsigned int>> links(n);
    std::vector<unsigned int> sortedVertices;
    for (unsigned int i = 0; i < n; ++i) {
      auto& trackster = tracksters[i];
      const auto layerClusterIndices = vertices[i];
      trackster.vertices().assign(layerClusterIndices.begin(), layerClusterIndices.end());
      sortedVertices.assign(layerClusterIndices.begin(), layerClusterIndices.end());
      std::ranges::sort(sortedVertices);
      if (std::ranges::adjacent_find(sortedVertices) != sortedVertices.end()) {
        throw edm::Exception(edm::errors::LogicError)
            << "TracksterLinksFromSoAProducer: linked trackster " << i << " has a layer cluster twice.";
      }
      trackster.vertex_multiplicity().assign(layerClusterIndices.size(), 1.f);
      for (const auto layerCluster : trackster.vertices()) {
        if (layerCluster >= mask.size()) {
          throw edm::Exception(edm::errors::LogicError)
              << "TracksterLinksFromSoAProducer: a trackster uses layer cluster " << layerCluster << " of "
              << mask.size() << ".";
        }
        mask[layerCluster] = 0.f;
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
      if (trackster.vertices().size() <= 2) {
        ticl::assignGeometricFrame(trackster, layerClusters, rhtools_);
      }
      ticl::TracksterIdProbabilities idProbabilities = tracksterSoA.id()[i].idProbabilities();
      trackster.setProbabilities(idProbabilities.data());

      linkedTracksters[i] = {i};
      const auto members = linksSoA[i];
      links[i].assign(members.begin(), members.end());
    }

    iEvent.emplace(trackstersToken_, std::move(tracksters));
    iEvent.emplace(linkedTrackstersToken_, std::move(linkedTracksters));
    iEvent.emplace(linksToken_, std::move(links));
    iEvent.emplace(maskToken_, std::move(mask));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("ticlTracksterLinksSoA"))->setComment("The linked tracksters.");
    desc.add<edm::InputTag>("links", edm::InputTag("ticlTracksterLinksSoA", "links"))
        ->setComment("The input tracksters of each linked trackster.");
    desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"))
        ->setComment("The layer clusters that the vertices of the tracksters index.");
    desc.add<std::vector<edm::InputTag>>("original_masks",
                                         {edm::InputTag("hgcalMergeLayerClusters", "InitialLayerClustersMask")})
        ->setComment(
            "The masks whose product, with 0 for the layer clusters of the linked tracksters, is the output mask.");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const edm::EDGetTokenT<ticl::TracksterHostCollection> tracksterSoAToken_;
  const edm::EDGetTokenT<ticl::TracksterLinksHostCollection> linksSoAToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClustersToken_;
  std::vector<edm::EDGetTokenT<std::vector<float>>> originalMasksTokens_;
  const edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
  const edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
  const edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;
  const edm::EDPutTokenT<std::vector<ticl::Trackster>> trackstersToken_;
  const edm::EDPutTokenT<std::vector<std::vector<unsigned int>>> linkedTrackstersToken_;
  const edm::EDPutTokenT<std::vector<std::vector<unsigned int>>> linksToken_;
  const edm::EDPutTokenT<std::vector<float>> maskToken_;
  ticlgeom::Tools rhtools_;
};

DEFINE_FWK_MODULE(TracksterLinksFromSoAProducer);
