/*
Host step of the heterogeneous superclustering DNN: turns the pair scores into superclusters.

For each candidate, in increasing pT rank, the best-scoring seed above nnWorkingPoint absorbs it, as in
TracksterLinkingbySuperClusteringDNN::linkTracksters. A candidate absorbed once can no longer be a seed, which makes
the association sequential, hence on the host.

Produces the same products as TracksterLinksProducer, so EGammaSuperclusterProducer reads it unchanged.
The trackster indices in the pairs refer to the trackster SoA, which must be in the order of the legacy tracksters.
*/

#include <memory>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoTICL/Superclustering/interface/SuperclusteringDNNHostCollections.h"

namespace ticl {

  class SuperclustersFromDNNScoresProducer : public edm::stream::EDProducer<> {
  public:
    explicit SuperclustersFromDNNScoresProducer(edm::ParameterSet const& ps)
        : trackstersToken_{consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("tracksters"))},
          pairsToken_{consumes<SuperclusteringDNNPairsHost>(ps.getParameter<edm::InputTag>("pairs"))},
          scoresToken_{consumes<SuperclusteringDNNPairScoresHost>(ps.getParameter<edm::InputTag>("scores"))},
          layerClustersToken_{consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"))},
          clustersTimeToken_{
              consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("layer_clustersTime"))},
          nnWorkingPoint_{static_cast<float>(ps.getParameter<double>("nnWorkingPoint"))} {
      produces<std::vector<Trackster>>();
      produces<std::vector<std::vector<unsigned int>>>();
      produces<std::vector<std::vector<unsigned int>>>("linkedTracksterIdToInputTracksterId");
      produces<std::vector<float>>();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersCLUEstering"));
      desc.add<edm::InputTag>("pairs", edm::InputTag("ticlSuperclusteringDNNPairFeatures"));
      desc.add<edm::InputTag>("scores", edm::InputTag("ticlSuperclusteringDNNInference"));
      desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
      desc.add<edm::InputTag>("layer_clustersTime", edm::InputTag("hgcalMergeLayerClusters", "timeLayerCluster"));
      desc.add<double>("nnWorkingPoint", 0.57247);
      // TODO: seedPtThreshold, filterByTracksterPID, tracksterPIDCategoriesToFilter, emissionPIDThreshold for the
      // single-trackster superclusters
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(edm::Event& evt, edm::EventSetup const&) override {
      auto const& tracksters = evt.get(trackstersToken_);
      auto const& pairs = evt.get(pairsToken_);
      auto const& scores = evt.get(scoresToken_);

      auto resultTracksters = std::make_unique<std::vector<Trackster>>();
      auto superclusters = std::make_unique<std::vector<std::vector<unsigned int>>>();
      auto linkedTracksterIdToInputTracksterId = std::make_unique<std::vector<std::vector<unsigned int>>>();
      auto resultMask = std::make_unique<std::vector<float>>(evt.get(layerClustersToken_).size(), 1.f);

      // TODO: best seed per candidate over pairs/scores, then the single-trackster superclusters, as in
      // TracksterLinkingbySuperClusteringDNN::linkTracksters
      // TODO: the layer-cluster mask and assignPCAtoTracksters, as in TracksterLinksProducer::produce

      evt.put(std::move(superclusters));
      evt.put(std::move(resultMask));
      evt.put(std::move(resultTracksters));
      evt.put(std::move(linkedTracksterIdToInputTracksterId), "linkedTracksterIdToInputTracksterId");
    }

  private:
    edm::EDGetTokenT<std::vector<Trackster>> const trackstersToken_;
    edm::EDGetTokenT<SuperclusteringDNNPairsHost> const pairsToken_;
    edm::EDGetTokenT<SuperclusteringDNNPairScoresHost> const scoresToken_;
    edm::EDGetTokenT<std::vector<reco::CaloCluster>> const layerClustersToken_;
    edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> const clustersTimeToken_;
    float const nnWorkingPoint_;
  };

}  // namespace ticl

using ticl::SuperclustersFromDNNScoresProducer;
DEFINE_FWK_MODULE(SuperclustersFromDNNScoresProducer);
