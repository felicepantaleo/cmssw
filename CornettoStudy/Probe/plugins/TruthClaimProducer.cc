// Study producer, not for integration. The truth claim of the tracks: for each layer cluster, the track whose particles
// (the deep match of the track, DeepTrackMatch.h, and its descendants) hold more than minFraction of the simulated
// energy of the cluster (hits times hit fractions), else -1. A particle that the deep match of several tracks holds
// belongs to the track with the highest momentum. Product: std::vector<int> per layer cluster, the claimOverride input
// of TICLInterpretationProducer.

#include <unordered_map>
#include <vector>

#include "CornettoStudy/Probe/plugins/DeepTrackMatch.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"

namespace {
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
  bool isHGCal(uint32_t rawId) {
    const auto det = DetId(rawId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }
}  // namespace

class TruthClaimProducer : public edm::global::EDProducer<> {
public:
  explicit TruthClaimProducer(const edm::ParameterSet&);
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<std::vector<reco::Track>> trackToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<MapType> tkR2tToken_;
  const edm::EDGetTokenT<std::vector<unsigned int>> assignableToken_;
  const double minFraction_;
};

TruthClaimProducer::TruthClaimProducer(const edm::ParameterSet& cfg)
    : trackToken_(consumes<std::vector<reco::Track>>(cfg.getParameter<edm::InputTag>("tracks"))),
      lcToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))),
      graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      tkR2tToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("trackRecoToTruth"))),
      assignableToken_(consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("assignableRoots"))),
      minFraction_(cfg.getParameter<double>("minFraction")) {
  produces<std::vector<int>>();
}

void TruthClaimProducer::produce(edm::StreamID, edm::Event& event, const edm::EventSetup&) const {
  auto const& tracks = event.get(trackToken_);
  auto const& lcs = event.get(lcToken_);
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  auto const& tkR2t = event.get(tkR2tToken_).getMap();
  const unsigned int nP = graph.particles().size();

  // The track of each particle: the deep match of the track with the highest momentum that holds it.
  const cornettostudy::DeepTrackMatch deepMatch(graph, tkR2t, event.get(assignableToken_));
  std::vector<int> trackOf(nP, -1);
  auto take = [&](unsigned int p, unsigned int t) {
    if (trackOf[p] < 0 || tracks[t].p() > tracks[trackOf[p]].p())
      trackOf[p] = t;
  };
  for (unsigned int t = 0; t < tracks.size(); ++t) {
    float purity;
    const int u = deepMatch(t, purity);
    if (u < 0)
      continue;
    take(u, t);
    for (auto const& d : truth::Particle(&graph, u).descendants())
      take(d.id(), t);
  }
  // Simulated HGCAL energy per cell and calo-boundary particle.
  std::unordered_map<uint32_t, std::vector<std::pair<unsigned int, float>>> cellTruth;
  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  for (unsigned int p = 0; p < nP; ++p) {
    if (!graph.particles()[p].isAtLevel(truth::LevelFlag::CaloBoundary))
      continue;
    hits.clear();
    index.appendSubgraphHits(truth::HitChannel::Calo, p, hits);
    for (auto const& h : hits)
      if (isHGCal(h.detId) && h.energy > 0.f)
        cellTruth[h.detId].emplace_back(p, h.energy);
  }
  auto out = std::make_unique<std::vector<int>>(lcs.size(), -1);
  std::unordered_map<int, double> perTrack;
  for (size_t v = 0; v < lcs.size(); ++v) {
    perTrack.clear();
    double total = 0.;
    for (auto const& [id, frac] : lcs[v].hitsAndFractions()) {
      auto it = cellTruth.find(id.rawId());
      if (it == cellTruth.end())
        continue;
      for (auto const& [p, e] : it->second) {
        total += frac * e;
        if (trackOf[p] >= 0)
          perTrack[trackOf[p]] += frac * e;
      }
    }
    if (!(total > 0.))
      continue;
    for (auto const& [t, e] : perTrack)
      if (e > minFraction_ * total)
        (*out)[v] = t;
  }
  event.put(std::move(out));
}

DEFINE_FWK_MODULE(TruthClaimProducer);
