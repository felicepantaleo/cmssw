// Study producer, not for integration. Oracle shares of the neutral candidates: per interpretation trackster, the share
// that a perfect model with a given target gives. The candidates come from a reference arbitration (share 1). For each
// neutral candidate the footprint truth (simulated HGCAL energy of its layer clusters, divided by the vertex
// multiplicity) is split by calo-boundary particle. The energy of the particles that the deep match of a charged
// candidate track holds (DeepTrackMatch.h) is removed in every target. Targets, as fractions of the footprint truth:
//   A1: the rest.
//   T:  the rest without the out-of-time particles (bunch crossing not 0).
//   N:  the rest without the charged particles. A particle is charged or neutral by its ancestor at the generator
//       final-state level (ReconstructableFinalState), or by its own pdg id if it has no such ancestor. A pdg id in
//       neither list is kept and counted.
//   TN: the rest without both.
// A candidate with no footprint truth gets share 0. A trackster that is not a neutral candidate gets -1 (no override).
// Products: std::vector<float> per interpretation trackster, instances A1, T, N, TN, and One (share 1, a wiring check).
// Tree "oracle": one row per neutral candidate with trackster pT below maxPt, with the truth split.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CornettoStudy/Probe/plugins/DeepTrackMatch.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/InteractionId.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"

#include "TTree.h"

namespace {
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
  bool isHGCal(uint32_t rawId) {
    const auto det = DetId(rawId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }
  // 1 neutral, 2 charged, 0 in neither list.
  int chargeClass(int pdg) {
    switch (std::abs(pdg)) {
      case 22:
      case 111:
      case 130:
      case 310:
      case 2112:
      case 3122:
      case 3212:
      case 3322:
        return 1;
      case 11:
      case 13:
      case 211:
      case 321:
      case 2212:
      case 3112:
      case 3222:
      case 3312:
      case 3334:
        return 2;
      default:
        return 0;
    }
  }
}  // namespace

class NeutralShareOracleProducer : public edm::one::EDProducer<edm::one::SharedResources> {
public:
  explicit NeutralShareOracleProducer(const edm::ParameterSet&);
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  // Class of a calo-boundary particle: chargeClass of its generator final-state ancestor, or of itself.
  struct Class {
    int cls = -1;
    bool ancestor = false;
    int pdg = 0;
  };
  Class classify(const truth::Graph& graph, unsigned int p, std::vector<Class>& cache) const;

  const edm::EDGetTokenT<std::vector<TICLCandidate>> candToken_;
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksterToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const edm::EDGetTokenT<MapType> tkR2tToken_;
  const edm::EDGetTokenT<std::vector<unsigned int>> assignableToken_;
  const double maxPt_;
  TTree* tree_;
  int event_, nTs_;
  float eta_, pt_, candE_, truthE_, trackedE_, inNeutralE_, inChargedE_, earlyNeutralE_, earlyChargedE_, lateNeutralE_,
      lateChargedE_, unknownE_, noAncestorE_, signalE_, share_[4];
  // Footprint truth energy of the pdg ids in neither list, and of the particles with no final-state ancestor.
  std::map<int, double> unknownPdgE_;
  double noAncestorTotalE_ = 0., totalE_ = 0.;
};

NeutralShareOracleProducer::NeutralShareOracleProducer(const edm::ParameterSet& cfg)
    : candToken_(consumes<std::vector<TICLCandidate>>(cfg.getParameter<edm::InputTag>("candidates"))),
      tracksterToken_(consumes<std::vector<ticl::Trackster>>(cfg.getParameter<edm::InputTag>("interpretations"))),
      graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      lcToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))),
      tkR2tToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("trackRecoToTruth"))),
      assignableToken_(consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("assignableRoots"))),
      maxPt_(cfg.getParameter<double>("maxPt")) {
  usesResource(TFileService::kSharedResource);
  for (const char* instance : {"A1", "T", "N", "TN", "One"})
    produces<std::vector<float>>(instance);
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("oracle", "oracle");
  tree_->Branch("event", &event_);
  tree_->Branch("nTs", &nTs_);
  for (auto [n, p] : std::vector<std::pair<const char*, float*>>{{"eta", &eta_},
                                                                {"pt", &pt_},
                                                                {"candE", &candE_},
                                                                {"truthE", &truthE_},
                                                                {"trackedE", &trackedE_},
                                                                {"inNeutralE", &inNeutralE_},
                                                                {"inChargedE", &inChargedE_},
                                                                {"earlyNeutralE", &earlyNeutralE_},
                                                                {"earlyChargedE", &earlyChargedE_},
                                                                {"lateNeutralE", &lateNeutralE_},
                                                                {"lateChargedE", &lateChargedE_},
                                                                {"unknownE", &unknownE_},
                                                                {"noAncestorE", &noAncestorE_},
                                                                {"signalE", &signalE_}})
    tree_->Branch(n, p);
  tree_->Branch("share", share_, "share[4]/F");
}

NeutralShareOracleProducer::Class NeutralShareOracleProducer::classify(const truth::Graph& graph,
                                                                       unsigned int p,
                                                                       std::vector<Class>& cache) const {
  if (cache[p].cls >= 0)
    return cache[p];
  Class c;
  unsigned int q = p;
  for (int depth = 0; depth < 256; ++depth) {
    if (graph.particles()[q].isAtLevel(truth::LevelFlag::ReconstructableFinalState)) {
      c.ancestor = true;
      break;
    }
    const auto parents = truth::Particle(&graph, q).parents();
    if (parents.empty())
      break;
    q = parents.front().id();
  }
  c.pdg = graph.particles()[c.ancestor ? q : p].pdgId;
  c.cls = chargeClass(c.pdg);
  cache[p] = c;
  return c;
}

void NeutralShareOracleProducer::produce(edm::Event& event, const edm::EventSetup&) {
  auto const& cands = event.get(candToken_);
  const auto tracksters = event.getHandle(tracksterToken_);
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  auto const& layerClusters = event.get(lcToken_);
  auto const& tkR2t = event.get(tkR2tToken_).getMap();
  event_ = event.id().event();
  const unsigned int nP = graph.particles().size();

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
  // Particles that the deep match of a charged candidate track holds, with their descendants.
  const cornettostudy::DeepTrackMatch deepMatch(graph, tkR2t, event.get(assignableToken_));
  std::vector<char> tracked(nP, 0);
  for (auto const& c : cands) {
    if (c.charge() == 0 || c.trackPtrs().empty() || c.trackPtrs()[0].isNull())
      continue;
    float purity;
    const int u = deepMatch(c.trackPtrs()[0].key(), purity);
    if (u < 0 || tracked[u])
      continue;
    tracked[u] = 1;
    for (auto const& d : truth::Particle(&graph, u).descendants())
      tracked[d.id()] = 1;
  }

  std::vector<Class> cache(nP);
  std::vector<std::vector<float>> out(5, std::vector<float>(tracksters->size(), -1.f));
  std::unordered_map<unsigned int, double> comp;
  for (auto const& cand : cands) {
    if (cand.charge() != 0 || cand.tracksters().empty())
      continue;
    const auto first = cand.tracksters().front();
    if (first.id() != tracksters.id())
      throw cms::Exception("LogicError") << "NeutralShareOracleProducer: the candidate tracksters are not the "
                                            "interpretation tracksters";
    comp.clear();
    double tracksterE = 0.;
    for (auto const& ts : cand.tracksters()) {
      tracksterE += ts->regressed_energy();
      for (size_t i = 0; i < ts->vertices().size(); ++i) {
        const auto lc = ts->vertices()[i];
        if (lc >= layerClusters.size())
          continue;
        const float share = 1.f / std::max<float>(1.f, ts->vertex_multiplicity()[i]);
        for (auto const& [id, frac] : layerClusters[lc].hitsAndFractions()) {
          auto it = cellTruth.find(id.rawId());
          if (it == cellTruth.end())
            continue;
          for (auto const& [p, e] : it->second)
            comp[p] += share * frac * e;
        }
      }
    }
    double truth = 0., trackedE = 0., removeT = 0., removeN = 0., removeTN = 0.;
    double inN = 0., inC = 0., earlyN = 0., earlyC = 0., lateN = 0., lateC = 0., unknown = 0., noAnc = 0., signal = 0.;
    for (auto const& [p, e] : comp) {
      truth += e;
      if (tracked[p]) {
        trackedE += e;
        continue;
      }
      const auto cl = classify(graph, p, cache);
      const int bx = truth::bunchCrossingOf(graph.particles()[p].eventId);
      const bool charged = cl.cls == 2;
      if (cl.cls == 0) {
        unknown += e;
        unknownPdgE_[cl.pdg] += e;
      }
      if (!cl.ancestor) {
        noAnc += e;
        noAncestorTotalE_ += e;
      }
      if (graph.particles()[p].eventId == 0)
        signal += e;
      if (bx < 0)
        (charged ? earlyC : earlyN) += e;
      else if (bx > 0)
        (charged ? lateC : lateN) += e;
      else
        (charged ? inC : inN) += e;
      if (bx != 0)
        removeT += e;
      if (charged)
        removeN += e;
      if (bx != 0 || charged)
        removeTN += e;
    }
    totalE_ += truth;
    const auto keep = [truth, trackedE](double removed) {
      return truth > 0. ? static_cast<float>(std::clamp((truth - trackedE - removed) / truth, 0., 1.)) : 0.f;
    };
    share_[0] = keep(0.);
    share_[1] = keep(removeT);
    share_[2] = keep(removeN);
    share_[3] = keep(removeTN);
    for (int t = 0; t < 4; ++t)
      out[t][first.key()] = share_[t];
    out[4][first.key()] = 1.f;
    pt_ = tracksterE / std::cosh(cand.eta());
    if (pt_ >= maxPt_)
      continue;
    nTs_ = cand.tracksters().size();
    eta_ = cand.eta();
    candE_ = cand.energy();
    truthE_ = truth;
    trackedE_ = trackedE;
    inNeutralE_ = inN;
    inChargedE_ = inC;
    earlyNeutralE_ = earlyN;
    earlyChargedE_ = earlyC;
    lateNeutralE_ = lateN;
    lateChargedE_ = lateC;
    unknownE_ = unknown;
    noAncestorE_ = noAnc;
    signalE_ = signal;
    tree_->Fill();
  }
  const char* instances[5] = {"A1", "T", "N", "TN", "One"};
  for (int t = 0; t < 5; ++t)
    event.put(std::make_unique<std::vector<float>>(std::move(out[t])), instances[t]);
}

void NeutralShareOracleProducer::endJob() {
  edm::LogPrint("NeutralShareOracle") << "NeutralShareOracle: footprint truth " << totalE_
                                      << " GeV, with no final-state ancestor " << noAncestorTotalE_ << " GeV";
  for (auto const& [pdg, e] : unknownPdgE_)
    edm::LogPrint("NeutralShareOracle") << "NeutralShareOracle: pdg id in neither list " << pdg << ": " << e << " GeV";
}

DEFINE_FWK_MODULE(NeutralShareOracleProducer);
