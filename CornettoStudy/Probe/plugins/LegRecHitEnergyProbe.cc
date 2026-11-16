// Study probe, not for integration. One row per member of a truth level in 1.5 < |eta| < 3.0: the HGCAL rechit
// energy of the member as the truth-graph associators weight it. A cell gives the member its simulated share of the
// cell (the member subgraph over all particles) times the rechit energy of the cell.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"

#include "TTree.h"

namespace {
  bool isHGCal(uint32_t rawId) {
    const auto det = DetId(rawId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }
}  // namespace

class LegRecHitEnergyProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit LegRecHitEnergyProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> recHitTokens_;
  std::vector<edm::EDGetTokenT<std::vector<unsigned int>>> levelTokens_;

  TTree* tree_;
  unsigned long long event_;
  int level_, particle_, pdg_, nCells_;
  // eSim: simulated HGCAL energy of the member; eRecHit: its rechit energy; eRecHitOwn: the part on cells where the
  // member holds more than half of the simulated energy.
  float pt_, eta_, eSim_, eRecHit_, eRecHitOwn_;
};

LegRecHitEnergyProbe::LegRecHitEnergyProbe(const edm::ParameterSet& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))) {
  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("hgcalRecHits"))
    recHitTokens_.push_back(consumes<HGCRecHitCollection>(tag));
  for (auto name : cfg.getParameter<std::vector<std::string>>("truthLevels")) {
    name[0] = std::toupper(static_cast<unsigned char>(name[0]));
    levelTokens_.push_back(
        consumes<std::vector<unsigned int>>(edm::InputTag("truthBranchTargets", "truthToRecoTargets" + name)));
  }
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("members", "members");
  tree_->Branch("event", &event_);
  tree_->Branch("level", &level_);
  tree_->Branch("particle", &particle_);
  tree_->Branch("pdg", &pdg_);
  tree_->Branch("pt", &pt_);
  tree_->Branch("eta", &eta_);
  tree_->Branch("nCells", &nCells_);
  tree_->Branch("eSim", &eSim_);
  tree_->Branch("eRecHit", &eRecHit_);
  tree_->Branch("eRecHitOwn", &eRecHitOwn_);
}

void LegRecHitEnergyProbe::analyze(const edm::Event& event, const edm::EventSetup&) {
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  const uint32_t nP = graph.nParticles();
  event_ = event.id().event();

  std::unordered_map<uint32_t, float> recHitE;
  for (auto const& token : recHitTokens_)
    for (auto const& h : event.get(token))
      recHitE[h.id().rawId()] += h.energy();
  // The simulated energy of all particles on each cell.
  std::unordered_map<uint32_t, float> cellSim;
  for (uint32_t p = 0; p < nP; ++p)
    for (auto const& h : index.directHits(truth::HitChannel::Calo, p))
      if (isHGCal(h.detId))
        cellSim[h.detId] += h.energy;

  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  std::unordered_map<uint32_t, float> mine;
  for (std::size_t l = 0; l < levelTokens_.size(); ++l) {
    const auto targetsH = event.getHandle(levelTokens_[l]);
    if (!targetsH.isValid())
      continue;
    level_ = static_cast<int>(l);
    for (const uint32_t p : *targetsH) {
      if (p >= nP || truth::isInvisible(graph.particles()[p].pdgId))
        continue;
      auto const& pd = graph.particles()[p];
      if (pd.momentum.pt() <= 0. || std::abs(pd.momentum.eta()) < 1.5 || std::abs(pd.momentum.eta()) > 3.0)
        continue;
      particle_ = static_cast<int>(p);
      pdg_ = pd.pdgId;
      pt_ = pd.momentum.pt();
      eta_ = pd.momentum.eta();
      hits.clear();
      index.appendSubgraphHits(truth::HitChannel::Calo, p, hits);
      mine.clear();
      for (auto const& h : hits)
        if (isHGCal(h.detId) && h.energy > 0.f)
          mine[h.detId] += h.energy;
      eSim_ = eRecHit_ = eRecHitOwn_ = 0.f;
      nCells_ = 0;
      for (auto const& [id, e] : mine) {
        eSim_ += e;
        auto const r = recHitE.find(id);
        auto const s = cellSim.find(id);
        if (r == recHitE.end() || s == cellSim.end() || s->second <= 0.f)
          continue;
        const float share = std::min(1.f, e / s->second);
        eRecHit_ += share * r->second;
        if (share > 0.5f)
          eRecHitOwn_ += share * r->second;
        ++nCells_;
      }
      tree_->Fill();
    }
  }
}

DEFINE_FWK_MODULE(LegRecHitEnergyProbe);
