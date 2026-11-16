// Study probe, not for integration. One row per HGCAL scintillator cell with a simulated deposit or a rechit: the
// simulated energy from the calo-boundary particles of the truth graph (signal = eventId 0, and pileup), the rechit
// energy, and the layer cluster that holds the cell (its number of hits and energy).

#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

#include "TTree.h"

class SciCellProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SciCellProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<HGCRecHitCollection> recHitToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  TTree* tree_;
  int event_, layer_, ring_, lcNHits_;
  float simSignal_, simPileup_, recE_, lcE_;
};

SciCellProbe::SciCellProbe(const edm::ParameterSet& cfg)
    : recHitToken_(consumes<HGCRecHitCollection>(cfg.getParameter<edm::InputTag>("recHits"))),
      lcToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))),
      graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))) {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("cells", "cells");
  tree_->Branch("event", &event_);
  tree_->Branch("layer", &layer_);
  tree_->Branch("ring", &ring_);
  tree_->Branch("lcNHits", &lcNHits_);
  tree_->Branch("simSignal", &simSignal_);
  tree_->Branch("simPileup", &simPileup_);
  tree_->Branch("recE", &recE_);
  tree_->Branch("lcE", &lcE_);
}

void SciCellProbe::analyze(const edm::Event& event, const edm::EventSetup&) {
  auto const& recHits = event.get(recHitToken_);
  auto const& lcs = event.get(lcToken_);
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  event_ = event.id().event();

  struct Cell {
    float simSignal = 0.f, simPileup = 0.f, recE = -1.f, lcE = -1.f;
    int lcNHits = 0;
  };
  std::unordered_map<uint32_t, Cell> cells;
  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  for (unsigned int p = 0; p < graph.particles().size(); ++p) {
    auto const& pd = graph.particles()[p];
    if (!pd.isAtLevel(truth::LevelFlag::CaloBoundary))
      continue;
    hits.clear();
    index.appendSubgraphHits(truth::HitChannel::Calo, p, hits);
    for (auto const& h : hits)
      if (DetId(h.detId).det() == DetId::HGCalHSc && h.energy > 0.f)
        (pd.eventId == 0 ? cells[h.detId].simSignal : cells[h.detId].simPileup) += h.energy;
  }
  for (auto const& rh : recHits)
    cells[rh.detid().rawId()].recE = rh.energy();
  for (auto const& lc : lcs) {
    auto const& hf = lc.hitsAndFractions();
    if (hf.empty() || hf[0].first.det() != DetId::HGCalHSc)
      continue;
    for (auto const& [id, f] : hf) {
      auto& c = cells[id.rawId()];
      c.lcNHits = static_cast<int>(hf.size());
      c.lcE = lc.energy();
    }
  }
  for (auto const& [id, c] : cells) {
    const HGCScintillatorDetId sid(id);
    layer_ = sid.layer();
    ring_ = sid.ring();
    simSignal_ = c.simSignal;
    simPileup_ = c.simPileup;
    recE_ = c.recE;
    lcE_ = c.lcE;
    lcNHits_ = c.lcNHits;
    tree_->Fill();
  }
}

DEFINE_FWK_MODULE(SciCellProbe);
