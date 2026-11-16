// Study probe, not for integration. e/gamma objects of a particle-gun sample matched to the gun particles with the
// truth graph. The HGCAL simulated energy of a gun particle is the energy of the calorimeter hits of its subgraph.
// The share of an object is the simulated energy of the gun particle in the cells of the object (weighted by the
// cell fractions) over the HGCAL simulated energy of the gun particle. One row per gun particle (tree "parts") with,
// per object kind, the largest share of an object whose largest gun contributor is that particle and the energy of
// that object. One row per object above minObjectPt (tree "objs") with the share of its simulated energy that comes
// from the gun particles.

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"

#include "TTree.h"

namespace {
  bool isHGCal(uint32_t rawId) {
    const auto det = DetId(rawId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }

  using CellMap = std::unordered_map<uint32_t, float>;

  // Object kinds, in the order of the per-kind branches.
  enum Kind { kSC = 0, kEle, kPho, kTiclE, kTiclG, kNKinds };
  const char* const kKindNames[kNKinds] = {"sc", "ele", "pho", "ticlE", "ticlG"};

  struct Object {
    int kind;
    float pt, eta, energy;
    std::vector<std::pair<uint32_t, float>> cells;  // (detId, fraction)
  };
}  // namespace

class EGammaTruthProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit EGammaTruthProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<truth::Graph> graphToken_;
  edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  edm::EDGetTokenT<std::vector<reco::SuperCluster>> scToken_;
  edm::EDGetTokenT<std::vector<reco::GsfElectron>> eleToken_;
  edm::EDGetTokenT<std::vector<reco::Photon>> phoToken_;
  edm::EDGetTokenT<std::vector<TICLCandidate>> candToken_;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const double minAbsEta_, maxAbsEta_, minObjectPt_;

  TTree* parts_;
  TTree* objs_;
  int pEvent_, pPdg_;
  float pPt_, pEta_, pE_, pSimE_;
  float share_[kNKinds], energy_[kNKinds], scRaw_;
  int oEvent_, oKind_;
  float oPt_, oEta_, oGunShare_, oSimE_;
};

EGammaTruthProbe::EGammaTruthProbe(const edm::ParameterSet& cfg)
    : graphToken_(consumes(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes(cfg.getParameter<edm::InputTag>("hitIndex"))),
      scToken_(consumes(cfg.getParameter<edm::InputTag>("superClusters"))),
      eleToken_(consumes(cfg.getParameter<edm::InputTag>("electrons"))),
      phoToken_(consumes(cfg.getParameter<edm::InputTag>("photons"))),
      candToken_(consumes(cfg.getParameter<edm::InputTag>("candidates"))),
      lcToken_(consumes(cfg.getParameter<edm::InputTag>("layerClusters"))),
      minAbsEta_(cfg.getParameter<double>("minAbsEta")),
      maxAbsEta_(cfg.getParameter<double>("maxAbsEta")),
      minObjectPt_(cfg.getParameter<double>("minObjectPt")) {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  parts_ = fs->make<TTree>("parts", "gun particles");
  parts_->Branch("event", &pEvent_);
  parts_->Branch("pdg", &pPdg_);
  parts_->Branch("pt", &pPt_);
  parts_->Branch("eta", &pEta_);
  parts_->Branch("energy", &pE_);
  parts_->Branch("simE", &pSimE_);
  parts_->Branch("scRaw", &scRaw_);
  for (int k = 0; k < kNKinds; ++k) {
    parts_->Branch((std::string(kKindNames[k]) + "Share").c_str(), &share_[k]);
    parts_->Branch((std::string(kKindNames[k]) + "E").c_str(), &energy_[k]);
  }
  objs_ = fs->make<TTree>("objs", "objects");
  objs_->Branch("event", &oEvent_);
  objs_->Branch("kind", &oKind_);
  objs_->Branch("pt", &oPt_);
  objs_->Branch("eta", &oEta_);
  objs_->Branch("gunShare", &oGunShare_);
  objs_->Branch("simE", &oSimE_);
}

void EGammaTruthProbe::analyze(const edm::Event& event, const edm::EventSetup&) {
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  auto const& lcs = event.get(lcToken_);
  pEvent_ = oEvent_ = event.id().event();

  // Simulated energy of every HGCAL cell, from all particles.
  CellMap all;
  for (uint32_t i = 0; i < index.nParticles(); ++i)
    for (auto const& h : index.directHits(truth::HitChannel::Calo, i))
      if (isHGCal(h.detId))
        all[h.detId] += h.energy;

  // Gun particles: signal generator e and photons, stable, in the acceptance.
  std::vector<uint32_t> guns;
  std::vector<CellMap> gunCells;
  std::vector<float> gunSim;
  for (uint32_t i = 0; i < graph.particles().size(); ++i) {
    auto const& pd = graph.particles()[i];
    const truth::Particle p(&graph, i);
    const int a = std::abs(pd.pdgId);
    const double ae = std::abs(pd.momentum.eta());
    if ((a != 11 && a != 22) || !pd.isSignal() || !p.hasGen() || p.status() != 1 || ae < minAbsEta_ || ae > maxAbsEta_)
      continue;
    std::vector<truth::LogicalGraphHitIndex::Hit> hits;
    index.appendSubgraphHits(truth::HitChannel::Calo, i, hits);
    CellMap cells;
    float sum = 0.f;
    for (auto const& h : hits)
      if (isHGCal(h.detId)) {
        cells[h.detId] += h.energy;
        sum += h.energy;
      }
    guns.push_back(i);
    gunCells.push_back(std::move(cells));
    gunSim.push_back(sum);
  }

  // Objects and their cells.
  std::vector<Object> objects;
  auto clusterCells = [](reco::CaloCluster const& c, float weight, std::vector<std::pair<uint32_t, float>>& out) {
    for (auto const& [id, f] : c.hitsAndFractions())
      out.emplace_back(id.rawId(), f * weight);
  };
  auto scCells = [&](reco::SuperCluster const& sc) {
    std::vector<std::pair<uint32_t, float>> out;
    for (auto const& c : sc.clusters())
      clusterCells(*c, 1.f, out);
    return out;
  };
  std::vector<float> scRaw;
  for (auto const& sc : event.get(scToken_)) {
    if (std::abs(sc.position().eta()) < minAbsEta_)
      continue;
    const float pt = sc.energy() / std::cosh(sc.position().eta());
    objects.push_back({kSC, pt, static_cast<float>(sc.position().eta()), static_cast<float>(sc.energy()), scCells(sc)});
    scRaw.push_back(sc.rawEnergy());
  }
  for (auto const& e : event.get(eleToken_))
    objects.push_back({kEle,
                       static_cast<float>(e.pt()),
                       static_cast<float>(e.eta()),
                       static_cast<float>(e.energy()),
                       scCells(*e.superCluster())});
  for (auto const& g : event.get(phoToken_))
    objects.push_back({kPho,
                       static_cast<float>(g.pt()),
                       static_cast<float>(g.eta()),
                       static_cast<float>(g.energy()),
                       scCells(*g.superCluster())});
  for (auto const& c : event.get(candToken_)) {
    const int a = std::abs(c.pdgId());
    if (a != 11 && a != 22 && a != 111)
      continue;
    std::vector<std::pair<uint32_t, float>> cells;
    for (auto const& t : c.tracksters())
      for (unsigned int k = 0; k < t->vertices().size(); ++k)
        clusterCells(lcs[t->vertices()[k]], 1.f / t->vertex_multiplicity()[k], cells);
    objects.push_back({a == 11 ? kTiclE : kTiclG,
                       static_cast<float>(c.pt()),
                       static_cast<float>(c.eta()),
                       static_cast<float>(c.energy()),
                       std::move(cells)});
  }

  // Share of each gun particle in each object, the best gun of each object, and the object rows.
  std::vector<std::vector<float>> shares(objects.size(), std::vector<float>(guns.size(), 0.f));
  std::vector<int> bestGun(objects.size(), -1);
  for (unsigned int o = 0; o < objects.size(); ++o) {
    float sim = 0.f, gunSum = 0.f, best = 0.f;
    for (auto const& [id, f] : objects[o].cells)
      if (auto it = all.find(id); it != all.end())
        sim += f * it->second;
    for (unsigned int g = 0; g < guns.size(); ++g) {
      float s = 0.f;
      for (auto const& [id, f] : objects[o].cells)
        if (auto it = gunCells[g].find(id); it != gunCells[g].end())
          s += f * it->second;
      gunSum += s;
      shares[o][g] = gunSim[g] > 0.f ? s / gunSim[g] : 0.f;
      if (s > best) {
        best = s;
        bestGun[o] = g;
      }
    }
    if (objects[o].pt > minObjectPt_) {
      oKind_ = objects[o].kind;
      oPt_ = objects[o].pt;
      oEta_ = objects[o].eta;
      oSimE_ = sim;
      oGunShare_ = sim > 0.f ? gunSum / sim : 0.f;
      objs_->Fill();
    }
  }

  for (unsigned int g = 0; g < guns.size(); ++g) {
    auto const& pd = graph.particles()[guns[g]];
    pPdg_ = pd.pdgId;
    pPt_ = pd.momentum.pt();
    pEta_ = pd.momentum.eta();
    pE_ = pd.momentum.E();
    pSimE_ = gunSim[g];
    scRaw_ = -1.f;
    for (int k = 0; k < kNKinds; ++k)
      share_[k] = energy_[k] = 0.f;
    // The superclusters are the first objects, in the order of scRaw.
    for (unsigned int o = 0; o < objects.size(); ++o) {
      const int k = objects[o].kind;
      if (bestGun[o] == static_cast<int>(g) && shares[o][g] > share_[k]) {
        share_[k] = shares[o][g];
        energy_[k] = objects[o].energy;
        if (k == kSC)
          scRaw_ = scRaw[o];
      }
    }
    parts_->Fill();
  }
}

DEFINE_FWK_MODULE(EGammaTruthProbe);
