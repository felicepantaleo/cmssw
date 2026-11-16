// Study probe, not for integration. One TTree row per signal truth target (tau leg) in the endcap:
// the leg topology from the truth graph, what the linking inputs hold of it, the oracle linking
// ceiling, and the outcome of the output collections.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"

#include "TTree.h"

namespace {
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;

  bool isHGCal(uint32_t rawId) {
    const auto det = DetId(rawId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }

  double hgcalEnergy(truth::LogicalGraphHitIndex const& index, uint32_t particle) {
    std::vector<truth::LogicalGraphHitIndex::Hit> hits;
    index.appendSubgraphHits(truth::HitChannel::Calo, particle, hits);
    double e = 0.;
    for (auto const& h : hits)
      if (isHGCal(h.detId))
        e += h.energy;
    return e;
  }
}  // namespace

class TauLegTopologyProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TauLegTopologyProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  struct Coll {
    edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksters;
    edm::EDGetTokenT<MapType> truthToReco;
    edm::EDGetTokenT<MapType> recoToTruth;
  };
  Coll makeColl(std::string const& associator, std::string const& name);

  edm::EDGetTokenT<truth::Graph> graphToken_;
  edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  edm::EDGetTokenT<std::vector<unsigned int>> targetsToken_;
  std::vector<Coll> inputs_, outputs_;
  const double minAbsEta_, maxAbsEta_, minPt_;

  static constexpr int kMaxOut = 4;
  static constexpr int kNS = 5, kNT = 6;
  TTree* tree_;
  TTree* pieces_;
  int pEvent_, pLegPdg_, pTimeValid_, pPileup_;
  float pDtNs_, pPull_, pLeadErr_, pErr_, pPurity_, pE_, pS_, pDT_, pLeadE_;
  int event_, pdgId_, nCB_, inN5_, oracleN_;
  float pt_, eta_, energy_, eHgcal_, cbLeadFrac_, cbDR12_, cb2ndFrac_;
  float inSum_, inLeadFrac_, inLeadE_, in2ndFrac_, in2ndE_, oracleFrac_, oraclePur_;
  int outInd_[kMaxOut];
  float outLead_[kMaxOut], outLeadScore_[kMaxOut];
  float phi_, nearestLegDR_;
  float sigE_[kNS * kNT], bkgE_[kNS * kNT];
};

TauLegTopologyProbe::Coll TauLegTopologyProbe::makeColl(std::string const& associator, std::string const& name) {
  return {consumes<std::vector<ticl::Trackster>>(edm::InputTag(name)),
          consumes<MapType>(edm::InputTag(associator, name + "TruthToReco")),
          consumes<MapType>(edm::InputTag(associator, name + "RecoToTruthFixed"))};
}

TauLegTopologyProbe::TauLegTopologyProbe(const edm::ParameterSet& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      targetsToken_(consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("targets"))),
      minAbsEta_(cfg.getParameter<double>("minAbsEta")),
      maxAbsEta_(cfg.getParameter<double>("maxAbsEta")),
      minPt_(cfg.getParameter<double>("minPt")) {
  usesResource(TFileService::kSharedResource);
  const auto associator = cfg.getParameter<std::string>("associator");
  for (auto const& n : cfg.getParameter<std::vector<std::string>>("inputs"))
    inputs_.push_back(makeColl(associator, n));
  for (auto const& n : cfg.getParameter<std::vector<std::string>>("outputs"))
    outputs_.push_back(makeColl(associator, n));
  if (outputs_.size() > kMaxOut)
    throw cms::Exception("Configuration") << "at most " << kMaxOut << " outputs";

  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("legs", "legs");
  tree_->Branch("event", &event_);
  pieces_ = fs->make<TTree>("pieces", "input tracksters around the lead piece of each leg");
  pieces_->Branch("event", &pEvent_);
  pieces_->Branch("legPdg", &pLegPdg_);
  pieces_->Branch("timeValid", &pTimeValid_);
  pieces_->Branch("pileup", &pPileup_);
  pieces_->Branch("dtNs", &pDtNs_);
  pieces_->Branch("pull", &pPull_);
  pieces_->Branch("leadErr", &pLeadErr_);
  pieces_->Branch("err", &pErr_);
  pieces_->Branch("purity", &pPurity_);
  pieces_->Branch("energy", &pE_);
  pieces_->Branch("leadE", &pLeadE_);
  pieces_->Branch("s", &pS_);
  pieces_->Branch("dT", &pDT_);
  tree_->Branch("pdgId", &pdgId_);
  tree_->Branch("pt", &pt_);
  tree_->Branch("eta", &eta_);
  tree_->Branch("phi", &phi_);
  tree_->Branch("nearestLegDR", &nearestLegDR_);
  tree_->Branch("sigE", sigE_, ("sigE[" + std::to_string(kNS * kNT) + "]/F").c_str());
  tree_->Branch("bkgE", bkgE_, ("bkgE[" + std::to_string(kNS * kNT) + "]/F").c_str());
  tree_->Branch("energy", &energy_);
  tree_->Branch("eHgcal", &eHgcal_);
  tree_->Branch("nCB", &nCB_);
  tree_->Branch("cbLeadFrac", &cbLeadFrac_);
  tree_->Branch("cb2ndFrac", &cb2ndFrac_);
  tree_->Branch("cbDR12", &cbDR12_);
  tree_->Branch("inSum", &inSum_);
  tree_->Branch("inN5", &inN5_);
  tree_->Branch("inLeadFrac", &inLeadFrac_);
  tree_->Branch("inLeadE", &inLeadE_);
  tree_->Branch("in2ndFrac", &in2ndFrac_);
  tree_->Branch("in2ndE", &in2ndE_);
  tree_->Branch("oracleN", &oracleN_);
  tree_->Branch("oracleFrac", &oracleFrac_);
  tree_->Branch("oraclePur", &oraclePur_);
  tree_->Branch("outInd", outInd_, ("outInd[" + std::to_string(kMaxOut) + "]/I").c_str());
  tree_->Branch("outLead", outLead_, ("outLead[" + std::to_string(kMaxOut) + "]/F").c_str());
  tree_->Branch("outLeadScore", outLeadScore_, ("outLeadScore[" + std::to_string(kMaxOut) + "]/F").c_str());
}

void TauLegTopologyProbe::analyze(const edm::Event& event, const edm::EventSetup&) {
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  auto const& targets = event.get(targetsToken_);
  event_ = event.id().event();

  struct View {
    std::vector<ticl::Trackster> const* ts;
    ticl::mapWithSharedEnergyAndScore const* t2r;
    ticl::mapWithSharedEnergyAndScore const* r2t;
  };
  auto view = [&event](Coll const& c) {
    return View{&event.get(c.tracksters), &event.get(c.truthToReco).getMap(), &event.get(c.recoToTruth).getMap()};
  };
  std::vector<View> ins, outs;
  for (auto const& c : inputs_)
    ins.push_back(view(c));
  for (auto const& c : outputs_)
    outs.push_back(view(c));
  auto recoScore = [](View const& v, unsigned int r, unsigned int b) {
    if (r >= v.r2t->size())
      return 1.f;
    for (auto const& m : (*v.r2t)[r])
      if (m.index() == b)
        return m.score();
    return 1.f;
  };

  std::vector<uint32_t> signalLegs;
  for (unsigned int b : targets)
    if (b < graph.particles().size() && graph.particles()[b].isSignal() && graph.particles()[b].momentum.pt() > 0.)
      signalLegs.push_back(b);

  for (unsigned int b : targets) {
    if (b >= graph.particles().size())
      continue;
    auto const& pd = graph.particles()[b];
    if (!pd.isSignal())
      continue;
    auto const& p4 = pd.momentum;
    if (p4.pt() < minPt_ || std::abs(p4.eta()) < minAbsEta_ || std::abs(p4.eta()) > maxAbsEta_)
      continue;
    pdgId_ = pd.pdgId;
    pt_ = p4.pt();
    eta_ = p4.eta();
    phi_ = p4.phi();
    nearestLegDR_ = 99.f;
    for (uint32_t o : signalLegs)
      if (o != b) {
        auto const& q = graph.particles()[o].momentum;
        nearestLegDR_ = std::min(nearestLegDR_, static_cast<float>(reco::deltaR(p4.eta(), p4.phi(), q.eta(), q.phi())));
      }
    energy_ = p4.E();
    eHgcal_ = hgcalEnergy(index, b);

    // Calo-boundary particles of the leg, the leg itself included when it is one.
    std::vector<std::pair<double, uint32_t>> cb;
    const truth::Particle leg(&graph, b);
    if (pd.isAtLevel(truth::LevelFlag::CaloBoundary))
      cb.emplace_back(eHgcal_, b);
    else
      for (auto const& d : leg.descendants())
        if (d.data().isAtLevel(truth::LevelFlag::CaloBoundary))
          cb.emplace_back(hgcalEnergy(index, d.id()), d.id());
    std::sort(cb.begin(), cb.end(), [](auto const& x, auto const& y) { return x.first > y.first; });
    const double den = eHgcal_ > 0. ? eHgcal_ : 1.;
    nCB_ = std::count_if(cb.begin(), cb.end(), [den](auto const& c) { return c.first > 0.05 * den; });
    cbLeadFrac_ = cb.empty() ? 0.f : cb[0].first / den;
    cb2ndFrac_ = cb.size() > 1 ? cb[1].first / den : 0.f;
    cbDR12_ = -1.f;
    if (cb.size() > 1) {
      auto const& a = graph.particles()[cb[0].second].momentum;
      auto const& c = graph.particles()[cb[1].second].momentum;
      if (a.pt() > 0. && c.pt() > 0.)
        cbDR12_ = reco::deltaR(a.eta(), a.phi(), c.eta(), c.phi());
    }

    // What the linking inputs hold of the leg, and the oracle merge of the input tracksters
    // the leg dominates (reco purity >= 0.5).
    std::vector<std::pair<float, float>> frags;  // (fraction of the leg, raw energy)
    inSum_ = 0.f;
    double oE = 0., oPurE = 0.;
    oracleFrac_ = 0.f;
    oracleN_ = 0;
    for (auto const& v : ins) {
      if (b >= v.t2r->size())
        continue;
      for (auto const& m : (*v.t2r)[b]) {
        if (m.index() >= v.ts->size())
          continue;
        const float frac = m.sharedEnergy();
        const float e = (*v.ts)[m.index()].raw_energy();
        frags.emplace_back(frac, e);
        inSum_ += frac;
        const float purity = 1.f - recoScore(v, m.index(), b);
        if (purity >= 0.5f) {
          ++oracleN_;
          oracleFrac_ += frac;
          oE += e;
          oPurE += purity * e;
        }
      }
    }
    oraclePur_ = oE > 0. ? oPurE / oE : 0.f;
    std::sort(frags.begin(), frags.end(), [](auto const& x, auto const& y) { return x.first > y.first; });
    inN5_ = std::count_if(frags.begin(), frags.end(), [](auto const& f) { return f.first > 0.05f; });
    inLeadFrac_ = frags.empty() ? 0.f : frags[0].first;
    inLeadE_ = frags.empty() ? 0.f : frags[0].second;
    in2ndFrac_ = frags.size() > 1 ? frags[1].first : 0.f;
    in2ndE_ = frags.size() > 1 ? frags[1].second : 0.f;

    // Signal (leg) and background energy of every input trackster around the largest input piece,
    // in bins of longitudinal s and transverse dT along the projective axis of that piece.
    std::fill(sigE_, sigE_ + kNS * kNT, 0.f);
    std::fill(bkgE_, bkgE_ + kNS * kNT, 0.f);
    ticl::Trackster const* leadTs = nullptr;
    float leadFrac = -1.f;
    for (auto const& v : ins)
      if (b < v.t2r->size())
        for (auto const& m : (*v.t2r)[b])
          if (m.index() < v.ts->size() && m.sharedEnergy() > leadFrac) {
            leadFrac = m.sharedEnergy();
            leadTs = &(*v.ts)[m.index()];
          }
    if (leadTs != nullptr) {
      const auto lb = leadTs->barycenter();
      const float lm = std::sqrt(lb.Mag2());
      const auto axis = lm > 0.f ? lb / lm : lb;
      static const float sEdges[kNS - 1] = {-20.f, 0.f, 20.f, 60.f};
      static const float tEdges[kNT - 1] = {2.f, 5.f, 10.f, 20.f, 40.f};
      for (auto const& v : ins)
        for (unsigned int r = 0; r < v.ts->size(); ++r) {
          auto const& tr = (*v.ts)[r];
          const auto rb = tr.barycenter();
          if (rb.z() * lb.z() <= 0.f || std::abs(rb.eta() - lb.eta()) > 0.5f ||
              std::abs(reco::deltaPhi(rb.phi(), lb.phi())) > 0.5f)
            continue;
          const auto D = rb - lb;
          const float sP = D.Dot(axis);
          const float dT = std::sqrt(std::max(0.f, static_cast<float>(D.Mag2()) - sP * sP));
          const int is = std::upper_bound(sEdges, sEdges + kNS - 1, sP) - sEdges;
          const int it = std::upper_bound(tEdges, tEdges + kNT - 1, dT) - tEdges;
          const float purity = 1.f - recoScore(v, r, b);
          sigE_[is * kNT + it] += purity * tr.raw_energy();
          bkgE_[is * kNT + it] += (1.f - purity) * tr.raw_energy();
          if (&tr != leadTs && sP > -20.f && sP < 60.f && dT < 10.f) {
            pEvent_ = event_;
            pLegPdg_ = pdgId_;
            pPurity_ = purity;
            pE_ = tr.raw_energy();
            pLeadE_ = leadTs->raw_energy();
            pS_ = sP;
            pDT_ = dT;
            pLeadErr_ = leadTs->timeError();
            pErr_ = tr.timeError();
            pTimeValid_ = (leadTs->timeError() > 0.f && tr.timeError() > 0.f) ? 1 : 0;
            pDtNs_ = pTimeValid_ ? tr.time() - leadTs->time() : -999.f;
            pPull_ = pTimeValid_ ? pDtNs_ / std::sqrt(pLeadErr_ * pLeadErr_ + pErr_ * pErr_) : -999.f;
            // Pileup piece: its best reco-to-truth match is a pileup particle.
            pPileup_ = -1;
            if (r < v.r2t->size() && !(*v.r2t)[r].empty()) {
              auto const& row = (*v.r2t)[r];
              auto bestIt = std::min_element(
                  row.begin(), row.end(), [](auto const& x, auto const& y) { return x.score() < y.score(); });
              if (bestIt->index() < graph.particles().size())
                pPileup_ = graph.particles()[bestIt->index()].isFromPileup() ? 1 : 0;
            }
            pieces_->Fill();
          }
        }
    }

    for (int o = 0; o < kMaxOut; ++o) {
      outInd_[o] = -1;
      outLead_[o] = outLeadScore_[o] = -1.f;
      if (o >= static_cast<int>(outs.size()))
        continue;
      auto const& v = outs[o];
      float best = -1.f;
      unsigned int bestR = 0;
      if (b < v.t2r->size())
        for (auto const& m : (*v.t2r)[b])
          if (m.sharedEnergy() > best) {
            best = m.sharedEnergy();
            bestR = m.index();
          }
      outLead_[o] = best;
      outLeadScore_[o] = best >= 0.f ? recoScore(v, bestR, b) : -1.f;
      outInd_[o] = (best > 0.5f && outLeadScore_[o] < 0.6f) ? 1 : 0;
    }
    tree_->Fill();
  }
}

DEFINE_FWK_MODULE(TauLegTopologyProbe);
