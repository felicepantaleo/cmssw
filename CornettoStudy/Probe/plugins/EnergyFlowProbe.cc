// Study probe, not for integration. Where the HGCAL energy of each track's particle ends up, from the truth graph.
//   tracks: one row per charged TICL candidate whose track matches a truth particle (Fixed reco-to-truth): the deposit of
//           the particle and its descendants in HGCAL (simulated energy) split into its own candidate, other charged
//           candidates, neutral candidates (EM-typed or not) and no candidate; the reco energy of neutral candidates
//           that belongs to it (double counted); its track and its first decay or interaction vertex.
//   frags:  one row per (neutral candidate, tracked particle) with a share of the neutral's truth energy: the share, the
//           neutral, its distance to the propagated track, and the calo-boundary particle that carries most of the
//           shared energy with its production process and vertex.

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTICL/Common/interface/TICLUtils.h"
#include "RecoTICL/Interpretation/interface/TrackImpact.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/Vertex.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "TTree.h"

namespace {
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
  bool isHGCal(uint32_t rawId) {
    const auto det = DetId(rawId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }
}  // namespace

class EnergyFlowProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit EnergyFlowProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<std::vector<TICLCandidate>> candToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const edm::EDGetTokenT<std::vector<reco::Track>> trackToken_;
  const edm::EDGetTokenT<MapType> r2tToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<int>> modeToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propToken_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hgconsToken_;
  TTree* tracks_;
  TTree* frags_;
  // tracks
  int event_, row_, tPdg_, tSignal_, tMode_, tHasTs_, tReachesCalo_, tEndReason_, tMissOuter_, tCandPdg_;
  float tP_, tPt_, tEta_, tOuterZ_, tOuterR_, tEndR_, tEndZ_, tDep_, tDepOwn_, tDepOther_, tDepNeutral_, tDepNeutralEM_,
      tDepNone_, tDoubleE_, tCandE_;
  // frags
  int fRow_, fPdg_, fNLC_, fCarrierPdg_, fCarrierReason_;
  float fE_, fShare_, fShareOtherTracked_, fSharePileup_, fShareSignalUntracked_, fDist_, fBaryZ_, fBaryR_, fEmFrac_,
      fCarrierR_, fCarrierZ_, fPointing_;
};

EnergyFlowProbe::EnergyFlowProbe(const edm::ParameterSet& cfg)
    : candToken_(consumes<std::vector<TICLCandidate>>(cfg.getParameter<edm::InputTag>("candidates"))),
      lcToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))),
      trackToken_(consumes<std::vector<reco::Track>>(cfg.getParameter<edm::InputTag>("tracks"))),
      r2tToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("trackRecoToTruth"))),
      graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      modeToken_(consumes<std::vector<int>>(cfg.getParameter<edm::InputTag>("trackMode"))),
      fieldToken_(esConsumes()),
      propToken_(esConsumes(edm::ESInputTag("", cfg.getParameter<std::string>("propagator")))),
      hgconsToken_(esConsumes(edm::ESInputTag("", "HGCalEESensitive"))) {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  tracks_ = fs->make<TTree>("tracks", "tracks");
  for (auto [n, v] : std::vector<std::pair<const char*, int*>>{{"event", &event_},
                                                               {"row", &row_},
                                                               {"pdg", &tPdg_},
                                                               {"signal", &tSignal_},
                                                               {"mode", &tMode_},
                                                               {"hasTs", &tHasTs_},
                                                               {"reachesCalo", &tReachesCalo_},
                                                               {"endReason", &tEndReason_},
                                                               {"missOuter", &tMissOuter_},
                                                               {"candPdg", &tCandPdg_}})
    tracks_->Branch(n, v);
  for (auto [n, v] : std::vector<std::pair<const char*, float*>>{{"p", &tP_},
                                                                 {"pt", &tPt_},
                                                                 {"eta", &tEta_},
                                                                 {"outerZ", &tOuterZ_},
                                                                 {"outerR", &tOuterR_},
                                                                 {"endR", &tEndR_},
                                                                 {"endZ", &tEndZ_},
                                                                 {"dep", &tDep_},
                                                                 {"depOwn", &tDepOwn_},
                                                                 {"depOther", &tDepOther_},
                                                                 {"depNeutral", &tDepNeutral_},
                                                                 {"depNeutralEM", &tDepNeutralEM_},
                                                                 {"depNone", &tDepNone_},
                                                                 {"doubleE", &tDoubleE_},
                                                                 {"candE", &tCandE_}})
    tracks_->Branch(n, v);
  frags_ = fs->make<TTree>("frags", "frags");
  for (auto [n, v] : std::vector<std::pair<const char*, int*>>{{"event", &event_},
                                                               {"row", &fRow_},
                                                               {"pdg", &fPdg_},
                                                               {"nLC", &fNLC_},
                                                               {"carrierPdg", &fCarrierPdg_},
                                                               {"carrierReason", &fCarrierReason_}})
    frags_->Branch(n, v);
  for (auto [n, v] : std::vector<std::pair<const char*, float*>>{{"E", &fE_},
                                                                 {"share", &fShare_},
                                                                 {"shareOtherTracked", &fShareOtherTracked_},
                                                                 {"sharePileup", &fSharePileup_},
                                                                 {"shareSignalUntracked", &fShareSignalUntracked_},
                                                                 {"dist", &fDist_},
                                                                 {"baryZ", &fBaryZ_},
                                                                 {"baryR", &fBaryR_},
                                                                 {"emFrac", &fEmFrac_},
                                                                 {"carrierR", &fCarrierR_},
                                                                 {"carrierZ", &fCarrierZ_},
                                                                 {"pointing", &fPointing_}})
    frags_->Branch(n, v);
}

void EnergyFlowProbe::analyze(const edm::Event& event, const edm::EventSetup& es) {
  auto const& cands = event.get(candToken_);
  auto const& lcs = event.get(lcToken_);
  const auto tkH = event.getHandle(trackToken_);
  auto const& tracks = *tkH;
  auto const& r2t = event.get(r2tToken_).getMap();
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  auto const& mode = event.get(modeToken_);
  auto const* field = &es.getData(fieldToken_);
  auto const& prop = es.getData(propToken_);
  const auto frontDisks = ticl::utils::buildHGCalFirstDisks(es.getData(hgconsToken_));
  event_ = event.id().event();
  const unsigned int nP = graph.particles().size();

  // Charged candidates with a matched track: the truth particle, and the owner (row) of it and of its descendants.
  struct Charged {
    size_t cand, track;
    unsigned int particle;
  };
  std::vector<Charged> charged;
  std::vector<int> owner(nP, -1);  // row of the charged candidate whose particle is this particle or an ancestor
  for (size_t c = 0; c < cands.size(); ++c) {
    auto const& cd = cands[c];
    if (cd.charge() == 0 || cd.trackPtrs().empty() || cd.trackPtrs()[0].isNull() || cd.trackPtrs()[0].id() != tkH.id())
      continue;
    const size_t t = cd.trackPtrs()[0].key();
    if (t >= r2t.size() || r2t[t].empty())
      continue;
    auto best = std::min_element(
        r2t[t].begin(), r2t[t].end(), [](auto const& a, auto const& b) { return a.score() < b.score(); });
    const unsigned int u = best->index();
    if (u >= nP || owner[u] >= 0)
      continue;
    const int row = static_cast<int>(charged.size());
    charged.push_back({c, t, u});
    owner[u] = row;
    for (auto const& d : truth::Particle(&graph, u).descendants())
      if (owner[d.id()] < 0)
        owner[d.id()] = row;
  }

  // Simulated HGCAL energy per cell and calo-boundary particle.
  std::unordered_map<uint32_t, std::vector<std::pair<unsigned int, float>>> cellTruth;
  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  std::vector<float> depositOfRow(charged.size(), 0.f);
  for (unsigned int p = 0; p < nP; ++p) {
    if (!graph.particles()[p].isAtLevel(truth::LevelFlag::CaloBoundary))
      continue;
    hits.clear();
    index.appendSubgraphHits(truth::HitChannel::Calo, p, hits);
    for (auto const& h : hits)
      if (isHGCal(h.detId) && h.energy > 0.f) {
        cellTruth[h.detId].emplace_back(p, h.energy);
        if (owner[p] >= 0)
          depositOfRow[owner[p]] += h.energy;
      }
  }

  // Truth content of each candidate: energy per calo-boundary particle. A layer cluster shared by several candidates is
  // counted once per candidate, as the candidates count it.
  std::vector<std::unordered_map<unsigned int, float>> content(cands.size());
  std::vector<float> contentTotal(cands.size(), 0.f);
  for (size_t c = 0; c < cands.size(); ++c)
    for (auto const& t : cands[c].tracksters())
      for (auto v : t->vertices()) {
        if (v >= lcs.size())
          continue;
        for (auto const& [id, f] : lcs[v].hitsAndFractions()) {
          auto it = cellTruth.find(id.rawId());
          if (it == cellTruth.end())
            continue;
          for (auto const& [p, e] : it->second) {
            content[c][p] += f * e;
            contentTotal[c] += f * e;
          }
        }
      }

  // Per tracked particle: where its deposit is; and the fragments in neutral candidates.
  std::vector<float> own(charged.size(), 0.f), other(charged.size(), 0.f), neutral(charged.size(), 0.f),
      neutralEM(charged.size(), 0.f), doubleE(charged.size(), 0.f);
  std::vector<ticl::TrackImpact> impacts(charged.size());
  for (size_t r = 0; r < charged.size(); ++r)
    impacts[r] = ticl::propagateToHGCalFront(tracks[charged[r].track], field, prop, frontDisks);
  for (size_t c = 0; c < cands.size(); ++c) {
    if (contentTotal[c] <= 0.f)
      continue;
    const bool isNeutral = cands[c].charge() == 0;
    const int a = std::abs(cands[c].pdgId());
    const bool em = a == 22 || a == 111;
    // Shares per owner row, and the carrier particle per row.
    std::unordered_map<int, float> byRow;
    std::unordered_map<int, std::pair<unsigned int, float>> carrier;
    float pileup = 0.f, signalUntracked = 0.f;
    for (auto const& [p, e] : content[c]) {
      const int row = owner[p];
      if (row < 0) {
        (graph.particles()[p].eventId == 0 ? signalUntracked : pileup) += e;
        continue;
      }
      byRow[row] += e;
      if (e > carrier[row].second)
        carrier[row] = {p, e};
    }
    float trackedTotal = 0.f;
    for (auto const& [row, e] : byRow)
      trackedTotal += e;
    for (auto const& [row, e] : byRow) {
      if (!isNeutral) {
        (charged[row].cand == c ? own[row] : other[row]) += e;
        continue;
      }
      (em ? neutralEM[row] : neutral[row]) += e;
      const float share = e / contentTotal[c];
      doubleE[row] += share * cands[c].energy();
      // Fragment row.
      auto const& ts = *cands[c].tracksters()[0];
      fRow_ = row;
      fPdg_ = cands[c].pdgId();
      fNLC_ = ts.vertices().size();
      fE_ = cands[c].energy();
      fShare_ = share;
      fShareOtherTracked_ = (trackedTotal - e) / contentTotal[c];
      fSharePileup_ = pileup / contentTotal[c];
      fShareSignalUntracked_ = signalUntracked / contentTotal[c];
      fDist_ = impacts[row].valid ? ticl::impactTransverseDistance(impacts[row], ts.barycenter()) : -1.f;
      fBaryZ_ = std::abs(ts.barycenter().z());
      fBaryR_ = std::hypot(ts.barycenter().x(), ts.barycenter().y());
      fEmFrac_ = ts.raw_energy() > 0.f ? ts.raw_em_energy() / ts.raw_energy() : -1.f;
      // Pointing: cosine between the trackster axis and the line from the origin to its barycenter.
      const auto axis = ts.eigenvectors(0);
      const auto bary = ts.barycenter();
      const float nb = std::sqrt(bary.mag2()), na = std::sqrt(axis.mag2());
      fPointing_ = nb > 0.f && na > 0.f ? std::abs(axis.Dot(bary)) / (na * nb) : -1.f;
      const unsigned int cp = carrier[row].first;
      fCarrierPdg_ = graph.particles()[cp].pdgId;
      fCarrierReason_ = -1;
      fCarrierR_ = fCarrierZ_ = -1.f;
      const auto pv = truth::Particle(&graph, cp).productionVertices();
      if (!pv.empty()) {
        fCarrierReason_ = pv.front().data().reason;
        fCarrierR_ = std::hypot(pv.front().position().x(), pv.front().position().y());
        fCarrierZ_ = std::abs(pv.front().position().z());
      }
      frags_->Fill();
    }
  }

  for (size_t r = 0; r < charged.size(); ++r) {
    auto const& ch = charged[r];
    auto const& tk = tracks[ch.track];
    auto const& pd = graph.particles()[ch.particle];
    row_ = static_cast<int>(r);
    tPdg_ = pd.pdgId;
    tSignal_ = pd.eventId == 0;
    tMode_ = ch.track < mode.size() ? mode[ch.track] : -9;
    tHasTs_ = !cands[ch.cand].tracksters().empty();
    tCandPdg_ = cands[ch.cand].pdgId();
    tCandE_ = cands[ch.cand].energy();
    tReachesCalo_ = pd.isAtLevel(truth::LevelFlag::CaloBoundary);
    tP_ = tk.p();
    tPt_ = tk.pt();
    tEta_ = tk.eta();
    tOuterZ_ = tk.outerOk() ? std::abs(tk.outerPosition().z()) : -1.f;
    tOuterR_ = tk.outerOk() ? tk.outerPosition().rho() : -1.f;
    tMissOuter_ = tk.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS);
    tEndReason_ = -1;
    tEndR_ = tEndZ_ = -1.f;
    const auto dv = truth::Particle(&graph, ch.particle).decayVertices();
    if (!dv.empty()) {
      tEndReason_ = dv.front().data().reason;
      tEndR_ = std::hypot(dv.front().position().x(), dv.front().position().y());
      tEndZ_ = std::abs(dv.front().position().z());
    }
    tDep_ = depositOfRow[r];
    tDepOwn_ = own[r];
    tDepOther_ = other[r];
    tDepNeutral_ = neutral[r];
    tDepNeutralEM_ = neutralEM[r];
    tDepNone_ = std::max(0.f, depositOfRow[r] - own[r] - other[r] - neutral[r] - neutralEM[r]);
    tDoubleE_ = doubleE[r];
    tracks_->Fill();
  }
}

DEFINE_FWK_MODULE(EnergyFlowProbe);
