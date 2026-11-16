// Study probe, not for integration. One row per TICLCandidate in minAbsEta < |eta| < maxAbsEta: the track (momentum,
// its error, quality, interpretation mode), the tracksters (raw, regressed and EM energy, PID, layer clusters), and the
// truth: the track particle and its energy, the dominant calo-boundary particle of the footprint with its purity, and the
// adaptive-associator truth particle of the lead trackster. Input of the track + trackster energy combination study.

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "CornettoStudy/Probe/plugins/DeepTrackMatch.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

#include "TTree.h"

namespace {
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
  bool isHGCal(uint32_t rawId) {
    const auto det = DetId(rawId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }
}  // namespace

class CandidateEnergyProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit CandidateEnergyProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<std::vector<TICLCandidate>> candToken_;
  const edm::EDGetTokenT<std::vector<reco::Track>> trackToken_;
  const edm::EDGetTokenT<std::vector<int>> trackModeToken_;
  const edm::EDGetTokenT<MapType> tkR2tToken_;
  const edm::EDGetTokenT<MapType> tsR2tToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksterToken_;
  const edm::EDGetTokenT<std::vector<unsigned int>> assignableToken_;
  const double minAbsEta_, maxAbsEta_;

  TTree* tree_;
  int event_, lumi_, pdg_, charge_, nTk_, nTs_, tkMode_, tkHits_, tkLostOuter_, tkHP_, nLC_;
  int tkTruthPdg_, tkTruthSignal_, domPdg_, domSignal_, domIsTkParticle_, hasGsf_, repPdg_, adPdg_, adSignal_, adFound_;
  float candE_, candRawE_, eta_, phi_;
  float tkP_, tkPt_, tkEta_, tkSigmaPOverP_, tkPtErr_, tkChi2n_;
  float tsRawE_, tsRegE_, tsRawEmE_, tsEta_, pid_[8];
  float tkTruthE_, tkTruthP_, tkTruthPurity_, tkFootE_, tkSubtreeE_;
  float footTruthE_, domE_, domFootE_, domSubtreeE_;
  float gsfP_, gsfPMode_, gsfSigmaPOverP_, gsfSigmaPModeOverP_;
  float domEventFootE_, signalFootE_, nearRawE_, sideRawE_, repE_, repFootE_, adE_, adFootE_, adScore_;
  // Footprint energy of the particles that a charged candidate's track already represents (the track's truth particle
  // and its descendants), and the jet target numerator: the adaptive particle's branch and the other signal particles,
  // without the represented ones.
  float dcFootE_, jetFootE_;
  // Variants of dcFootE. Id: the same definition, computed again from the per-particle weights. P50: a particle is
  // represented only by a track with truth purity >= 0.5. Deep: the represented particle is the deep match of the track
  // (DeepTrackMatch.h). DeepCov: as Deep, with the weight clip(sum of the energies of the charged candidates with the same
  // deep match / its energy, 0, 1).
  float dcFootEId_, dcFootEP50_, dcFootEDeep_, dcFootEDeepCov_;
  // Graph index of the best Fixed truth match of the track; -1 without a match. The same for the deep match, with its
  // pdg id, energy, momentum and purity.
  int tkTruthIdx_, tkTruthIdxDeep_, tkTruthPdgDeep_;
  float tkTruthEDeep_, tkTruthPDeep_, tkTruthPurityDeep_;
  float eig_[3], sigPCA_[3], sig_[3], lcMaxFrac_, lcCore2_, lcCore5_, lcRms_, lcZ0_, lcZ1_;
  // Layer clusters of the lead trackster of a neutral candidate (no track); empty for the other candidates.
  std::vector<float> lcE_, lcL_, lcR_, lcAbsZ_, lcNHits_, lcTruthE_, lcRepE_, lcAdE_;
  int nNear_;
};

CandidateEnergyProbe::CandidateEnergyProbe(const edm::ParameterSet& cfg)
    : candToken_(consumes<std::vector<TICLCandidate>>(cfg.getParameter<edm::InputTag>("candidates"))),
      trackToken_(consumes<std::vector<reco::Track>>(cfg.getParameter<edm::InputTag>("tracks"))),
      trackModeToken_(consumes<std::vector<int>>(cfg.getParameter<edm::InputTag>("trackMode"))),
      tkR2tToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("trackRecoToTruth"))),
      tsR2tToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("tracksterRecoToTruth"))),
      graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      lcToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))),
      tracksterToken_(consumes<std::vector<ticl::Trackster>>(cfg.getParameter<edm::InputTag>("tracksters"))),
      assignableToken_(consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("assignableRoots"))),
      minAbsEta_(cfg.getParameter<double>("minAbsEta")),
      maxAbsEta_(cfg.getParameter<double>("maxAbsEta")) {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("cands", "candidate energy inputs and truth");
  for (auto [n, p] : std::vector<std::pair<const char*, int*>>{{"event", &event_},
                                                               {"lumi", &lumi_},
                                                               {"pdg", &pdg_},
                                                               {"charge", &charge_},
                                                               {"nTk", &nTk_},
                                                               {"nTs", &nTs_},
                                                               {"tkMode", &tkMode_},
                                                               {"tkHits", &tkHits_},
                                                               {"tkLostOuter", &tkLostOuter_},
                                                               {"tkHP", &tkHP_},
                                                               {"nLC", &nLC_},
                                                               {"tkTruthPdg", &tkTruthPdg_},
                                                               {"tkTruthSignal", &tkTruthSignal_},
                                                               {"domPdg", &domPdg_},
                                                               {"repPdg", &repPdg_},
                                                               {"domSignal", &domSignal_},
                                                               {"adPdg", &adPdg_},
                                                               {"adSignal", &adSignal_},
                                                               {"adFound", &adFound_},
                                                               {"domIsTkParticle", &domIsTkParticle_},
                                                               {"hasGsf", &hasGsf_},
                                                               {"nNear", &nNear_},
                                                               {"tkTruthIdx", &tkTruthIdx_},
                                                               {"tkTruthIdxDeep", &tkTruthIdxDeep_},
                                                               {"tkTruthPdgDeep", &tkTruthPdgDeep_}})
    tree_->Branch(n, p);
  for (auto [n, p] : std::vector<std::pair<const char*, float*>>{{"candE", &candE_},
                                                                 {"candRawE", &candRawE_},
                                                                 {"eta", &eta_},
                                                                 {"phi", &phi_},
                                                                 {"tkP", &tkP_},
                                                                 {"tkPt", &tkPt_},
                                                                 {"tkEta", &tkEta_},
                                                                 {"tkSigmaPOverP", &tkSigmaPOverP_},
                                                                 {"tkPtErr", &tkPtErr_},
                                                                 {"tkChi2n", &tkChi2n_},
                                                                 {"tsRawE", &tsRawE_},
                                                                 {"tsRegE", &tsRegE_},
                                                                 {"tsRawEmE", &tsRawEmE_},
                                                                 {"tsEta", &tsEta_},
                                                                 {"tkTruthE", &tkTruthE_},
                                                                 {"tkTruthP", &tkTruthP_},
                                                                 {"tkTruthPurity", &tkTruthPurity_},
                                                                 {"tkFootE", &tkFootE_},
                                                                 {"tkSubtreeE", &tkSubtreeE_},
                                                                 {"footTruthE", &footTruthE_},
                                                                 {"domE", &domE_},
                                                                 {"domFootE", &domFootE_},
                                                                 {"domSubtreeE", &domSubtreeE_},
                                                                 {"gsfP", &gsfP_},
                                                                 {"gsfPMode", &gsfPMode_},
                                                                 {"gsfSigmaPOverP", &gsfSigmaPOverP_},
                                                                 {"gsfSigmaPModeOverP", &gsfSigmaPModeOverP_},
                                                                 {"domEventFootE", &domEventFootE_},
                                                                 {"repE", &repE_},
                                                                 {"repFootE", &repFootE_},
                                                                 {"adE", &adE_},
                                                                 {"adFootE", &adFootE_},
                                                                 {"adScore", &adScore_},
                                                                 {"lcMaxFrac", &lcMaxFrac_},
                                                                 {"lcCore2", &lcCore2_},
                                                                 {"lcCore5", &lcCore5_},
                                                                 {"lcRms", &lcRms_},
                                                                 {"lcZ0", &lcZ0_},
                                                                 {"lcZ1", &lcZ1_},
                                                                 {"signalFootE", &signalFootE_},
                                                                 {"dcFootE", &dcFootE_},
                                                                 {"dcFootEId", &dcFootEId_},
                                                                 {"dcFootEP50", &dcFootEP50_},
                                                                 {"dcFootEDeep", &dcFootEDeep_},
                                                                 {"dcFootEDeepCov", &dcFootEDeepCov_},
                                                                 {"tkTruthEDeep", &tkTruthEDeep_},
                                                                 {"tkTruthPDeep", &tkTruthPDeep_},
                                                                 {"tkTruthPurityDeep", &tkTruthPurityDeep_},
                                                                 {"jetFootE", &jetFootE_},
                                                                 {"nearRawE", &nearRawE_},
                                                                 {"sideRawE", &sideRawE_}})
    tree_->Branch(n, p);
  tree_->Branch("pid", pid_, "pid[8]/F");
  tree_->Branch("eig", eig_, "eig[3]/F");
  tree_->Branch("sigPCA", sigPCA_, "sigPCA[3]/F");
  tree_->Branch("sig", sig_, "sig[3]/F");
  for (auto [n, v] : std::vector<std::pair<const char*, std::vector<float>*>>{{"lcE", &lcE_},
                                                                              {"lcL", &lcL_},
                                                                              {"lcR", &lcR_},
                                                                              {"lcAbsZ", &lcAbsZ_},
                                                                              {"lcNHits", &lcNHits_},
                                                                              {"lcTruthE", &lcTruthE_},
                                                                              {"lcRepE", &lcRepE_},
                                                                              {"lcAdE", &lcAdE_}})
    tree_->Branch(n, v);
}

void CandidateEnergyProbe::analyze(const edm::Event& event, const edm::EventSetup&) {
  auto const& cands = event.get(candToken_);
  auto const& tracks = event.get(trackToken_);
  auto const& trackMode = event.get(trackModeToken_);
  auto const& tkR2t = event.get(tkR2tToken_).getMap();
  auto const& tsR2t = event.get(tsR2tToken_).getMap();
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  auto const& layerClusters = event.get(lcToken_);
  auto const& finalTracksters = event.get(tracksterToken_);
  // Raw energy of the final tracksters per HGCAL side (z < 0, z > 0).
  float sideE[2] = {0.f, 0.f};
  for (auto const& ts : finalTracksters)
    sideE[ts.barycenter().z() > 0.f ? 1 : 0] += ts.raw_energy();
  event_ = event.id().event();
  lumi_ = event.id().luminosityBlock();
  const unsigned int nP = graph.particles().size();

  // Simulated HGCAL energy per cell and calo-boundary particle.
  std::unordered_map<uint32_t, std::vector<std::pair<int, float>>> cellTruth;
  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  for (unsigned int p = 0; p < nP; ++p) {
    if (!graph.particles()[p].isAtLevel(truth::LevelFlag::CaloBoundary))
      continue;
    hits.clear();
    index.appendSubgraphHits(truth::HitChannel::Calo, p, hits);
    for (auto const& h : hits)
      if (isHGCal(h.detId) && h.energy > 0.f)
        cellTruth[h.detId].emplace_back(static_cast<int>(p), h.energy);
  }
  // Total HGCAL energy of a particle and its descendants.
  auto subtreeEnergy = [&](int p, std::unordered_map<uint32_t, float>& cells) {
    cells.clear();
    if (p < 0)
      return 0.f;
    hits.clear();
    index.appendSubgraphHits(truth::HitChannel::Calo, p, hits);
    double total = 0.;
    for (auto const& h : hits)
      if (isHGCal(h.detId) && h.energy > 0.f) {
        cells[h.detId] += h.energy;
        total += h.energy;
      }
    return static_cast<float>(total);
  };
  std::unordered_map<uint32_t, float> tkCells, domCells, adCells;
  // Particles that a charged candidate's track represents: the best Fixed truth match of the track and its descendants.
  std::vector<bool> represented(nP, false);
  for (auto const& c : cands) {
    if (c.trackPtr().isNull())
      continue;
    const auto key = c.trackPtr().key();
    if (key >= tkR2t.size() || tkR2t[key].empty())
      continue;
    auto best = std::min_element(
        tkR2t[key].begin(), tkR2t[key].end(), [](auto const& a, auto const& b) { return a.score() < b.score(); });
    if (best->index() >= nP || represented[best->index()])
      continue;
    represented[best->index()] = true;
    for (auto const& d : graph.particle(best->index()).descendants())
      represented[d.id()] = true;
  }
  // The particle that a track follows (DeepTrackMatch.h).
  const cornettostudy::DeepTrackMatch deepMatch(graph, tkR2t, event.get(assignableToken_));
  // Per-root matches for the variants: any best match (Id), its purity >= 0.5 (P50), the deep match (Deep) and the sum
  // of the energies of the charged candidates with the same deep match (DeepCov).
  std::vector<char> rootAny(nP, 0), rootP50(nP, 0), deepAny(nP, 0);
  std::vector<double> deepCandE(nP, 0.);
  for (auto const& c : cands) {
    if (c.trackPtr().isNull())
      continue;
    const auto key = c.trackPtr().key();
    if (key >= tkR2t.size() || tkR2t[key].empty())
      continue;
    auto best = std::min_element(
        tkR2t[key].begin(), tkR2t[key].end(), [](auto const& a, auto const& b) { return a.score() < b.score(); });
    if (best->index() < nP) {
      rootAny[best->index()] = 1;
      if (1.f - best->score() >= 0.5f)
        rootP50[best->index()] = 1;
    }
    float purity;
    if (const int deep = deepMatch(key, purity); deep >= 0) {
      deepAny[deep] = 1;
      deepCandE[deep] += c.energy();
    }
  }
  // Weight of each particle: the largest weight of the represented particles whose subtree holds it.
  std::vector<float> wId(nP, 0.f), wP50(nP, 0.f), wDeep(nP, 0.f), wDeepCov(nP, 0.f);
  for (unsigned int r = 0; r < nP; ++r) {
    if (rootAny[r]) {
      const float p50 = rootP50[r] ? 1.f : 0.f;
      auto mark = [&](unsigned int p) {
        wId[p] = 1.f;
        wP50[p] = std::max(wP50[p], p50);
      };
      mark(r);
      for (auto const& d : graph.particle(r).descendants())
        mark(d.id());
    }
    if (deepAny[r]) {
      const double eR = graph.particles()[r].momentum.energy();
      const float cov = eR > 0. ? static_cast<float>(std::clamp(deepCandE[r] / eR, 0., 1.)) : 0.f;
      auto mark = [&](unsigned int p) {
        wDeep[p] = 1.f;
        wDeepCov[p] = std::max(wDeepCov[p], cov);
      };
      mark(r);
      for (auto const& d : graph.particle(r).descendants())
        mark(d.id());
    }
  }

  for (auto const& cand : cands) {
    if (std::abs(cand.eta()) < minAbsEta_ || std::abs(cand.eta()) > maxAbsEta_)
      continue;
    pdg_ = cand.pdgId();
    charge_ = cand.charge();
    candE_ = cand.energy();
    candRawE_ = cand.rawEnergy();
    eta_ = cand.eta();
    phi_ = cand.phi();
    nTs_ = cand.tracksters().size();
    nTk_ = cand.trackPtr().isNonnull() ? 1 : 0;

    // Track and its truth particle (best Fixed reco-to-truth match).
    tkMode_ = -2;
    tkHits_ = tkLostOuter_ = tkHP_ = 0;
    tkP_ = tkPt_ = tkEta_ = tkSigmaPOverP_ = tkPtErr_ = tkChi2n_ = -1.f;
    tkTruthPdg_ = tkTruthSignal_ = 0;
    tkTruthE_ = tkTruthP_ = tkTruthPurity_ = -1.f;
    int tkParticle = -1;
    tkTruthIdx_ = tkTruthIdxDeep_ = -1;
    tkTruthPdgDeep_ = 0;
    tkTruthEDeep_ = tkTruthPDeep_ = tkTruthPurityDeep_ = -1.f;
    if (nTk_) {
      float purity;
      if (const int deep = deepMatch(cand.trackPtr().key(), purity); deep >= 0) {
        auto const& pd = graph.particles()[deep];
        tkTruthIdxDeep_ = deep;
        tkTruthPdgDeep_ = pd.pdgId;
        tkTruthEDeep_ = pd.momentum.energy();
        tkTruthPDeep_ = pd.momentum.P();
        tkTruthPurityDeep_ = purity;
      }
    }
    if (nTk_) {
      const auto key = cand.trackPtr().key();
      auto const& tk = tracks[key];
      tkMode_ = key < trackMode.size() ? trackMode[key] : -2;
      tkP_ = tk.p();
      tkPt_ = tk.pt();
      tkEta_ = tk.eta();
      tkPtErr_ = tk.ptError();
      tkSigmaPOverP_ = tk.qoverpError() * tk.p();
      tkChi2n_ = tk.normalizedChi2();
      tkHits_ = tk.numberOfValidHits();
      tkLostOuter_ = tk.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS);
      tkHP_ = tk.quality(reco::TrackBase::highPurity) ? 1 : 0;
      if (key < tkR2t.size() && !tkR2t[key].empty()) {
        auto best = std::min_element(
            tkR2t[key].begin(), tkR2t[key].end(), [](auto const& a, auto const& b) { return a.score() < b.score(); });
        if (best->index() < nP) {
          tkParticle = static_cast<int>(best->index());
          auto const& pd = graph.particles()[tkParticle];
          tkTruthPdg_ = pd.pdgId;
          tkTruthSignal_ = pd.isSignal() ? 1 : 0;
          tkTruthE_ = pd.momentum.energy();
          tkTruthP_ = pd.momentum.P();
          tkTruthPurity_ = 1.f - best->score();
          tkTruthIdx_ = tkParticle;
        }
      }
    }

    // GSF track of an electron candidate: mean and mode momenta and their relative errors.
    hasGsf_ = cand.gsfTrackPtrs().empty() ? 0 : 1;
    gsfP_ = gsfPMode_ = gsfSigmaPOverP_ = gsfSigmaPModeOverP_ = -1.f;
    if (hasGsf_) {
      auto const& gsf = *cand.gsftrackPtr();
      gsfP_ = gsf.p();
      gsfPMode_ = gsf.pMode();
      gsfSigmaPOverP_ = gsf.qoverpError() * gsf.p();
      gsfSigmaPModeOverP_ = gsf.qoverpModeError() * gsf.pMode();
    }

    // Tracksters: energies, PID of the most energetic, and the truth composition of their layer clusters.
    tsRawE_ = tsRegE_ = tsRawEmE_ = 0.f;
    tsEta_ = -99.f;
    nLC_ = 0;
    std::fill(pid_, pid_ + 8, -1.f);
    std::unordered_map<int, double> comp;
    float leadE = -1.f;
    const ticl::Trackster* lead = nullptr;
    size_t leadKey = 0;
    for (auto const& ts : cand.tracksters()) {
      tsRawE_ += ts->raw_energy();
      tsRegE_ += ts->regressed_energy();
      tsRawEmE_ += ts->raw_em_energy();
      nLC_ += ts->vertices().size();
      if (ts->raw_energy() > leadE) {
        leadE = ts->raw_energy();
        lead = ts.get();
        leadKey = ts.key();
        tsEta_ = ts->barycenter().eta();
        for (int k = 0; k < 8; ++k)
          pid_[k] = ts->id_probabilities(k);
      }
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
    footTruthE_ = 0.f;
    int dom = -1;
    double domFoot = 0.;
    for (auto const& [p, e] : comp) {
      footTruthE_ += e;
      if (e > domFoot) {
        domFoot = e;
        dom = p;
      }
    }
    domFootE_ = domFoot;
    // Footprint energy from the interaction of the dominant particle, and from the signal interaction (eventId 0).
    domEventFootE_ = signalFootE_ = 0.f;
    if (dom >= 0) {
      const auto domEvent = graph.particles()[dom].eventId;
      for (auto const& [p, e] : comp) {
        const auto ev = graph.particles()[p].eventId;
        if (ev == domEvent)
          domEventFootE_ += e;
        if (ev == 0)
          signalFootE_ += e;
      }
    }
    // Particle that the candidate represents: the dominant particle, or the photon, electron, pi0 or eta it comes from
    // through a single parent (the photons of a pi0 merge in one shower). Footprint energy of it and its descendants.
    repPdg_ = 0;
    repE_ = -1.f;
    repFootE_ = 0.f;
    std::unordered_set<uint32_t> members;
    if (dom >= 0) {
      auto isEm = [](int pdg) {
        const int a = std::abs(pdg);
        return a == 22 || a == 11 || a == 111 || a == 221;
      };
      auto rep = graph.particle(dom);
      while (true) {
        const auto parents = rep.parents();
        if (parents.size() != 1 || !isEm(parents[0].pdgId()))
          break;
        rep = parents[0];
      }
      members.insert(rep.id());
      if (rep.id() != static_cast<uint32_t>(dom))
        for (auto const& d : rep.descendants())
          members.insert(d.id());
      for (auto const& [p, e] : comp)
        if (members.count(p))
          repFootE_ += e;
      repPdg_ = rep.pdgId();
      repE_ = rep.momentum().energy();
    }
    // Shape of the lead trackster: PCA eigenvalues and spreads, and from its layer clusters the largest energy share,
    // the energy share within 2 and 5 cm of the main axis, the energy-weighted transverse rms (cm), and the |z| extent.
    std::fill(eig_, eig_ + 3, -1.f);
    std::fill(sigPCA_, sigPCA_ + 3, -1.f);
    std::fill(sig_, sig_ + 3, -1.f);
    lcMaxFrac_ = lcCore2_ = lcCore5_ = lcRms_ = lcZ0_ = lcZ1_ = -1.f;
    if (lead != nullptr) {
      for (int k = 0; k < 3; ++k) {
        eig_[k] = lead->eigenvalues()[k];
        sigPCA_[k] = lead->sigmasPCA()[k];
        sig_[k] = lead->sigmas()[k];
      }
      const auto& b = lead->barycenter();
      auto axis = lead->eigenvectors(0);
      if (axis.mag2() <= 0.f || !std::isfinite(axis.mag2()))
        axis = b;
      axis = axis.unit();
      double eSum = 0., eMax = 0., e2 = 0., e5 = 0., r2Sum = 0.;
      float z0 = 1e9f, z1 = -1e9f;
      for (size_t i = 0; i < lead->vertices().size(); ++i) {
        const auto lc = lead->vertices()[i];
        if (lc >= layerClusters.size())
          continue;
        const float e = layerClusters[lc].energy() / std::max<float>(1.f, lead->vertex_multiplicity()[i]);
        const auto& pos = layerClusters[lc].position();
        const ticl::Trackster::Vector d(pos.x() - b.x(), pos.y() - b.y(), pos.z() - b.z());
        const float r = std::sqrt(std::max(0.f, d.mag2() - std::pow(d.Dot(axis), 2.f)));
        eSum += e;
        eMax = std::max<double>(eMax, e);
        e2 += r < 2.f ? e : 0.f;
        e5 += r < 5.f ? e : 0.f;
        r2Sum += e * r * r;
        z0 = std::min(z0, std::abs(static_cast<float>(pos.z())));
        z1 = std::max(z1, std::abs(static_cast<float>(pos.z())));
      }
      if (eSum > 0.) {
        lcMaxFrac_ = eMax / eSum;
        lcCore2_ = e2 / eSum;
        lcCore5_ = e5 / eSum;
        lcRms_ = std::sqrt(r2Sum / eSum);
        lcZ0_ = z0;
        lcZ1_ = z1;
      }
    }
    // Truth particle of the lead trackster from the adaptive associator (AdaptiveNominal), and the footprint energy of
    // its subgraph: energy from its own branch is not contamination.
    adPdg_ = adSignal_ = adFound_ = 0;
    adE_ = -1.f;
    adFootE_ = 0.f;
    adScore_ = -1.f;
    adCells.clear();
    if (lead != nullptr && leadKey < tsR2t.size() && !tsR2t[leadKey].empty()) {
      auto best = std::min_element(tsR2t[leadKey].begin(), tsR2t[leadKey].end(), [](auto const& a, auto const& b) {
        return a.score() < b.score();
      });
      if (best->index() < nP) {
        const int ad = static_cast<int>(best->index());
        adFound_ = 1;
        adPdg_ = graph.particles()[ad].pdgId;
        adSignal_ = graph.particles()[ad].isSignal() ? 1 : 0;
        adE_ = graph.particles()[ad].momentum.energy();
        adScore_ = best->score();
        subtreeEnergy(ad, adCells);
        for (auto const& ts : cand.tracksters())
          for (size_t i = 0; i < ts->vertices().size(); ++i) {
            const auto lc = ts->vertices()[i];
            if (lc >= layerClusters.size())
              continue;
            const float share = 1.f / std::max<float>(1.f, ts->vertex_multiplicity()[i]);
            for (auto const& [id, frac] : layerClusters[lc].hitsAndFractions())
              if (auto it = adCells.find(id.rawId()); it != adCells.end())
                adFootE_ += share * frac * it->second;
          }
      }
    }
    // Split of the footprint truth energy: represented by a track (double counted), and the jet target numerator.
    dcFootE_ = jetFootE_ = 0.f;
    dcFootEId_ = dcFootEP50_ = dcFootEDeep_ = dcFootEDeepCov_ = 0.f;
    for (auto const& [p, e] : comp) {
      dcFootEId_ += wId[p] * e;
      dcFootEP50_ += wP50[p] * e;
      dcFootEDeep_ += wDeep[p] * e;
      dcFootEDeepCov_ += wDeepCov[p] * e;
    }
    {
      std::unordered_set<uint32_t> branch;
      if (adFound_ && lead != nullptr && leadKey < tsR2t.size() && !tsR2t[leadKey].empty()) {
        auto best = std::min_element(tsR2t[leadKey].begin(), tsR2t[leadKey].end(), [](auto const& a, auto const& b) {
          return a.score() < b.score();
        });
        branch.insert(best->index());
        for (auto const& d : graph.particle(best->index()).descendants())
          branch.insert(d.id());
      }
      for (auto const& [p, e] : comp) {
        if (represented[p]) {
          dcFootE_ += e;
          continue;
        }
        if (branch.count(p) || graph.particles()[p].eventId == 0)
          jetFootE_ += e;
      }
    }
    // Layer clusters of the lead trackster of a neutral candidate: energy (shared by the multiplicity), position along
    // and across the main axis (cm), |z|, number of hits, truth energy and the part from the represented particle.
    for (auto* v : {&lcE_, &lcL_, &lcR_, &lcAbsZ_, &lcNHits_, &lcTruthE_, &lcRepE_, &lcAdE_})
      v->clear();
    if (lead != nullptr && nTk_ == 0) {
      const auto& b = lead->barycenter();
      auto axis = lead->eigenvectors(0);
      if (axis.mag2() <= 0.f || !std::isfinite(axis.mag2()))
        axis = b;
      axis = axis.unit();
      for (size_t i = 0; i < lead->vertices().size(); ++i) {
        const auto lc = lead->vertices()[i];
        if (lc >= layerClusters.size())
          continue;
        const float share = 1.f / std::max<float>(1.f, lead->vertex_multiplicity()[i]);
        const auto& pos = layerClusters[lc].position();
        const ticl::Trackster::Vector d(pos.x() - b.x(), pos.y() - b.y(), pos.z() - b.z());
        const float along = d.Dot(axis);
        double truthE = 0., repE = 0., adE = 0.;
        for (auto const& [id, frac] : layerClusters[lc].hitsAndFractions()) {
          if (auto ia = adCells.find(id.rawId()); ia != adCells.end())
            adE += share * frac * ia->second;
          auto it = cellTruth.find(id.rawId());
          if (it == cellTruth.end())
            continue;
          for (auto const& [p, e] : it->second) {
            truthE += share * frac * e;
            if (members.count(p))
              repE += share * frac * e;
          }
        }
        lcE_.push_back(layerClusters[lc].energy() * share);
        lcL_.push_back(along);
        lcR_.push_back(std::sqrt(std::max(0.f, d.mag2() - along * along)));
        lcAbsZ_.push_back(std::abs(static_cast<float>(pos.z())));
        lcNHits_.push_back(layerClusters[lc].hitsAndFractions().size());
        lcTruthE_.push_back(truthE);
        lcRepE_.push_back(repE);
        lcAdE_.push_back(adE);
      }
    }
    // Reco density around the lead trackster: other final tracksters within dR 0.2 on the same side, and the side total.
    nearRawE_ = sideRawE_ = 0.f;
    nNear_ = 0;
    if (lead != nullptr) {
      const auto& b = lead->barycenter();
      sideRawE_ = sideE[b.z() > 0.f ? 1 : 0];
      for (auto const& ts : finalTracksters) {
        const auto& c = ts.barycenter();
        if (&ts == lead || c.z() * b.z() <= 0.f || reco::deltaR(c.eta(), c.phi(), b.eta(), b.phi()) >= 0.2f)
          continue;
        nearRawE_ += ts.raw_energy();
        ++nNear_;
      }
    }
    domPdg_ = dom >= 0 ? graph.particles()[dom].pdgId : 0;
    domSignal_ = dom >= 0 && graph.particles()[dom].isSignal() ? 1 : 0;
    domE_ = dom >= 0 ? graph.particles()[dom].momentum.energy() : -1.f;
    domSubtreeE_ = subtreeEnergy(dom, domCells);

    // Energy of the track particle (and its descendants) held by the footprint.
    tkSubtreeE_ = subtreeEnergy(tkParticle, tkCells);
    tkFootE_ = 0.f;
    domIsTkParticle_ = (tkParticle >= 0 && tkParticle == dom) ? 1 : 0;
    if (tkParticle >= 0) {
      for (auto const& ts : cand.tracksters())
        for (size_t i = 0; i < ts->vertices().size(); ++i) {
          const auto lc = ts->vertices()[i];
          if (lc >= layerClusters.size())
            continue;
          const float share = 1.f / std::max<float>(1.f, ts->vertex_multiplicity()[i]);
          for (auto const& [id, frac] : layerClusters[lc].hitsAndFractions())
            if (auto it = tkCells.find(id.rawId()); it != tkCells.end())
              tkFootE_ += share * frac * it->second;
        }
    }
    tree_->Fill();
  }
}

DEFINE_FWK_MODULE(CandidateEnergyProbe);
