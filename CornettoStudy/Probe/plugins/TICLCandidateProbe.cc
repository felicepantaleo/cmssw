// Study probe, not for integration. One TTree row per signal truth target (tau leg) in the endcap: the
// best final trackster and track of the leg, the TICL candidates that hold them, and the type and energy
// of the chosen candidate (the one holding the matched track, else the one holding the matched trackster). The
// calorimeter part of a candidate is matched like one trackster made of the layer clusters of all its tracksters:
// the best candidate by shared fraction, its score, the sum of the shares over all candidates, and the share and
// score of the chosen candidate.
// A second TTree has one row per signal generator tau: its visible decay, the number of its legs in total and
// in the endcap, and the HPS tau that holds the largest energy share of its endcap legs. A leg is in an HPS tau
// when the particle-flow candidate made from its chosen TICL candidate is a signal constituent of the HPS tau.

#include <cmath>
#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/Vertex.h"

#include "TTree.h"

namespace {
  // What a tau decayed into, from the generator decay chain: the visible momentum, the charged hadrons, the pi0s and
  // the lepton (11 or 13, 0 for a hadronic decay).
  struct TauContent {
    int prongs = 0;
    int pi0s = 0;
    int lepton = 0;
    math::XYZTLorentzVectorD visible;
  };

  void collectTau(truth::Particle const& particle, TauContent& content) {
    const int a = std::abs(particle.pdgId());
    if (truth::isInvisible(particle.pdgId()))
      return;
    if (a == 111 || a == 130 || a == 310) {
      if (a == 111)
        ++content.pi0s;
      content.visible += particle.momentum();
      return;
    }
    bool decayed = false;
    for (auto const& vertex : particle.decayVertices()) {
      if (!vertex.hasGen())
        continue;
      decayed = true;
      for (auto const& child : vertex.outgoingParticles())
        if (child != particle)
          collectTau(child, content);
    }
    if (decayed)
      return;
    if (a == 211 || a == 321)
      ++content.prongs;
    if (a == 11 || a == 13)
      content.lepton = a;
    content.visible += particle.momentum();
  }

  // The last copy of the nearest generator tau above a particle, if there is one.
  std::optional<uint32_t> tauOf(truth::Graph const& graph, uint32_t id) {
    const truth::Particle p(&graph, id);
    for (int pdg : {15, -15})
      if (auto t = p.firstAncestorWithPdgId(pdg))
        return t->lastCopy().id();
    return std::nullopt;
  }
}  // namespace

class TICLCandidateProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TICLCandidateProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
  edm::EDGetTokenT<truth::Graph> graphToken_;
  edm::EDGetTokenT<std::vector<unsigned int>> targetsToken_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksterToken_;
  edm::EDGetTokenT<MapType> tsT2rToken_, tsR2tToken_;
  edm::EDGetTokenT<std::vector<reco::Track>> trackToken_;
  edm::EDGetTokenT<MapType> tkT2rToken_, tkR2tToken_;
  edm::EDGetTokenT<std::vector<TICLCandidate>> candToken_;
  edm::EDGetTokenT<MapType> cT2rToken_, cR2tToken_;
  edm::EDGetTokenT<std::vector<int>> trackModeToken_;
  edm::EDGetTokenT<reco::PFTauCollection> hpsToken_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> hpsDmfToken_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfToken_, pfTmpToken_, pfBarrelToken_;
  const double minAbsEta_, maxAbsEta_, minPt_, minTrackRecoPurity_;

  TTree* tree_;
  int event_, pdgId_, tsIdx_, tsInd_, tsCand_, tkIdx_, tkCand_, candPdg_, candCharge_, candNTs_, candNTk_, tkMode_,
      tkHP_, tkMissOuter_, tkMissInner_, tkHits_;
  float pt_, eta_, energy_, tsFrac_, tsScore_, candE_, candRawE_, tkP_;
  float phi_;
  int tau_;
  int cIdx_, cInd_;
  float cFrac_, cScore_, cSum_, chFrac_, chScore_;

  // Tau tree. hpsFrac: energy of the endcap legs in the HPS tau / energy of all endcap legs, for the HPS tau with
  // the largest share; hpsDM is -1 and hpsPt 0 when no HPS tau holds a leg. hpsDMF is the decay mode finding with
  // the new decay modes.
  TTree* taus_;
  int tEvent_, tIdx_, tProngs_, tPi0s_, tLepton_, tNLeg_, tNLegEC_, tHpsDM_, tHpsDMF_;
  float tVisPt_, tVisEta_, tVisPhi_, tHpsPt_, tHpsFrac_;
};

TICLCandidateProbe::TICLCandidateProbe(const edm::ParameterSet& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      targetsToken_(consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("targets"))),
      minAbsEta_(cfg.getParameter<double>("minAbsEta")),
      maxAbsEta_(cfg.getParameter<double>("maxAbsEta")),
      minPt_(cfg.getParameter<double>("minPt")),
      minTrackRecoPurity_(cfg.getParameter<double>("minTrackRecoPurity")) {
  usesResource(TFileService::kSharedResource);
  const auto ts = cfg.getParameter<std::string>("tracksters");
  const auto tsAssoc = cfg.getParameter<std::string>("tracksterAssociator");
  tracksterToken_ = consumes<std::vector<ticl::Trackster>>(edm::InputTag(ts));
  tsT2rToken_ = consumes<MapType>(edm::InputTag(tsAssoc, ts + "TruthToReco"));
  tsR2tToken_ = consumes<MapType>(edm::InputTag(tsAssoc, ts + "RecoToTruthFixed"));
  const auto tk = cfg.getParameter<std::string>("tracks");
  const auto tkAssoc = cfg.getParameter<std::string>("trackAssociator");
  trackToken_ = consumes<std::vector<reco::Track>>(edm::InputTag(tk));
  tkT2rToken_ = consumes<MapType>(edm::InputTag(tkAssoc, tk + "TruthToReco"));
  tkR2tToken_ = consumes<MapType>(edm::InputTag(tkAssoc, tk + "RecoToTruthFixed"));
  const auto candTag = cfg.getParameter<edm::InputTag>("candidates");
  candToken_ = consumes<std::vector<TICLCandidate>>(candTag);
  const auto cAssoc = cfg.getUntrackedParameter<std::string>("candidateAssociator", "truthBranchCandidateAssociators");
  cT2rToken_ = consumes<MapType>(edm::InputTag(cAssoc, candTag.label() + "TruthToReco"));
  cR2tToken_ = consumes<MapType>(edm::InputTag(cAssoc, candTag.label() + "RecoToTruthFixed"));
  if (cfg.existsAs<edm::InputTag>("trackMode"))
    trackModeToken_ = consumes<std::vector<int>>(cfg.getParameter<edm::InputTag>("trackMode"));
  hpsToken_ = consumes<reco::PFTauCollection>(
      cfg.getUntrackedParameter<edm::InputTag>("hpsTaus", edm::InputTag("hpsPFTauProducer")));
  hpsDmfToken_ = consumes<reco::PFTauDiscriminator>(cfg.getUntrackedParameter<edm::InputTag>(
      "hpsDecayModeFinding", edm::InputTag("hpsPFTauDiscriminationByDecayModeFindingNewDMs")));
  pfToken_ = consumes<reco::PFCandidateCollection>(edm::InputTag("particleFlow"));
  pfTmpToken_ = consumes<reco::PFCandidateCollection>(edm::InputTag("particleFlowTmp"));
  pfBarrelToken_ = consumes<reco::PFCandidateCollection>(edm::InputTag("particleFlowTmpBarrel"));

  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("cands", "cands");
  for (auto [name, ptr] : std::vector<std::pair<const char*, int*>>{{"event", &event_},
                                                                    {"pdgId", &pdgId_},
                                                                    {"tsIdx", &tsIdx_},
                                                                    {"tsInd", &tsInd_},
                                                                    {"tsCand", &tsCand_},
                                                                    {"tkIdx", &tkIdx_},
                                                                    {"tkCand", &tkCand_},
                                                                    {"candPdg", &candPdg_},
                                                                    {"candCharge", &candCharge_},
                                                                    {"candNTs", &candNTs_},
                                                                    {"candNTk", &candNTk_},
                                                                    {"tkMode", &tkMode_},
                                                                    {"tkHP", &tkHP_},
                                                                    {"tkMissOuter", &tkMissOuter_},
                                                                    {"tkMissInner", &tkMissInner_},
                                                                    {"tkHits", &tkHits_},
                                                                    {"tau", &tau_},
                                                                    {"cIdx", &cIdx_},
                                                                    {"cInd", &cInd_}})
    tree_->Branch(name, ptr);
  for (auto [name, ptr] : std::vector<std::pair<const char*, float*>>{{"pt", &pt_},
                                                                      {"eta", &eta_},
                                                                      {"energy", &energy_},
                                                                      {"tsFrac", &tsFrac_},
                                                                      {"tsScore", &tsScore_},
                                                                      {"candE", &candE_},
                                                                      {"candRawE", &candRawE_},
                                                                      {"tkP", &tkP_},
                                                                      {"phi", &phi_},
                                                                      {"cFrac", &cFrac_},
                                                                      {"cScore", &cScore_},
                                                                      {"cSum", &cSum_},
                                                                      {"chFrac", &chFrac_},
                                                                      {"chScore", &chScore_}})
    tree_->Branch(name, ptr);

  taus_ = fs->make<TTree>("taus", "taus");
  for (auto [name, ptr] : std::vector<std::pair<const char*, int*>>{{"event", &tEvent_},
                                                                    {"tau", &tIdx_},
                                                                    {"prongs", &tProngs_},
                                                                    {"pi0s", &tPi0s_},
                                                                    {"lepton", &tLepton_},
                                                                    {"nLeg", &tNLeg_},
                                                                    {"nLegEC", &tNLegEC_},
                                                                    {"hpsDM", &tHpsDM_},
                                                                    {"hpsDMF", &tHpsDMF_}})
    taus_->Branch(name, ptr);
  for (auto [name, ptr] : std::vector<std::pair<const char*, float*>>{{"visPt", &tVisPt_},
                                                                      {"visEta", &tVisEta_},
                                                                      {"visPhi", &tVisPhi_},
                                                                      {"hpsPt", &tHpsPt_},
                                                                      {"hpsFrac", &tHpsFrac_}})
    taus_->Branch(name, ptr);
}

void TICLCandidateProbe::analyze(const edm::Event& event, const edm::EventSetup&) {
  auto const& graph = event.get(graphToken_);
  auto const& targets = event.get(targetsToken_);
  const auto tsH = event.getHandle(tracksterToken_);
  auto const& tsT2r = event.get(tsT2rToken_).getMap();
  auto const& tsR2t = event.get(tsR2tToken_).getMap();
  const auto tkH = event.getHandle(trackToken_);
  auto const& tkT2r = event.get(tkT2rToken_).getMap();
  auto const& tkR2t = event.get(tkR2tToken_).getMap();
  auto const& cands = event.get(candToken_);
  auto const& cT2r = event.get(cT2rToken_).getMap();
  auto const& cR2t = event.get(cR2tToken_).getMap();
  const std::vector<int> noModes;
  auto const& trackMode = trackModeToken_.isUninitialized() ? noModes : event.get(trackModeToken_);
  event_ = event.id().event();

  // Candidate index of each final trackster and of each track, by product id and key.
  std::map<unsigned int, int> candOfTs, candOfTk;
  for (unsigned int c = 0; c < cands.size(); ++c) {
    for (auto const& p : cands[c].tracksters())
      if (p.isNonnull() && p.id() == tsH.id())
        candOfTs[p.key()] = c;
    for (auto const& p : cands[c].trackPtrs())
      if (p.isNonnull() && p.id() == tkH.id())
        candOfTk[p.key()] = c;
  }
  auto recoScore = [](const ticl::mapWithSharedEnergyAndScore& r2t, unsigned int r, unsigned int b) {
    if (r >= r2t.size())
      return 1.f;
    for (auto const& m : r2t[r])
      if (m.index() == b)
        return m.score();
    return 1.f;
  };

  // Signal generator taus, by the graph id of their last copy, and the number of their legs above minPt in total
  // and in the endcap acceptance.
  std::map<uint32_t, int> tauIndex;
  std::vector<int> nLeg, nLegEC;
  for (uint32_t i = 0; i < graph.particles().size(); ++i) {
    auto const& pd = graph.particles()[i];
    const truth::Particle p(&graph, i);
    if (std::abs(pd.pdgId) != 15 || !pd.isSignal() || !p.hasGen() || p.lastCopy().id() != i)
      continue;
    tauIndex.emplace(i, static_cast<int>(nLeg.size()));
    nLeg.push_back(0);
    nLegEC.push_back(0);
  }
  for (unsigned int b : targets) {
    if (b >= graph.particles().size())
      continue;
    auto const& pd = graph.particles()[b];
    if (!pd.isSignal() || pd.momentum.pt() < minPt_)
      continue;
    const auto t = tauOf(graph, b);
    if (!t || !tauIndex.count(*t))
      continue;
    const int k = tauIndex[*t];
    ++nLeg[k];
    const double ae = std::abs(pd.momentum.eta());
    if (ae >= minAbsEta_ && ae <= maxAbsEta_)
      ++nLegEC[k];
  }

  // Energy and chosen candidate of each endcap leg, per tau.
  std::vector<std::vector<std::pair<float, int>>> legsOfTau(nLeg.size());
  for (unsigned int b : targets) {
    if (b >= graph.particles().size())
      continue;
    auto const& pd = graph.particles()[b];
    auto const& p4 = pd.momentum;
    if (!pd.isSignal() || p4.pt() < minPt_ || std::abs(p4.eta()) < minAbsEta_ || std::abs(p4.eta()) > maxAbsEta_)
      continue;
    pdgId_ = pd.pdgId;
    pt_ = p4.pt();
    eta_ = p4.eta();
    phi_ = p4.phi();
    energy_ = p4.E();
    const auto t = tauOf(graph, b);
    tau_ = (t && tauIndex.count(*t)) ? tauIndex[*t] : -1;

    // Best final trackster: largest shared fraction of the leg.
    tsIdx_ = -1;
    tsFrac_ = -1.f;
    tsScore_ = 1.f;
    if (b < tsT2r.size())
      for (auto const& m : tsT2r[b])
        if (m.sharedEnergy() > tsFrac_) {
          tsFrac_ = m.sharedEnergy();
          tsIdx_ = m.index();
        }
    if (tsIdx_ >= 0)
      tsScore_ = recoScore(tsR2t, tsIdx_, b);
    tsInd_ = (tsFrac_ > 0.5f && tsScore_ < 0.6f) ? 1 : 0;
    tsCand_ = (tsIdx_ >= 0 && candOfTs.count(tsIdx_)) ? candOfTs[tsIdx_] : -1;

    // Best track: highest reco purity, at least minTrackRecoPurity.
    tkIdx_ = -1;
    float bestPurity = -1.f;
    if (b < tkT2r.size())
      for (auto const& m : tkT2r[b]) {
        const float purity = 1.f - recoScore(tkR2t, m.index(), b);
        if (purity >= minTrackRecoPurity_ && purity > bestPurity) {
          bestPurity = purity;
          tkIdx_ = m.index();
        }
      }
    tkP_ = tkIdx_ >= 0 ? (*tkH)[tkIdx_].p() : -1.f;
    tkHP_ = tkMissOuter_ = tkMissInner_ = tkHits_ = -1;
    if (tkIdx_ >= 0) {
      auto const& tk = (*tkH)[tkIdx_];
      tkHP_ = tk.quality(reco::TrackBase::highPurity) ? 1 : 0;
      tkMissOuter_ = tk.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS);
      tkMissInner_ = tk.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
      tkHits_ = tk.hitPattern().numberOfValidHits();
    }
    tkCand_ = (tkIdx_ >= 0 && candOfTk.count(tkIdx_)) ? candOfTk[tkIdx_] : -1;
    tkMode_ = (tkIdx_ >= 0 && tkIdx_ < static_cast<int>(trackMode.size())) ? trackMode[tkIdx_] : -9;

    const int c = tkCand_ >= 0 ? tkCand_ : tsCand_;
    candPdg_ = c >= 0 ? cands[c].pdgId() : 0;
    candCharge_ = c >= 0 ? cands[c].charge() : 0;
    candE_ = c >= 0 ? cands[c].energy() : -1.f;
    candRawE_ = c >= 0 ? cands[c].rawEnergy() : -1.f;
    candNTs_ = c >= 0 ? cands[c].tracksters().size() : -1;
    candNTk_ = c >= 0 ? cands[c].trackPtrs().size() : -1;

    // Calorimeter part of the candidates.
    cIdx_ = -1;
    cFrac_ = -1.f;
    cSum_ = 0.f;
    chFrac_ = 0.f;
    if (b < cT2r.size())
      for (auto const& m : cT2r[b]) {
        cSum_ += m.sharedEnergy();
        if (m.sharedEnergy() > cFrac_) {
          cFrac_ = m.sharedEnergy();
          cIdx_ = m.index();
        }
        if (static_cast<int>(m.index()) == c)
          chFrac_ = m.sharedEnergy();
      }
    cScore_ = cIdx_ >= 0 ? recoScore(cR2t, cIdx_, b) : 1.f;
    cInd_ = (cFrac_ > 0.5f && cScore_ < 0.6f) ? 1 : 0;
    chScore_ = c >= 0 ? recoScore(cR2t, c, b) : 1.f;
    if (tau_ >= 0)
      legsOfTau[tau_].emplace_back(energy_, c);
    tree_->Fill();
  }

  // TICL candidate index of each particle-flow candidate: particleFlow points to particleFlowTmp, which is
  // particleFlowTmpBarrel followed by pfTICL, one entry per TICL candidate in order.
  const auto pfH = event.getHandle(pfToken_);
  const auto pfTmpH = event.getHandle(pfTmpToken_);
  // Without particle flow (a job that stops at the TICL candidates) there is no tau tree.
  if (!pfH.isValid() || !pfTmpH.isValid())
    return;
  const auto nBarrel = event.get(pfBarrelToken_).size();
  if (pfTmpH->size() != nBarrel + cands.size())
    throw cms::Exception("LogicError") << "particleFlowTmp has " << pfTmpH->size() << " entries, expected " << nBarrel
                                       << " + " << cands.size();
  auto ticlCandOf = [&](reco::CandidatePtr const& ptr) {
    if (ptr.id() != pfH.id())
      return -1;
    auto const src = (*pfH)[ptr.key()].sourceCandidatePtr(0);
    if (src.isNull() || src.id() != pfTmpH.id() || src.key() < nBarrel)
      return -1;
    return static_cast<int>(src.key() - nBarrel);
  };
  const auto hpsH = event.getHandle(hpsToken_);
  const auto dmfH = event.getHandle(hpsDmfToken_);
  std::vector<std::vector<int>> hpsCands(hpsH->size());
  for (unsigned int h = 0; h < hpsH->size(); ++h)
    for (auto const& sc : (*hpsH)[h].signalCands())
      if (const int c = ticlCandOf(sc); c >= 0)
        hpsCands[h].push_back(c);

  for (auto const& [id, k] : tauIndex) {
    TauContent content;
    collectTau(truth::Particle(&graph, id), content);
    tEvent_ = event_;
    tIdx_ = k;
    tProngs_ = content.prongs;
    tPi0s_ = content.pi0s;
    tLepton_ = content.lepton;
    tNLeg_ = nLeg[k];
    tNLegEC_ = nLegEC[k];
    tVisPt_ = content.visible.pt();
    tVisEta_ = tVisPt_ > 0.f ? content.visible.eta() : 0.f;
    tVisPhi_ = content.visible.phi();
    tHpsPt_ = 0.f;
    tHpsFrac_ = 0.f;
    tHpsDM_ = -1;
    tHpsDMF_ = 0;
    float total = 0.f;
    for (auto const& [e, c] : legsOfTau[k])
      total += e;
    for (unsigned int h = 0; h < hpsH->size() && total > 0.f; ++h) {
      float held = 0.f;
      for (auto const& [e, c] : legsOfTau[k])
        if (c >= 0 && std::find(hpsCands[h].begin(), hpsCands[h].end(), c) != hpsCands[h].end())
          held += e;
      if (held / total <= tHpsFrac_)
        continue;
      auto const& tau = (*hpsH)[h];
      tHpsFrac_ = held / total;
      tHpsPt_ = tau.pt();
      tHpsDM_ = tau.decayMode();
      tHpsDMF_ = dmfH.isValid() && (*dmfH)[reco::PFTauRef(hpsH, h)] > 0.5f ? 1 : 0;
    }
    taus_->Fill();
  }
}

DEFINE_FWK_MODULE(TICLCandidateProbe);
