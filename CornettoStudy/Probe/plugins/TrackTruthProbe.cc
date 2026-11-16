// Study probe, not for integration. Tracks in the HGCAL acceptance against the truth graph.
//   tracks: one row per track with minAbsEta < |eta| < maxAbsEta: kinematics and quality, whether the interpretation
//           selection (cutTk) passes, the interpretation track mode and linked trackster energy, whether a candidate
//           holds it, and its truth (Fixed reco-to-truth match): species, signal or pileup, match purity, and the
//           HGCAL deposit of the matched particle and its descendants.
//   legs:   one row per signal charged target in the same window: the best track (TruthToReco, reco purity >= 0.75),
//           its selection, track mode and candidate.
// For tracks matched to a particle with HGCAL hits, the transverse distance between the energy-weighted position of
// those hits (CE-E and CE-H separately) and the track extension: straight from the outermost state, or propagated
// through the field to the HGCAL front and straight from there.

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTICL/Common/interface/TICLUtils.h"
#include "RecoTICL/Interpretation/interface/TrackImpact.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
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

class TrackTruthProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TrackTruthProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<std::vector<reco::Track>> trackToken_;
  const edm::EDGetTokenT<MapType> r2tToken_, t2rToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<unsigned int>> legsToken_;
  const edm::EDGetTokenT<std::vector<TICLCandidate>> candToken_;
  const edm::EDGetTokenT<std::vector<reco::GenParticle>> genToken_;
  edm::EDGetTokenT<std::vector<int>> modeToken_, targetToken_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>> interpToken_;
  const StringCutObjectSelector<reco::Track> cut_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propToken_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hgconsToken_;
  const double minAbsEta_, maxAbsEta_;
  TTree* tracks_;
  TTree* legs_;
  int event_, hp_, missOuter_, missInner_, nValid_, pass_, mode_, inCand_, candPdg_, truthPdg_, signal_;
  float pt_, eta_, p_, relErr_, chi2n_, dz_, tsE_, truthPurity_, truthDeposit_, truthE_, dStraightEE_, dHelixEE_,
      dStraightEH_, dHelixEH_;
  int lPdg_, lHasTrack_, lPass_, lMode_, lInCand_, lCandPdg_, lHp_, lMissOuter_;
  float lPt_, lEta_, lE_, lDeposit_, lTrackP_, lRelErr_;
};

TrackTruthProbe::TrackTruthProbe(const edm::ParameterSet& cfg)
    : trackToken_(consumes<std::vector<reco::Track>>(cfg.getParameter<edm::InputTag>("tracks"))),
      r2tToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("trackRecoToTruth"))),
      t2rToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("trackTruthToReco"))),
      graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      legsToken_(consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("legs"))),
      candToken_(consumes<std::vector<TICLCandidate>>(cfg.getParameter<edm::InputTag>("candidates"))),
      genToken_(consumes<std::vector<reco::GenParticle>>(cfg.getParameter<edm::InputTag>("genParticles"))),
      cut_(cfg.getParameter<std::string>("cutTk")),
      geomToken_(esConsumes()),
      fieldToken_(esConsumes()),
      propToken_(esConsumes(edm::ESInputTag("", cfg.getParameter<std::string>("propagator")))),
      hgconsToken_(esConsumes(edm::ESInputTag("", "HGCalEESensitive"))),
      minAbsEta_(cfg.getParameter<double>("minAbsEta")),
      maxAbsEta_(cfg.getParameter<double>("maxAbsEta")) {
  if (const auto interp = cfg.getParameter<std::string>("interpretations"); !interp.empty()) {
    modeToken_ = consumes<std::vector<int>>(edm::InputTag(interp, "trackMode"));
    targetToken_ = consumes<std::vector<int>>(edm::InputTag(interp, "trackToTrackster"));
    interpToken_ = consumes<std::vector<ticl::Trackster>>(edm::InputTag(interp));
  }
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  tracks_ = fs->make<TTree>("tracks", "tracks");
  for (auto [n, v] : std::vector<std::pair<const char*, int*>>{{"event", &event_},
                                                               {"hp", &hp_},
                                                               {"missOuter", &missOuter_},
                                                               {"missInner", &missInner_},
                                                               {"nValid", &nValid_},
                                                               {"pass", &pass_},
                                                               {"mode", &mode_},
                                                               {"inCand", &inCand_},
                                                               {"candPdg", &candPdg_},
                                                               {"truthPdg", &truthPdg_},
                                                               {"signal", &signal_}})
    tracks_->Branch(n, v);
  for (auto [n, v] : std::vector<std::pair<const char*, float*>>{{"pt", &pt_},
                                                                 {"eta", &eta_},
                                                                 {"p", &p_},
                                                                 {"relErr", &relErr_},
                                                                 {"chi2n", &chi2n_},
                                                                 {"dz", &dz_},
                                                                 {"tsE", &tsE_},
                                                                 {"truthPurity", &truthPurity_},
                                                                 {"truthDeposit", &truthDeposit_},
                                                                 {"truthE", &truthE_},
                                                                 {"dStraightEE", &dStraightEE_},
                                                                 {"dHelixEE", &dHelixEE_},
                                                                 {"dStraightEH", &dStraightEH_},
                                                                 {"dHelixEH", &dHelixEH_}})
    tracks_->Branch(n, v);
  legs_ = fs->make<TTree>("legs", "legs");
  for (auto [n, v] : std::vector<std::pair<const char*, int*>>{{"event", &event_},
                                                               {"pdg", &lPdg_},
                                                               {"hasTrack", &lHasTrack_},
                                                               {"pass", &lPass_},
                                                               {"mode", &lMode_},
                                                               {"inCand", &lInCand_},
                                                               {"candPdg", &lCandPdg_},
                                                               {"hp", &lHp_},
                                                               {"missOuter", &lMissOuter_}})
    legs_->Branch(n, v);
  for (auto [n, v] : std::vector<std::pair<const char*, float*>>{{"pt", &lPt_},
                                                                 {"eta", &lEta_},
                                                                 {"E", &lE_},
                                                                 {"deposit", &lDeposit_},
                                                                 {"trackP", &lTrackP_},
                                                                 {"relErr", &lRelErr_}})
    legs_->Branch(n, v);
}

void TrackTruthProbe::analyze(const edm::Event& event, const edm::EventSetup& es) {
  hgcal::RecHitTools rhtools;
  rhtools.setGeometry(es.getData(geomToken_));
  auto const* field = &es.getData(fieldToken_);
  auto const& prop = es.getData(propToken_);
  const auto frontDisks = ticl::utils::buildHGCalFirstDisks(es.getData(hgconsToken_));
  const auto tkH = event.getHandle(trackToken_);
  auto const& tracks = *tkH;
  auto const& r2t = event.get(r2tToken_).getMap();
  auto const& t2r = event.get(t2rToken_).getMap();
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  auto const& legs = event.get(legsToken_);
  auto const& cands = event.get(candToken_);
  auto const& gen = event.get(genToken_);
  const std::vector<int> none;
  auto const& mode = modeToken_.isUninitialized() ? none : event.get(modeToken_);
  auto const& target = targetToken_.isUninitialized() ? none : event.get(targetToken_);
  const std::vector<ticl::Trackster> noTs;
  auto const& interp = interpToken_.isUninitialized() ? noTs : event.get(interpToken_);
  event_ = event.id().event();
  const double pvZ = gen.empty() ? 0. : gen.front().vz();
  const unsigned int nP = graph.particles().size();

  // Candidate holding each track.
  std::unordered_map<size_t, int> candOfTrack;
  for (auto const& c : cands)
    for (auto const& t : c.trackPtrs())
      if (t.isNonnull() && t.id() == tkH.id())
        candOfTrack[t.key()] = c.pdgId();
  // HGCAL deposit of a particle and its descendants.
  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  auto deposit = [&](unsigned int u) {
    hits.clear();
    index.appendSubgraphHits(truth::HitChannel::Calo, u, hits);
    double d = 0.;
    for (auto const& h : hits)
      if (isHGCal(h.detId))
        d += h.energy;
    return static_cast<float>(d);
  };

  for (size_t i = 0; i < tracks.size(); ++i) {
    auto const& tk = tracks[i];
    if (!(std::abs(tk.eta()) > minAbsEta_ && std::abs(tk.eta()) < maxAbsEta_))
      continue;
    pt_ = tk.pt();
    eta_ = tk.eta();
    p_ = tk.p();
    relErr_ = tk.pt() > 0. ? tk.ptError() / tk.pt() : -1.f;
    chi2n_ = tk.normalizedChi2();
    dz_ = tk.vz() - pvZ;
    hp_ = tk.quality(reco::TrackBase::highPurity);
    missOuter_ = tk.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS);
    missInner_ = tk.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
    nValid_ = tk.hitPattern().numberOfValidHits();
    pass_ = cut_(tk);
    mode_ = i < mode.size() ? mode[i] : -9;
    tsE_ = (i < target.size() && target[i] >= 0 && target[i] < static_cast<int>(interp.size()))
               ? interp[target[i]].raw_energy()
               : -1.f;
    auto it = candOfTrack.find(i);
    inCand_ = it != candOfTrack.end();
    candPdg_ = inCand_ ? it->second : 0;
    truthPdg_ = signal_ = 0;
    truthPurity_ = truthDeposit_ = truthE_ = -1.f;
    dStraightEE_ = dHelixEE_ = dStraightEH_ = dHelixEH_ = -1.f;
    if (i < r2t.size() && !r2t[i].empty()) {
      auto best = std::min_element(
          r2t[i].begin(), r2t[i].end(), [](auto const& a, auto const& b) { return a.score() < b.score(); });
      if (best->index() < nP) {
        auto const& pd = graph.particles()[best->index()];
        truthPdg_ = pd.pdgId;
        signal_ = pd.eventId == 0 ? 1 : 0;
        truthPurity_ = 1.f - best->score();
        truthE_ = pd.momentum.E();
        truthDeposit_ = deposit(best->index());
        // Energy-weighted hit position of the particle in CE-E and in CE-H (hits filled by deposit()).
        double w[2] = {0., 0.}, x[2] = {0., 0.}, y[2] = {0., 0.}, z[2] = {0., 0.};
        for (auto const& h : hits) {
          const DetId id(h.detId);
          if (!isHGCal(h.detId) || !(h.energy > 0.f))
            continue;
          const int k = id.det() == DetId::HGCalEE ? 0 : 1;
          const GlobalPoint pos = rhtools.getPosition(id);
          w[k] += h.energy;
          x[k] += h.energy * pos.x();
          y[k] += h.energy * pos.y();
          z[k] += h.energy * pos.z();
        }
        const auto impact = ticl::propagateToHGCalFront(tk, field, prop, frontDisks);
        float* dS[2] = {&dStraightEE_, &dStraightEH_};
        float* dH[2] = {&dHelixEE_, &dHelixEH_};
        for (int k = 0; k < 2; ++k) {
          if (w[k] <= 0.)
            continue;
          const GlobalPoint c(x[k] / w[k], y[k] / w[k], z[k] / w[k]);
          *dS[k] = ticl::trackTransverseDistance(ticl::TrackImpact{}, tk, c);
          *dH[k] = impact.valid ? ticl::impactTransverseDistance(impact, c) : -2.f;
        }
      }
    }
    tracks_->Fill();
  }

  for (unsigned int b : legs) {
    if (b >= nP)
      continue;
    auto const& pd = graph.particles()[b];
    const int a = std::abs(pd.pdgId);
    const bool charged = a == 211 || a == 321 || a == 2212 || a == 11 || a == 13;
    if (pd.eventId != 0 || !charged ||
        !(std::abs(pd.momentum.eta()) > minAbsEta_ && std::abs(pd.momentum.eta()) < maxAbsEta_))
      continue;
    lPdg_ = pd.pdgId;
    lPt_ = pd.momentum.pt();
    lEta_ = pd.momentum.eta();
    lE_ = pd.momentum.E();
    lDeposit_ = deposit(b);
    int tk = -1;
    float bestPurity = 0.75f;
    if (b < t2r.size())
      for (auto const& m : t2r[b]) {
        const unsigned int t = m.index();
        if (t >= r2t.size())
          continue;
        for (auto const& r : r2t[t])
          if (r.index() == b && 1.f - r.score() >= bestPurity) {
            bestPurity = 1.f - r.score();
            tk = static_cast<int>(t);
          }
      }
    lHasTrack_ = tk >= 0;
    lPass_ = tk >= 0 ? cut_(tracks[tk]) : 0;
    lMode_ = tk >= 0 && tk < static_cast<int>(mode.size()) ? mode[tk] : -9;
    auto it = tk >= 0 ? candOfTrack.find(tk) : candOfTrack.end();
    lInCand_ = it != candOfTrack.end();
    lCandPdg_ = lInCand_ ? it->second : 0;
    lHp_ = tk >= 0 ? tracks[tk].quality(reco::TrackBase::highPurity) : -1;
    lMissOuter_ = tk >= 0 ? tracks[tk].hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS) : -1;
    lTrackP_ = tk >= 0 ? tracks[tk].p() : -1.f;
    lRelErr_ = tk >= 0 && tracks[tk].pt() > 0. ? tracks[tk].ptError() / tracks[tk].pt() : -1.f;
    legs_->Fill();
  }
}

DEFINE_FWK_MODULE(TrackTruthProbe);
