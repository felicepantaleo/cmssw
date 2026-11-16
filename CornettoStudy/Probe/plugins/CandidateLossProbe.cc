// Study probe, not for integration. One row per member of a truth level: the rules of TruthBranchCandidateValidator
// (calorimeter identity at one adaptive working point, track identity from the Fixed track map) decide whether the
// member is reconstructed, and the simulated HGCAL energy of the member is split by where its layer clusters are:
// a candidate on the member, a candidate on another member, a candidate on an ancestor, a candidate on pileup or
// outside the level, a trackster that no candidate uses (claim or other), a CLUE3D trackster only, or no trackster.

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <limits>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "RecoTICL/Interpretation/interface/AssignmentMaps.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "Validation/TruthInfo/interface/CandidateTruthMatching.h"

#include "TTree.h"

namespace {
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
  using truth::candidate::LevelProjection;
  constexpr uint32_t kNone = std::numeric_limits<uint32_t>::max();

  bool isHGCal(uint32_t rawId) {
    const auto det = DetId(rawId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }

  // Where a layer cluster is, seen from one member. Lower value wins when a cluster is in several candidates.
  enum Category {
    kOwn,
    kOtherCharged,
    kOtherNeutral,
    kMerged,
    kPileup,
    kOutside,
    kNoIdentity,
    kClaim,
    kFinalNoCandidate,
    kClue3DOnly,
    kNoTrackster,
    kNoLayerCluster,
    kNCategories
  };
  constexpr std::array<const char*, kNCategories> kCategoryNames{"fOwn",
                                                                 "fOtherCharged",
                                                                 "fOtherNeutral",
                                                                 "fMerged",
                                                                 "fPileup",
                                                                 "fOutside",
                                                                 "fNoIdentity",
                                                                 "fClaim",
                                                                 "fFinalNoCand",
                                                                 "fClue3DOnly",
                                                                 "fNoTrackster",
                                                                 "fNoLC"};
}  // namespace

class CandidateLossProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit CandidateLossProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  struct LevelTokens {
    edm::EDGetTokenT<std::vector<unsigned int>> targets, eligibility;
  };
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<TICLCandidate>> candToken_;
  const edm::EDGetTokenT<MapType> wpMapToken_, truthToRecoToken_, trackMapToken_;
  const edm::EDGetTokenT<std::vector<reco::Track>> trackToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> finalToken_, clue3dToken_;
  edm::EDGetTokenT<std::vector<int>> trackToTracksterToken_, neutralIdxToken_, trackModeToken_;
  const bool hasMaps_;
  const double minTrackPurity_;
  std::vector<LevelTokens> levels_;

  TTree* tree_;
  int run_, lumi_, level_, particle_, pdg_, cls_, reconstructed_, nCand_, otherClass_, otherMode_, ownCharged_;
  // The candidate that takes most of the energy in the own and merged categories: its kind, the pdg id of its
  // calorimeter identity and its energy. Kind: the track mode (1 to 5), 10 a TICLv5 track, 20 a residual, 21 a neutral.
  int ownKind_, mergedKind_, mergedPdg_;
  float mergedE_, mergedShare_;
  // The signal energy in the HGCAL footprint of the merged candidate, split into the member, the charged members and
  // the other members of the level; and the sum, in simulated energy.
  float mergedSigOwn_, mergedSigCharged_, mergedSigNeutral_, mergedSig_;
  // Per reconstruction stage: the trackster that holds most of the member energy, its share of that energy, and the
  // signal composition of its footprint (the member, the charged members, the other members).
  static constexpr int kMaxStages = 4;
  std::vector<edm::EDGetTokenT<std::vector<ticl::Trackster>>> stageTokens_;
  int nStages_;
  std::array<float, kMaxStages> stageShare_, stageOwn_, stageCharged_, stageNeutral_;
  unsigned long long event_;
  unsigned int failed_;
  float e_, pt_, eta_, phi_, eHgc_, maxShare_, dRCharged_, eRep_;
  std::array<float, kNCategories> f_;
  // The part of fFinalNoCand in a jet trackster, in a neutral trackster, and in a trackster of another track mode.
  float fNoCandJet_, fNoCandNeutral_, fNoCandOther_;
};

CandidateLossProbe::CandidateLossProbe(const edm::ParameterSet& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      candToken_(consumes<std::vector<TICLCandidate>>(cfg.getParameter<edm::InputTag>("candidates"))),
      wpMapToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("candidateRecoToTruth"))),
      truthToRecoToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("candidateTruthToReco"))),
      trackMapToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("trackRecoToTruth"))),
      trackToken_(consumes<std::vector<reco::Track>>(cfg.getParameter<edm::InputTag>("tracks"))),
      lcToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))),
      finalToken_(consumes<std::vector<ticl::Trackster>>(cfg.getParameter<edm::InputTag>("finalTracksters"))),
      clue3dToken_(consumes<std::vector<ticl::Trackster>>(cfg.getParameter<edm::InputTag>("clue3dTracksters"))),
      hasMaps_(!cfg.getParameter<std::string>("interpretation").empty()),
      minTrackPurity_(cfg.getParameter<double>("minTrackPurity")) {
  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("stageTracksters"))
    stageTokens_.push_back(consumes<std::vector<ticl::Trackster>>(tag));
  if (stageTokens_.size() > kMaxStages)
    throw cms::Exception("Configuration") << "at most " << kMaxStages << " stages";
  nStages_ = static_cast<int>(stageTokens_.size());
  if (hasMaps_) {
    const auto label = cfg.getParameter<std::string>("interpretation");
    trackToTracksterToken_ = consumes<std::vector<int>>(edm::InputTag(label, "trackToTrackster"));
    neutralIdxToken_ = consumes<std::vector<int>>(edm::InputTag(label, "neutralIdx"));
    trackModeToken_ = consumes<std::vector<int>>(edm::InputTag(label, "trackMode"));
  }
  for (auto name : cfg.getParameter<std::vector<std::string>>("truthLevels")) {
    name[0] = std::toupper(static_cast<unsigned char>(name[0]));
    const std::string instance = "truthToRecoTargets" + name;
    levels_.push_back(
        {consumes<std::vector<unsigned int>>(edm::InputTag("truthBranchTargets", instance)),
         consumes<std::vector<unsigned int>>(edm::InputTag("truthBranchTargets", instance + "Eligibility"))});
  }
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("members", "members");
  tree_->Branch("run", &run_);
  tree_->Branch("lumi", &lumi_);
  tree_->Branch("event", &event_);
  tree_->Branch("level", &level_);
  tree_->Branch("particle", &particle_);
  tree_->Branch("pdg", &pdg_);
  tree_->Branch("cls", &cls_);
  tree_->Branch("failed", &failed_);
  tree_->Branch("reconstructed", &reconstructed_);
  tree_->Branch("nCand", &nCand_);
  tree_->Branch("otherClass", &otherClass_);
  tree_->Branch("otherMode", &otherMode_);
  tree_->Branch("ownCharged", &ownCharged_);
  tree_->Branch("ownKind", &ownKind_);
  tree_->Branch("mergedKind", &mergedKind_);
  tree_->Branch("mergedPdg", &mergedPdg_);
  tree_->Branch("mergedE", &mergedE_);
  tree_->Branch("mergedShare", &mergedShare_);
  tree_->Branch("mergedSigOwn", &mergedSigOwn_);
  tree_->Branch("mergedSigCharged", &mergedSigCharged_);
  tree_->Branch("mergedSigNeutral", &mergedSigNeutral_);
  tree_->Branch("mergedSig", &mergedSig_);
  tree_->Branch("nStages", &nStages_);
  tree_->Branch("stageShare", stageShare_.data(), "stageShare[4]/F");
  tree_->Branch("stageOwn", stageOwn_.data(), "stageOwn[4]/F");
  tree_->Branch("stageCharged", stageCharged_.data(), "stageCharged[4]/F");
  tree_->Branch("stageNeutral", stageNeutral_.data(), "stageNeutral[4]/F");
  for (auto [n, v] : std::vector<std::pair<const char*, float*>>{{"E", &e_},
                                                                 {"pt", &pt_},
                                                                 {"eta", &eta_},
                                                                 {"phi", &phi_},
                                                                 {"eHgc", &eHgc_},
                                                                 {"maxShare", &maxShare_},
                                                                 {"dRCharged", &dRCharged_},
                                                                 {"eRep", &eRep_}})
    tree_->Branch(n, v);
  for (int k = 0; k < kNCategories; ++k)
    tree_->Branch(kCategoryNames[k], &f_[k]);
  tree_->Branch("fNoCandJet", &fNoCandJet_);
  tree_->Branch("fNoCandNeutral", &fNoCandNeutral_);
  tree_->Branch("fNoCandOther", &fNoCandOther_);
}

void CandidateLossProbe::analyze(const edm::Event& event, const edm::EventSetup&) {
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  auto const& cands = event.get(candToken_);
  auto const& wpMap = event.get(wpMapToken_).getMap();
  auto const& truthToReco = event.get(truthToRecoToken_).getMap();
  auto const& trackMap = event.get(trackMapToken_).getMap();
  const auto tracksH = event.getHandle(trackToken_);
  auto const& lcs = event.get(lcToken_);
  auto const& finalTs = event.get(finalToken_);
  auto const& clue3d = event.get(clue3dToken_);
  const uint32_t nP = graph.nParticles();
  run_ = event.id().run();
  lumi_ = event.id().luminosityBlock();
  event_ = event.id().event();

  // Candidate identities, as in the validator.
  const std::size_t nC = cands.size();
  std::vector<uint32_t> caloId(nC, kNone), trackId(nC, kNone);
  std::vector<int> kind(nC, 21);
  std::vector<int> const* trackMode = hasMaps_ ? &event.get(trackModeToken_) : nullptr;
  for (std::size_t c = 0; c < nC; ++c) {
    if (c < wpMap.size() && !wpMap[c].empty() && wpMap[c][0].index() < nP)
      caloId[c] = wpMap[c][0].index();
    auto const track = cands[c].trackPtr();
    if (track.isNull() || track.id() != tracksH.id())
      continue;
    kind[c] = trackMode == nullptr ? 10 : track.key() < trackMode->size() ? (*trackMode)[track.key()] : -1;
    if (track.key() < trackMap.size() && !trackMap[track.key()].empty()) {
      auto const& best = trackMap[track.key()][0];
      if (1.f - best.score() >= minTrackPurity_ && best.index() < nP)
        trackId[c] = best.index();
    }
  }

  // Layer clusters of each candidate, of the final tracksters no candidate uses, and of the CLUE3D tracksters.
  const std::size_t nLC = lcs.size();
  std::vector<std::vector<uint32_t>> candsOfLC(nLC);
  for (std::size_t c = 0; c < nC; ++c)
    for (auto const& ts : cands[c].tracksters())
      if (ts.isNonnull())
        for (auto v : ts->vertices())
          if (v < nLC && (candsOfLC[v].empty() || candsOfLC[v].back() != c))
            candsOfLC[v].push_back(c);
  std::vector<char> inFinal(nLC, 0), inClaim(nLC, 0), inClue3d(nLC, 0);
  // The track mode that references each final trackster, 0 for a neutral trackster, -9 for none (a claim).
  std::vector<int> refMode(finalTs.size(), -9);
  if (hasMaps_) {
    auto const& trackToTrackster = event.get(trackToTracksterToken_);
    for (std::size_t k = 0; k < trackToTrackster.size(); ++k) {
      const int i = trackToTrackster[k];
      if (i >= 0 && static_cast<std::size_t>(i) < finalTs.size() && refMode[i] == -9)
        refMode[i] = k < trackMode->size() ? (*trackMode)[k] : -1;
    }
    for (int i : event.get(neutralIdxToken_))
      if (i >= 0 && static_cast<std::size_t>(i) < finalTs.size() && refMode[i] == -9)
        refMode[i] = 0;
  }
  std::vector<int> modeOfLC(nLC, -9);
  for (std::size_t i = 0; i < finalTs.size(); ++i)
    for (auto v : finalTs[i].vertices())
      if (v < nLC) {
        inFinal[v] = 1;
        if (hasMaps_ && refMode[i] == -9)
          inClaim[v] = 1;
        if (modeOfLC[v] == -9)
          modeOfLC[v] = refMode[i];
      }
  for (auto const& ts : clue3d)
    for (auto v : ts.vertices())
      if (v < nLC)
        inClue3d[v] = 1;
  // A TICLv6 candidate without a track is a residual when a track references its trackster.
  if (hasMaps_)
    for (std::size_t c = 0; c < nC; ++c)
      if (cands[c].trackPtr().isNull() && !cands[c].tracksters().empty() && cands[c].tracksters()[0].isNonnull() &&
          cands[c].tracksters()[0].key() < refMode.size() && refMode[cands[c].tracksters()[0].key()] > 0)
        kind[c] = 20;
  // The tracksters of each stage that hold each layer cluster.
  std::vector<std::vector<ticl::Trackster> const*> stages;
  std::vector<std::vector<std::vector<uint32_t>>> stageOfLC;
  for (auto const& token : stageTokens_) {
    stages.push_back(&event.get(token));
    stageOfLC.emplace_back(nLC);
    for (std::size_t i = 0; i < stages.back()->size(); ++i)
      for (auto v : (*stages.back())[i].vertices())
        if (v < nLC && (stageOfLC.back()[v].empty() || stageOfLC.back()[v].back() != i))
          stageOfLC.back()[v].push_back(i);
  }
  std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>> lcsOfCell;
  for (std::size_t v = 0; v < nLC; ++v)
    for (auto const& [id, frac] : lcs[v].hitsAndFractions())
      if (frac > 0.f)
        lcsOfCell[id.rawId()].emplace_back(v, frac);

  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  for (std::size_t l = 0; l < levels_.size(); ++l) {
    const auto targetsH = event.getHandle(levels_[l].targets);
    if (!targetsH.isValid())
      continue;
    auto const& members = *targetsH;
    const auto eligH = event.getHandle(levels_[l].eligibility);
    const bool hasElig = eligH.isValid() && eligH->size() == members.size();
    LevelProjection projection(graph, members);
    std::vector<truth::candidate::ParticleClass> memberClass;
    for (const unsigned int m : members)
      memberClass.push_back(m < nP
                                ? truth::candidate::truthClass(graph.particles()[m].pdgId, graph.particle(m).charge())
                                : truth::candidate::ParticleClass::Other);
    // The simulated HGCAL energy of every member on each cell.
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>> membersOfCell;
    for (std::size_t u = 0; u < members.size(); ++u) {
      if (members[u] >= nP)
        continue;
      hits.clear();
      index.appendSubgraphHits(truth::HitChannel::Calo, members[u], hits);
      for (auto const& h : hits)
        if (isHGCal(h.detId) && h.energy > 0.f)
          membersOfCell[h.detId].emplace_back(u, h.energy);
    }
    std::vector<std::vector<uint32_t>> belonging(members.size());
    for (std::size_t c = 0; c < nC; ++c) {
      const uint32_t cs = projection.memberSlot(caloId[c]), ts = projection.memberSlot(trackId[c]);
      if (cs != LevelProjection::kNoMember)
        belonging[cs].push_back(c);
      if (ts != LevelProjection::kNoMember && ts != cs)
        belonging[ts].push_back(c);
    }
    level_ = static_cast<int>(l);
    for (std::size_t t = 0; t < members.size(); ++t) {
      const uint32_t p = members[t];
      if (p >= nP || truth::isInvisible(graph.particles()[p].pdgId))
        continue;
      auto const& pd = graph.particles()[p];
      if (pd.momentum.pt() <= 0.)
        continue;
      const double aeta = std::abs(pd.momentum.eta());
      if (aeta < 1.5 || aeta > 3.0)
        continue;
      particle_ = static_cast<int>(p);
      pdg_ = pd.pdgId;
      cls_ = static_cast<int>(memberClass[t]);
      failed_ = hasElig ? (*eligH)[t] : 0u;
      e_ = pd.momentum.energy();
      pt_ = pd.momentum.pt();
      eta_ = pd.momentum.eta();
      phi_ = pd.momentum.phi();
      reconstructed_ = belonging[t].empty() ? 0 : 1;
      nCand_ = static_cast<int>(belonging[t].size());
      maxShare_ = 0.f;
      if (p < truthToReco.size())
        for (auto const& e : truthToReco[p])
          maxShare_ = std::max(maxShare_, e.sharedEnergy());
      // The representative energy: the candidate on the member with the largest share.
      eRep_ = -1.f;
      ownCharged_ = 0;
      float bestShare = -1.f;
      for (const uint32_t c : belonging[t]) {
        float share = 0.f;
        if (p < truthToReco.size())
          for (auto const& e : truthToReco[p])
            if (e.index() == c)
              share = e.sharedEnergy();
        if (share > bestShare) {
          bestShare = share;
          eRep_ = cands[c].energy();
          ownCharged_ = cands[c].trackPtr().isNonnull() ? 1 : 0;
        }
      }
      // The nearest charged member of the level that is not the member.
      dRCharged_ = 99.f;
      for (std::size_t u = 0; u < members.size(); ++u) {
        if (u == t || !truth::candidate::isCharged(memberClass[u]) || members[u] >= nP)
          continue;
        auto const& q = graph.particles()[members[u]].momentum;
        if (q.pt() > 0.)
          dRCharged_ =
              std::min<float>(dRCharged_, reco::deltaR(q.eta(), q.phi(), pd.momentum.eta(), pd.momentum.phi()));
      }

      f_.fill(0.f);
      fNoCandJet_ = fNoCandNeutral_ = fNoCandOther_ = 0.f;
      std::array<float, truth::candidate::kNClasses> otherByClass{};
      std::unordered_map<int, float> otherByMode, ownByCand, mergedByCand;
      hits.clear();
      index.appendSubgraphHits(truth::HitChannel::Calo, p, hits);
      double total = 0.;
      for (auto const& h : hits) {
        if (!isHGCal(h.detId) || h.energy <= 0.f)
          continue;
        total += h.energy;
        float clustered = 0.f;
        auto it = lcsOfCell.find(h.detId);
        if (it != lcsOfCell.end()) {
          for (auto const& [v, frac] : it->second) {
            clustered += frac;
            const float w = frac * h.energy;
            int best = kNCategories;
            int bestOtherSlotClass = -1, bestOtherMode = -2;
            uint32_t bestCand = kNone;
            for (const uint32_t c : candsOfLC[v]) {
              int cat;
              const uint32_t cs = projection.memberSlot(caloId[c]), ts = projection.memberSlot(trackId[c]);
              if (cs == t || ts == t)
                cat = kOwn;
              else if (cs != LevelProjection::kNoMember)
                cat = cands[c].trackPtr().isNonnull() ? kOtherCharged : kOtherNeutral;
              else if (caloId[c] == kNone)
                cat = kNoIdentity;
              else if (projection.isAbove(caloId[c]))
                cat = kMerged;
              else if (graph.particles()[caloId[c]].isFromPileup())
                cat = kPileup;
              else
                cat = kOutside;
              if (cat < best) {
                best = cat;
                bestCand = c;
                if (cat == kOtherCharged || cat == kOtherNeutral) {
                  bestOtherSlotClass = static_cast<int>(memberClass[cs]);
                  bestOtherMode = kind[c];
                }
              }
            }
            if (best == kNCategories)
              best = inClaim[v] ? kClaim : inFinal[v] ? kFinalNoCandidate : inClue3d[v] ? kClue3DOnly : kNoTrackster;
            f_[best] += w;
            if (best == kFinalNoCandidate)
              (modeOfLC[v] == static_cast<int>(ticl::TrackMode::kJetMember) ? fNoCandJet_
               : modeOfLC[v] == 0                                           ? fNoCandNeutral_
                                                                            : fNoCandOther_) += w;
            if (best == kOwn)
              ownByCand[bestCand] += w;
            if (best == kMerged)
              mergedByCand[bestCand] += w;
            if (bestOtherSlotClass >= 0) {
              otherByClass[bestOtherSlotClass] += w;
              otherByMode[bestOtherMode] += w;
            }
          }
        }
        if (clustered < 1.f)
          f_[kNoLayerCluster] += (1.f - clustered) * h.energy;
      }
      eHgc_ = static_cast<float>(total);
      if (total > 0.) {
        for (auto& x : f_)
          x = static_cast<float>(x / total);
        fNoCandJet_ /= total;
        fNoCandNeutral_ /= total;
        fNoCandOther_ /= total;
      }
      otherClass_ = -1;
      float bestOther = 0.f;
      for (std::size_t k = 0; k < otherByClass.size(); ++k)
        if (otherByClass[k] > bestOther) {
          bestOther = otherByClass[k];
          otherClass_ = static_cast<int>(k);
        }
      auto argmax = [](std::unordered_map<int, float> const& m) {
        int arg = -3;
        float best = 0.f;
        for (auto const& [k, w] : m)
          if (w > best) {
            best = w;
            arg = k;
          }
        return arg;
      };
      otherMode_ = argmax(otherByMode);
      const int ownCand = argmax(ownByCand), mergedCand = argmax(mergedByCand);
      ownKind_ = ownCand >= 0 ? kind[ownCand] : -3;
      mergedKind_ = mergedCand >= 0 ? kind[mergedCand] : -3;
      mergedPdg_ = mergedCand >= 0 ? graph.particles()[caloId[mergedCand]].pdgId : 0;
      mergedE_ = mergedCand >= 0 ? cands[mergedCand].energy() : -1.f;
      // The share of the merged candidate in the member energy, from the truth-to-reco map.
      mergedShare_ = 0.f;
      if (mergedCand >= 0 && p < truthToReco.size())
        for (auto const& e : truthToReco[p])
          if (e.index() == static_cast<uint32_t>(mergedCand))
            mergedShare_ = e.sharedEnergy();
      mergedSigOwn_ = mergedSigCharged_ = mergedSigNeutral_ = mergedSig_ = 0.f;
      if (mergedCand >= 0) {
        for (auto const& ts : cands[mergedCand].tracksters()) {
          if (ts.isNull())
            continue;
          for (auto v : ts->vertices()) {
            if (v >= nLC)
              continue;
            for (auto const& [id, frac] : lcs[v].hitsAndFractions()) {
              auto it = membersOfCell.find(id.rawId());
              if (it == membersOfCell.end())
                continue;
              for (auto const& [u, e] : it->second) {
                const float w = frac * e;
                mergedSig_ += w;
                if (u == t)
                  mergedSigOwn_ += w;
                else if (truth::candidate::isCharged(memberClass[u]))
                  mergedSigCharged_ += w;
                else
                  mergedSigNeutral_ += w;
              }
            }
          }
        }
        if (mergedSig_ > 0.f) {
          mergedSigOwn_ /= mergedSig_;
          mergedSigCharged_ /= mergedSig_;
          mergedSigNeutral_ /= mergedSig_;
        }
      }
      // The stage trace.
      stageShare_.fill(-1.f);
      stageOwn_.fill(-1.f);
      stageCharged_.fill(-1.f);
      stageNeutral_.fill(-1.f);
      for (std::size_t k = 0; k < stages.size(); ++k) {
        std::unordered_map<uint32_t, float> inTrackster;
        for (auto const& h : hits) {
          if (!isHGCal(h.detId) || h.energy <= 0.f)
            continue;
          auto it = lcsOfCell.find(h.detId);
          if (it == lcsOfCell.end())
            continue;
          for (auto const& [v, frac] : it->second)
            for (const uint32_t i : stageOfLC[k][v])
              inTrackster[i] += frac * h.energy;
        }
        uint32_t bestTs = kNone;
        float bestE = 0.f;
        for (auto const& [i, e] : inTrackster)
          if (e > bestE) {
            bestE = e;
            bestTs = i;
          }
        if (bestTs == kNone || total <= 0.)
          continue;
        stageShare_[k] = static_cast<float>(bestE / total);
        float own = 0.f, charged = 0.f, neutral = 0.f;
        for (auto v : (*stages[k])[bestTs].vertices()) {
          if (v >= nLC)
            continue;
          for (auto const& [id, frac] : lcs[v].hitsAndFractions()) {
            auto it = membersOfCell.find(id.rawId());
            if (it == membersOfCell.end())
              continue;
            for (auto const& [u, e] : it->second)
              (u == t ? own : truth::candidate::isCharged(memberClass[u]) ? charged : neutral) += frac * e;
          }
        }
        const float sig = own + charged + neutral;
        if (sig > 0.f) {
          stageOwn_[k] = own / sig;
          stageCharged_[k] = charged / sig;
          stageNeutral_[k] = neutral / sig;
        }
      }
      tree_->Fill();
    }
  }
}

DEFINE_FWK_MODULE(CandidateLossProbe);
