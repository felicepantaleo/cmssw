// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

// Reco-side DQM validator for TICLCandidates against the truth Branch graph.
//
// A TICLCandidate is not a single-channel object: it mixes a track (tracker hits)
// with tracksters (HGCAL calo hits), and the truth Branch it should represent has
// BOTH a tracker and a calo footprint. So, unlike the generic single-channel
// BranchTracksterRecoValidator / BranchTrackRecoValidator, this validator matches a
// candidate to the truth on the two channels at once and asks the questions a
// physicist actually has about candidate quality:
//
//   Q1  How often is a truth particle turned into a candidate at all (efficiency),
//       and where does that efficiency fall off (species, p_T, eta)?
//   Q2  When a candidate exists, is it the RIGHT kind - the "match -> +charge ->
//       +PID -> +energy" ladder. The gaps between consecutive steps localize the
//       loss (linking vs charge assignment vs PID vs energy regression).
//   Q3  For charged candidates, does the track and the calo point to the SAME truth
//       particle (track<->trackster linking consistency)? A candidate whose track
//       comes from a different particle than its tracksters is a linking failure
//       even if each half matches something.
//   Q4  When a candidate does NOT cleanly represent one particle, why: fake (matches
//       no branch), merged (absorbs >=2 branches), or split (one branch reconstructed
//       as >=2 candidates -> duplicate)?
//   Q5  How is the energy measured (response E_reco/E_truth), on the regressed and
//       raw scales, vs p_T/eta/energy?
//
// The branch (truth) side is restricted to the interesting particles
// (interestingPdgIds); for a single-particle gun set onlyGenPrimaries=True so the
// reference is the fired GEN particle - a clean antichain (a Branch subgraph
// aggregates descendants, so against the full graph the reference degenerates).
// Numerator/denominator histograms are turned into efficiencies/rates by the
// DQMGenericClient harvester (truthGraphDQMHarvester_cff); distributions and
// profiles are final as booked.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"
#include "PhysicsTools/TruthInfo/interface/RecoHitAdapters.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace {
  // Coarse reconstructible particle classes: what a TICLCandidate can actually get
  // right. Truth PDG id and candidate PDG id are compared at this granularity, so a
  // photon reconstructed as pi0 or a K0L as a neutron is not a PID failure, while a
  // charged pion labelled a photon is.
  enum class PidClass { NeutralEM, ChargedEM, Muon, ChargedHadron, NeutralHadron, Other };
  PidClass pidClass(int pdgId) {
    switch (std::abs(pdgId)) {
      case 22:
      case 111:
        return PidClass::NeutralEM;
      case 11:
        return PidClass::ChargedEM;
      case 13:
        return PidClass::Muon;
      case 211:
      case 321:
      case 2212:
        return PidClass::ChargedHadron;
      case 130:
      case 310:
      case 2112:
        return PidClass::NeutralHadron;
      default:
        return PidClass::Other;
    }
  }
  // A truth particle is charged if its class carries a track.
  bool isChargedClass(PidClass c) {
    return c == PidClass::ChargedEM || c == PidClass::Muon || c == PidClass::ChargedHadron;
  }

  // Categorical outcome per candidate (Q4). Kept in sync with the bin labels booked
  // in bookHistograms.
  enum CandOutcome { kMatched = 1, kFake = 2, kMerged = 3, kTrackCaloMismatch = 4 };
}  // namespace

class BranchTICLCandidateValidator : public DQMEDAnalyzer {
public:
  explicit BranchTICLCandidateValidator(edm::ParameterSet const&);
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  [[nodiscard]] bool selected(double eta, double x) const {
    return std::abs(eta) >= minAbsEta_ && std::abs(eta) <= maxAbsEta_ && x >= minX_;
  }

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<TICLCandidate>> candidateToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClusterToken_;
  const std::vector<int> interestingPdgIds_;
  const std::string folder_;
  const double xMax_;
  const double energyMax_;
  const double minX_;
  const double minAbsEta_;
  const double maxAbsEta_;
  const double matchThreshold_;
  const double mergeThreshold_;
  const double energyResponseMin_;
  const double energyResponseMax_;
  const bool onlyGenPrimaries_;

  [[nodiscard]] bool isRootParticle(truth::Graph const& g, uint32_t i) const {
    auto const& pd = g.particles()[i];
    if (onlyGenPrimaries_ && !pd.hasGen())
      return false;
    return interestingPdgIds_.empty() ||
           std::find(interestingPdgIds_.begin(), interestingPdgIds_.end(), pd.pdgId) != interestingPdgIds_.end();
  }

  // Truth (branch) side, vs eta / pt / energy: denominator and the reconstruction
  // ladder numerators (Q1, Q2). Index 0=eta, 1=pt, 2=energy.
  std::array<MonitorElement*, 3> denom_{};
  std::array<MonitorElement*, 3> effNum_{};     // reconstructed (a candidate assigned)
  std::array<MonitorElement*, 3> chargeNum_{};  // + charged/neutral correct
  std::array<MonitorElement*, 3> pidNum_{};     // + PID class correct
  std::array<MonitorElement*, 3> energyNum_{};  // + energy response in window
  std::array<MonitorElement*, 3> dupNum_{};     // split into >=2 candidates
  // Reco (candidate) side, vs eta / pt / energy (Q4).
  std::array<MonitorElement*, 3> recoDenom_{};
  std::array<MonitorElement*, 3> fakeNum_{};
  std::array<MonitorElement*, 3> mergeNum_{};
  // Track<->calo linking consistency, vs eta / pt (Q3). No energy axis: it is a
  // charged-candidate question and pt is the natural variable.
  std::array<MonitorElement*, 2> trackCaloDenom_{};
  std::array<MonitorElement*, 2> trackCaloConsistentNum_{};
  // Distributions / profiles (Q5).
  MonitorElement* purityCalo_ = nullptr;
  MonitorElement* purityTrack_ = nullptr;
  MonitorElement* response_ = nullptr;
  MonitorElement* responseRaw_ = nullptr;
  std::array<MonitorElement*, 3> responseVs_{};
  MonitorElement* responseVsRaw_ = nullptr;
  MonitorElement* candOutcome_ = nullptr;
  // Fragmentation: number of candidates a single truth particle is split into (Q4).
  MonitorElement* nCandPerBranch_ = nullptr;

  static constexpr std::array<char const*, 3> kAxis{"eta", "pt", "energy"};
};

BranchTICLCandidateValidator::BranchTICLCandidateValidator(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      candidateToken_(consumes<std::vector<TICLCandidate>>(cfg.getParameter<edm::InputTag>("recoCollection"))),
      layerClusterToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))),
      interestingPdgIds_(cfg.getParameter<std::vector<int>>("interestingPdgIds")),
      folder_(cfg.getParameter<std::string>("folder")),
      xMax_(cfg.getParameter<double>("xMax")),
      energyMax_(cfg.getParameter<double>("energyMax")),
      minX_(cfg.getParameter<double>("minX")),
      minAbsEta_(cfg.getParameter<double>("minAbsEta")),
      maxAbsEta_(cfg.getParameter<double>("maxAbsEta")),
      matchThreshold_(cfg.getParameter<double>("matchThreshold")),
      mergeThreshold_(cfg.getParameter<double>("mergeThreshold")),
      energyResponseMin_(cfg.getParameter<double>("energyResponseMin")),
      energyResponseMax_(cfg.getParameter<double>("energyResponseMax")),
      onlyGenPrimaries_(cfg.getParameter<bool>("onlyGenPrimaries")) {}

void BranchTICLCandidateValidator::bookHistograms(DQMStore::IBooker& ib, edm::Run const&, edm::EventSetup const&) {
  ib.setCurrentFolder(folder_);
  constexpr int kEtaBins = 40;
  constexpr int kPtBins = 50;
  constexpr int kEnergyBins = 50;
  const double etaMax = maxAbsEta_ + 0.2;
  const std::array<int, 3> nbins{kEtaBins, kPtBins, kEnergyBins};
  const std::array<double, 3> lo{-etaMax, 0., 0.};
  // Separate maxima for the pt and energy axes: at 1.5<|eta|<3.1 a pt<=200 GeV
  // particle reaches E ~ pt*cosh(eta) up to ~2 TeV, so the energy axis needs a much
  // larger range than pt or everything piles into the overflow.
  const std::array<double, 3> hi{etaMax, xMax_, energyMax_};
  const std::array<char const*, 3> axisTitle{"#eta", "p_{T} [GeV]", "E [GeV]"};

  auto book = [&](std::array<MonitorElement*, 3>& mes, char const* stem, char const* what) {
    for (int a = 0; a < 3; ++a)
      mes[a] = ib.book1D(std::string(stem) + "_" + kAxis[a],
                         std::string(what) + " vs " + axisTitle[a] + ";" + axisTitle[a] + ";" + what,
                         nbins[a],
                         lo[a],
                         hi[a]);
  };

  book(denom_, "denom", "Selected branches");
  book(effNum_, "effnum", "Branches reconstructed");
  book(chargeNum_, "chargenum", "Branches reco + charge correct");
  book(pidNum_, "pidnum", "Branches reco + PID correct");
  book(energyNum_, "energynum", "Branches reco + energy correct");
  book(dupNum_, "dupnum", "Branches split (>=2 candidates)");
  book(recoDenom_, "recodenom", "Candidates");
  book(fakeNum_, "fakenum", "Fake candidates");
  book(mergeNum_, "mergenum", "Merged candidates (>=2 branches)");

  for (int a = 0; a < 2; ++a) {  // eta, pt only
    trackCaloDenom_[a] = ib.book1D(
        std::string("trackcalo_denom_") + kAxis[a],
        std::string("Charged candidates, track+calo matched vs ") + axisTitle[a] + ";" + axisTitle[a] + ";candidates",
        nbins[a],
        lo[a],
        hi[a]);
    trackCaloConsistentNum_[a] =
        ib.book1D(std::string("trackcalo_consistentnum_") + kAxis[a],
                  std::string("Track and calo same branch vs ") + axisTitle[a] + ";" + axisTitle[a] + ";candidates",
                  nbins[a],
                  lo[a],
                  hi[a]);
  }

  purityCalo_ = ib.book1D("purity_calo", "Best-branch calo purity;purity;candidates", 52, -0.01, 1.03);
  purityTrack_ = ib.book1D("purity_track", "Best-branch track purity;purity;candidates", 52, -0.01, 1.03);
  response_ = ib.book1D("energy_response", "Candidate energy response;E_{reco}/E_{truth};candidates", 120, 0., 3.);
  responseRaw_ =
      ib.book1D("energy_response_raw", "Candidate raw-energy response;E_{raw}/E_{truth};candidates", 120, 0., 3.);
  for (int a = 0; a < 3; ++a)
    responseVs_[a] =
        ib.bookProfile(std::string("energy_response_vs_") + kAxis[a],
                       std::string("Energy response vs ") + axisTitle[a] + ";" + axisTitle[a] + ";E_{reco}/E_{truth}",
                       nbins[a],
                       lo[a],
                       hi[a],
                       0.,
                       3.);
  responseVsRaw_ = ib.bookProfile("energy_response_raw_vs_energy",
                                  "Raw-energy response vs E;E [GeV];E_{raw}/E_{truth}",
                                  kEnergyBins,
                                  0.,
                                  energyMax_,
                                  0.,
                                  3.);
  nCandPerBranch_ = ib.book1D(
      "n_candidates_per_branch", "Candidates per truth particle (fragmentation);N candidates;particles", 40, 0.5, 40.5);

  candOutcome_ = ib.book1D("cand_outcome", "Candidate outcome;;candidates", 4, 0.5, 4.5);
  candOutcome_->setBinLabel(kMatched, "matched");
  candOutcome_->setBinLabel(kFake, "fake");
  candOutcome_->setBinLabel(kMerged, "merged");
  candOutcome_->setBinLabel(kTrackCaloMismatch, "track-calo mismatch");
}

void BranchTICLCandidateValidator::analyze(edm::Event const& event, edm::EventSetup const&) {
  auto const& graph = event.get(graphToken_);
  auto const& hitIndex = event.get(hitIndexToken_);
  auto const& candidates = event.get(candidateToken_);
  auto const& layerClusters = event.get(layerClusterToken_);

  // Branch roots: the interesting particles (a clean antichain for a gun when
  // onlyGenPrimaries). emptyRootsMeansAll=false so a restriction that selected
  // nothing this event matches nothing rather than falling back to the whole graph.
  const bool restricted = !interestingPdgIds_.empty() || onlyGenPrimaries_;
  std::vector<uint32_t> roots;
  if (restricted)
    for (uint32_t i = 0; i < graph.nParticles(); ++i)
      if (isRootParticle(graph, i))
        roots.push_back(i);
  truth::BranchHitAssociator caloAssoc(
      hitIndex, roots, truth::BranchHitAssociator::Metric::SharedEnergy, truth::HitChannel::HGCalCalo, !restricted);
  truth::BranchHitAssociator trackAssoc(
      hitIndex, roots, truth::BranchHitAssociator::Metric::SharedHits, truth::HitChannel::Tracker, !restricted);

  // Per-branch reconstruction outcome, evaluated on the REPRESENTATIVE candidate: the
  // one that covers the most of the branch (max shared energy), i.e. the main particle
  // candidate - NOT the highest-purity one, which for a fragmented shower is a tiny
  // pure splinter and would misreport the charge/PID/energy. nCandidates counts every
  // candidate matched to the branch (the fragmentation multiplicity).
  struct BranchReco {
    int nCandidates = 0;
    double repShared = -1.;    // shared energy of the representative candidate
    double repEnergy = 0.;     // its regressed energy (for the energy response)
    double repRawEnergy = 0.;  // its raw (trackster) energy
    bool chargeOk = false;
    bool pidOk = false;
    bool energyOk = false;
  };
  std::unordered_map<uint32_t, BranchReco> branchReco;

  // --- Reco -> sim: one pass over candidates (Q3, Q4, Q5, and fill branchReco). ---
  for (auto const& cand : candidates) {
    const double eta = cand.eta();
    const double pt = cand.pt();
    const double energy = cand.energy();
    const std::array<double, 3> xval{eta, pt, energy};

    // Calo side: union of the candidate's tracksters' cells.
    auto caloHits = truth::recoHits(cand, layerClusters);
    // Track side: the valid rechits of the candidate's track(s).
    std::vector<truth::RecoHit> trackHits;
    for (auto const& trkPtr : cand.trackPtrs()) {
      if (trkPtr.isNull())
        continue;
      auto th = truth::recoHits(*trkPtr);
      trackHits.insert(trackHits.end(), th.begin(), th.end());
    }
    const bool hasTrack = !trackHits.empty();

    if (caloHits.empty() && trackHits.empty())
      continue;

    // Pick the best branch by the SAME quantity we report as purity: the largest
    // shared energy (calo) / shared-hit count (track). bestBranches() sorts by score
    // (squared per-cell excess), which need not agree with linear shared energy in an
    // overlap region, so front() is not necessarily the highest-purity branch; scan
    // for the max instead so "best branch", its purity, and the identity are consistent.
    auto maxSharedBranch = [](std::vector<truth::BranchMatch> const& ms) -> truth::BranchMatch const* {
      truth::BranchMatch const* best = nullptr;
      for (auto const& m : ms)
        if (best == nullptr || m.sharedEnergy > best->sharedEnergy)
          best = &m;
      return best;
    };

    // Calo best branch (shared-energy purity = shared / candidate cell-fraction sum).
    double caloWeight = 0.;
    for (auto const& h : caloHits)
      caloWeight += static_cast<double>(h.fraction) * h.energy;
    if (caloWeight <= 0.)
      caloWeight = 1.;
    auto caloMatches = caloAssoc.bestBranches(std::span<const truth::RecoHit>(caloHits));
    truth::BranchMatch const* bestCalo = maxSharedBranch(caloMatches);
    const double caloPurity = bestCalo ? bestCalo->sharedEnergy / caloWeight : 0.;
    const bool caloMatched = bestCalo != nullptr && caloPurity >= matchThreshold_;

    // Track best branch (shared-hit purity = shared hits / candidate track hits).
    const double trackWeight = hasTrack ? static_cast<double>(trackHits.size()) : 1.;
    auto trackMatches = hasTrack ? trackAssoc.bestBranches(std::span<const truth::RecoHit>(trackHits))
                                 : std::vector<truth::BranchMatch>{};
    truth::BranchMatch const* bestTrack = maxSharedBranch(trackMatches);
    const double trackPurity = bestTrack ? bestTrack->sharedEnergy / trackWeight : 0.;
    const bool trackMatched = hasTrack && bestTrack != nullptr && trackPurity >= matchThreshold_;

    if (!caloHits.empty())
      purityCalo_->Fill(caloPurity);
    if (hasTrack)
      purityTrack_->Fill(trackPurity);

    // The candidate's identity branch: its track when it has a matched one (a charged
    // candidate is defined by its track), otherwise its calo match.
    const bool matched = trackMatched || caloMatched;
    const uint32_t identityBranch =
        trackMatched ? bestTrack->rootParticleId : (caloMatched ? bestCalo->rootParticleId : 0u);

    // Merge: >=2 calo branches share this candidate above mergeThreshold.
    int sharedCaloBranches = 0;
    for (auto const& m : caloMatches)
      if (m.sharedEnergy / caloWeight >= mergeThreshold_)
        ++sharedCaloBranches;

    // Track<->calo consistency (Q3): only meaningful when both halves matched.
    const bool bothMatched = trackMatched && caloMatched;
    const bool trackCaloConsistent = bothMatched && bestTrack->rootParticleId == bestCalo->rootParticleId;

    // Reco-side categorical outcome + rates, on candidates passing the selection.
    if (selected(eta, pt)) {
      for (int a = 0; a < 3; ++a)
        recoDenom_[a]->Fill(xval[a]);
      if (!matched) {
        candOutcome_->Fill(kFake);
        for (int a = 0; a < 3; ++a)
          fakeNum_[a]->Fill(xval[a]);
      } else if (bothMatched && !trackCaloConsistent) {
        candOutcome_->Fill(kTrackCaloMismatch);
      } else if (sharedCaloBranches >= 2) {
        candOutcome_->Fill(kMerged);
      } else {
        candOutcome_->Fill(kMatched);
      }
      // mergerate is an independent rate over ALL candidates sharing >=2 branches; the
      // categorical cand_outcome above instead assigns a single label with track-calo
      // mismatch taking precedence over merge, so a candidate can be both a mismatch in
      // cand_outcome and counted here - the two answer different questions.
      if (sharedCaloBranches >= 2)
        for (int a = 0; a < 3; ++a)
          mergeNum_[a]->Fill(xval[a]);
      if (bothMatched) {
        trackCaloDenom_[0]->Fill(eta);
        trackCaloDenom_[1]->Fill(pt);
        if (trackCaloConsistent) {
          trackCaloConsistentNum_[0]->Fill(eta);
          trackCaloConsistentNum_[1]->Fill(pt);
        }
      }
    }

    if (!matched)
      continue;
    // Defensive: identityBranch indexes graph.particles(). In the shipped (restricted)
    // configuration it is always a valid graph particle id; guard the fully-unrestricted
    // case where the associator can key on hitIndex ids if the two producers disagree.
    if (identityBranch >= graph.nParticles())
      continue;

    // branchReco ladder bookkeeping (Q1, Q2, Q5). The energy response is measured
    // per-branch on the representative candidate (filled in the sim->reco loop), NOT
    // per-candidate here: a fragmented shower yields many tiny high-purity candidates
    // whose per-candidate response would swamp the histogram near zero.
    auto const& branch = graph.particles()[identityBranch];
    const double truthEnergy = branch.momentum.energy();

    auto& br = branchReco[identityBranch];
    ++br.nCandidates;
    // Representative = the candidate covering the most of THIS branch (max shared
    // energy / shared hits), i.e. the main particle candidate.
    const double coverage = caloMatched ? bestCalo->sharedEnergy : (bestTrack ? bestTrack->sharedEnergy : 0.);
    if (coverage > br.repShared) {
      br.repShared = coverage;
      br.repEnergy = energy;
      br.repRawEnergy = cand.rawEnergy();
      const bool truthCharged = isChargedClass(pidClass(branch.pdgId));
      br.chargeOk = (truthCharged == (cand.charge() != 0));
      br.pidOk = (pidClass(cand.pdgId()) == pidClass(branch.pdgId));
      br.energyOk = truthEnergy > 0. && (energy / truthEnergy) >= energyResponseMin_ &&
                    (energy / truthEnergy) <= energyResponseMax_;
    }
  }

  // --- Sim -> reco: efficiency ladder + split, over the selected branches. ---
  const uint32_t nP = graph.nParticles();
  for (uint32_t r = 0; r < nP; ++r) {
    if (!isRootParticle(graph, r))
      continue;
    // The branch must have a footprint in at least one channel to be reconstructible.
    const bool hasCalo = !hitIndex.subgraphHits(truth::HitChannel::HGCalCalo, r).empty();
    const bool hasTracker = !hitIndex.subgraphHits(truth::HitChannel::Tracker, r).empty();
    if (!hasCalo && !hasTracker)
      continue;
    auto const& p = graph.particles()[r].momentum;
    const double eta = p.eta();
    const double pt = p.pt();
    const double energy = p.energy();
    if (!selected(eta, pt))
      continue;
    const std::array<double, 3> xval{eta, pt, energy};
    for (int a = 0; a < 3; ++a)
      denom_[a]->Fill(xval[a]);

    auto it = branchReco.find(r);
    if (it == branchReco.end())
      continue;
    auto const& br = it->second;
    for (int a = 0; a < 3; ++a)
      effNum_[a]->Fill(xval[a]);
    if (br.chargeOk)
      for (int a = 0; a < 3; ++a)
        chargeNum_[a]->Fill(xval[a]);
    if (br.chargeOk && br.pidOk)
      for (int a = 0; a < 3; ++a)
        pidNum_[a]->Fill(xval[a]);
    if (br.chargeOk && br.pidOk && br.energyOk)
      for (int a = 0; a < 3; ++a)
        energyNum_[a]->Fill(xval[a]);
    if (br.nCandidates >= 2)
      for (int a = 0; a < 3; ++a)
        dupNum_[a]->Fill(xval[a]);

    // Fragmentation multiplicity: how many candidates this one particle was split into.
    nCandPerBranch_->Fill(std::min(br.nCandidates, 39));

    // Energy response (Q5), measured once per particle on the representative candidate.
    if (energy > 0.) {
      const double response = br.repEnergy / energy;
      response_->Fill(response);
      if (br.repRawEnergy > 0.) {
        responseRaw_->Fill(br.repRawEnergy / energy);
        responseVsRaw_->Fill(energy, br.repRawEnergy / energy);
      }
      responseVs_[0]->Fill(eta, response);
      responseVs_[1]->Fill(pt, response);
      responseVs_[2]->Fill(energy, response);
    }
  }
}

void BranchTICLCandidateValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<edm::InputTag>("recoCollection", edm::InputTag("ticlCandidate"));
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<std::vector<int>>("interestingPdgIds", {})
      ->setComment("Restrict the branch side to these PDG ids (empty = all branches).");
  desc.add<std::string>("folder", "HGCAL/BranchValidator/TICLCandidate");
  desc.add<double>("xMax", 200.)->setComment("Upper edge of the p_T axis.");
  desc.add<double>("energyMax", 2500.)->setComment("Upper edge of the energy axis (E >> p_T at high |eta|).");
  desc.add<double>("minX", 0.);
  desc.add<double>("minAbsEta", 1.5);
  desc.add<double>("maxAbsEta", 3.0);
  desc.add<double>("matchThreshold", 0.5)->setComment("Min best-branch purity for a candidate to count as matched.");
  desc.add<double>("mergeThreshold", 0.3)
      ->setComment("Min shared calo fraction for a branch to count toward a merge (>=2 -> merged candidate).");
  desc.add<double>("energyResponseMin", 0.7)->setComment("Lower edge of the 'energy correct' response window.");
  desc.add<double>("energyResponseMax", 1.3)->setComment("Upper edge of the 'energy correct' response window.");
  desc.add<bool>("onlyGenPrimaries", false)
      ->setComment("Restrict the branch side to GEN primaries (a clean antichain for single-particle guns).");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(BranchTICLCandidateValidator);
