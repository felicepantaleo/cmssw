#include <numeric>
#include <iomanip>
#include <sstream>

#include "Validation/HGCalValidation/interface/TICLCandidateValidator.h"
#include "RecoHGCal/TICL/interface/commons.h"

TICLCandidateValidator::TICLCandidateValidator(
    edm::EDGetTokenT<std::vector<TICLCandidate>> ticlCandidates,
    edm::EDGetTokenT<std::vector<TICLCandidate>> simTICLCandidatesToken,
    edm::EDGetTokenT<std::vector<reco::Track>> recoTracksToken,
    edm::EDGetTokenT<std::vector<ticl::Trackster>> trackstersToken,
    edm::EDGetTokenT<hgcal::RecoToSimCollectionSimTracksters> associatorMapRtSToken,
    edm::EDGetTokenT<hgcal::SimToRecoCollectionSimTracksters> associatorMapStRToken)
    : TICLCandidatesToken_(ticlCandidates),
      simTICLCandidatesToken_(simTICLCandidatesToken),
      recoTracksToken_(recoTracksToken),
      trackstersToken_(trackstersToken),
      associatorMapRtSToken_(associatorMapRtSToken),
      associatorMapStRToken_(associatorMapStRToken) {}

TICLCandidateValidator::~TICLCandidateValidator() {}

// TODO split this in different folders
void TICLCandidateValidator::bookCandidatesHistos(DQMStore::IBooker& ibook, std::string baseDir) {
  // book CAND histos
  h_tracksters_in_candidate = ibook.book1D("N of tracksters in candidate", "N of tracksters in candidate", 100, 0, 99);
  h_candidate_raw_energy = ibook.book1D("Candidates raw energy", "Candidates raw energy;E (GeV)", 250, 0, 250);
  h_candidate_regressed_energy =
      ibook.book1D("Candidates regressed energy", "Candidates regressed energy;E (GeV)", 250, 0, 250);
  h_candidate_pT = ibook.book1D("Candidates pT", "Candidates pT;p_{T}", 250, 0, 250);
  h_candidate_charge = ibook.book1D("Candidates charge", "Candidates charge;Charge", 3, -1.5, 1.5);
  h_candidate_pdgId = ibook.book1D("Candidates PDG Id", "Candidates PDG ID", 100, -220, 220);

  // neutral: photon, pion, hadron
  const std::vector<std::string> neutrals{"photons", "neutral_pions", "neutral_hadrons"};
  for (long unsigned int i = 0; i < neutrals.size(); i++) {
    ibook.setCurrentFolder(baseDir + "/" + neutrals[i]);
    h_den_neut_energy_candidate.push_back(
        ibook.book1D("den_cand_vs_energy_" + neutrals[i], neutrals[i] + " vs energy;E (GeV)", 250, 0, 250));
    h_num_neut_energy_candidate_pdgId.push_back( ibook.book1D("num_pid_cand_vs_energy_" + neutrals[i],
                                                        neutrals[i] + " track and PID efficiency vs energy;E (GeV)",
                                                        250,
                                                        0,
                                                        250));
    h_num_neut_energy_candidate_energy.push_back(
        ibook.book1D("num_energy_cand_vs_energy_" + neutrals[i],
                     neutrals[i] + " track, PID and energy efficiency vs energy;E (GeV)",
                     250,
                     0,
                     250));
    h_den_neut_pt_candidate.push_back(
        ibook.book1D("den_cand_vs_pt_" + neutrals[i], neutrals[i] + " vs pT;p_{T} (GeV)", 250, 0, 250));
    h_num_neut_pt_candidate_pdgId.push_back( ibook.book1D(
        "num_pid_cand_vs_pt_" + neutrals[i], neutrals[i] + " track and PID efficiency vs pT;p_{T} (GeV)", 250, 0, 250));
    h_num_neut_pt_candidate_energy.push_back(
        ibook.book1D("num_energy_cand_vs_pt_" + neutrals[i],
                     neutrals[i] + " track, PID and energy efficiency vs pT;p_{T} (GeV)",
                     250,
                     0,
                     250));
    h_den_neut_eta_candidate.push_back(
        ibook.book1D("den_cand_vs_eta_" + neutrals[i], neutrals[i] + " vs eta;#eta (GeV)", 100, -3, 3));
    h_num_neut_eta_candidate_pdgId.push_back( ibook.book1D(
        "num_pid_cand_vs_eta_" + neutrals[i], neutrals[i] + " track and PID efficiency vs eta;#eta (GeV)", 100, -3, 3));
    h_num_neut_eta_candidate_energy.push_back(
        ibook.book1D("num_energy_cand_vs_eta_" + neutrals[i],
                     neutrals[i] + " track, PID and energy efficiency vs eta;#eta (GeV)",
                     100,
                     -3,
                     3));
    h_den_neut_phi_candidate.push_back(
        ibook.book1D("den_cand_vs_phi_" + neutrals[i], neutrals[i] + " vs phi;#phi (GeV)", 100, -3.14159, 3.14159));
    h_num_neut_phi_candidate_pdgId.push_back( ibook.book1D("num_pid_cand_vs_phi_" + neutrals[i],
                                                     neutrals[i] + " track and PID efficiency vs phi;#phi (GeV)",
                                                     100,
                                                     -3.14159,
                                                     3.14159));
    h_num_neut_phi_candidate_energy.push_back(
        ibook.book1D("num_energy_cand_vs_phi_" + neutrals[i],
                     neutrals[i] + " track, PID and energy efficiency vs phi;#phi (GeV)",
                     100,
                     -3.14159,
                     3.14159));
  }
  // charged: electron, pion, hadron
  const std::vector<std::string> charged{"electrons", "charged_pions", "charged_hadrons"};
  for (long unsigned int i = 0; i < charged.size(); i++) {
    ibook.setCurrentFolder(baseDir + "/" + charged[i]);
    h_den_chg_energy_candidate.push_back(
        ibook.book1D("den_cand_vs_energy_" + charged[i], charged[i] + " vs energy;E (GeV)", 250, 0, 250));
    h_num_chg_energy_candidate_track.push_back( ibook.book1D(
        "num_track_cand_vs_energy_" + charged[i], charged[i] + " track efficiency vs energy;E (GeV)", 250, 0, 250));
    h_num_chg_energy_candidate_pdgId.push_back( ibook.book1D(
        "num_pid_cand_vs_energy_" + charged[i], charged[i] + " track and PID efficiency vs energy;E (GeV)", 250, 0, 250));
    h_num_chg_energy_candidate_energy.push_back(
        ibook.book1D("num_energy_cand_vs_energy_" + charged[i],
                     charged[i] + " track, PID and energy efficiency vs energy;E (GeV)",
                     250,
                     0,
                     250));
    h_den_chg_pt_candidate.push_back(
        ibook.book1D("den_cand_vs_pt_" + charged[i], charged[i] + " vs pT;p_{T} (GeV)", 250, 0, 250));
    h_num_chg_pt_candidate_track.push_back( ibook.book1D(
        "num_track_cand_vs_pt_" + charged[i], charged[i] + " track efficiency vs pT;p_{T} (GeV)", 250, 0, 250));
    h_num_chg_pt_candidate_pdgId.push_back( ibook.book1D(
        "num_pid_cand_vs_pt_" + charged[i], charged[i] + " track and PID efficiency vs pT;p_{T} (GeV)", 250, 0, 250));
    h_num_chg_pt_candidate_energy.push_back( ibook.book1D("num_energy_cand_vs_pt_" + charged[i],
                                                    charged[i] + " track, PID and energy efficiency vs pT;p_{T} (GeV)",
                                                    250,
                                                    0,
                                                    250));
    h_den_chg_eta_candidate.push_back(
        ibook.book1D("den_cand_vs_eta_" + charged[i], charged[i] + " vs eta;#eta (GeV)", 100, -3, 3));
    h_num_chg_eta_candidate_track.push_back( ibook.book1D(
        "num_track_cand_vs_eta_" + charged[i], charged[i] + " track efficiency vs eta;#eta (GeV)", 100, -3, 3));
    h_num_chg_eta_candidate_pdgId.push_back( ibook.book1D(
        "num_pid_cand_vs_eta_" + charged[i], charged[i] + " track and PID efficiency vs eta;#eta (GeV)", 100, -3, 3));
    h_num_chg_eta_candidate_energy.push_back( ibook.book1D("num_energy_cand_vs_eta_" + charged[i],
                                                     charged[i] + " track, PID and energy efficiency vs eta;#eta (GeV)",
                                                     100,
                                                     -3,
                                                     3));
    h_den_chg_phi_candidate.push_back(
        ibook.book1D("den_cand_vs_phi_" + charged[i], charged[i] + " vs phi;#phi (GeV)", 100, -3.14159, 3.14159));
    h_num_chg_phi_candidate_track.push_back( ibook.book1D("num_track_cand_vs_phi_" + charged[i],
                                                    charged[i] + " track efficiency vs phi;#phi (GeV)",
                                                    100,
                                                    -3.14159,
                                                    3.14159));
    h_num_chg_phi_candidate_pdgId.push_back( ibook.book1D("num_pid_cand_vs_phi_" + charged[i],
                                                    charged[i] + " track and PID efficiency vs phi;#phi (GeV)",
                                                    100,
                                                    -3.14159,
                                                    3.14159));
    h_num_chg_phi_candidate_energy.push_back( ibook.book1D("num_energy_cand_vs_phi_" + charged[i],
                                                     charged[i] + " track, PID and energy efficiency vs phi;#phi (GeV)",
                                                     100,
                                                     -3.14159,
                                                     3.14159));
  }
}

void TICLCandidateValidator::fillCandidateHistos(const edm::Event& event,
                                                 edm::Handle<ticl::TracksterCollection> simTrackstersCP_h) {
  edm::Handle<std::vector<TICLCandidate>> TICLCandidates_h;
  event.getByToken(TICLCandidatesToken_, TICLCandidates_h);
  auto TICLCandidates = *TICLCandidates_h;

  edm::Handle<std::vector<TICLCandidate>> simTICLCandidates_h;
  event.getByToken(simTICLCandidatesToken_, simTICLCandidates_h);
  auto simTICLCandidates = *simTICLCandidates_h;

  edm::Handle<std::vector<reco::Track>> recoTracks_h;
  event.getByToken(recoTracksToken_, recoTracks_h);
  auto recoTracks = *recoTracks_h;

  edm::Handle<std::vector<ticl::Trackster>> Tracksters_h;
  event.getByToken(trackstersToken_, Tracksters_h);
  auto trackstersMerged = *Tracksters_h;

  edm::Handle<hgcal::RecoToSimCollectionSimTracksters> mergeTsRecoToSim_h;
  event.getByToken(associatorMapRtSToken_, mergeTsRecoToSim_h);
  auto const& mergeTsRecoToSimMap = *mergeTsRecoToSim_h;

  edm::Handle<hgcal::SimToRecoCollectionSimTracksters> mergeTsSimToReco_h;
  event.getByToken(associatorMapStRToken_, mergeTsSimToReco_h);
  auto const& mergeTsSimToRecoMap = *mergeTsSimToReco_h;

  // candidates plots
  for (const auto& cand : TICLCandidates) {
    h_tracksters_in_candidate->Fill(cand.tracksters().size());
    h_candidate_raw_energy->Fill(cand.rawEnergy());
    h_candidate_regressed_energy->Fill(cand.energy());
    h_candidate_pT->Fill(cand.pt());
    h_candidate_charge->Fill(cand.charge());
    h_candidate_pdgId->Fill(cand.pdgId());
  }

  std::cout << "-------EVENT-------\n";
  std::cout << "simTICLCandidates: " << simTICLCandidates.size() << "\n";
  std::cout << "TICLCandidates: " << TICLCandidates.size() << "\n";
  std::cout << "tracks: " << recoTracks.size() << "\n";

  std::vector<int> chargedCandidates;
  std::vector<int> neutralCandidates;
  chargedCandidates.reserve(simTICLCandidates.size());
  neutralCandidates.reserve(simTICLCandidates.size());

  for (size_t i = 0; i < simTICLCandidates.size(); ++i) {
    const auto& simCand = simTICLCandidates[i];
    const auto particleType = ticl::tracksterParticleTypeFromPdgId(simCand.pdgId(), 1);
    if (particleType == ticl::Trackster::ParticleType::electron or particleType == ticl::Trackster::ParticleType::muon or
        particleType == ticl::Trackster::ParticleType::charged_hadron)
      chargedCandidates.emplace_back(i);
    else if (particleType == ticl::Trackster::ParticleType::photon or particleType == ticl::Trackster::ParticleType::neutral_pion or
             particleType == ticl::Trackster::ParticleType::neutral_hadron)
      neutralCandidates.emplace_back(i);
    // should consider also unknown ?
  }

  chargedCandidates.shrink_to_fit();
  neutralCandidates.shrink_to_fit();

  for (const auto i : chargedCandidates) {
    const auto& simCand = simTICLCandidates[i];
    auto index = std::log2(int(ticl::tracksterParticleTypeFromPdgId(simCand.pdgId(), 1)));
    /* 11 (type 1) becomes 0
     * 13 (type 2) becomes 1
     * 211 (type 4) becomes 2
     */
    std::cout << " --- simCand CHARGED, type: " << index << " --- \n";
    int32_t simCandTrackIdx = -1;
    if (simCand.trackPtr().get() != nullptr)
      simCandTrackIdx = simCand.trackPtr().get() - edm::Ptr<reco::Track>(recoTracks_h, 0).get();
    else { 
      std::cout << "no reco track, but simCand is charged --> SKIP\n";
      continue;
    }
    std::cout << "track Idx = " << simCandTrackIdx << "\n";
    if (simCand.trackPtr().get()->pt() < 1 or simCand.trackPtr().get()->missingOuterHits() > 5 or
        not simCand.trackPtr().get()->quality(reco::TrackBase::highPurity)) {
      std::cout << "track does not pass cuts: pt " << simCand.trackPtr().get()->pt() << " > 1, moh "
                << simCand.trackPtr().get()->missingOuterHits() << " < 5 and quality "
                << simCand.trackPtr().get()->quality(reco::TrackBase::highPurity) << " --> SKIP\n";
      continue;
    }
    // +1 to all denominators
    h_den_chg_energy_candidate[index]->Fill(simCand.rawEnergy());
    h_den_chg_pt_candidate[index]->Fill(simCand.pt());
    h_den_chg_eta_candidate[index]->Fill(simCand.eta());
    h_den_chg_phi_candidate[index]->Fill(simCand.phi());

    int32_t cand_idx = -1;
    const edm::Ref<ticl::TracksterCollection> stsRef(simTrackstersCP_h, i);
    const auto ts_iter = mergeTsSimToRecoMap.find(stsRef);
    float shared_energy = 0.;
    // search for reco cand associated
    if (ts_iter != mergeTsSimToRecoMap.end()) {
      const auto& tsAssoc = (ts_iter->val);
      std::vector<uint32_t> MergeTracksters_simToReco;
      std::vector<float> MergeTracksters_simToReco_score;
      std::vector<float> MergeTracksters_simToReco_sharedE;
      MergeTracksters_simToReco.reserve(tsAssoc.size());
      MergeTracksters_simToReco_score.reserve(tsAssoc.size());
      MergeTracksters_simToReco_sharedE.reserve(tsAssoc.size());
      for (auto& ts : tsAssoc) {
        auto ts_id = (ts.first).get() - (edm::Ref<ticl::TracksterCollection>(Tracksters_h, 0)).get();
        MergeTracksters_simToReco.push_back(ts_id);
        MergeTracksters_simToReco_score.push_back(ts.second.second);
        MergeTracksters_simToReco_sharedE.push_back(ts.second.first);
      }
      auto min_idx = std::min_element(MergeTracksters_simToReco_score.begin(), MergeTracksters_simToReco_score.end());
      if (*min_idx != 1) {
        cand_idx = MergeTracksters_simToReco[min_idx - MergeTracksters_simToReco_score.begin()];
        shared_energy = MergeTracksters_simToReco_sharedE[min_idx - MergeTracksters_simToReco_score.begin()];
      }
    }

    std::cout << "reco cand assoc idx = " << cand_idx << "\n";
    // no reco associated to sim
    if (cand_idx == -1)
      continue;

    const auto& recoCand = TICLCandidates[cand_idx];
    if (recoCand.trackPtr().get() != nullptr) {
      const auto candTrackIdx = recoCand.trackPtr().get() - edm::Ptr<reco::Track>(recoTracks_h, 0).get();
      std::cout << "reco cand has track associated and ";
      if (simCandTrackIdx == candTrackIdx) {
        // +1 to track num
        std::cout << "is correct\n";
        h_num_chg_energy_candidate_track[index]->Fill(simCand.rawEnergy());
        h_num_chg_pt_candidate_track[index]->Fill(simCand.pt());
        h_num_chg_eta_candidate_track[index]->Fill(simCand.eta());
        h_num_chg_phi_candidate_track[index]->Fill(simCand.phi());
      } else {
        std::cout << "is wrong\n";
        continue;
      }
    } else {
      std::cout << "reco cand has NO track associated.\n";
      continue;
    }

    //step 2: PID
    std::cout << "PID: sim = " << simCand.pdgId() << " and reco = " << recoCand.pdgId() << "\n";
    if (simCand.pdgId() == recoCand.pdgId()) {
      // +1 to num pdg id
      h_num_chg_energy_candidate_pdgId[index]->Fill(simCand.rawEnergy());
      h_num_chg_pt_candidate_pdgId[index]->Fill(simCand.pt());
      h_num_chg_eta_candidate_pdgId[index]->Fill(simCand.eta());
      h_num_chg_phi_candidate_pdgId[index]->Fill(simCand.phi());

      //step 3: energy
      std::cout << "shared energy is " << shared_energy << ", raw energy " << simCand.rawEnergy()
                << ", passes threshold: " << (shared_energy / simCand.rawEnergy() > 0.5) << std::endl;
      if (shared_energy / simCand.rawEnergy() > 0.5) {
        // +1 to ene num
        h_num_chg_energy_candidate_energy[index]->Fill(simCand.rawEnergy());
        h_num_chg_pt_candidate_energy[index]->Fill(simCand.pt());
        h_num_chg_eta_candidate_energy[index]->Fill(simCand.eta());
        h_num_chg_phi_candidate_energy[index]->Fill(simCand.phi());
      }
    }
  }

  for (const auto i : neutralCandidates) {
    const auto& simCand = simTICLCandidates[i];
    auto index = int(ticl::tracksterParticleTypeFromPdgId(simCand.pdgId(), 1)) / 2;
    /* 22 (type 0) becomes 0
     * 111 (type 3) becomes 1
     * 130 (type 5) becomes 2
     */
    std::cout << " --- simCand NEUTRAL, type: " << index << " --- \n";
    if (simCand.trackPtr().get() != nullptr)
      std::cout << "ERROR: NEUTRAL WITH TRACK\n";

    h_den_neut_energy_candidate[index]->Fill(simCand.rawEnergy());
    h_den_neut_pt_candidate[index]->Fill(simCand.pt());
    h_den_neut_eta_candidate[index]->Fill(simCand.eta());
    h_den_neut_phi_candidate[index]->Fill(simCand.phi());

    int32_t cand_idx = -1;
    const edm::Ref<ticl::TracksterCollection> stsRef(simTrackstersCP_h, i);
    const auto ts_iter = mergeTsSimToRecoMap.find(stsRef);
    float shared_energy = 0.;
    // search for reco cand associated
    if (ts_iter != mergeTsSimToRecoMap.end()) {
      const auto& tsAssoc = (ts_iter->val);
      std::vector<uint32_t> MergeTracksters_simToReco;
      std::vector<float> MergeTracksters_simToReco_score;
      std::vector<float> MergeTracksters_simToReco_sharedE;
      MergeTracksters_simToReco.reserve(tsAssoc.size());
      MergeTracksters_simToReco_score.reserve(tsAssoc.size());
      MergeTracksters_simToReco_sharedE.reserve(tsAssoc.size());
      for (auto& ts : tsAssoc) {
        auto ts_id = (ts.first).get() - (edm::Ref<ticl::TracksterCollection>(Tracksters_h, 0)).get();
        MergeTracksters_simToReco.push_back(ts_id);
        MergeTracksters_simToReco_score.push_back(ts.second.second);
        MergeTracksters_simToReco_sharedE.push_back(ts.second.first);
      }
      auto min_idx = std::min_element(MergeTracksters_simToReco_score.begin(), MergeTracksters_simToReco_score.end());
      if (*min_idx != 1) {
        cand_idx = MergeTracksters_simToReco[min_idx - MergeTracksters_simToReco_score.begin()];
        shared_energy = MergeTracksters_simToReco_sharedE[min_idx - MergeTracksters_simToReco_score.begin()];
      }
    }

    std::cout << "reco cand assoc idx = " << cand_idx << "\n";
    // no reco associated to sim
    if (cand_idx == -1)
      continue;

    const auto& recoCand = TICLCandidates[cand_idx];
    if (recoCand.trackPtr().get() != nullptr) {
      std::cout << "ERROR: reco cand has track associated but sim was neutral\n";
      continue;
    }

    //step 2: PID
    std::cout << "PID: sim = " << simCand.pdgId() << " and reco = " << recoCand.pdgId() << "\n";
    if (simCand.pdgId() == recoCand.pdgId()) {
      // +1 to num pdg id
      h_num_neut_energy_candidate_pdgId[index]->Fill(simCand.rawEnergy());
      h_num_neut_pt_candidate_pdgId[index]->Fill(simCand.pt());
      h_num_neut_eta_candidate_pdgId[index]->Fill(simCand.eta());
      h_num_neut_phi_candidate_pdgId[index]->Fill(simCand.phi());

      //step 3: energy
      std::cout << "shared energy is " << shared_energy << ", raw energy " << simCand.rawEnergy()
                << ", passes threshold: " << (shared_energy / simCand.rawEnergy() > 0.5) << std::endl;
      if (shared_energy / simCand.rawEnergy() > 0.5) {
        // +1 to ene num
        h_num_neut_energy_candidate_energy[index]->Fill(simCand.rawEnergy());
        h_num_neut_pt_candidate_energy[index]->Fill(simCand.pt());
        h_num_neut_eta_candidate_energy[index]->Fill(simCand.eta());
        h_num_neut_phi_candidate_energy[index]->Fill(simCand.phi());
      }
    }
  }
}
