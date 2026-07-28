// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "Validation/TruthInfo/interface/TruthBranchHistoProducerAlgo.h"

namespace truth {

  TruthBranchHistoProducerAlgo::TruthBranchHistoProducerAlgo(edm::ParameterSet const& pset)
      : nintPt_(pset.getParameter<int>("nintPt")),
        nintEta_(pset.getParameter<int>("nintEta")),
        nintPhi_(pset.getParameter<int>("nintPhi")),
        nintScore_(pset.getParameter<int>("nintScore")),
        nintShared_(pset.getParameter<int>("nintShared")),
        minPt_(pset.getParameter<double>("minPt")),
        maxPt_(pset.getParameter<double>("maxPt")),
        minEta_(pset.getParameter<double>("minEta")),
        maxEta_(pset.getParameter<double>("maxEta")),
        minPhi_(pset.getParameter<double>("minPhi")),
        maxPhi_(pset.getParameter<double>("maxPhi")),
        minScore_(pset.getParameter<double>("minScore")),
        maxScore_(pset.getParameter<double>("maxScore")),
        minShared_(pset.getParameter<double>("minShared")),
        maxShared_(pset.getParameter<double>("maxShared")) {}

  void TruthBranchHistoProducerAlgo::bookHistos(dqm::implementation::IBooker& booker,
                                                TruthBranchHistograms& h) const {
    // The names are the harvesting API: DQMGenericClient forms every ratio from these
    // by string, so renaming one silently drops a plot rather than failing.
    h.h_simul_pt.push_back(booker.book1D("num_simul_pt", "Selected branches vs p_{T}", nintPt_, minPt_, maxPt_));
    h.h_simul_eta.push_back(booker.book1D("num_simul_eta", "Selected branches vs #eta", nintEta_, minEta_, maxEta_));
    h.h_simul_phi.push_back(booker.book1D("num_simul_phi", "Selected branches vs #phi", nintPhi_, minPhi_, maxPhi_));

    h.h_assoc_simToReco_pt.push_back(
        booker.book1D("num_assoc(simToReco)_pt", "Associated branches vs p_{T}", nintPt_, minPt_, maxPt_));
    h.h_assoc_simToReco_eta.push_back(
        booker.book1D("num_assoc(simToReco)_eta", "Associated branches vs #eta", nintEta_, minEta_, maxEta_));
    h.h_assoc_simToReco_phi.push_back(
        booker.book1D("num_assoc(simToReco)_phi", "Associated branches vs #phi", nintPhi_, minPhi_, maxPhi_));

    h.h_reco_pt.push_back(booker.book1D("num_reco_pt", "Reco objects vs p_{T}", nintPt_, minPt_, maxPt_));
    h.h_reco_eta.push_back(booker.book1D("num_reco_eta", "Reco objects vs #eta", nintEta_, minEta_, maxEta_));
    h.h_reco_phi.push_back(booker.book1D("num_reco_phi", "Reco objects vs #phi", nintPhi_, minPhi_, maxPhi_));

    h.h_assoc_recoToSim_pt.push_back(
        booker.book1D("num_assoc(recoToSim)_pt", "Matched reco objects vs p_{T}", nintPt_, minPt_, maxPt_));
    h.h_assoc_recoToSim_eta.push_back(
        booker.book1D("num_assoc(recoToSim)_eta", "Matched reco objects vs #eta", nintEta_, minEta_, maxEta_));
    h.h_assoc_recoToSim_phi.push_back(
        booker.book1D("num_assoc(recoToSim)_phi", "Matched reco objects vs #phi", nintPhi_, minPhi_, maxPhi_));

    h.h_duplicate_pt.push_back(
        booker.book1D("num_duplicate_pt", "Branches matched more than once vs p_{T}", nintPt_, minPt_, maxPt_));
    h.h_duplicate_eta.push_back(
        booker.book1D("num_duplicate_eta", "Branches matched more than once vs #eta", nintEta_, minEta_, maxEta_));

    h.h_score.push_back(booker.book1D("association_score", "Association score", nintScore_, minScore_, maxScore_));
    h.h_sharedQuantity.push_back(
        booker.book1D("shared_quantity", "Shared hits or energy", nintShared_, minShared_, maxShared_));
  }

  void TruthBranchHistoProducerAlgo::fill_simul(TruthBranchHistograms const& h,
                                                std::size_t i,
                                                double pt,
                                                double eta,
                                                double phi,
                                                bool associated,
                                                bool duplicate) const {
    h.h_simul_pt[i]->Fill(pt);
    h.h_simul_eta[i]->Fill(eta);
    h.h_simul_phi[i]->Fill(phi);
    if (associated) {
      h.h_assoc_simToReco_pt[i]->Fill(pt);
      h.h_assoc_simToReco_eta[i]->Fill(eta);
      h.h_assoc_simToReco_phi[i]->Fill(phi);
    }
    if (duplicate) {
      h.h_duplicate_pt[i]->Fill(pt);
      h.h_duplicate_eta[i]->Fill(eta);
    }
  }

  void TruthBranchHistoProducerAlgo::fill_reco(
      TruthBranchHistograms const& h, std::size_t i, double pt, double eta, double phi, bool associated) const {
    h.h_reco_pt[i]->Fill(pt);
    h.h_reco_eta[i]->Fill(eta);
    h.h_reco_phi[i]->Fill(phi);
    if (associated) {
      h.h_assoc_recoToSim_pt[i]->Fill(pt);
      h.h_assoc_recoToSim_eta[i]->Fill(eta);
      h.h_assoc_recoToSim_phi[i]->Fill(phi);
    }
  }

  void TruthBranchHistoProducerAlgo::fill_match(TruthBranchHistograms const& h,
                                                std::size_t i,
                                                double score,
                                                double sharedQuantity) const {
    h.h_score[i]->Fill(score);
    h.h_sharedQuantity[i]->Fill(sharedQuantity);
  }

}  // namespace truth
