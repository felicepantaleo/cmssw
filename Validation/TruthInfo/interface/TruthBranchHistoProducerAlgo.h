// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// Histogram definitions for the truth-branch validation, kept apart from the analyzer
// the way MTVHistoProducerAlgoForTracker and HGVHistoProducerAlgo are: a POD struct
// that owns only MonitorElement pointers, plus a stateless algorithm whose fill_*
// methods take that struct by CONST reference. That is what lets the analyzer be a
// DQMGlobalEDAnalyzer, where booking and filling are both const and the MEs live in a
// per-run cache.
//
// Only num/denom histograms are booked. Every efficiency, fake rate and duplicate rate
// is formed downstream by DQMGenericClient from the string configuration, so this
// package contains no harvesting C++ at all.

#ifndef Validation_TruthInfo_TruthBranchHistoProducerAlgo_h
#define Validation_TruthInfo_TruthBranchHistoProducerAlgo_h

#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace truth {

  // One entry per (collection, working point), appended in booking order. The fill
  // side indexes with the same counter, which is the contract the whole design rests
  // on: booking order == fill index.
  struct TruthBranchHistograms {
    using METype = dqm::reco::MonitorElement*;

    // Truth side: the denominator is every selected branch, the numerator those that
    // a reco object was associated to.
    std::vector<METype> h_simul_pt, h_simul_eta, h_simul_phi;
    std::vector<METype> h_assoc_simToReco_pt, h_assoc_simToReco_eta, h_assoc_simToReco_phi;

    // Reco side: the denominator is every reco object, the numerator those matched to
    // a branch. One minus that ratio is the fake rate.
    std::vector<METype> h_reco_pt, h_reco_eta, h_reco_phi;
    std::vector<METype> h_assoc_recoToSim_pt, h_assoc_recoToSim_eta, h_assoc_recoToSim_phi;

    // A branch matched by more than one reco object.
    std::vector<METype> h_duplicate_pt, h_duplicate_eta;

    // Quality of the match itself.
    std::vector<METype> h_score, h_sharedQuantity;
  };

  class TruthBranchHistoProducerAlgo {
  public:
    explicit TruthBranchHistoProducerAlgo(edm::ParameterSet const& pset);

    // Book one full set of histograms into the current folder. Call once per
    // (collection, working point), in the order the fill side will index them.
    void bookHistos(dqm::implementation::IBooker& booker, TruthBranchHistograms& histograms) const;

    void fill_simul(TruthBranchHistograms const& histograms,
                    std::size_t index,
                    double pt,
                    double eta,
                    double phi,
                    bool associated,
                    bool duplicate) const;

    void fill_reco(TruthBranchHistograms const& histograms,
                   std::size_t index,
                   double pt,
                   double eta,
                   double phi,
                   bool associated) const;

    void fill_match(TruthBranchHistograms const& histograms,
                    std::size_t index,
                    double score,
                    double sharedQuantity) const;

  private:
    int nintPt_, nintEta_, nintPhi_, nintScore_, nintShared_;
    double minPt_, maxPt_, minEta_, maxEta_, minPhi_, maxPhi_;
    double minScore_, maxScore_, minShared_, maxShared_;
  };

}  // namespace truth

#endif
