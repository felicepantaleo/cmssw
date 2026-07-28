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
#include <vector>

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace truth {

  // One entry per (collection, working point), appended in booking order. The fill
  // side indexes with the same counter, which is the contract the whole design rests
  // on: booking order == fill index.
  // The x variables the efficiency-style plots are binned in, following the
  // MTVHistoProducerAlgoForTracker set restricted to what a truth branch can supply:
  // kinematics from the root particle, position from its production vertex, and the
  // hit count from the branch's own footprint in the hit index.
  enum class Variable { Pt, Eta, Phi, Nhits, Vertpos, Zpos, Dxy, Dz };
  inline static const std::vector<std::string> kVariableNames = {
      "pt", "eta", "phi", "nhits", "vertpos", "zpos", "dxy", "dz"};

  struct TruthBranchHistograms {
    using METype = dqm::reco::MonitorElement*;

    // Each vector is indexed [entry][variable], entry being the (collection, working
    // point) counter and variable the enum above, so booking order and fill index stay
    // in step exactly as in MTV.
    using MERow = std::vector<METype>;

    // Truth side: denominator every selected branch, numerator those a reco object was
    // associated to.
    std::vector<MERow> h_simul, h_assoc_simToReco;

    // Reco side: denominator every reco object, numerator those matched to a branch.
    // One minus that ratio is the fake rate.
    std::vector<MERow> h_reco, h_assoc_recoToSim;

    // A branch matched by more than one reco object, and a reco object matched only to
    // a branch from a pileup interaction.
    std::vector<MERow> h_duplicate, h_pileup;

    // Efficiency and duplicate rate against the Geant4 process that CREATED the
    // branch, which only the graph can supply: the production vertex of the branch
    // root carries its VertexReason, so a loss can be attributed to the physics that
    // made the particle rather than only to where it landed.
    std::vector<METype> h_simul_reason, h_assoc_simToReco_reason, h_duplicate_reason;

    // Quality of the match itself.
    std::vector<METype> h_score, h_sharedQuantity;

    // Resolution inputs: 2D of (reco - truth)/truth against the truth variable, which
    // the harvester turns into _Mean and _Sigma by a Gaussian fit per slice.
    std::vector<METype> h_ptres_vs_eta, h_ptres_vs_pt, h_etares_vs_eta, h_phires_vs_eta;
  };

  class TruthBranchHistoProducerAlgo {
  public:
    explicit TruthBranchHistoProducerAlgo(edm::ParameterSet const& pset);

    // Book one full set of histograms into the current folder. Call once per
    // (collection, working point), in the order the fill side will index them.
    void bookHistos(dqm::implementation::IBooker& booker, TruthBranchHistograms& histograms) const;

    // Values of every x variable for one object, in the enum order.
    struct Kinematics {
      double pt = 0., eta = 0., phi = 0., nhits = 0., vertpos = 0., zpos = 0., dxy = 0., dz = 0.;
      std::vector<double> asVector() const { return {pt, eta, phi, nhits, vertpos, zpos, dxy, dz}; }
    };

    void fill_simul(TruthBranchHistograms const& histograms,
                    std::size_t index,
                    Kinematics const& kin,
                    bool associated,
                    bool duplicate) const;

    void fill_reco(TruthBranchHistograms const& histograms,
                   std::size_t index,
                   Kinematics const& kin,
                   bool associated,
                   bool pileup) const;

    // Categorical fill against the VertexReason of the branch root's production
    // vertex, passed as its underlying integer so this header stays free of the
    // graph data formats.
    void fill_reason(TruthBranchHistograms const& histograms,
                     std::size_t index,
                     unsigned int reason,
                     bool associated,
                     bool duplicate) const;

    void fill_match(TruthBranchHistograms const& histograms,
                    std::size_t index,
                    double score,
                    double sharedQuantity) const;

    // Called once per matched pair, with the truth branch kinematics and the matched
    // reco object's pt/eta/phi, to fill the resolution inputs.
    void fill_resolution(TruthBranchHistograms const& histograms,
                         std::size_t index,
                         Kinematics const& truth,
                         double recoPt,
                         double recoEta,
                         double recoPhi) const;

  private:
    struct Axis {
      int nbins;
      double min, max;
    };
    std::vector<Axis> axes_;  // one per Variable, in enum order
    int nintScore_, nintShared_, nintRes_;
    double minScore_, maxScore_, minShared_, maxShared_, minRes_, maxRes_;
    // The resolution 2D uses its OWN, coarser x binning: each x slice is fitted with a
    // Gaussian, so it needs enough entries per slice to constrain the fit, which the
    // efficiency binning does not provide.
    Axis resEtaAxis_, resPtAxis_;
  };

}  // namespace truth

#endif
