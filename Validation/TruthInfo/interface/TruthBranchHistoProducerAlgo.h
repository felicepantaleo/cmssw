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
//
// The truth side and the reco side are binned in DIFFERENT variable sets, because they
// describe different objects. Efficiency and duplicate rate divide two truth-side
// histograms and are binned in branch variables, which every domain supplies. Purity,
// fake rate and pileup rate divide two reco-side histograms and are binned in the reco
// object's own variables, which a vertex and a trackster do not share with a track.
// Booking a variable a domain cannot fill would put a spike at zero into every such
// plot and read as a real feature.

#ifndef Validation_TruthInfo_TruthBranchHistoProducerAlgo_h
#define Validation_TruthInfo_TruthBranchHistoProducerAlgo_h

#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace truth {

  // The x variables, following the MTVHistoProducerAlgoForTracker set restricted to
  // what a truth branch can supply, plus two the graph alone can supply: depth is how
  // far down the event history the branch root sits, and root_footprint_fraction is how much of the
  // branch footprint belongs to the root particle itself rather than to its
  // descendants.
  enum class Variable { Pt, Eta, Phi, Nhits, Vertpos, Zpos, Dxy, Dz, Depth, RootFootprintFraction, CaloEta };
  inline static const std::vector<std::string> kVariableNames = {
      "pt", "eta", "phi", "nhits", "vertpos", "zpos", "dxy", "dz", "depth", "root_footprint_fraction", "caloeta"};

  // caloeta of a branch that never reached the calorimeter. Far outside every axis
  // range, so such a branch lands in the underflow of BOTH numerator and denominator
  // and the calorimeter-entrance axis shows only what a calorimeter could have seen.
  inline constexpr double kNoCaloEntry = -999.;

  struct TruthBranchHistograms {
    using METype = dqm::reco::MonitorElement*;

    // Each vector is indexed [entry][variable], variable being the position within that
    // side's variable list, so booking order and fill index stay in step exactly as in
    // MTV. The two sides carry INDEPENDENT entry counters: truth-driven rows are
    // indexed by (collection, level) in truth-entry booking order, reco-driven rows by
    // (collection, working point) in wp-entry booking order.
    using MERow = std::vector<METype>;

    // Truth side, one row per (collection, level): denominator every target at that
    // level, numerator those a reco object was associated to. The cumulative numerator
    // also accepts targets covered only by several reco objects together, so it is a
    // superset of the individual one by construction.
    std::vector<MERow> h_simul, h_assoc_simToReco, h_assoc_simToReco_cumulative;

    // The two ways a truth object can be reconstructed as one object more than once or
    // in pieces, mutually exclusive so that individual + duplicate + split + lost = 1.
    //   duplicate  more than one reco object individually reconstructs the whole thing
    //   split      no single object does, but several together cover the subgraph
    // h_duplicate is left EMPTY for a calorimetric domain, where the outcome cannot
    // occur: two reco objects built from disjoint layer clusters cannot each miss less
    // than maxSimToRecoScoreForDuplicate of the same branch energy, since the two scores
    // sum to at least one. Measured on 200 no-PU ttbar events: ticlCandidate,
    // ticlTrackstersCLUE3DHigh and ticlTracksterLinks each use every layer cluster in at
    // most one trackster. A collection whose objects SHARE hits would make it reachable
    // again and would have to book it. Split carries the calorimetric pathology instead.
    std::vector<MERow> h_duplicate, h_split;

    // Reco side, one row per (collection, working point): denominator every reco
    // object, numerator those matched to a branch; one minus that ratio is the fake
    // rate. Pileup counts objects matched only to an overlaid interaction.
    //
    // h_assoc_recoToSim counts matched objects, one entry each, so the fake rate is a
    // FRACTION OF OBJECTS the way MultiTrackValidator's is. h_recopurity fills the same
    // objects weighted by the purity of the match, so its ratio to h_reco is the mean
    // purity. The two must stay separate: filling one histogram with the purity as a
    // weight and reading it as a count turns the fake rate into one minus the mean
    // purity, which on no-PU ttbar reads 0.83 where the fake rate is 0.003.
    //
    // h_assoc_strict is the calorimetric domains' second, STRICTER numerator: matched
    // AND below maxRecoToSimScore, which is HGCalValidator's non-fake criterion. It is
    // a separate histogram because that criterion mixes "corresponds to nothing" with
    // "corresponds to something but shares its cells with other truth", and only the
    // first is a fake. Booked for calorimetric domains only; empty elsewhere.
    std::vector<MERow> h_reco, h_assoc_recoToSim, h_recopurity, h_pileup, h_assoc_strict;

    // Efficiency and duplicate rate against the Geant4 process that CREATED the
    // branch, which only the graph can supply: the production vertex of the branch
    // root carries its VertexReason, so a loss can be attributed to the physics that
    // made the particle rather than only to where it landed. Truth side.
    std::vector<METype> h_simul_reason, h_assoc_simToReco_reason, h_duplicate_reason;

    // Quality of the match itself, one per direction. The denominator is what the name
    // says: reco purity divides by the reco object (reco side), truth purity by the
    // truth object (truth side).
    std::vector<METype> h_score, h_sharedQuantity, h_recoPurity, h_truthPurity;

    // DOMINANCE of the leading truth contributor, the axis a fake criterion built on
    // "no truth dominates the little contaminations" would cut on. leading_truth_share
    // is the leading branch's shared energy over the shared energy of ALL candidate
    // branches; dominance_ratio is leading over runner-up, capped at 20. Reco side, and
    // both are read from the FIRST working point's map, the only one that carries every
    // candidate. Filled for every reco object with at least one candidate.
    std::vector<METype> h_leadingShare, h_dominanceRatio;

    // The axis the calorimetric efficiency cut acts on: shared energy over the truth
    // branch's own energy. Booked only by the domains judged on it, so it is empty for
    // every other one. Truth side.
    std::vector<METype> h_sharedEnergyFraction;

    // Resolution inputs: 2D of (reco - truth)/truth against the truth variable, which
    // the harvester turns into _Mean and _Sigma by a Gaussian fit per slice. Reco side:
    // the pair comes from the reco-driven match, so it depends on the working point.
    std::vector<METype> h_ptres_vs_eta, h_ptres_vs_pt, h_etares_vs_eta, h_phires_vs_eta;
  };

  class TruthBranchHistoProducerAlgo {
  public:
    explicit TruthBranchHistoProducerAlgo(edm::ParameterSet const& pset);

    // Book one set of histograms into the current folder, appending one row to each of
    // that side's vectors. Call bookRecoHistos once per (collection, working point) and
    // bookTruthHistos once per (collection, level), each in the order the fill side
    // will index that list.
    // calorimetric additionally books the strict numerator described above.
    void bookRecoHistos(dqm::implementation::IBooker& booker,
                        TruthBranchHistograms& histograms,
                        bool calorimetric) const;
    // calorimetric books the shared-energy-fraction monitor element, the axis those
    // domains gate efficiency on, and skips the duplicate ones the same domains cannot
    // fill. It must be the same for every truth entry of one module, so the row index
    // stays shared with the other truth vectors.
    void bookTruthHistos(dqm::implementation::IBooker& booker,
                         TruthBranchHistograms& histograms,
                         bool calorimetric) const;

    // Values of every x variable for one object, in the enum order. A domain fills only
    // the ones it has; which of them are booked is decided by the variable lists.
    struct Kinematics {
      double pt = 0., eta = 0., phi = 0., nhits = 0., vertpos = 0., zpos = 0., dxy = 0., dz = 0.;
      double depth = 0., root_footprint_fraction = 0., caloeta = kNoCaloEntry;
      std::vector<double> asVector() const {
        return {pt, eta, phi, nhits, vertpos, zpos, dxy, dz, depth, root_footprint_fraction, caloeta};
      }
    };

    // How one truth object was reconstructed. Exactly one of these is true.
    enum class TruthOutcome { Individual, Duplicate, Split, Lost };

    // cumulative is true when the collection as a whole covers the truth object,
    // whether by one reco object or by several together.
    void fill_simul(TruthBranchHistograms const& histograms,
                    std::size_t index,
                    Kinematics const& kin,
                    TruthOutcome outcome,
                    bool cumulative) const;

    // Truth purity of the leading reco object, filled once per truth object that has
    // any overlap at all.
    void fill_truth_purity(TruthBranchHistograms const& histograms, std::size_t index, double truthPurity) const;

    // Shared energy fraction of the leading reco object, filled once per truth object
    // that has any overlap at all, by the domains that booked it.
    void fill_shared_energy_fraction(TruthBranchHistograms const& histograms,
                                     std::size_t index,
                                     double sharedEnergyFraction) const;

    // matchQuality is the purity of the match: 1 minus the reco-normalised score for a
    // hit-based domain, the leading truth vertex's share of the constituents for a
    // composite one. It weights the h_recopurity fill only. The h_assoc_recoToSim fill
    // is always unweighted, so num_assoc(recoToSim)/num_reco stays a fraction of
    // objects for every domain. A composite object always matches something, so there
    // that fraction is near one by construction and the purity is the number to read.
    // strictMatch is the calorimetric non-fake criterion (matched AND below
    // maxRecoToSimScore). It fills h_assoc_strict only, never h_assoc_recoToSim, so the
    // fake rate keeps meaning "matched to nothing" in every domain.
    void fill_reco(TruthBranchHistograms const& histograms,
                   std::size_t index,
                   Kinematics const& kin,
                   bool associated,
                   bool pileup,
                   double matchQuality = 1.,
                   bool strictMatch = false) const;

    // Categorical fill against the VertexReason of the branch root's production
    // vertex, passed as its underlying integer so this header stays free of the
    // graph data formats.
    void fill_reason(TruthBranchHistograms const& histograms,
                     std::size_t index,
                     unsigned int reason,
                     TruthOutcome outcome) const;

    // Negative values mean the object had no candidate at all and are not filled.
    void fill_dominance(TruthBranchHistograms const& histograms,
                        std::size_t index,
                        double leadingShare,
                        double dominanceRatio) const;

    void fill_match(TruthBranchHistograms const& histograms,
                    std::size_t index,
                    double score,
                    double sharedQuantity,
                    double recoPurity) const;

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
    // Which entries of Kinematics::asVector each side books, in booking order.
    std::vector<std::size_t> truthVars_, recoVars_;
    std::vector<std::string> truthVarNames_, recoVarNames_;
    std::vector<Axis> truthAxes_, recoAxes_;

    int nintScore_, nintShared_, nintRes_;
    double minScore_, maxScore_, minShared_, maxShared_, minRes_, maxRes_;
    // The resolution 2D uses its OWN, coarser x binning: each x slice is fitted with a
    // Gaussian, so it needs enough entries per slice to constrain the fit, which the
    // efficiency binning does not provide.
    Axis resEtaAxis_, resPtAxis_;
  };

}  // namespace truth

#endif
