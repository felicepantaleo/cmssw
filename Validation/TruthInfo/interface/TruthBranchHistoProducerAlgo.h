// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// Histogram definitions for the truth-branch validation. TruthBranchHistograms holds only
// MonitorElement pointers. The fill_* methods are const and take it by const reference,
// as a DQMGlobalEDAnalyzer requires.
//
// Only num/denom histograms are booked. DQMGenericClient forms every ratio from the
// string configuration.
//
// Truth-side ratios (efficiency, duplicate rate) use branch variables. Reco-side ratios
// (purity, fake rate, pileup rate) use the variables of the reco object. Do not book a
// variable that a domain cannot fill: it puts a false spike at zero into the plot.

#ifndef Validation_TruthInfo_TruthBranchHistoProducerAlgo_h
#define Validation_TruthInfo_TruthBranchHistoProducerAlgo_h

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/TruthInfo/interface/BranchSelector.h"

namespace truth {

  // Acceptance regions in absolute pseudorapidity. The two endcaps are pooled. Every num_*
  // row is booked once inclusively and once per booked region, in a sub-folder of that
  // name. On no-PU TenTau, 58.6% of the taus enter the calorimeter in the barrel, where no
  // trackster exists. Mixing the regions moves the calorimetric efficiency from 0.49 to 0.19.
  // Index 0 is the inclusive row and is always filled. An object outside every band, or
  // one whose region variable is undefined, fills only index 0.
  enum class EtaRegion { Inclusive = 0, Barrel, Endcap, Forward };
  inline constexpr std::size_t kNEtaRegions = 4;
  inline static const std::vector<std::string> kEtaRegionFolders = {"", "etaLt15", "eta15to30", "eta30to45"};

  // Which band an absolute pseudorapidity falls in; Inclusive when it is in none.
  [[nodiscard]] inline EtaRegion etaRegionOf(double absEta) {
    if (absEta < 1.5)
      return EtaRegion::Barrel;
    if (absEta < 3.0)
      return EtaRegion::Endcap;
    if (absEta < 4.5)
      return EtaRegion::Forward;
    return EtaRegion::Inclusive;
  }

  // The x variables. depth is the number of ancestors of the branch root.
  // root_footprint_fraction is the fraction of the branch footprint that belongs to the
  // root particle itself and not to its descendants.
  enum class Variable { Pt, Eta, Phi, Nhits, Vertpos, Zpos, Dxy, Dz, Depth, RootFootprintFraction, CaloEta, Flavour };
  inline static const std::vector<std::string> kVariableNames = {"pt",
                                                                 "eta",
                                                                 "phi",
                                                                 "nhits",
                                                                 "vertpos",
                                                                 "zpos",
                                                                 "dxy",
                                                                 "dz",
                                                                 "depth",
                                                                 "root_footprint_fraction",
                                                                 "caloeta",
                                                                 "flavour"};

  // The species of the truth object root, as a bin index. A root that is not a quark or
  // a gluon fills the Other bin.
  enum class FlavourBin { Other = 0, Down, Up, Strange, Charm, Bottom, Top, Gluon };
  inline static constexpr int kNFlavourBins = 8;
  inline static const std::vector<std::string> kFlavourBinNames = {"other", "d", "u", "s", "c", "b", "t", "g"};

  [[nodiscard]] inline double flavourBin(int32_t pdgId) {
    const int32_t a = std::abs(pdgId);
    if (a >= 1 && a <= 6)
      return static_cast<double>(a) + 0.5;
    if (a == 21)
      return static_cast<double>(FlavourBin::Gluon) + 0.5;
    return static_cast<double>(FlavourBin::Other) + 0.5;
  }

  // caloeta of a branch that does not reach the calorimeter. The value is outside every
  // axis range, so the branch fills the underflow of numerator and denominator.
  inline constexpr double kNoCaloEntry = -999.;

  struct TruthBranchHistograms {
    using METype = dqm::reco::MonitorElement*;

    // Each entry has rowsPerEntry() rows: the inclusive row, then one row per booked
    // region. A fill writes at most two rows: the inclusive row and the object region row.
    // Each vector is indexed [row][variable], with variable the position in the variable
    // list of that side. Truth rows count per (collection, level) and reco rows per
    // (collection, working point), with independent entry counters.
    using MERow = std::vector<METype>;

    // Truth side. Denominator: every target at the level. Numerator: targets that one
    // reco object reconstructs. The cumulative numerator also accepts targets that only
    // several reco objects together cover.
    std::vector<MERow> h_simul, h_assoc_simToReco, h_assoc_simToReco_cumulative;

    // The outcomes are exclusive: individual + duplicate + split + lost = 1.
    //   duplicate  more than one reco object reconstructs the whole truth object
    //   split      no single reco object does, but several together cover the subgraph
    // h_duplicate is empty for a calorimetric domain, where the outcome cannot occur: two
    // reco objects with disjoint layer clusters cannot both have a score below
    // maxSimToRecoScoreForDuplicate, because the two scores sum to at least one. On 200
    // no-PU ttbar events, ticlCandidate, ticlTrackstersCLUE3DHigh and ticlTracksterLinks
    // use each layer cluster in at most one trackster. A collection whose objects share
    // hits must book h_duplicate.
    std::vector<MERow> h_duplicate, h_split;

    // Reco side. h_reco is the denominator: every reco object.
    // h_dominated is the fake-rate numerator: the object is matched and, where dominance
    // is defined, one truth branch of the antichain owns at least minLeadingTruthShare of
    // the shared quantity.
    // h_levelCandidate counts the objects where dominance is defined. A matched object
    // with no candidate at the dominance level is not a fake. On no-PU ttbar that is 32.5%
    // of tracksters and 36.8% of tracks, against 0.3% of tracks matched to nothing.
    // h_assoc_recoToSim counts the objects matched to anything (the no-candidate rate).
    // h_recopurity counts matched objects weighted by the match purity. Its ratio to
    // h_reco is the mean purity, with 0 for an unmatched object. Read as a count, it gives
    // a fake rate of 0.83 on no-PU ttbar, where the fake rate is 0.003.
    // h_pileup counts objects matched only to an overlaid interaction.
    // h_assoc_strict: calorimetric domains only. HGCalValidator's non-fake criterion,
    // matched and below maxRecoToSimScore, for comparison with HGCalValidator. It is not a
    // fake rate: it is normalised to the total truth energy of the cell, so pileup moves
    // it towards 1 also for a good match.
    std::vector<MERow> h_reco, h_dominated, h_levelCandidate, h_assoc_recoToSim, h_recopurity, h_pileup, h_assoc_strict;

    // Efficiency and duplicate rate against the VertexReason (the Geant4 creation
    // process) of the production vertex of the branch root. Truth side.
    std::vector<METype> h_simul_reason, h_assoc_simToReco_reason, h_duplicate_reason;

    // Quality of the match itself, one per direction. The denominator is what the name
    // says: reco purity divides by the reco object (reco side), truth purity by the
    // truth object (truth side).
    std::vector<METype> h_score, h_sharedQuantity, h_recoPurity, h_truthPurity;

    // Dominance of the leading truth contributor, the axis of the fake criterion.
    // leading_truth_share is the shared quantity of the leading antichain member over the
    // sum for all antichain members. dominance_ratio is leading over runner-up, capped at
    // 20. Reco side. Both come from the map of the first working point, the only map that
    // carries every candidate. Filled for every reco object with a candidate at the
    // dominance level.
    std::vector<METype> h_leadingShare, h_dominanceRatio;

    // The axis of the calorimetric efficiency cut: shared energy over the truth branch
    // energy. Booked for calorimetric domains only. Truth side.
    std::vector<METype> h_sharedEnergyFraction;

    // Resolution inputs: 2D of (reco - truth)/truth against the truth variable, which
    // the harvester turns into _Mean and _Sigma by a Gaussian fit per slice. Reco side:
    // the pair comes from the reco-driven match, so it depends on the working point.
    std::vector<METype> h_ptres_vs_eta, h_ptres_vs_pt, h_etares_vs_eta, h_phires_vs_eta;
  };

  class TruthBranchHistoProducerAlgo {
  public:
    explicit TruthBranchHistoProducerAlgo(edm::ParameterSet const& pset);

    // Book one entry: rowsPerEntry() rows in each vector of that side, with the region
    // rows in sub-folders, and the diagnostics. Call bookRecoHistos once per (collection,
    // working point) and bookTruthHistos once per (collection, level), in fill order.
    // calorimetric also books h_assoc_strict.
    void bookRecoHistos(dqm::implementation::IBooker& booker,
                        TruthBranchHistograms& histograms,
                        bool calorimetric) const;
    // calorimetric books h_sharedEnergyFraction and skips the duplicate histograms. It
    // must be the same for every truth entry of one module, so that all truth vectors
    // share one index.
    void bookTruthHistos(dqm::implementation::IBooker& booker,
                         TruthBranchHistograms& histograms,
                         bool calorimetric) const;

    // One region's worth of rows, into the booker's current folder. bookTruthHistos and
    // bookRecoHistos call these once per region.
    void bookTruthRow(dqm::implementation::IBooker& booker, TruthBranchHistograms& histograms, bool calorimetric) const;
    void bookRecoRow(dqm::implementation::IBooker& booker, TruthBranchHistograms& histograms, bool calorimetric) const;

    // The once-per-entry distributions, booked in the base folder only.
    void bookTruthDiagnostics(dqm::implementation::IBooker& booker,
                              TruthBranchHistograms& histograms,
                              bool calorimetric) const;
    void bookRecoDiagnostics(dqm::implementation::IBooker& booker,
                             TruthBranchHistograms& histograms,
                             bool calorimetric) const;

    // Values of every x variable for one object, in the enum order. A domain fills only
    // the ones it has; which of them are booked is decided by the variable lists.
    struct Kinematics {
      double pt = 0., eta = 0., phi = 0., nhits = 0., vertpos = 0., zpos = 0., dxy = 0., dz = 0.;
      double depth = 0., root_footprint_fraction = 0., caloeta = kNoCaloEntry;
      double flavour = static_cast<double>(FlavourBin::Other) + 0.5;
      std::array<double, 12> asVector() const {
        return {pt, eta, phi, nhits, vertpos, zpos, dxy, dz, depth, root_footprint_fraction, caloeta, flavour};
      }
    };

    // How one truth object was reconstructed. Exactly one of these is true.
    enum class TruthOutcome { Individual, Duplicate, Split, Lost };

    // Row-level fills. fill_simul and fill_reco call them for the inclusive row and for
    // the region row of the object.
    // cumulative is true when the collection covers the truth object, with one reco
    // object or with several together.
    // failedCuts is a BranchSelector::CutBit mask of the plotted-axis cuts that the object
    // fails. A variable is filled only when the object fails no cut except the cut on
    // that variable, so the efficiency against pt keeps the objects that fail the pt cut.
    void fill_simul_row(TruthBranchHistograms const& histograms,
                        std::size_t index,
                        Kinematics const& kin,
                        TruthOutcome outcome,
                        bool cumulative,
                        uint32_t failedCuts) const;

    void fill_simul(TruthBranchHistograms const& histograms,
                    std::size_t index,
                    Kinematics const& kin,
                    TruthOutcome outcome,
                    bool cumulative,
                    uint32_t failedCuts) const;

    // linthresh > 0 selects symlog binning: one linear bin [min, linthresh], then
    // log-spaced bins up to max. A log axis cannot show 0, and on DY 20.5% of the signal
    // level has pt exactly 0 (the pre-ISR copy of the resonance).
    // binEdges returns float because the variable-bin overload of the DQM booker takes float.
    struct SymlogAxis {
      int nbins;
      double min, max, linthresh;
    };
    [[nodiscard]] static std::vector<float> binEdges(SymlogAxis const& axis);

    // The cut bit of the cut that acts on a truth variable, or 0 when no cut acts on it.
    // Only pt and eta have a cut bit.
    [[nodiscard]] static uint32_t cutBitOfVariable(std::string const& name);

    // Truth purity of the leading reco object, filled once per truth object that has
    // any overlap at all.
    void fill_truth_purity(TruthBranchHistograms const& histograms, std::size_t index, double truthPurity) const;

    // Shared energy fraction of the leading reco object, filled once per truth object
    // that has any overlap at all, by the domains that booked it.
    void fill_shared_energy_fraction(TruthBranchHistograms const& histograms,
                                     std::size_t index,
                                     double sharedEnergyFraction) const;

    // How one reco object relates to the truth.
    struct RecoOutcome {
      // Not a fake: matched, and not contaminated beyond attribution. Its complement is
      // the fake rate.
      bool dominated = false;
      // Matched to any truth object. An unmatched object is a fake.
      bool associated = false;
      // Dominance is defined for this object: at least one candidate projects onto the
      // antichain. Its complement is a separate page and is not a fake.
      bool hasLevelCandidate = false;
      bool pileup = false;
      // Calorimetric only, HGCalValidator's non-fake criterion. Fills h_assoc_strict
      // and nothing else, so it can never move the fake rate.
      bool strictMatch = false;
      // Purity of the match: 1 minus the reco-normalised score for a hit-based domain,
      // the pt^2 share of the constituents from the leading truth vertex for a composite one. It
      // weights the h_recopurity fill only; every other fill here is a count.
      double matchQuality = 1.;
    };

    void fill_reco(TruthBranchHistograms const& histograms,
                   std::size_t index,
                   Kinematics const& kin,
                   RecoOutcome const& outcome) const;

    void fill_reco_row(TruthBranchHistograms const& histograms,
                       std::size_t index,
                       Kinematics const& kin,
                       RecoOutcome const& outcome) const;

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

    // How many rows one entry occupies: the inclusive row plus the booked regions.
    [[nodiscard]] std::size_t rowsPerEntry() const { return 1 + bookedRegions_.size(); }

  private:
    struct Axis {
      int nbins;
      double min, max;
      double linthresh = 0.;
    };
    // Which entries of Kinematics::asVector each side books, in booking order.
    std::vector<std::size_t> truthVars_, recoVars_;
    std::vector<std::string> truthVarNames_, recoVarNames_;
    // Cut bit per truth variable, resolved once so the fill loop does no string work.
    std::vector<uint32_t> truthCutBits_;
    std::vector<Axis> truthAxes_, recoAxes_;

    // The regions this domain books, in booking order. regionSlot_ is the row offset of
    // each region, or -1 for a region that is not booked.
    std::vector<EtaRegion> bookedRegions_;
    std::array<int, kNEtaRegions> regionSlot_{};

    int nintScore_, nintShared_, nintRes_;
    double minScore_, maxScore_, minShared_, maxShared_, minRes_, maxRes_;
    // The resolution 2D histograms use a coarser x binning, so that each x slice has
    // enough entries for its Gaussian fit.
    Axis resEtaAxis_, resPtAxis_;
  };

}  // namespace truth

#endif
