// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include <algorithm>
#include <iterator>

#include "FWCore/Utilities/interface/Exception.h"
#include "SimDataFormats/TruthInfo/interface/VertexData.h"
#include "Validation/TruthInfo/interface/TruthBranchHistoProducerAlgo.h"

namespace {
  // One bin per VertexReason, the enum being contiguous from Unknown to Other, plus one
  // synthetic bin. VertexReason is derived from the Geant4 creator-process subtype of a
  // SimVertex, so a GEN-only vertex has no process and reads as Unknown. That is a
  // different statement from "the process is not one we map", and in a pileup sample it
  // is the dominant category: collapsePileupGen replaces each pileup interaction with
  // one GEN-only vertex carrying all its stable particles. Giving it its own bin keeps
  // Unknown meaning what it says.
  constexpr int kNReasons = static_cast<int>(truth::VertexReason::Other) + 1;
  constexpr int kGenOnlyBin = kNReasons;
  constexpr int kNReasonBins = kNReasons + 1;
}  // namespace

namespace truth {

  TruthBranchHistoProducerAlgo::TruthBranchHistoProducerAlgo(edm::ParameterSet const& pset)
      : nintScore_(pset.getParameter<int>("nintScore")),
        nintShared_(pset.getParameter<int>("nintShared")),
        nintRes_(pset.getParameter<int>("nintRes")),
        minScore_(pset.getParameter<double>("minScore")),
        maxScore_(pset.getParameter<double>("maxScore")),
        minShared_(pset.getParameter<double>("minShared")),
        maxShared_(pset.getParameter<double>("maxShared")),
        minRes_(pset.getParameter<double>("minRes")),
        maxRes_(pset.getParameter<double>("maxRes")),
        resEtaAxis_{pset.getParameter<int>("nint_res_eta"),
                    pset.getParameter<double>("min_res_eta"),
                    pset.getParameter<double>("max_res_eta")},
        resPtAxis_{pset.getParameter<int>("nint_res_pt"),
                   pset.getParameter<double>("min_res_pt"),
                   pset.getParameter<double>("max_res_pt")} {
    // Resolve a variable name to its position in Kinematics::asVector, so a typo in the
    // configuration is a configuration error and not a silently missing plot.
    auto resolve = [&](std::vector<std::string> const& names,
                       std::vector<std::size_t>& indices,
                       std::vector<std::string>& kept,
                       std::vector<Axis>& axes) {
      for (auto const& name : names) {
        const auto it = std::find(kVariableNames.begin(), kVariableNames.end(), name);
        if (it == kVariableNames.end()) {
          throw cms::Exception("Configuration") << "unknown truth-branch plot variable '" << name << "'";
        }
        indices.push_back(static_cast<std::size_t>(std::distance(kVariableNames.begin(), it)));
        kept.push_back(name);
        axes.push_back({pset.getParameter<int>("nint_" + name),
                        pset.getParameter<double>("min_" + name),
                        pset.getParameter<double>("max_" + name)});
      }
    };
    resolve(pset.getParameter<std::vector<std::string>>("truthVariables"), truthVars_, truthVarNames_, truthAxes_);
    resolve(pset.getParameter<std::vector<std::string>>("recoVariables"), recoVars_, recoVarNames_, recoAxes_);
  }

  void TruthBranchHistoProducerAlgo::bookHistos(dqm::implementation::IBooker& booker, TruthBranchHistograms& h) const {
    // The ME names are the harvesting API: DQMGenericClient forms every ratio from
    // these by string, so a rename silently drops a plot rather than failing.
    auto bookRow = [&](std::vector<TruthBranchHistograms::MERow>& rows,
                       std::string const& prefix,
                       std::vector<std::string> const& names,
                       std::vector<Axis> const& axes) {
      TruthBranchHistograms::MERow row;
      for (std::size_t v = 0; v < names.size(); ++v) {
        auto const& name = names[v];
        auto const& axis = axes[v];
        row.push_back(booker.book1D(prefix + "_" + name, prefix + " vs " + name, axis.nbins, axis.min, axis.max));
      }
      rows.push_back(std::move(row));
    };

    bookRow(h.h_simul, "num_simul", truthVarNames_, truthAxes_);
    bookRow(h.h_assoc_simToReco, "num_assoc(simToReco)", truthVarNames_, truthAxes_);
    bookRow(h.h_duplicate, "num_duplicate", truthVarNames_, truthAxes_);
    bookRow(h.h_reco, "num_reco", recoVarNames_, recoAxes_);
    bookRow(h.h_assoc_recoToSim, "num_assoc(recoToSim)", recoVarNames_, recoAxes_);
    bookRow(h.h_pileup, "num_pileup", recoVarNames_, recoAxes_);

    // Categorical axis: one labelled bin per Geant4 creation process.
    auto bookReason = [&](std::vector<TruthBranchHistograms::METype>& v, std::string const& name) {
      auto* me = booker.book1D(name, name, kNReasonBins, -0.5, kNReasonBins - 0.5);
      for (int r = 0; r < kNReasons; ++r) {
        me->setBinLabel(r + 1, truth::vertexReasonName(static_cast<truth::VertexReason>(r)));
      }
      me->setBinLabel(kGenOnlyBin + 1, "GenOnly");
      v.push_back(me);
    };
    bookReason(h.h_simul_reason, "num_simul_reason");
    bookReason(h.h_assoc_simToReco_reason, "num_assoc(simToReco)_reason");
    bookReason(h.h_duplicate_reason, "num_duplicate_reason");

    h.h_score.push_back(booker.book1D("association_score", "Association score", nintScore_, minScore_, maxScore_));
    h.h_sharedQuantity.push_back(
        booker.book1D("shared_quantity", "Shared hits or energy", nintShared_, minShared_, maxShared_));

    // 2D inputs for the Gaussian slice fit the harvester runs. Same naming as MTV so
    // the resolution strings and the plot script read the same way.
    auto const& etaAxis = resEtaAxis_;
    auto const& ptAxis = resPtAxis_;
    h.h_ptres_vs_eta.push_back(booker.book2D("ptres_vs_eta",
                                             "Relative p_{T} residual vs #eta",
                                             etaAxis.nbins,
                                             etaAxis.min,
                                             etaAxis.max,
                                             nintRes_,
                                             minRes_,
                                             maxRes_));
    h.h_ptres_vs_pt.push_back(booker.book2D("ptres_vs_pt",
                                            "Relative p_{T} residual vs p_{T}",
                                            ptAxis.nbins,
                                            ptAxis.min,
                                            ptAxis.max,
                                            nintRes_,
                                            minRes_,
                                            maxRes_));
    h.h_etares_vs_eta.push_back(booker.book2D(
        "etares_vs_eta", "#eta residual vs #eta", etaAxis.nbins, etaAxis.min, etaAxis.max, nintRes_, minRes_, maxRes_));
    h.h_phires_vs_eta.push_back(booker.book2D(
        "phires_vs_eta", "#phi residual vs #eta", etaAxis.nbins, etaAxis.min, etaAxis.max, nintRes_, minRes_, maxRes_));
  }

  void TruthBranchHistoProducerAlgo::fill_simul(
      TruthBranchHistograms const& h, std::size_t i, Kinematics const& kin, bool associated, bool duplicate) const {
    const auto values = kin.asVector();
    for (std::size_t v = 0; v < truthVars_.size(); ++v) {
      const double x = values[truthVars_[v]];
      h.h_simul[i][v]->Fill(x);
      if (associated) {
        h.h_assoc_simToReco[i][v]->Fill(x);
      }
      if (duplicate) {
        h.h_duplicate[i][v]->Fill(x);
      }
    }
  }

  void TruthBranchHistoProducerAlgo::fill_reco(TruthBranchHistograms const& h,
                                               std::size_t i,
                                               Kinematics const& kin,
                                               bool associated,
                                               bool pileup,
                                               double matchQuality) const {
    const auto values = kin.asVector();
    for (std::size_t v = 0; v < recoVars_.size(); ++v) {
      const double x = values[recoVars_[v]];
      h.h_reco[i][v]->Fill(x);
      if (associated) {
        h.h_assoc_recoToSim[i][v]->Fill(x, matchQuality);
      }
      if (pileup) {
        h.h_pileup[i][v]->Fill(x);
      }
    }
  }

  void TruthBranchHistoProducerAlgo::fill_reason(
      TruthBranchHistograms const& h, std::size_t i, unsigned int reason, bool associated, bool duplicate) const {
    const double bin =
        (reason < static_cast<unsigned int>(kNReasonBins)) ? reason : static_cast<double>(truth::VertexReason::Other);
    h.h_simul_reason[i]->Fill(bin);
    if (associated) {
      h.h_assoc_simToReco_reason[i]->Fill(bin);
    }
    if (duplicate) {
      h.h_duplicate_reason[i]->Fill(bin);
    }
  }

  void TruthBranchHistoProducerAlgo::fill_match(TruthBranchHistograms const& h,
                                                std::size_t i,
                                                double score,
                                                double sharedQuantity) const {
    h.h_score[i]->Fill(score);
    h.h_sharedQuantity[i]->Fill(sharedQuantity);
  }

  void TruthBranchHistoProducerAlgo::fill_resolution(TruthBranchHistograms const& h,
                                                     std::size_t i,
                                                     Kinematics const& truthKin,
                                                     double recoPt,
                                                     double recoEta,
                                                     double recoPhi) const {
    if (truthKin.pt > 0.) {
      const double dpt = (recoPt - truthKin.pt) / truthKin.pt;
      h.h_ptres_vs_eta[i]->Fill(truthKin.eta, dpt);
      h.h_ptres_vs_pt[i]->Fill(truthKin.pt, dpt);
    }
    h.h_etares_vs_eta[i]->Fill(truthKin.eta, recoEta - truthKin.eta);
    h.h_phires_vs_eta[i]->Fill(truthKin.eta, recoPhi - truthKin.phi);
  }

}  // namespace truth
