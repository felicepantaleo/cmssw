// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "Validation/TruthInfo/interface/TruthBranchHistoProducerAlgo.h"

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
        maxRes_(pset.getParameter<double>("maxRes")) {
    // One axis per Variable, read in the enum order so the vectors line up.
    for (auto const& name : kVariableNames) {
      axes_.push_back({pset.getParameter<int>("nint_" + name),
                       pset.getParameter<double>("min_" + name),
                       pset.getParameter<double>("max_" + name)});
    }
  }

  void TruthBranchHistoProducerAlgo::bookHistos(dqm::implementation::IBooker& booker, TruthBranchHistograms& h) const {
    // The ME names are the harvesting API: DQMGenericClient forms every ratio from
    // these by string, so a rename silently drops a plot rather than failing.
    auto bookRow = [&](std::vector<TruthBranchHistograms::MERow>& rows, std::string const& prefix) {
      TruthBranchHistograms::MERow row;
      for (std::size_t v = 0; v < kVariableNames.size(); ++v) {
        auto const& name = kVariableNames[v];
        auto const& axis = axes_[v];
        row.push_back(booker.book1D(prefix + "_" + name, prefix + " vs " + name, axis.nbins, axis.min, axis.max));
      }
      rows.push_back(std::move(row));
    };

    bookRow(h.h_simul, "num_simul");
    bookRow(h.h_assoc_simToReco, "num_assoc(simToReco)");
    bookRow(h.h_reco, "num_reco");
    bookRow(h.h_assoc_recoToSim, "num_assoc(recoToSim)");
    bookRow(h.h_duplicate, "num_duplicate");
    bookRow(h.h_pileup, "num_pileup");

    h.h_score.push_back(booker.book1D("association_score", "Association score", nintScore_, minScore_, maxScore_));
    h.h_sharedQuantity.push_back(
        booker.book1D("shared_quantity", "Shared hits or energy", nintShared_, minShared_, maxShared_));

    // 2D inputs for the Gaussian slice fit the harvester runs. Same naming as MTV so
    // the resolution strings and the plot script read the same way.
    auto const& etaAxis = axes_[static_cast<std::size_t>(Variable::Eta)];
    auto const& ptAxis = axes_[static_cast<std::size_t>(Variable::Pt)];
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
    for (std::size_t v = 0; v < values.size(); ++v) {
      h.h_simul[i][v]->Fill(values[v]);
      if (associated) {
        h.h_assoc_simToReco[i][v]->Fill(values[v]);
      }
      if (duplicate) {
        h.h_duplicate[i][v]->Fill(values[v]);
      }
    }
  }

  void TruthBranchHistoProducerAlgo::fill_reco(
      TruthBranchHistograms const& h, std::size_t i, Kinematics const& kin, bool associated, bool pileup) const {
    const auto values = kin.asVector();
    for (std::size_t v = 0; v < values.size(); ++v) {
      h.h_reco[i][v]->Fill(values[v]);
      if (associated) {
        h.h_assoc_recoToSim[i][v]->Fill(values[v]);
      }
      if (pileup) {
        h.h_pileup[i][v]->Fill(values[v]);
      }
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
