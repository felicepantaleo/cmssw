#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CondFormats/RunInfo/interface/LHCInfoPerFill.h"
#include "CondFormats/DataRecord/interface/LHCInfoPerFillRcd.h"
#include "TOffsetFitter_crate_2.h"
#include "TFile.h"
#include "TFitResult.h"
#include "TFitResultPtr.h"
#include "TGraphErrors.h"
#include "TProfile.h"

class HGCalRecHitTimeHarvester : public DQMEDHarvester {
public:
  explicit HGCalRecHitTimeHarvester(const edm::ParameterSet& config)
      : DQMEDHarvester(config),
        inputFile_(config.getParameter<std::string>("inputFile")),
        lhcInfoToken_(esConsumes(config.getParameter<edm::ESInputTag>("lhcInfo"))) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    DQMEDHarvester::fillDescription(desc);
    desc.add<std::string>("inputFile", "");
    desc.add<edm::ESInputTag>("lhcInfo", edm::ESInputTag("lhcPhaseShiftESSource", "lhcPhaseShift"));
    descriptions.add("hgcalRecHitTimeHarvester", desc);
  }

  void beginRun(const edm::Run& run, const edm::EventSetup& iSetup) override {
    const auto& lhcInfo = iSetup.getData(lhcInfoToken_);
    const auto collidingBunchBitset = lhcInfo.bunchBitsetForBeam1() & lhcInfo.bunchBitsetForBeam2();
    trainStartEdges_.clear();
    for (size_t i = 0; i < LHCInfoPerFill::bunchSlots; i++)
      if ((i == 0 || collidingBunchBitset[i - 1] == 0) && collidingBunchBitset[i] == 1)
        trainStartEdges_.push_back(i);
  }

private:
  const std::string inputFile_;
  const edm::ESGetToken<LHCInfoPerFill, LHCInfoPerFillRcd> lhcInfoToken_;
  std::vector<int> trainStartEdges_;

  void dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) override {
    std::unique_ptr<TFile> source;
    if (!inputFile_.empty()) {
      source.reset(TFile::Open(inputFile_.c_str(), "READ"));
      if (!source || source->IsZombie())
        throw std::runtime_error("Cannot open DQM input file: " + inputFile_);
    }
    for (const std::string endcap : {"Minus", "Plus"}) {
      const std::string folder = "HGCAL/EndCap_" + endcap;
      TProfile* profile = nullptr;
      if (source) {
        // Standalone test input: the saved Run 1 histogram file.
        const std::string path = "DQMData/Run 1/HGCAL/Run summary/EndCap_" + endcap + "/rechittimevscrateProfile";
        source->GetObject(path.c_str(), profile);
      } else {
        auto* input = getter.get(folder + "/rechittimevscrateProfile");
        if (input)
          profile = input->getTProfile();
      }
      if (!profile || !std::string(profile->GetErrorOption()).empty()) {
        edm::LogWarning("HGCalRecHitTimeHarvester") << "Expected a TProfile with errors on the mean in " << folder;
        continue;
      }

      booker.setCurrentFolder(folder);
      const int nbins = profile->GetNbinsX();
      const double xmin = profile->GetXaxis()->GetXmin();
      const double xmax = profile->GetXaxis()->GetXmax();
      const std::string title = endcap + ";Crate number;Fitted " + profile->GetYaxis()->GetTitle();
      auto* output = booker.book1D("rechittimevscrateFit", title, nbins, xmin, xmax);
      auto* valid = booker.book1D(
          "rechittimevscrateFitValid", endcap + ";Crate number;Valid fit result (1=yes, 0=no)", nbins, xmin, xmax);
      output->Reset();
      valid->Reset();

      TGraphErrors graph;
      std::vector<int> bins;
      for (int bin = 1; bin <= nbins; ++bin) {
        if (profile->GetBinEntries(bin) <= 0.)
          continue;
        const double mean = profile->GetBinContent(bin);
        const double error = profile->GetBinError(bin);
        if (!std::isfinite(mean) || !std::isfinite(error) || error <= 0.)
          continue;
        const int i = bins.size();
        bins.push_back(bin);
        // dummy BX=0; crate information only
        graph.SetPoint(i, i * LHCInfoPerFill::bunchSlots, mean);
        graph.SetPointError(i, 0., error);
      }
      if (bins.empty()) {
        edm::LogWarning("HGCalRecHitTimeHarvester") << "No crate bins with usable errors in " << folder;
        continue;
      }

      const unsigned ncrates = bins.size();
      TOffsetFitter_crate_2 helper(ncrates);
      helper.setTrainStartEdges(trainStartEdges_);
      TF1 model(
          ("crateTimeModel" + endcap).c_str(), helper, 0., ncrates * LHCInfoPerFill::bunchSlots - 1., helper.npars());
      // no BX info for now
      model.FixParameter(ncrates, 0.);
      auto result = graph.Fit(&model, "SQN");
      if (int(result) != 0 || !result.Get() || !result->IsValid() || result->CovMatrixStatus() != 3) {
        edm::LogWarning("HGCalRecHitTimeHarvester") << "Crate fit failed in " << folder;
        continue;
      }
      for (unsigned i = 0; i < ncrates; ++i) {
        output->setBinContent(bins[i], result->Parameter(i));
        output->setBinError(bins[i], result->ParError(i));
        valid->setBinContent(bins[i], 1.);
      }
    }
  }
};

DEFINE_FWK_MODULE(HGCalRecHitTimeHarvester);
