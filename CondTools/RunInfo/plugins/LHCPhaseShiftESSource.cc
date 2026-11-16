#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordInfiniteIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "CondFormats/RunInfo/interface/LHCInfoPerFill.h"
#include "CondFormats/DataRecord/interface/LHCInfoPerFillRcd.h"
#include "CondCore/CondDB/interface/WebUtils.h"
#include <CLHEP/Units/SystemOfUnits.h>
#include <cmath>
#include <nlohmann/json.hpp>

class LHCPhaseShiftESSource : public edm::ESProducer, public edm::EventSetupRecordInfiniteIntervalFinder {
public:
  explicit LHCPhaseShiftESSource(const edm::ParameterSet& iConfig)
      : schema_(iConfig.getParameter<std::string>("schema")),
        V0_(iConfig.getParameter<double>("V0")),
        ppb_(iConfig.getParameter<double>("ppb")),
        b_len_(iConfig.getParameter<double>("b_len")) {
    auto cc = setWhatProduced(this, "lhcPhaseShift");
    lhcInfoToken_ = cc.consumes(edm::ESInputTag("", ""));
    findingRecord<LHCInfoPerFillRcd>();
  };
  ~LHCPhaseShiftESSource() override {};

  std::unique_ptr<LHCInfoPerFill> produce(const LHCInfoPerFillRcd&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  static constexpr double RF = 400800000;  //RF radio frequency [Hz]
  static constexpr double RoverQ = 45;     //R/Q of the cavity [Ohm]
  static constexpr double RFxPI = RF * M_PI;

  const std::string schema_;
  const double V0_, ppb_, b_len_;
  edm::ESGetToken<LHCInfoPerFill, LHCInfoPerFillRcd> lhcInfoToken_;

  std::vector<double> calculatePhase(const std::bitset<LHCInfoPerFill::bunchSlots + 1>& beams) {
    const auto t1 = RFxPI * b_len_;
    const auto fbp = 3 * (-t1 * std::cos(t1) + std::sin(t1)) / (t1 * t1 * t1);
    const auto ibt = 2 * fbp * ppb_ * CLHEP::e_SI / 25.0e-9;
    const auto fbx = double(beams.count()) / LHCInfoPerFill::bunchSlots;
    const auto domega = -RFxPI * RoverQ * ibt * fbx / V0_;
    std::vector<double> phasevec(LHCInfoPerFill::bunchSlots);
    double ssum(0.), integ(0.);
    for (size_t i = 0; i < LHCInfoPerFill::bunchSlots; i++) {
      integ += domega * 25.0e-9 * (1. - beams[i] / fbx);
      const auto phase = -0.5 * integ * 1.0e9 / RFxPI;  // [ns]
      phasevec[i] = phase;
      ssum += phase;
    }
    const auto mean = ssum / LHCInfoPerFill::bunchSlots;
    for (auto& phase : phasevec)
      phase -= mean;
    return phasevec;
  };
};

std::unique_ptr<LHCInfoPerFill> LHCPhaseShiftESSource::produce(const LHCInfoPerFillRcd& iRecord) {
  auto lhcInfo = std::make_unique<LHCInfoPerFill>();

  // load LHC information
  const auto& hLHCInfoFill = iRecord.getHandle(lhcInfoToken_);
  if (hLHCInfoFill.isValid())
    lhcInfo.reset(hLHCInfoFill->cloneFill());

  // set beam information
  const auto schema = lhcInfo->injectionScheme().empty() ? schema_ : lhcInfo->injectionScheme();
  if (!schema.empty()) {
    std::string lhcfile;
    auto status = cond::httpGet(
        "https://gitlab.cern.ch/lhc-injection-scheme/injection-schemes/raw/master/" + schema + ".json", lhcfile);
    if (status != 200 && status != 201)
      throw cms::Exception("LogicError") << "Failed to access lhc-injection-scheme gitlab website";
    const nlohmann::ordered_json lhc_data = nlohmann::ordered_json::parse(lhcfile, nullptr, true, true);
    std::bitset<LHCInfoPerFill::bunchSlots + 1> beam1, beam2;
    for (const auto& b : {1, 2})
      for (const auto& beam : lhc_data[b == 1 ? "beam1" : "beam2"])
        for (const auto& batch : beam["batches"])
          for (size_t i = 0; i < batch["bunches"]; i++)
            (b == 1 ? beam1 : beam2).set(int(beam["lhcbunch"]) + int(batch["injBunch"]) + i);
    lhcInfo->setInjectionScheme(lhc_data["schemeName"]);
    lhcInfo->setLhcComment(lhc_data["description"]);
    lhcInfo->setBunchBitsetForBeam1(beam1);
    lhcInfo->setBunchBitsetForBeam2(beam2);
    lhcInfo->setBunchesInBeam1(beam1.count());
    lhcInfo->setBunchesInBeam2(beam2.count());
    lhcInfo->setCollidingBunches((beam1 & beam2).count());
  }

  // calculate phase shifts per beam
  const auto phaseShift1 = calculatePhase(lhcInfo->bunchBitsetForBeam1());
  const auto phaseShift2 = calculatePhase(lhcInfo->bunchBitsetForBeam2());
  std::vector<float> phaseShift(LHCInfoPerFill::bunchSlots);
  for (size_t i = 0; i < LHCInfoPerFill::bunchSlots; i++)
    phaseShift[i] = (phaseShift1[i] + phaseShift2[i]) / 2.;
  lhcInfo->setPhaseShift(phaseShift);

  return lhcInfo;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void LHCPhaseShiftESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("schema", "25ns_2352b_2340_2004_2133_108bpi_24inj")->setComment("LHC schema");
  desc.add<double>("V0", 1500000)->setComment("Voltage of one cavity [V]");
  desc.add<double>("ppb", 1.15e11)->setComment("Protons per bunch");
  desc.add<double>("b_len", 1e-9)->setComment("Bunch length in seconds (4 sigma) [sec]");
  descriptions.add("lhcPhaseShiftESSource", desc);
}

DEFINE_FWK_EVENTSETUP_SOURCE(LHCPhaseShiftESSource);
