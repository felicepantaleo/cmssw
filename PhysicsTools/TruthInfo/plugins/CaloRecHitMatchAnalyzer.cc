// Diagnostic: does the DetId-keyed truth hit index actually match the reco rechit
// collections of EVERY calorimeter (HGCAL endcap, ECAL barrel, HCAL)?
//
// The index is built at DIGI from sim-hits and stores RECO DetIds (HGCAL hexagon
// unpacking and HcalHitRelabeller are applied at build time). The association layer
// matches by DetId, so the check is: for each calorimeter, what fraction of the index
// DetIds is present in that calorimeter's reco rechit collection.

#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace {
  // Bucket an index DetId by the calorimeter it claims to belong to.
  std::string caloName(uint32_t rawId) {
    const DetId id(rawId);
    switch (id.det()) {
      case DetId::HGCalEE:
        return "HGCAL EE   (DetId::HGCalEE)";
      case DetId::HGCalHSi:
        return "HGCAL HSi  (DetId::HGCalHSi)";
      case DetId::HGCalHSc:
        return "HGCAL HSc  (DetId::HGCalHSc)";
      case DetId::Ecal:
        return "ECAL       (DetId::Ecal, subdet " + std::to_string(id.subdetId()) + ")";
      case DetId::Hcal:
        return "HCAL       (DetId::Hcal, subdet " + std::to_string(id.subdetId()) + ")";
      case DetId::Forward:
        return "Forward/HGCAL-old (subdet " + std::to_string(id.subdetId()) + ")";
      default:
        return "OTHER det " + std::to_string(id.det());
    }
  }
}  // namespace

class CaloRecHitMatchAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit CaloRecHitMatchAnalyzer(edm::ParameterSet const&);
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> indexToken_;
  const edm::EDGetTokenT<HGCRecHitCollection> eeToken_;
  const edm::EDGetTokenT<HGCRecHitCollection> hefToken_;
  const edm::EDGetTokenT<HGCRecHitCollection> hebToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> ebToken_;
  const edm::EDGetTokenT<HBHERecHitCollection> hbheToken_;
};

CaloRecHitMatchAnalyzer::CaloRecHitMatchAnalyzer(edm::ParameterSet const& cfg)
    : indexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      eeToken_(consumes<HGCRecHitCollection>(cfg.getParameter<edm::InputTag>("hgcalEE"))),
      hefToken_(consumes<HGCRecHitCollection>(cfg.getParameter<edm::InputTag>("hgcalHEF"))),
      hebToken_(consumes<HGCRecHitCollection>(cfg.getParameter<edm::InputTag>("hgcalHEB"))),
      ebToken_(consumes<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("ecalEB"))),
      hbheToken_(consumes<HBHERecHitCollection>(cfg.getParameter<edm::InputTag>("hcalHBHE"))) {}

void CaloRecHitMatchAnalyzer::analyze(edm::Event const& event, edm::EventSetup const&) {
  edm::Handle<truth::LogicalGraphHitIndex> hIndex;
  event.getByToken(indexToken_, hIndex);
  if (!hIndex.isValid()) {
    edm::LogPrint("CaloMatch") << "hit index NOT FOUND";
    return;
  }

  // Every reco calo DetId available in this event, plus per-collection sets so a hit
  // can be attributed to the right calorimeter.
  std::unordered_set<uint32_t> hgcal, ecal, hcal;

  auto fillHGC = [&](edm::EDGetTokenT<HGCRecHitCollection> const& token) {
    edm::Handle<HGCRecHitCollection> h;
    event.getByToken(token, h);
    if (h.isValid())
      for (auto const& rh : *h)
        hgcal.insert(rh.detid().rawId());
  };
  fillHGC(eeToken_);
  fillHGC(hefToken_);
  fillHGC(hebToken_);

  edm::Handle<EcalRecHitCollection> hEB;
  event.getByToken(ebToken_, hEB);
  if (hEB.isValid())
    for (auto const& rh : *hEB)
      ecal.insert(rh.detid().rawId());

  edm::Handle<HBHERecHitCollection> hHBHE;
  event.getByToken(hbheToken_, hHBHE);
  if (hHBHE.isValid())
    for (auto const& rh : *hHBHE)
      hcal.insert(rh.id().rawId());

  edm::LogPrint("CaloMatch") << "=== event " << event.id().event() << ": reco rechits  HGCAL=" << hgcal.size()
                             << "  ECAL(EB)=" << ecal.size() << "  HCAL(HBHE)=" << hcal.size();

  // Walk the flat Calo-channel storage: every (particle, cell) hit in the index.
  auto const& channel = hIndex->channel(truth::HitChannel::Calo);

  struct Stat {
    std::size_t cells = 0;
    std::size_t matchedCells = 0;
    double energy = 0.;
    double matchedEnergy = 0.;
  };
  std::map<std::string, Stat> stats;

  // Sum sim energy per cell first: the same cell appears once per contributing
  // particle, and the association denominator is the per-cell total.
  std::map<uint32_t, double> cellEnergy;
  for (auto const& hit : channel.directHits)
    cellEnergy[hit.detId] += hit.energy;

  for (auto const& [detId, energy] : cellEnergy) {
    const bool matched = hgcal.count(detId) || ecal.count(detId) || hcal.count(detId);
    auto& s = stats[caloName(detId)];
    ++s.cells;
    s.energy += energy;
    if (matched) {
      ++s.matchedCells;
      s.matchedEnergy += energy;
    }
  }

  for (auto const& [name, s] : stats) {
    const double cellFrac = s.cells ? 100.0 * double(s.matchedCells) / double(s.cells) : 0.0;
    const double eFrac = s.energy > 0. ? 100.0 * s.matchedEnergy / s.energy : 0.0;
    edm::LogPrint("CaloMatch") << "    " << name << " : cells " << s.matchedCells << "/" << s.cells << " (" << cellFrac
                               << "%)   simE " << s.matchedEnergy << "/" << s.energy << " (" << eFrac << "%)";
  }
}

void CaloRecHitMatchAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<edm::InputTag>("hgcalEE", edm::InputTag("HGCalRecHit", "HGCEERecHits"));
  desc.add<edm::InputTag>("hgcalHEF", edm::InputTag("HGCalRecHit", "HGCHEFRecHits"));
  desc.add<edm::InputTag>("hgcalHEB", edm::InputTag("HGCalRecHit", "HGCHEBRecHits"));
  desc.add<edm::InputTag>("ecalEB", edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("hcalHBHE", edm::InputTag("hbhereco"));
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CaloRecHitMatchAnalyzer);
