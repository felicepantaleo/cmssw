#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "CondFormats/HGCalObjects/interface/TICLGeom.h"
#include <map>
#include <memory>
#include <vector>

class TICLGeomESProducer : public edm::ESProducer {
public:
  explicit TICLGeomESProducer(const edm::ParameterSet& p);
  ~TICLGeomESProducer() override = default;

  std::unique_ptr<TICLGeom> produce(const CaloGeometryRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  std::vector<std::string> detectors_;
  std::string myLabel_;
};

TICLGeomESProducer::TICLGeomESProducer(const edm::ParameterSet& p)
    : detectors_(p.getParameter<std::vector<std::string>>("detectors")) {
  myLabel_ = p.getParameter<std::string>("label");
  geomToken_ = setWhatProduced(this, myLabel_).consumes<CaloGeometry>(edm::ESInputTag{""});
}

std::unique_ptr<TICLGeom> TICLGeomESProducer::produce(const CaloGeometryRecord& iRecord) {
  const auto& geom = iRecord.get(geomToken_);

  // Map of detector names to pair of DetId::Detector and subdet id
  std::map<std::string, std::pair<int, int>> detMap = {{"EB", {3, 1}},
                                                       {"EE", {3, 2}},
                                                       {"ES", {3, 3}},
                                                       {"HB", {4, 1}},
                                                       {"HE", {4, 2}},
                                                       {"HF", {4, 4}},
                                                       {"HO", {4, 3}},
                                                       {"HGCEE", {8, 0}},
                                                       {"HGCHESil", {9, 0}},
                                                       {"HGCHESci", {10, 0}},
                                                       {"HFNose", {6, 6}}};

  std::map<std::string, std::vector<std::string>> detGroups = {{"ECAL", {"EB", "EE", "ES"}},
                                                               {"HCAL", {"HB", "HE", "HF", "HO"}},
                                                               {"HGCal", {"HGCEE", "HGCHESil", "HGCHESci"}},
                                                               {"HFNose", {"HFNose"}}};

  std::vector<DetId> validIds;

  for (const auto& group : detectors_) {
    // Check if the group is in the map
    // If the group is not in the map, search for the detector in the map
    if (detGroups.find(group) != detGroups.end()) {
      for (const auto& det : detGroups[group]) {
        const auto& ids = geom.getValidDetIds((DetId::Detector)(detMap[det].first), detMap[det].second);
        validIds.insert(validIds.end(), ids.begin(), ids.end());
      }
    } else {
      if (detMap.find(group) == detMap.end()) {
        throw cms::Exception("TICLGeomInvalidDetector") << "Detector " << group << " is not a valid detector name";
      }
      const auto& ids = geom.getValidDetIds((DetId::Detector)(detMap[group].first), detMap[group].second);
      validIds.insert(validIds.end(), ids.begin(), ids.end());
    }
  }

  auto nValidIds = validIds.size();
  auto ticlGeom = std::make_unique<TICLGeom>(nValidIds);
  auto& ticlGeomView = ticlGeom->hostCollection->view();

  for (unsigned int i = 0; i < nValidIds; i++) {
    auto id = validIds[i];
    ticlGeom->detIdToIndexMap.emplace(id.rawId(), i);
    auto pos = geom.getPosition(id);
    ticlGeomView.rawDetId()[i] = id.rawId();
    ticlGeomView.x()[i] = pos.x();
    ticlGeomView.y()[i] = pos.y();
    ticlGeomView.z()[i] = pos.z();
    ticlGeomView.eta()[i] = pos.eta();
    ticlGeomView.phi()[i] = pos.phi();
  }

  return ticlGeom;
}

void TICLGeomESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<std::string>>("detectors", {"ECAL", "HCAL", "HGCal", "HFNose"})
      ->setComment(
          "List of detectors or subdetectors to include in the TICL geometry (valid options: ECAL, HCAL, HGCal, "
          "HFNose, EB, EE, ES, HB, HE, HF, HO, HGCEE, HGCHESil, HGCHESci)");
  desc.add<std::string>("label", "all");
  descriptions.add("TICLGeomESProducer", desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(TICLGeomESProducer);

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(TICLGeom);
