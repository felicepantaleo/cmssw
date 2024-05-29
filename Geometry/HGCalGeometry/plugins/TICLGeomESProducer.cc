#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/HGCalGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "CondFormats/HGCalObjects/interface/TICLGeom.h"
#include <map>
#include <memory>

class TICLGeomESProducer : public edm::ESProducer {
public:
  explicit TICLGeomESProducer(const edm::ParameterSet& p);
  ~TICLGeomESProducer() override = default;

  std::unique_ptr<TICLGeom> produce(const CaloGeometryRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
};

TICLGeomESProducer::TICLGeomESProducer(const edm::ParameterSet& p)
    : geomToken_{setWhatProduced(this).consumes<CaloGeometry>(edm::ESInputTag{""})} {}

std::unique_ptr<TICLGeom> TICLGeomESProducer::produce(const CaloGeometryRecord& iRecord) {
  const auto& geom = iRecord.get(geomToken_);
  auto validIds = geom.getValidDetIds();
  auto nValidIds = validIds.size();
  // Create an instance of TICLGeom with the size of validIds
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
  descriptions.add("TICLGeomESProducer", desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(TICLGeomESProducer);

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(TICLGeom);