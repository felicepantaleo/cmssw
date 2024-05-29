#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/HGCalObjects/interface/TICLGeom.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

class TICLGeomAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit TICLGeomAnalyzer(const edm::ParameterSet&);
  ~TICLGeomAnalyzer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::ESGetToken<TICLGeom, CaloGeometryRecord> ticlGeomToken_;
};

TICLGeomAnalyzer::TICLGeomAnalyzer(const edm::ParameterSet& iConfig)
    : ticlGeomToken_(esConsumes<TICLGeom, CaloGeometryRecord>()) {}

void TICLGeomAnalyzer::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  auto const& ticlGeom = iSetup.getData(ticlGeomToken_);
  auto& ticlGeomView = ticlGeom.hostCollection->view();

  // Use ticlGeom as needed
  // For example, you can loop over detIdToIndexMap and print the values
  for (const auto& [detId, index] : ticlGeom.detIdToIndexMap) {
    std::cout << "DetId: " << detId << " -> Index: " << index << std::endl;
    // you can access the values in the hostCollection using the index
    std::cout << "x: " << ticlGeomView.x()[index] << std::endl;
    std::cout << "y: " << ticlGeomView.y()[index] << std::endl;
    std::cout << "z: " << ticlGeomView.z()[index] << std::endl;
    std::cout << "eta: " << ticlGeomView.eta()[index] << std::endl;
    std::cout << "phi: " << ticlGeomView.phi()[index] << std::endl;
  }
}

void TICLGeomAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // You can add default parameters here if needed
  descriptions.add("TICLGeomAnalyzer", desc);
}
// Define this as a plug-in
DEFINE_FWK_MODULE(TICLGeomAnalyzer);