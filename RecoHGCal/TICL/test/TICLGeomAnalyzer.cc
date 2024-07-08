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
  std::string label_;

  edm::ESGetToken<TICLGeom, CaloGeometryRecord> ticlGeomToken_;
};

TICLGeomAnalyzer::TICLGeomAnalyzer(const edm::ParameterSet& iConfig)
    : label_(iConfig.getParameter<std::string>("label")), 
    ticlGeomToken_(esConsumes<TICLGeom, CaloGeometryRecord>(edm::ESInputTag("",label_))) {}

void TICLGeomAnalyzer::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  auto const& ticlGeom = iSetup.getData(ticlGeomToken_);
  auto& ticlGeomView = ticlGeom.hostCollection->view();

  // Use ticlGeom as needed
  // For example, you can loop over detIdToIndexMap and print the values

    for (const auto& [detId, index] : ticlGeom.detIdToIndexMap) {
      std::cout << label_ << "\t DetId: " << detId << "\tIndex: " << index << "\tx: " << ticlGeomView.x()[index] << "\ty: " << ticlGeomView.y()[index] << "\tz: " << ticlGeomView.z()[index] << "\teta: " << ticlGeomView.eta()[index] << "\tphi " << ticlGeomView.phi()[index] << std::endl;
    }
  
}

void TICLGeomAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("label", "all");
  descriptions.add("TICLGeomAnalyzer", desc);
}
// Define this as a plug-in
DEFINE_FWK_MODULE(TICLGeomAnalyzer);