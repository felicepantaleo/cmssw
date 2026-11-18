#include <Eigen/Core>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "DataFormats/HGCalReco/test/HGCalRecoSoATestValues.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edmtest {

  // Reads the HGCalReco SoA host collections named in 'products' and throws if one differs from the known values of
  // TestWriteHGCalRecoSoA.
  class TestReadHGCalRecoSoA : public edm::global::EDAnalyzer<> {
  public:
    explicit TestReadHGCalRecoSoA(edm::ParameterSet const& config) {
      const auto source = config.getParameter<edm::InputTag>("source");
      const auto products = config.getParameter<std::vector<std::string>>("products");
      const auto has = [&](const char* name) { return std::ranges::find(products, name) != products.end(); };
      if (has("extra"))
        extraToken_ = consumes(source);
      if (has("mask"))
        maskToken_ = consumes(source);
      if (has("tracksters"))
        trackstersToken_ = consumes(source);
      if (has("links"))
        linksToken_ = consumes(source);
      if (has("pid"))
        pidToken_ = consumes(source);
    }

    void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const override {
      const auto n = static_cast<int32_t>((event.id().event() - 1) % 4);
      if (not extraToken_.isUninitialized())
        hgcalrecotest::check(event.get(extraToken_).const_view(), n);
      if (not maskToken_.isUninitialized())
        hgcalrecotest::check(event.get(maskToken_).const_view(), n);
      if (not trackstersToken_.isUninitialized())
        hgcalrecotest::check(event.get(trackstersToken_).const_view(), n);
      if (not linksToken_.isUninitialized())
        hgcalrecotest::check(event.get(linksToken_).const_view(), n);
      if (not pidToken_.isUninitialized())
        hgcalrecotest::check(event.get(pidToken_).const_view(), n);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("source", edm::InputTag("writer"));
      desc.add<std::vector<std::string>>("products", {"extra", "mask", "tracksters", "links", "pid"})
          ->setComment("The collections to check: extra, mask, tracksters, links, pid.");
      descriptions.addDefault(desc);
    }

  private:
    edm::EDGetTokenT<HGCalSoALayerClustersExtraHostCollection> extraToken_;
    edm::EDGetTokenT<ticl::LayerClusterMaskHostCollection> maskToken_;
    edm::EDGetTokenT<ticl::TracksterHostCollection> trackstersToken_;
    edm::EDGetTokenT<ticl::TracksterLinksHostCollection> linksToken_;
    edm::EDGetTokenT<ticl::TracksterPidHostCollection> pidToken_;
  };

}  // namespace edmtest

using edmtest::TestReadHGCalRecoSoA;
DEFINE_FWK_MODULE(TestReadHGCalRecoSoA);
