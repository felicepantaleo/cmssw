#include <Eigen/Core>

#include <cstdint>

#include "DataFormats/HGCalReco/test/HGCalRecoSoATestValues.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

namespace edmtest {

  // Puts the HGCalReco SoA host collections with the known values of the tests: event e has (e - 1) % 4 elements.
  class TestWriteHGCalRecoSoA : public edm::global::EDProducer<> {
  public:
    explicit TestWriteHGCalRecoSoA(edm::ParameterSet const&)
        : extraToken_(produces()),
          maskToken_(produces()),
          trackstersToken_(produces()),
          linksToken_(produces()),
          pidToken_(produces()) {}

    void produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const override {
      const auto n = static_cast<int32_t>((event.id().event() - 1) % 4);

      HGCalSoALayerClustersExtraHostCollection extra(n);
      hgcalrecotest::fill(extra.view());
      event.emplace(extraToken_, std::move(extra));

      ticl::LayerClusterMaskHostCollection mask(n);
      hgcalrecotest::fill(mask.view());
      event.emplace(maskToken_, std::move(mask));

      auto tracksters = hgcalrecotest::makeTracksters(n);
      hgcalrecotest::fill(tracksters.view());
      event.emplace(trackstersToken_, std::move(tracksters));

      auto links = hgcalrecotest::makeLinks(n);
      hgcalrecotest::fill(links.view());
      event.emplace(linksToken_, std::move(links));

      ticl::TracksterPidHostCollection pid(n);
      hgcalrecotest::fill(pid.view());
      event.emplace(pidToken_, std::move(pid));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addDefault(desc);
    }

  private:
    const edm::EDPutTokenT<HGCalSoALayerClustersExtraHostCollection> extraToken_;
    const edm::EDPutTokenT<ticl::LayerClusterMaskHostCollection> maskToken_;
    const edm::EDPutTokenT<ticl::TracksterHostCollection> trackstersToken_;
    const edm::EDPutTokenT<ticl::TracksterLinksHostCollection> linksToken_;
    const edm::EDPutTokenT<ticl::TracksterPidHostCollection> pidToken_;
  };

}  // namespace edmtest

using edmtest::TestWriteHGCalRecoSoA;
DEFINE_FWK_MODULE(TestWriteHGCalRecoSoA);
