// Convert a legacy std::vector<ticl::Trackster> into the portable TracksterSoA.
//
// Generic by construction: it is configured by InputTag only, so one module type
// serves ticlTrackstersCLUE3DHigh, ticlTracksterLinks, ticlSimTracksters or any
// other std::vector<Trackster> product. It is the entry point of every Alpaka
// TICL step: an Alpaka consumer declares a device::EDGetToken against the host
// product produced here and the framework performs the host to device copy
// through the PortableHostCollection CopyToDevice specialisation, so no explicit
// memcpy is needed anywhere downstream.
//
// eta, phi and rawPt are materialised here, once per event on host, rather than
// being recomputed inside device inner loops.

#include <vector>

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TracksterHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

class LegacyTracksterToSoAProducer : public edm::global::EDProducer<> {
public:
  explicit LegacyTracksterToSoAProducer(const edm::ParameterSet &ps)
      : trackstersToken_(consumes<std::vector<ticl::Trackster>>(ps.getParameter<edm::InputTag>("tracksters"))),
        soaToken_(produces<TracksterHostCollection>()) {}

  void produce(edm::StreamID, edm::Event &evt, const edm::EventSetup &) const override {
    const auto &tracksters = evt.get(trackstersToken_);
    const int32_t nTracksters = static_cast<int32_t>(tracksters.size());

    TracksterHostCollection soa{cms::alpakatools::host(), nTracksters};
    auto view = soa.view();

    uint32_t verticesOffset = 0;
    for (int32_t i = 0; i < nTracksters; ++i) {
      const auto &ts = tracksters[i];
      const auto &bary = ts.barycenter();
      const auto &axis = ts.eigenvectors(0);
      const auto &eigenvalues = ts.eigenvalues();
      const auto &sigmas = ts.sigmasPCA();
      auto element = view[i];

      element.baryX() = bary.x();
      element.baryY() = bary.y();
      element.baryZ() = bary.z();

      element.axisX() = axis.x();
      element.axisY() = axis.y();
      element.axisZ() = axis.z();

      element.eta() = bary.eta();
      element.phi() = bary.phi();

      element.rawEnergy() = ts.raw_energy();
      element.regressedEnergy() = ts.regressed_energy();
      element.rawPt() = ts.raw_pt();

      element.time() = ts.time();
      element.timeError() = ts.timeError();

      element.eigenvalue0() = eigenvalues[0];
      element.eigenvalue1() = eigenvalues[1];
      element.eigenvalue2() = eigenvalues[2];
      element.sigmaPCA0() = sigmas[0];
      element.sigmaPCA1() = sigmas[1];
      element.sigmaPCA2() = sigmas[2];

      // Filled even though the companion vertices collection is not produced yet:
      // the range is what makes the SoA self-describing, and it costs nothing.
      const uint32_t nVertices = static_cast<uint32_t>(ts.vertices().size());
      element.verticesOffset() = verticesOffset;
      element.nVertices() = nVertices;
      verticesOffset += nVertices;
    }

    evt.emplace(soaToken_, std::move(soa));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersCLUE3DHigh"))
        ->setComment("Any std::vector<ticl::Trackster> collection to expose as a portable SoA.");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> trackstersToken_;
  const edm::EDPutTokenT<TracksterHostCollection> soaToken_;
};

DEFINE_FWK_MODULE(LegacyTracksterToSoAProducer);
