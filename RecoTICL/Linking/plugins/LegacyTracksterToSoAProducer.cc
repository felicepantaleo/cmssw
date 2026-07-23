// Flatten one or more legacy std::vector<ticl::Trackster> collections into a
// single portable TracksterSoA.
//
// The concatenation uses edm::MultiSpan<Trackster>, built in the SAME order and
// with the SAME empty-span skipping as TracksterLinksProducer (the CPU linking
// driver). That is what makes this faithful: the CPU TracksterLinkingAlgoBase
// links over exactly this MultiSpan, so its global index is the concatenated
// index space, and by filling SoA row i from multiSpan[i] the SoA row index IS
// that global index. A device linker then sees the same index space the CPU
// plugin does, and the component labels are directly comparable.
//
// (When cms-sw/cmssw PR #51458 lands an SoA row-concatenation view, this producer
// can consume the device collections directly and drop the host flatten; until
// then MultiSpan is the concatenation primitive and it is host-only.)
//
// Configured by VInputTag only, so one module type serves any set of trackster
// collections. eta/phi/rawPt are materialised here, once per event on host.

#include <vector>

#include "DataFormats/Common/interface/MultiSpan.h"
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
      : soaToken_(produces<TracksterHostCollection>()) {
    for (auto const &tag : ps.getParameter<std::vector<edm::InputTag>>("tracksters_collections"))
      trackstersTokens_.push_back(consumes<std::vector<ticl::Trackster>>(tag));
  }

  void produce(edm::StreamID, edm::Event &evt, const edm::EventSetup &) const override {
    // Build the MultiSpan exactly as TracksterLinksProducer does: add each
    // collection in configuration order; edm::MultiSpan::add skips empty spans.
    edm::MultiSpan<ticl::Trackster> tracksters;
    for (const auto &token : trackstersTokens_)
      tracksters.add(evt.get(token));

    const int32_t n = static_cast<int32_t>(tracksters.size());
    TracksterHostCollection soa{cms::alpakatools::host(), n};
    auto view = soa.view();

    uint32_t verticesOffset = 0;
    for (int32_t i = 0; i < n; ++i) {
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

      const uint32_t nVertices = static_cast<uint32_t>(ts.vertices().size());
      element.verticesOffset() = verticesOffset;
      element.nVertices() = nVertices;
      verticesOffset += nVertices;
    }

    evt.emplace(soaToken_, std::move(soa));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<edm::InputTag>>("tracksters_collections", {edm::InputTag("ticlTrackstersCLUE3DHigh")})
        ->setComment("Trackster collections to concatenate (in order) into one portable SoA.");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  std::vector<edm::EDGetTokenT<std::vector<ticl::Trackster>>> trackstersTokens_;
  const edm::EDPutTokenT<TracksterHostCollection> soaToken_;
};

DEFINE_FWK_MODULE(LegacyTracksterToSoAProducer);
