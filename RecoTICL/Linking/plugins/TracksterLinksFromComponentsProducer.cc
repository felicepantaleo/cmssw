// Host tail of the device Cornetto: turn per-trackster component labels into the
// legacy linking products.
//
// This is the tail of TracksterLinkingByCornetto::linkTracksters (the merge loop),
// unchanged: group by label, mergeTracksters, emit EVERY component including
// singletons. It rebuilds the SAME edm::MultiSpan<Trackster> the SoA producer
// flattened (same tracksters_collections, same order), so the component labels
// (which are SoA row indices = MultiSpan global indices) resolve back to the right
// tracksters, and the merge uses the MultiSpan overload just like the CPU plugin.
// Keeping this on host means the emitted products are structurally identical to
// the CPU plugin's, which is what makes the backend comparison a plain diff.

#include <vector>

#include "DataFormats/Common/interface/MultiSpan.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TracksterComponentsHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"

class TracksterLinksFromComponentsProducer : public edm::global::EDProducer<> {
public:
  explicit TracksterLinksFromComponentsProducer(const edm::ParameterSet &ps)
      : componentsToken_(consumes<TracksterComponentsHostCollection>(ps.getParameter<edm::InputTag>("components"))),
        tracksterToken_(produces<std::vector<ticl::Trackster>>()),
        linksToken_(produces<std::vector<std::vector<unsigned int>>>("linkedTracksterIdToInputTracksterId")) {
    for (auto const &tag : ps.getParameter<std::vector<edm::InputTag>>("tracksters_collections"))
      trackstersTokens_.push_back(consumes<std::vector<ticl::Trackster>>(tag));
  }

  void produce(edm::StreamID, edm::Event &evt, const edm::EventSetup &) const override {
    edm::MultiSpan<ticl::Trackster> tracksters;
    for (const auto &token : trackstersTokens_)
      tracksters.add(evt.get(token));

    const auto view = evt.get(componentsToken_).const_view();
    const int32_t n = static_cast<int32_t>(tracksters.size());
    if (view.metadata().size() != n) {
      throw edm::Exception(edm::errors::LogicError)
          << "component label count " << view.metadata().size() << " != trackster count " << n;
    }

    // components[r] collects the members whose label is r. The label is the
    // smallest input index in the component, so r is itself a member and the
    // per-component member lists come out in increasing index order.
    std::vector<std::vector<unsigned int>> components(n);
    for (int32_t i = 0; i < n; ++i) {
      const int32_t r = view[i].label();
      if (r < 0 or r >= n) {
        throw edm::Exception(edm::errors::LogicError) << "trackster " << i << " has out-of-range label " << r;
      }
      components[r].push_back(static_cast<unsigned int>(i));
    }

    auto resultTracksters = std::make_unique<std::vector<ticl::Trackster>>();
    auto linkedTracksterIdToInputTracksterId = std::make_unique<std::vector<std::vector<unsigned int>>>();
    for (int32_t r = 0; r < n; ++r) {
      if (components[r].empty())
        continue;
      ticl::Trackster merged;
      merged.mergeTracksters(tracksters, components[r]);
      resultTracksters->push_back(std::move(merged));
      linkedTracksterIdToInputTracksterId->push_back(components[r]);
    }

    evt.put(tracksterToken_, std::move(resultTracksters));
    evt.put(linksToken_, std::move(linkedTracksterIdToInputTracksterId));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<edm::InputTag>>("tracksters_collections", {edm::InputTag("ticlTrackstersCLUE3DHigh")})
        ->setComment("Same collections, same order, as the SoA producer; the merge reads their layer clusters.");
    desc.add<edm::InputTag>("components", edm::InputTag("ticlCornettoLinksAlpaka"))
        ->setComment("Per-trackster component labels from the Cornetto device producer.");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  std::vector<edm::EDGetTokenT<std::vector<ticl::Trackster>>> trackstersTokens_;
  const edm::EDGetTokenT<TracksterComponentsHostCollection> componentsToken_;
  const edm::EDPutTokenT<std::vector<ticl::Trackster>> tracksterToken_;
  const edm::EDPutTokenT<std::vector<std::vector<unsigned int>>> linksToken_;
};

DEFINE_FWK_MODULE(TracksterLinksFromComponentsProducer);
