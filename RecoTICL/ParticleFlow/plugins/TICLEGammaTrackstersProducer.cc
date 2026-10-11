// The final tracksters of the interpretation stage that it types as electrons or as EM neutrals, the input of the HGCAL
// PF clusters of the e/gamma objects, and their indices in the final tracksters (interpretationIndex).

#include <string>
#include <utility>
#include <vector>

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTICL/Interpretation/interface/AssignmentMaps.h"

class TICLEGammaTrackstersProducer : public edm::global::EDProducer<> {
public:
  explicit TICLEGammaTrackstersProducer(const edm::ParameterSet &ps);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void produce(edm::StreamID, edm::Event &evt, const edm::EventSetup &) const override;

private:
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksters_token_;
  const edm::EDGetTokenT<std::vector<int>> trackToTrackster_token_;
  const edm::EDGetTokenT<std::vector<int>> trackMode_token_;
  const edm::EDGetTokenT<std::vector<int>> neutralIdx_token_;
  const edm::EDGetTokenT<std::vector<int>> neutralPdg_token_;
  const std::vector<int> neutralPdgs_;
  const edm::EDPutTokenT<std::vector<ticl::Trackster>> put_token_;
  const edm::EDPutTokenT<std::vector<int>> index_put_token_;
};

namespace {
  // InputTag of a product instance of the interpretation stage.
  edm::InputTag interpretationTag(const edm::ParameterSet &ps, const std::string &instance) {
    const auto &tag = ps.getParameter<edm::InputTag>("interpretations");
    return edm::InputTag(tag.label(), instance, tag.process());
  }
}  // namespace

TICLEGammaTrackstersProducer::TICLEGammaTrackstersProducer(const edm::ParameterSet &ps)
    : tracksters_token_(consumes(ps.getParameter<edm::InputTag>("interpretations"))),
      trackToTrackster_token_(consumes(interpretationTag(ps, "trackToTrackster"))),
      trackMode_token_(consumes(interpretationTag(ps, "trackMode"))),
      neutralIdx_token_(consumes(interpretationTag(ps, "neutralIdx"))),
      neutralPdg_token_(consumes(interpretationTag(ps, "neutralPdg"))),
      neutralPdgs_(ps.getParameter<std::vector<int>>("neutralPdgs")),
      put_token_(produces()),
      index_put_token_(produces<std::vector<int>>("interpretationIndex")) {}

void TICLEGammaTrackstersProducer::produce(edm::StreamID, edm::Event &evt, const edm::EventSetup &) const {
  const auto &tracksters = evt.get(tracksters_token_);
  const auto &trackToTrackster = evt.get(trackToTrackster_token_);
  const auto &trackMode = evt.get(trackMode_token_);
  const auto &neutralIdx = evt.get(neutralIdx_token_);
  const auto &neutralPdg = evt.get(neutralPdg_token_);
  if (trackToTrackster.size() != trackMode.size() || neutralIdx.size() != neutralPdg.size())
    throw cms::Exception("LogicError") << "TICLEGammaTrackstersProducer: the interpretation maps differ in size";
  auto indices = ticl::eGammaTracksterIndices(trackToTrackster, trackMode, neutralIdx, neutralPdg, neutralPdgs_);
  std::vector<ticl::Trackster> result;
  result.reserve(indices.size());
  for (int i : indices) {
    if (i >= static_cast<int>(tracksters.size()))
      throw cms::Exception("LogicError") << "TICLEGammaTrackstersProducer: trackster index " << i << " out of range";
    result.push_back(tracksters[i]);
  }
  evt.emplace(put_token_, std::move(result));
  evt.emplace(index_put_token_, std::move(indices));
}

void TICLEGammaTrackstersProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("interpretations", edm::InputTag("ticlTracksterInterpretations"))
      ->setComment("Final tracksters of the interpretation stage; its maps are the instances of the same label.");
  desc.add<std::vector<int>>("neutralPdgs", {22, 111})
      ->setComment("Pdg ids of the neutral types that are selected, in addition to the electrons.");
  descriptions.add("ticlEGammaTrackstersProducer", desc);
}

DEFINE_FWK_MODULE(TICLEGammaTrackstersProducer);
