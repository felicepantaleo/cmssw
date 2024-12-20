// Author: Felice Pantaleo, Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 09/2018

// user include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"

#include "TracksterFilterFactory.h"
#include "TracksterFilterBase.h"

#include <string>

class FilteredTrackstersProducer : public edm::stream::EDProducer<> {
public:
  FilteredTrackstersProducer(const edm::ParameterSet&);
  ~FilteredTrackstersProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksters_token_;
  edm::EDGetTokenT<std::vector<float>> trackstersMask_token_;
  std::string clusterFilter_;
  std::string iteration_label_;
  std::unique_ptr<const ticl::TracksterFilterBase> theFilter_;
};

DEFINE_FWK_MODULE(FilteredTrackstersProducer);

FilteredTrackstersProducer::FilteredTrackstersProducer(const edm::ParameterSet& ps) {
  tracksters_token_ = consumes<std::vector<ticl::Trackster>>(ps.getParameter<edm::InputTag>("Tracksters"));
  trackstersMask_token_ = consumes<std::vector<float>>(ps.getParameter<edm::InputTag>("TrackstersInputMask"));
  clusterFilter_ = ps.getParameter<std::string>("clusterFilter");
  theFilter_ = TracksterFilterFactory::get()->create(clusterFilter_, ps);
  iteration_label_ = ps.getParameter<std::string>("iteration_label");
  produces<std::vector<float>>(iteration_label_);
}

void FilteredTrackstersProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("Tracksters", edm::InputTag("ticlTrackstersCLUE3DHigh"));
  desc.add<edm::InputTag>("TrackstersInputMask",
                          edm::InputTag("ticlTrackstersCLUE3DHigh", "InitialTrackstersMask"));
  desc.add<std::string>("clusterFilter", "TracksterFilterBySize");
  desc.add<int>("min_num_layerclusters", 0);
  desc.add<int>("max_num_layerclusters", 9999);
  descriptions.add("filteredTrackstersProducer", desc);
}

void FilteredTrackstersProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<std::vector<ticl::Trackster>> trackstersHandle;
  edm::Handle<std::vector<float>> inputTrackstersMaskHandle;
  evt.getByToken(tracksters_token_, trackstersHandle);
  evt.getByToken(trackstersMask_token_, inputTrackstersMaskHandle);
  const auto& inputTrackstersMask = *inputTrackstersMaskHandle;

  // Transfer input mask in output
  auto trackstersMask = std::make_unique<std::vector<float>>(inputTrackstersMask);

  const auto& tracksters = *trackstersHandle;
  if (theFilter_) {
    theFilter_->filter(tracksters, *trackstersMask);
  }

  evt.put(std::move(trackstersMask), iteration_label_);
}
