// Author: Felice Pantaleo, Wahid Redjeb - felice.pantaleo@cern.ch, wahid.redjeb@cern.ch
// Date: 01/2023

// user include files

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "RecoHGCal/TICL/interface/commons.h"
#include "SimDataFormats/Track/interface/UniqueSimTrackId.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

// #include "TrackstersPCA.h"
#include <vector>
#include <map>
#include <iterator>
#include <algorithm>

using namespace ticl;

class SimTICLCandidatesProducer : public edm::stream::EDProducer<> {
public:
  explicit SimTICLCandidatesProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  std::string detector_;
  const bool doNose_ = false;

  const edm::EDGetTokenT<std::vector<SimCluster>> simclusters_token_;
  const edm::EDGetTokenT<std::vector<CaloParticle>> caloparticles_token_;
  const edm::EDGetTokenT<std::vector<Trackster>> simTrackstersToken_;
  const edm::EDGetTokenT<std::vector<TrackingParticle>> trackingParticleToken_;
  const edm::EDGetTokenT<std::vector<reco::Track>> recoTracksToken_;
  const StringCutObjectSelector<reco::Track> cutTk_;

  const edm::EDGetTokenT<reco::SimToRecoCollection> associatormapStRsToken_;
  const edm::EDGetTokenT<reco::RecoToSimCollection> associatormapRtSsToken_;
  const edm::EDGetTokenT<SimTrackToTPMap> associationSimTrackToTPToken_;
};
DEFINE_FWK_MODULE(SimTICLCandidatesProducer);

SimTICLCandidatesProducer::SimTICLCandidatesProducer(const edm::ParameterSet& ps)
    : detector_(ps.getParameter<std::string>("detector")),
      doNose_(detector_ == "HFNose"),
      simclusters_token_(consumes(ps.getParameter<edm::InputTag>("simclusters"))),
      caloparticles_token_(consumes(ps.getParameter<edm::InputTag>("caloparticles"))),
      simTrackstersToken_(consumes(ps.getParameter<edm::InputTag>("simTracksters"))),
      trackingParticleToken_(
          consumes<std::vector<TrackingParticle>>(ps.getParameter<edm::InputTag>("trackingParticles"))),
      recoTracksToken_(consumes<std::vector<reco::Track>>(ps.getParameter<edm::InputTag>("recoTracks"))),
      cutTk_(ps.getParameter<std::string>("cutTk")),
      associatormapStRsToken_(consumes(ps.getParameter<edm::InputTag>("tpToTrack"))),
      associatormapRtSsToken_(consumes(ps.getParameter<edm::InputTag>("trackToTp"))),
      associationSimTrackToTPToken_(consumes(ps.getParameter<edm::InputTag>("simTrackToTPMap"))) {
  produces<std::vector<TICLCandidate>>();
}

void SimTICLCandidatesProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCAL");
  desc.add<edm::InputTag>("recoTracks", edm::InputTag("generalTracks"));
  desc.add<std::string>("cutTk",
                        "1.48 < abs(eta) < 3.0 && pt > 1. && quality(\"highPurity\") && "
                        "hitPattern().numberOfLostHits(\"MISSING_OUTER_HITS\") < 5");
  desc.add<edm::InputTag>("tpToTrack", edm::InputTag("trackingParticleRecoTrackAsssociation"));
  desc.add<edm::InputTag>("trackToTp", edm::InputTag("trackingParticleRecoTrackAsssociation"));
  desc.add<edm::InputTag>("simclusters", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("caloparticles", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("simTracksters", edm::InputTag("ticlSimTracksters"));
  desc.add<edm::InputTag>("trackingParticles", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("simTrackToTPMap", edm::InputTag("simHitTPAssocProducer", "simTrackToTP"));
  descriptions.addWithDefaultLabel(desc);
}

void SimTICLCandidatesProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  auto result = std::make_unique<std::vector<TICLCandidate>>();

  edm::Handle<std::vector<CaloParticle>> caloParticles_h;
  evt.getByToken(caloparticles_token_, caloParticles_h);
  edm::Handle<std::vector<TrackingParticle>> trackingParticles_h;
  evt.getByToken(trackingParticleToken_, trackingParticles_h);
  edm::Handle<std::vector<reco::Track>> recoTracks_h;
  evt.getByToken(recoTracksToken_, recoTracks_h);
  edm::Handle<std::vector<Trackster>> simTracksters_h;
  evt.getByToken(simTrackstersToken_, simTracksters_h);
  const auto& simClusters = evt.get(simclusters_token_);
  const auto& caloParticles = *caloParticles_h;
  const auto& simTracksters = *simTracksters_h;
  const auto& trackingParticles = *trackingParticles_h;
  const auto& TPtoRecoTrackMap = evt.get(associatormapStRsToken_);
  const auto& simTrackToTPMap = evt.get(associationSimTrackToTPToken_);
  const auto& recoTracks = *recoTracks_h;
  result->reserve(simTracksters.size());

  std::vector<int> usedSimTrackster(simTracksters.size(), 0);

  // Creating the map from TrackingParticle to SimTrackster
  std::unordered_map<unsigned int, std::vector<unsigned int>> TPtoSimTracksterMap;
  for (unsigned int i = 0; i < simTracksters.size(); ++i) {
    const auto& simTrack = (simTracksters[i].seedID() == caloParticles_h.id())
                               ? caloParticles[simTracksters[i].seedIndex()].g4Tracks()[0]
                               : simClusters[simTracksters[i].seedIndex()].g4Tracks()[0];
    UniqueSimTrackId simTkIds(simTrack.trackId(), simTrack.eventId());
    auto ipos = simTrackToTPMap.mapping.find(simTkIds);
    if (ipos != simTrackToTPMap.mapping.end()) {
      auto tpIdx = (ipos->second).get() - (edm::Ref<std::vector<TrackingParticle>>(trackingParticles_h, 0)).get();
      TPtoSimTracksterMap[tpIdx].push_back(i);
    }
  }

  for (const auto& [tpKey, associatedRecoTracks] : TPtoRecoTrackMap) {
    auto const& tp = *(tpKey);
    auto tpIndex = &tp - &trackingParticles[0];

    if (!associatedRecoTracks.empty()) {
      if (associatedRecoTracks[0].second > 0.75f) {
        auto const& track = associatedRecoTracks[0].first;
        auto trackIndex = &(*track) - &recoTracks[0];

        if (cutTk_(*track)) {
          if (!TPtoSimTracksterMap[tpIndex].empty()) {
            TICLCandidate tmpCand;
            tmpCand.zeroProbabilities();
            auto const particleType = tracksterParticleTypeFromPdgId(tp.pdgId(), 1);
            tmpCand.setIdProbability(particleType, 1.f);
            tmpCand.setTrackPtr(edm::Ptr<reco::Track>(recoTracks_h, trackIndex));
            tmpCand.setPdgId(tp.pdgId());
            tmpCand.setCharge(tp.charge());
            float rawEnergy = 0.f;
            float regressedEnergy = 0.f;
            for (auto simTracksterIndex : TPtoSimTracksterMap[tpIndex]) {
              if (usedSimTrackster[simTracksterIndex] == 0) {
                usedSimTrackster[simTracksterIndex] = 1;
                rawEnergy += simTracksters[simTracksterIndex].raw_energy();
                regressedEnergy += simTracksters[simTracksterIndex].regressed_energy();
                tmpCand.addTrackster(edm::Ptr<Trackster>(simTracksters_h, simTracksterIndex));
              }
            }
            math::XYZTLorentzVector p4(regressedEnergy * track->momentum().unit().x(),
                                       regressedEnergy * track->momentum().unit().y(),
                                       regressedEnergy * track->momentum().unit().z(),
                                       regressedEnergy);
            tmpCand.setP4(p4);
            tmpCand.setRawEnergy(rawEnergy);
            result->push_back(tmpCand);
          }
        }
      }
    }
  }

  for (unsigned int i = 0; i < simTracksters.size(); ++i) {
    if (usedSimTrackster[i] == 0) {
      usedSimTrackster[i] = 1;
      TICLCandidate tmpCand;
      const auto& simTrackster = simTracksters[i];
      tmpCand.setIdProbabilities(simTrackster.id_probabilities());
      const auto pdgId = (simTracksters[i].seedID() == caloParticles_h.id())
                             ? caloParticles[simTrackster.seedIndex()].pdgId()
                             : simClusters[simTrackster.seedIndex()].pdgId();
      tmpCand.setPdgId(pdgId);
      tmpCand.setCharge(0);
      tmpCand.setRawEnergy(simTrackster.raw_energy());
      float regressedEnergy = simTrackster.regressed_energy();
      math::XYZTLorentzVector p4(regressedEnergy * simTrackster.barycenter().unit().x(),
                                 regressedEnergy * simTrackster.barycenter().unit().y(),
                                 regressedEnergy * simTrackster.barycenter().unit().z(),
                                 regressedEnergy);
      tmpCand.setP4(p4);
      result->push_back(tmpCand);
    }
  }

  result->shrink_to_fit();

  evt.put(std::move(result));
}
