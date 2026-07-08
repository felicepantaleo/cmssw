// Author: Felice Pantaleo, Wahid Redjeb (CERN) - felice.pantaleo@cern.ch, wahid.redjeb@cern.ch
// Date: 12/2023
#include <memory>  // unique_ptr
#include "DataFormats/Common/interface/MultiSpan.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoHGCal/TICL/interface/TICLONNXGlobalCache.h"
#include "RecoHGCal/TICL/interface/TICLInterpretationAlgoBase.h"
#include "RecoHGCal/TICL/plugins/TICLInterpretationPluginFactory.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/HGCalReco/interface/MtdHostCollection.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include <numeric>

#include "DataFormats/Common/interface/OrphanHandle.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include "RecoHGCal/TICL/interface/TracksterLinkingAlgoBase.h"
#include "RecoHGCal/TICL/plugins/TracksterLinkingPluginFactory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoFactory.h"

#include "TrackstersPCA.h"

using namespace ticl;
using cms::Ort::ONNXRuntime;

class TracksterLinksProducer
    : public edm::stream::EDProducer<edm::GlobalCache<ticl::TICLONNXGlobalCache>, edm::stream::WatchRuns> {
public:
  explicit TracksterLinksProducer(const edm::ParameterSet &ps, const ticl::TICLONNXGlobalCache *cache);
  ~TracksterLinksProducer() override {};
  void produce(edm::Event &, const edm::EventSetup &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginRun(edm::Run const &iEvent, edm::EventSetup const &es) override;
  static std::unique_ptr<ticl::TICLONNXGlobalCache> initializeGlobalCache(const edm::ParameterSet &iConfig);
  static void globalEndJob(const ticl::TICLONNXGlobalCache *);

private:
  void printTrackstersDebug(const std::vector<Trackster> &, const char *label) const;
  void dumpTrackster(const Trackster &) const;

  std::unique_ptr<TracksterLinkingAlgoBase> linkingAlgo_;

  // Interpretation mode (runInterpretation): this instance hosts the track <->
  // trackster interpretations (masking passes or opinion arbitration) instead of a
  // linking algorithm, producing the final tracksters plus the per-track assignment
  // maps consumed by TICLCandidateProducer. This is what dissolves the GSF circular
  // dependency: PF clustering and GSF seeding depend only on this trackster module.
  void produceInterpretations(edm::Event &evt,
                              const edm::EventSetup &es,
                              const std::vector<reco::CaloCluster> &layerClusters,
                              const edm::ValueMap<std::pair<float, float>> &layerClustersTimes,
                              const edm::MultiSpan<Trackster> &trackstersManager,
                              std::unique_ptr<std::vector<Trackster>> resultTracksters,
                              std::unique_ptr<std::vector<float>> resultMask);
  std::string algoType_;

  std::vector<edm::EDGetTokenT<std::vector<Trackster>>> tracksters_tokens_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTime_token_;

  const bool regressionAndPid_;
  std::unique_ptr<TracksterInferenceAlgoBase> inferenceAlgo_;

  std::vector<edm::EDGetTokenT<std::vector<float>>> original_masks_tokens_;

  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;
  const std::string detector_;
  const std::string propName_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bfield_token_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagator_token_;
  const HGCalDDDConstants *hgcons_;
  hgcal::RecHitTools rhtools_;
  edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hdc_token_;

  const bool runInterpretation_;
  std::unique_ptr<TICLInterpretationAlgoBase<reco::Track>> generalInterpretationAlgo_;
  std::unique_ptr<TICLInterpretationAlgoBase<reco::Track>> muonInterpretationAlgo_;
  std::unique_ptr<TICLInterpretationAlgoBase<reco::Track>> egammaInterpretationAlgo_;
  std::unique_ptr<TICLInterpretationAlgoBase<reco::Track>> jetInterpretationAlgo_;
  std::vector<edm::EDGetTokenT<std::vector<Trackster>>> egamma_tracksters_tokens_;
  edm::EDGetTokenT<std::vector<reco::Track>> tracks_token_;
  edm::EDGetTokenT<std::vector<reco::Muon>> muons_token_;
  edm::EDGetTokenT<MtdHostCollection> inputTimingToken_;
  bool useMTDTiming_ = false;
  bool useArbitration_ = false;
  double arbitrationMaxSharedEnergyFraction_ = 0.2;
  const float tkEnergyCut_;
  const StringCutObjectSelector<reco::Track> cutTk_;
};

TracksterLinksProducer::TracksterLinksProducer(const edm::ParameterSet &ps, const ticl::TICLONNXGlobalCache *cache)
    : algoType_(ps.getParameter<edm::ParameterSet>("linkingPSet").getParameter<std::string>("type")),
      clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"))),
      clustersTime_token_(
          consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("layer_clustersTime"))),
      regressionAndPid_(ps.getParameter<bool>("regressionAndPid")),
      geometry_token_(esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>()),
      detector_(ps.getParameter<std::string>("detector")),
      propName_(ps.getParameter<std::string>("propagator")),
      bfield_token_(esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()),
      propagator_token_(
          esConsumes<Propagator, TrackingComponentsRecord, edm::Transition::BeginRun>(edm::ESInputTag("", propName_))),
      runInterpretation_(ps.getParameter<bool>("runInterpretation")),
      tkEnergyCut_(static_cast<float>(ps.getParameter<double>("tkEnergyCut"))),
      cutTk_(ps.getParameter<std::string>("cutTk")) {
  for (auto const &tag : ps.getParameter<std::vector<edm::InputTag>>("tracksters_collections")) {
    tracksters_tokens_.emplace_back(consumes<std::vector<Trackster>>(tag));
  }
  for (auto const &tag : ps.getParameter<std::vector<edm::InputTag>>("original_masks")) {
    original_masks_tokens_.emplace_back(consumes<std::vector<float>>(tag));
  }

  if (runInterpretation_) {
    for (auto const &tag : ps.getParameter<std::vector<edm::InputTag>>("egamma_tracksters_collections")) {
      egamma_tracksters_tokens_.emplace_back(consumes<std::vector<Trackster>>(tag));
    }
    tracks_token_ = consumes<std::vector<reco::Track>>(ps.getParameter<edm::InputTag>("tracks"));
    muons_token_ = consumes<std::vector<reco::Muon>>(ps.getParameter<edm::InputTag>("muons"));
    useMTDTiming_ = ps.getParameter<bool>("useMTDTiming");
    if (useMTDTiming_) {
      inputTimingToken_ = consumes<MtdHostCollection>(ps.getParameter<edm::InputTag>("timingSoA"));
    }
    useArbitration_ = ps.getParameter<bool>("useArbitration");
    arbitrationMaxSharedEnergyFraction_ = ps.getParameter<double>("arbitrationMaxSharedEnergyFraction");
    auto interpretationPSet = ps.getParameter<edm::ParameterSet>("interpretationDescPSet");
    generalInterpretationAlgo_ = TICLGeneralInterpretationPluginFactory::get()->create(
        interpretationPSet.getParameter<std::string>("type"), interpretationPSet, consumesCollector());
    auto muonInterpretationPSet = ps.getParameter<edm::ParameterSet>("muonInterpretationDescPSet");
    muonInterpretationAlgo_ = TICLGeneralInterpretationPluginFactory::get()->create(
        muonInterpretationPSet.getParameter<std::string>("type"), muonInterpretationPSet, consumesCollector());
    auto egammaInterpretationPSet = ps.getParameter<edm::ParameterSet>("egammaInterpretationDescPSet");
    egammaInterpretationAlgo_ = TICLGeneralInterpretationPluginFactory::get()->create(
        egammaInterpretationPSet.getParameter<std::string>("type"), egammaInterpretationPSet, consumesCollector());
    auto jetInterpretationPSet = ps.getParameter<edm::ParameterSet>("jetInterpretationDescPSet");
    jetInterpretationAlgo_ = TICLGeneralInterpretationPluginFactory::get()->create(
        jetInterpretationPSet.getParameter<std::string>("type"), jetInterpretationPSet, consumesCollector());
    produces<std::vector<int>>("trackToTrackster");
    produces<std::vector<int>>("trackMode");
    produces<std::vector<int>>("neutralIdx");
    produces<std::vector<int>>("neutralPdg");
  }
  produces<std::vector<Trackster>>();
  produces<std::vector<std::vector<unsigned int>>>();
  produces<std::vector<std::vector<unsigned int>>>("linkedTracksterIdToInputTracksterId");
  produces<std::vector<float>>();

  if (algoType_ == "Skeletons" || runInterpretation_) {
    std::string detectorName = (detector_ == "HFNose") ? "HGCalHFNoseSensitive" : "HGCalEESensitive";
    hdc_token_ = esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(
        edm::ESInputTag("", detectorName));
  }

  // Enforce presence of the superclustering DNN model when using a DNN-based linking plugin.
  // This fails fast at construction time, before any event processing.
  auto const linkingPSet = ps.getParameter<edm::ParameterSet>("linkingPSet");
  cms::Ort::ONNXRuntime const *linkingSession = nullptr;

  if (linkingPSet.existsAs<std::string>("onnxModelPath", true)) {
    auto const model = linkingPSet.getParameter<std::string>("onnxModelPath");
    linkingSession = cache ? cache->getByModelPathString(model) : nullptr;
  }

  linkingAlgo_ =
      TracksterLinkingPluginFactory::get()->create(algoType_, linkingPSet, consumesCollector(), linkingSession);

  // Initialize inference algorithm using the factory.
  // Do not build the inference plugin if it is disabled or if no model is configured (empty string => no session loaded).
  if (regressionAndPid_) {
    const std::string inferencePlugin = ps.getParameter<std::string>("inferenceAlgo");
    if (!inferencePlugin.empty()) {
      const edm::ParameterSet inferencePSet =
          ps.getParameter<edm::ParameterSet>("pluginInferenceAlgo" + inferencePlugin);

      // If the plugin config exposes model paths as std::string with default "",
      // the cache will only contain sessions for non-empty paths.
      const bool hasSingleModel = inferencePSet.existsAs<std::string>("onnxModelPath", true) &&
                                  !inferencePSet.getParameter<std::string>("onnxModelPath").empty();
      const bool hasPIDModel = inferencePSet.existsAs<std::string>("onnxPIDModelPath", true) &&
                               !inferencePSet.getParameter<std::string>("onnxPIDModelPath").empty();
      const bool hasEnergyModel = inferencePSet.existsAs<std::string>("onnxEnergyModelPath", true) &&
                                  !inferencePSet.getParameter<std::string>("onnxEnergyModelPath").empty();

      // Only instantiate the plugin if at least one model path is configured.
      if (hasSingleModel || hasPIDModel || hasEnergyModel) {
        inferenceAlgo_ = std::unique_ptr<TracksterInferenceAlgoBase>(
            TracksterInferenceAlgoFactory::get()->create(inferencePlugin, inferencePSet, cache));
      }
    }
  }
}

std::unique_ptr<ticl::TICLONNXGlobalCache> TracksterLinksProducer::initializeGlobalCache(
    const edm::ParameterSet &iConfig) {
  return ticl::TICLONNXGlobalCache::initialize(iConfig);
}

void TracksterLinksProducer::globalEndJob(const ticl::TICLONNXGlobalCache *) {}

void TracksterLinksProducer::beginRun(edm::Run const &iEvent, edm::EventSetup const &es) {
  if (algoType_ == "Skeletons" || runInterpretation_) {
    edm::ESHandle<HGCalDDDConstants> hdc = es.getHandle(hdc_token_);
    hgcons_ = hdc.product();
  }

  edm::ESHandle<CaloGeometry> geom = es.getHandle(geometry_token_);
  rhtools_.setGeometry(*geom);

  edm::ESHandle<MagneticField> bfield = es.getHandle(bfield_token_);
  edm::ESHandle<Propagator> propagator = es.getHandle(propagator_token_);

  linkingAlgo_->initialize(hgcons_, rhtools_, bfield, propagator);
  if (runInterpretation_) {
    generalInterpretationAlgo_->initialize(hgcons_, rhtools_, bfield, propagator);
    muonInterpretationAlgo_->initialize(hgcons_, rhtools_, bfield, propagator);
    egammaInterpretationAlgo_->initialize(hgcons_, rhtools_, bfield, propagator);
    jetInterpretationAlgo_->initialize(hgcons_, rhtools_, bfield, propagator);
  }
};

void TracksterLinksProducer::dumpTrackster(const Trackster &t) const {
  LogDebug("TracksterLinksProducer")
      << "\nTrackster raw_pt: " << t.raw_pt() << " raw_em_pt: " << t.raw_em_pt()
      << " eoh: " << (t.raw_em_pt() / ((t.raw_pt() - t.raw_em_pt()) != 0. ? (t.raw_pt() - t.raw_em_pt()) : 1.))
      << " barycenter: " << t.barycenter() << " eta,phi (baricenter): " << t.barycenter().eta() << ", "
      << t.barycenter().phi() << " eta,phi (eigen): " << t.eigenvectors(0).eta() << ", " << t.eigenvectors(0).phi()
      << " pt(eigen): " << std::sqrt(t.eigenvectors(0).Unit().perp2()) * t.raw_energy() << " seedID: " << t.seedID()
      << " seedIndex: " << t.seedIndex() << " size: " << t.vertices().size() << " average usage: "
      << (std::accumulate(std::begin(t.vertex_multiplicity()), std::end(t.vertex_multiplicity()), 0.) /
          (float)t.vertex_multiplicity().size())
      << " raw_energy: " << t.raw_energy() << " regressed energy: " << t.regressed_energy()
      << " probs(ga/e/mu/np/cp/nh/am/unk): ";
  for (auto const &p : t.id_probabilities()) {
    LogDebug("TracksterLinksProducer") << std::fixed << p << " ";
  }
  LogDebug("TracksterLinksProducer") << " sigmas: ";
  for (auto const &s : t.sigmas()) {
    LogDebug("TracksterLinksProducer") << s << " ";
  }
  LogDebug("TracksterLinksProducer") << std::endl;
}

void TracksterLinksProducer::produce(edm::Event &evt, const edm::EventSetup &es) {
  linkingAlgo_->setEvent(evt, es);

  auto resultTracksters = std::make_unique<std::vector<Trackster>>();

  auto linkedResultTracksters = std::make_unique<std::vector<std::vector<unsigned int>>>();

  const auto &layerClusters = evt.get(clusters_token_);
  const auto &layerClustersTimes = evt.get(clustersTime_token_);

  // loop over the original_masks_tokens_ and get the original masks collections and multiply them
  // to get the global mask
  std::vector<float> original_global_mask(layerClusters.size(), 1.f);
  for (unsigned int i = 0; i < original_masks_tokens_.size(); ++i) {
    const auto &tmp_mask = evt.get(original_masks_tokens_[i]);
    for (unsigned int j = 0; j < tmp_mask.size(); ++j) {
      original_global_mask[j] *= tmp_mask[j];
    }
  }

  auto resultMask = std::make_unique<std::vector<float>>(original_global_mask);

  std::vector<edm::Handle<std::vector<Trackster>>> tracksters_h(tracksters_tokens_.size());
  edm::MultiSpan<Trackster> trackstersManager;
  for (unsigned int i = 0; i < tracksters_tokens_.size(); ++i) {
    evt.getByToken(tracksters_tokens_[i], tracksters_h[i]);
    //Fill MultiSpan
    trackstersManager.add(*tracksters_h[i]);
  }

  if (runInterpretation_) {
    produceInterpretations(evt,
                           es,
                           layerClusters,
                           layerClustersTimes,
                           trackstersManager,
                           std::move(resultTracksters),
                           std::move(resultMask));
    return;
  }

  // Linking
  const typename TracksterLinkingAlgoBase::Inputs input(evt, es, layerClusters, layerClustersTimes, trackstersManager);
  auto linkedTracksterIdToInputTracksterId = std::make_unique<std::vector<std::vector<unsigned int>>>();

  // LinkTracksters will produce a vector of vector of indices of tracksters that:
  // 1) are linked together if more than one
  // 2) are isolated if only one
  // Result tracksters contains the final version of the trackster collection
  // linkedTrackstersToInputTrackstersMap contains the mapping between the linked tracksters and the input tracksters
  linkingAlgo_->linkTracksters(input, *resultTracksters, *linkedResultTracksters, *linkedTracksterIdToInputTracksterId);

  // Now we need to remove the tracksters that are not linked
  // We need to emplace_back in the resultTracksters only the tracksters that are linked

  for (auto const &resultTrackster : *resultTracksters) {
    for (auto const &clusterIndex : resultTrackster.vertices()) {
      (*resultMask)[clusterIndex] = 0.f;
    }
  }

  assignPCAtoTracksters(*resultTracksters,
                        layerClusters,
                        layerClustersTimes,
                        rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z(),
                        rhtools_,
                        true);

  if (regressionAndPid_ && inferenceAlgo_) {
    inferenceAlgo_->runInference(layerClusters, *resultTracksters, rhtools_);
  }

  evt.put(std::move(linkedResultTracksters));
  evt.put(std::move(resultMask));
  evt.put(std::move(resultTracksters));
  evt.put(std::move(linkedTracksterIdToInputTracksterId), "linkedTracksterIdToInputTracksterId");
}

void TracksterLinksProducer::printTrackstersDebug(const std::vector<Trackster> &tracksters, const char *label) const {
  int counter = 0;
  LogDebug("TracksterLinksProducer").log([&](auto &log) {
    for (auto const &t : tracksters) {
      log << counter++ << " TracksterLinksProducer (" << label << ") obj barycenter: " << t.barycenter()
          << " eta,phi (baricenter): " << t.barycenter().eta() << ", " << t.barycenter().phi()
          << " eta,phi (eigen): " << t.eigenvectors(0).eta() << ", " << t.eigenvectors(0).phi()
          << " pt(eigen): " << std::sqrt(t.eigenvectors(0).Unit().perp2()) * t.raw_energy() << " seedID: " << t.seedID()
          << " seedIndex: " << t.seedIndex() << " size: " << t.vertices().size() << " average usage: "
          << (std::accumulate(std::begin(t.vertex_multiplicity()), std::end(t.vertex_multiplicity()), 0.) /
              (float)t.vertex_multiplicity().size())
          << " raw_energy: " << t.raw_energy() << " regressed energy: " << t.regressed_energy()
          << " probs(ga/e/mu/np/cp/nh/am/unk): ";
      for (auto const &p : t.id_probabilities()) {
        log << std::fixed << p << " ";
      }
      log << " sigmas: ";
      for (auto const &s : t.sigmas()) {
        log << s << " ";
      }
      log << "\n";
    }
  });
}

static void interpretationsFilterTracks(edm::Handle<std::vector<reco::Track>> tkH,
                                        const edm::Handle<std::vector<reco::Muon>> &muons_h,
                                        const StringCutObjectSelector<reco::Track> cutTk_,
                                        const float tkEnergyCut_,
                                        std::vector<bool> &maskTracks) {
  auto const &tracks = *tkH;
  for (unsigned i = 0; i < tracks.size(); ++i) {
    const auto &tk = tracks[i];
    reco::TrackRef trackref = reco::TrackRef(tkH, i);

    // veto tracks associated to muons
    int muId = PFMuonAlgo::muAssocToTrack(trackref, *muons_h);
    const reco::MuonRef muonref = reco::MuonRef(muons_h, muId);

    if (!cutTk_((tk)) or (muId != -1 and PFMuonAlgo::isMuon(muonref) and not(*muons_h)[muId].isTrackerMuon())) {
      maskTracks[i] = false;
      continue;
    }

    // don't consider tracks below 2 GeV for linking
    if (std::sqrt(tk.p() * tk.p() + mpion2) < tkEnergyCut_) {
      maskTracks[i] = false;
      continue;
    }

    // record tracks that can be used to make a ticlcandidate
    maskTracks[i] = true;
  }
}

void TracksterLinksProducer::produceInterpretations(edm::Event &evt,
                                                    const edm::EventSetup &es,
                                                    const std::vector<reco::CaloCluster> &layerClusters,
                                                    const edm::ValueMap<std::pair<float, float>> &layerClustersTimes,
                                                    const edm::MultiSpan<Trackster> &trackstersManager,
                                                    std::unique_ptr<std::vector<Trackster>> resultTracksters,
                                                    std::unique_ptr<std::vector<float>> resultMask) {
  edm::Handle<reco::MuonCollection> muons_h;
  evt.getByToken(muons_token_, muons_h);
  edm::Handle<std::vector<reco::Track>> tracks_h;
  evt.getByToken(tracks_token_, tracks_h);
  const auto &tracks = *tracks_h;
  edm::Handle<MtdHostCollection> inputTiming_h;
  if (useMTDTiming_) {
    evt.getByToken(inputTimingToken_, inputTiming_h);
  }
  // The interpretation algorithms take the trackster links as part of their Inputs
  // but none of them uses it: pass an empty mapping.
  const std::vector<std::vector<unsigned>> generalTracksterLinksGlobalId;
  std::vector<bool> maskTracks;
  maskTracks.resize(tracks.size());
  interpretationsFilterTracks(tracks_h, muons_h, cutTk_, tkEnergyCut_, maskTracks);

  // Split the selected tracks: identified muons go to the muon interpretation pass
  // (built from the track momentum), the rest to the general pass.
  std::vector<bool> muonTrackMask(tracks.size(), false);
  std::vector<bool> generalTrackMask(maskTracks);
  for (size_t i = 0; i < tracks.size(); ++i) {
    if (!maskTracks[i])
      continue;
    auto trackRef = edm::Ref<reco::TrackCollection>(tracks_h, i);
    const int muId = PFMuonAlgo::muAssocToTrack(trackRef, *muons_h);
    const reco::MuonRef muonRef(muons_h, muId);
    if (muonRef.isNonnull() and PFMuonAlgo::isMuon(muonRef)) {
      muonTrackMask[i] = true;
      generalTrackMask[i] = false;
    }
  }

  const typename TICLInterpretationAlgoBase<reco::Track>::Inputs muonInput(evt,
                                                                           es,
                                                                           layerClusters,
                                                                           layerClustersTimes,
                                                                           trackstersManager,
                                                                           generalTracksterLinksGlobalId,
                                                                           tracks_h,
                                                                           muonTrackMask);
  auto trackToTrackster = std::make_unique<std::vector<int>>(tracks.size(), -1);
  // trackMode: -1 not selected, 0 selected but unassigned, then the winner modes.
  auto trackMode = std::make_unique<std::vector<int>>(tracks.size(), -1);
  for (size_t i = 0; i < maskTracks.size(); ++i)
    if (maskTracks[i])
      (*trackMode)[i] = 0;
  auto neutralIdx = std::make_unique<std::vector<int>>();
  auto neutralPdg = std::make_unique<std::vector<int>>();
  std::vector<int> muonInTrackIndices(tracks.size(), -1);
  std::vector<int> trackstersInTrackIndices(tracks.size(), -1);

  if (useArbitration_) {
    // === Arbitration mode: interpretations emit scored hypotheses over shared     ===
    // === inputs; the producer resolves them with strict energy exclusivity.       ===
    // The same physical shower can appear both in the Skeletons-linked (hadronic)
    // collection and in the superclustering (EM) collection: hypotheses from
    // different algorithms overlap by construction, and the arbitration below is
    // what guarantees each layer cluster's energy enters exactly one candidate.
    // e/gamma trackster span: the superclustering output collections.
    std::vector<edm::Handle<std::vector<Trackster>>> egamma_tracksters_h(egamma_tracksters_tokens_.size());
    edm::MultiSpan<Trackster> egammaTrackstersSpan;
    for (unsigned int i = 0; i < egamma_tracksters_tokens_.size(); ++i) {
      evt.getByToken(egamma_tracksters_tokens_[i], egamma_tracksters_h[i]);
      egammaTrackstersSpan.add(*egamma_tracksters_h[i]);
    }
    // Note: the superclustering links product indexes into its INPUT trackster
    // collection, not into the superclustered output span, so no global-id mapping is
    // built here; the e/gamma interpretation works on the superclustered tracksters
    // directly and does not use the links.
    const std::vector<std::vector<unsigned>> egammaTracksterLinksGlobalId;

    // --- Opinion collection. The general interpretation sees ALL selected tracks
    // (muons included): the muon and hadron hypotheses for the same track compete
    // and the arbitration decides.
    std::vector<Trackster> hypothesisTracksters;
    std::vector<Hypothesis> hypotheses;
    muonInterpretationAlgo_->makeOpinions(muonInput, inputTiming_h, hypothesisTracksters, hypotheses);
    const typename TICLInterpretationAlgoBase<reco::Track>::Inputs allInput(evt,
                                                                            es,
                                                                            layerClusters,
                                                                            layerClustersTimes,
                                                                            trackstersManager,
                                                                            generalTracksterLinksGlobalId,
                                                                            tracks_h,
                                                                            maskTracks);
    generalInterpretationAlgo_->makeOpinions(allInput, inputTiming_h, hypothesisTracksters, hypotheses);
    // The e/gamma interpretation uses the KF general tracks: the HGCAL GSF chain is
    // structurally downstream of ticlCandidate (via particleFlowClusterHGCal ->
    // particleFlowSuperClusterHGCal -> electron seeds) and cannot be consumed here.
    const typename TICLInterpretationAlgoBase<reco::Track>::Inputs egammaInput(evt,
                                                                               es,
                                                                               layerClusters,
                                                                               layerClustersTimes,
                                                                               egammaTrackstersSpan,
                                                                               egammaTracksterLinksGlobalId,
                                                                               tracks_h,
                                                                               maskTracks);
    egammaInterpretationAlgo_->makeOpinions(egammaInput, inputTiming_h, hypothesisTracksters, hypotheses);
    jetInterpretationAlgo_->makeOpinions(allInput, inputTiming_h, hypothesisTracksters, hypotheses);

    // Run the PID inference on the hypothesis tracksters and score the whole neutral
    // tier from the SAME classifier: P(gamma)+P(e)+P(pi0) for the photon reading vs
    // P(charged hadron)+P(neutral hadron) for the neutral-hadron reading of the same
    // energy. This makes the competing hypotheses commensurate (the input tracksters
    // carry no PID: upstream linking runs no inference and merging zeroes it).
    if (regressionAndPid_ && inferenceAlgo_ && !hypothesisTracksters.empty()) {
      assignPCAtoTracksters(hypothesisTracksters,
                            layerClusters,
                            layerClustersTimes,
                            rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z(),
                            rhtools_,
                            true);
      inferenceAlgo_->runInference(layerClusters, hypothesisTracksters, rhtools_);
      for (auto &h : hypotheses) {
        if (h.tracksterIdx < 0)
          continue;
        const auto &ts = hypothesisTracksters[h.tracksterIdx];
        if (h.type == Hypothesis::Type::NeutralHadron) {
          h.score = ts.id_probability(Trackster::ParticleType::charged_hadron) +
                    ts.id_probability(Trackster::ParticleType::neutral_hadron);
        } else if (h.type == Hypothesis::Type::Photon) {
          h.score = ts.id_probability(Trackster::ParticleType::photon) +
                    ts.id_probability(Trackster::ParticleType::electron) +
                    ts.id_probability(Trackster::ParticleType::neutral_pion);
        }
      }
    }

    // --- Arbitration: type priority (mu > e > charged hadron > gamma), then score.
    auto typePriority = [](Hypothesis::Type t) {
      switch (t) {
        case Hypothesis::Type::Muon:
          return 0;
        case Hypothesis::Type::Electron:
          return 1;
        case Hypothesis::Type::ChargedHadron:
          return 2;
        // A jet (multi-track) reading takes what no single-track hypothesis could
        // claim (the charged-hadron energy gate vetoes single tracks on
        // multi-particle tracksters).
        case Hypothesis::Type::Jet:
          return 3;
        // Single-track claim-and-attach recovery: below the tight charged hadrons and
        // the jets, above the neutral tier.
        case Hypothesis::Type::RecoveryChargedHadron:
          return 4;
        // Photon and NeutralHadron share a tier: the EM and the hadronic reading of
        // the same (neutral) energy compete on their scores, not on ordering.
        case Hypothesis::Type::Photon:
        case Hypothesis::Type::NeutralHadron:
          return 5;
      }
      return 4;
    };
    std::vector<unsigned> order(hypotheses.size());
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
      const int pa = typePriority(hypotheses[a].type);
      const int pb = typePriority(hypotheses[b].type);
      if (pa != pb)
        return pa < pb;
      return hypotheses[a].score > hypotheses[b].score;
    });

    // Energy exclusivity at the layer-cluster level: the constituents shared by all
    // merged-trackster collections, so overlaps between the hadronic and the EM view
    // of the same shower are detected regardless of which collection each came from.
    std::vector<bool> claimedLC(layerClusters.size(), false);
    auto footprintOverlap = [&](const Trackster &ts, double &footE, double &sharedE) {
      footE = 0.;
      sharedE = 0.;
      for (auto v : ts.vertices()) {
        const double e = layerClusters[v].energy();
        footE += e;
        if (claimedLC[v])
          sharedE += e;
      }
    };

    std::vector<bool> usedTrack(tracks.size(), false);
    std::vector<bool> accepted(hypotheses.size(), false);
    for (auto idx : order) {
      auto &h = hypotheses[idx];
      if (h.trackIdx >= 0 && usedTrack[h.trackIdx])
        continue;
      if (!h.trackIdxs.empty()) {
        // Jet: proceed with the tracks not already claimed by higher-tier winners;
        // fewer than two left means there is no multi-track reading. The member list
        // is filtered IN PLACE so the claim and the assignment maps below only ever
        // see the surviving members (a jet must not overwrite a muon's assignment).
        std::vector<int> freeMembers;
        for (int iTk : h.trackIdxs)
          if (!usedTrack[iTk])
            freeMembers.push_back(iTk);
        if (freeMembers.size() < 2)
          continue;
        h.trackIdxs.swap(freeMembers);
      }
      if (h.tracksterIdx >= 0) {
        double footE, sharedE;
        footprintOverlap(hypothesisTracksters[h.tracksterIdx], footE, sharedE);
        if (footE > 0. && sharedE / footE > arbitrationMaxSharedEnergyFraction_)
          continue;
      }
      accepted[idx] = true;
      if (h.trackIdx >= 0)
        usedTrack[h.trackIdx] = true;
      for (int iTk : h.trackIdxs)
        usedTrack[iTk] = true;
      if (h.tracksterIdx >= 0)
        for (auto v : hypothesisTracksters[h.tracksterIdx].vertices())
          claimedLC[v] = true;
    }

    // --- Result tracksters: the winners' tracksters plus the unclaimed leftovers of
    // the general span as neutrals (a leftover mostly claimed by a winner, e.g. the
    // Skeletons fragments of a superclustered photon, is dropped: its energy is
    // already in the winning candidate).
    std::vector<int> winnerResultIdx(hypotheses.size(), -1);
    for (unsigned idx = 0; idx < hypotheses.size(); ++idx) {
      if (!accepted[idx] || hypotheses[idx].tracksterIdx < 0)
        continue;
      winnerResultIdx[idx] = static_cast<int>(resultTracksters->size());
      resultTracksters->push_back(hypothesisTracksters[hypotheses[idx].tracksterIdx]);
    }
    std::vector<int> neutralResultIdx;
    for (unsigned iTs = 0; iTs < trackstersManager.size(); ++iTs) {
      const auto &ts = trackstersManager[iTs];
      double footE, sharedE;
      footprintOverlap(ts, footE, sharedE);
      if (footE <= 0.)
        continue;
      if (sharedE / footE > arbitrationMaxSharedEnergyFraction_)
        continue;
      neutralResultIdx.push_back(static_cast<int>(resultTracksters->size()));
      resultTracksters->push_back(ts);
    }

    assignPCAtoTracksters(*resultTracksters,
                          layerClusters,
                          layerClustersTimes,
                          rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z(),
                          rhtools_,
                          true);
    if (regressionAndPid_ && inferenceAlgo_) {
      inferenceAlgo_->runInference(layerClusters, *resultTracksters, rhtools_);
    }
    for (auto const &resultTrackster : *resultTracksters) {
      for (auto const &clusterIndex : resultTrackster.vertices()) {
        (*resultMask)[clusterIndex] = 0.f;
      }
    }
    evt.put(std::move(resultTracksters));

    // --- Assignment maps from the winning hypotheses.
    for (unsigned idx = 0; idx < hypotheses.size(); ++idx) {
      if (!accepted[idx])
        continue;
      const auto &h = hypotheses[idx];
      switch (h.type) {
        case Hypothesis::Type::Muon:
          (*trackMode)[h.trackIdx] = 1;
          (*trackToTrackster)[h.trackIdx] = winnerResultIdx[idx];
          break;
        case Hypothesis::Type::Electron:
          (*trackMode)[h.trackIdx] = 3;
          (*trackToTrackster)[h.trackIdx] = winnerResultIdx[idx];
          break;
        case Hypothesis::Type::ChargedHadron:
          (*trackMode)[h.trackIdx] = 2;
          (*trackToTrackster)[h.trackIdx] = winnerResultIdx[idx];
          break;
        case Hypothesis::Type::Photon:
          neutralIdx->push_back(winnerResultIdx[idx]);
          neutralPdg->push_back(22);
          break;
        case Hypothesis::Type::NeutralHadron:
          // The neutral-hadron hypothesis DEFENDS hadronic energy from photon claims
          // (its role in the arbitration), but the species label follows the trackster
          // PID like any leftover: a neutral-hadron win on an EM trackster (e.g. a jet
          // pi0 that was never superclustered, so no photon advocate existed) must not
          // force pdg 130 onto a photon.
          neutralIdx->push_back(winnerResultIdx[idx]);
          neutralPdg->push_back(0);
          break;
        case Hypothesis::Type::Jet:
          // One charged candidate per track (kinematics from the track); the assembly
          // adds the neutral residual per shared trackster.
          for (int iTk : h.trackIdxs) {
            (*trackMode)[iTk] = 4;
            (*trackToTrackster)[iTk] = winnerResultIdx[idx];
          }
          break;
        case Hypothesis::Type::RecoveryChargedHadron:
          // Same resolution as a single-member jet: charged candidate from the track,
          // neutral residual for the calorimetric excess.
          (*trackMode)[h.trackIdx] = 5;
          (*trackToTrackster)[h.trackIdx] = winnerResultIdx[idx];
          break;
      }
    }
    for (int iTrackster : neutralResultIdx) {
      neutralIdx->push_back(iTrackster);
      neutralPdg->push_back(0);
    }
  } else {
    // === Masking mode (default): sequential interpretation passes over a shared  ===
    // === trackster mask; a pass consumes what it claims.                         ===

    // Tracksters consumed across interpretation passes (indexed over the input span). The
    // muon pass runs first: it masks the MIP tracksters it consumes and reports, per
    // muon-candidate track, the consumed trackster (>=0), no trackster (-1, a track-only
    // muon), or a rejection (kMuonRejected: the trajectory points to a shower).
    std::vector<bool> maskedInputTracksters(trackstersManager.size(), false);
    muonInterpretationAlgo_->makeCandidates(
        muonInput, inputTiming_h, *resultTracksters, muonInTrackIndices, maskedInputTracksters);

    // A track the muon pass rejected is not a muon: route it back to the general pass so
    // it is reconstructed there (and no muon candidate is built for it below).
    for (size_t iTrack = 0; iTrack < tracks.size(); ++iTrack) {
      if (muonTrackMask[iTrack] && muonInTrackIndices[iTrack] == kMuonRejected) {
        muonTrackMask[iTrack] = false;
        generalTrackMask[iTrack] = true;
      }
    }

    const typename TICLInterpretationAlgoBase<reco::Track>::Inputs input(evt,
                                                                         es,
                                                                         layerClusters,
                                                                         layerClustersTimes,
                                                                         trackstersManager,
                                                                         generalTracksterLinksGlobalId,
                                                                         tracks_h,
                                                                         generalTrackMask);
    generalInterpretationAlgo_->makeCandidates(
        input, inputTiming_h, *resultTracksters, trackstersInTrackIndices, maskedInputTracksters);

    assignPCAtoTracksters(*resultTracksters,
                          layerClusters,
                          layerClustersTimes,
                          rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z(),
                          rhtools_,
                          true);
    if (regressionAndPid_) {
      // Run inference algorithm
      inferenceAlgo_->runInference(layerClusters, *resultTracksters, rhtools_);
    }

    std::vector<bool> maskTracksters(resultTracksters->size(), true);
    for (auto const &resultTrackster : *resultTracksters) {
      for (auto const &clusterIndex : resultTrackster.vertices()) {
        (*resultMask)[clusterIndex] = 0.f;
      }
    }
    evt.put(std::move(resultTracksters));

    // Muon tracks: mode 1 (p4 from the track), attaching the MIP trackster if any.
    for (size_t iTrack = 0; iTrack < tracks.size(); ++iTrack) {
      if (!muonTrackMask[iTrack])
        continue;
      (*trackMode)[iTrack] = 1;
      const int tracksterId = muonInTrackIndices[iTrack];
      if (tracksterId >= 0) {
        (*trackToTrackster)[iTrack] = tracksterId;
        maskTracksters[tracksterId] = false;
      }
    }

    // Charged (non-muon) tracks: mode 2.
    for (size_t iTrack = 0; iTrack < tracks.size(); iTrack++) {
      if (generalTrackMask[iTrack]) {
        auto const tracksterId = trackstersInTrackIndices[iTrack];
        if (tracksterId != -1 and !maskTracksters.empty()) {
          (*trackMode)[iTrack] = 2;
          (*trackToTrackster)[iTrack] = tracksterId;
          maskTracksters[tracksterId] = false;
        }
      }
    }

    // Neutral leftovers: pdg inferred from the trackster PID downstream.
    for (size_t iTrackster = 0; iTrackster < maskTracksters.size(); iTrackster++) {
      if (maskTracksters[iTrackster]) {
        neutralIdx->push_back(iTrackster);
        neutralPdg->push_back(0);
      }
    }
  }

  evt.put(std::move(trackToTrackster), "trackToTrackster");
  evt.put(std::move(trackMode), "trackMode");
  evt.put(std::move(neutralIdx), "neutralIdx");
  evt.put(std::move(neutralPdg), "neutralPdg");
  evt.put(std::move(resultMask));
  evt.put(std::make_unique<std::vector<std::vector<unsigned int>>>());
  evt.put(std::make_unique<std::vector<std::vector<unsigned int>>>(), "linkedTracksterIdToInputTracksterId");
}

void TracksterLinksProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  edm::ParameterSetDescription linkingDesc;
  linkingDesc.addNode(edm::PluginDescription<TracksterLinkingPluginFactory>("type", "Skeletons", true));
  // Inference Plugins
  edm::ParameterSetDescription inferenceDesc;
  inferenceDesc.addNode(edm::PluginDescription<TracksterInferenceAlgoFactory>("type", "TracksterInferenceByDNN", true));
  desc.add<edm::ParameterSetDescription>("pluginInferenceAlgoTracksterInferenceByDNN", inferenceDesc);

  edm::ParameterSetDescription inferenceDescPFN;
  inferenceDescPFN.addNode(
      edm::PluginDescription<TracksterInferenceAlgoFactory>("type", "TracksterInferenceByPFN", true));
  desc.add<edm::ParameterSetDescription>("pluginInferenceAlgoTracksterInferenceByPFN", inferenceDescPFN);
  desc.add<edm::ParameterSetDescription>("linkingPSet", linkingDesc);
  desc.add<std::vector<edm::InputTag>>("tracksters_collections", {edm::InputTag("ticlTrackstersCLUE3DHigh")});
  desc.add<std::vector<edm::InputTag>>("original_masks",
                                       {edm::InputTag("hgcalMergeLayerClusters", "InitialLayerClustersMask")});
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("layer_clustersTime", edm::InputTag("hgcalMergeLayerClusters", "timeLayerCluster"));
  desc.add<bool>("regressionAndPid", false);
  desc.add<std::string>("detector", "HGCAL");
  desc.add<std::string>("propagator", "PropagatorWithMaterial");
  desc.add<std::string>("inferenceAlgo", "");
  // Interpretation mode.
  desc.add<bool>("runInterpretation", false)
      ->setComment("Host the track <-> trackster interpretations instead of a linking algorithm.");
  edm::ParameterSetDescription interpretationDesc;
  interpretationDesc.addNode(
      edm::PluginDescription<TICLGeneralInterpretationPluginFactory>("type", "ChargedHadron", true));
  desc.add<edm::ParameterSetDescription>("interpretationDescPSet", interpretationDesc);
  edm::ParameterSetDescription jetInterpretationDesc;
  jetInterpretationDesc.addNode(edm::PluginDescription<TICLGeneralInterpretationPluginFactory>("type", "Jet", true));
  desc.add<edm::ParameterSetDescription>("jetInterpretationDescPSet", jetInterpretationDesc);
  edm::ParameterSetDescription muonInterpretationDesc;
  muonInterpretationDesc.addNode(edm::PluginDescription<TICLGeneralInterpretationPluginFactory>("type", "Muon", true));
  desc.add<edm::ParameterSetDescription>("muonInterpretationDescPSet", muonInterpretationDesc);
  edm::ParameterSetDescription egammaInterpretationDesc;
  egammaInterpretationDesc.addNode(
      edm::PluginDescription<TICLGeneralInterpretationPluginFactory>("type", "EGamma", true));
  desc.add<edm::ParameterSetDescription>("egammaInterpretationDescPSet", egammaInterpretationDesc);
  desc.add<bool>("useArbitration", false);
  desc.add<double>("arbitrationMaxSharedEnergyFraction", 0.2);
  desc.add<std::vector<edm::InputTag>>("egamma_tracksters_collections",
                                       {edm::InputTag("ticlTracksterLinksSuperclusteringDNN")});
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons1stStep"));
  desc.add<edm::InputTag>("timingSoA", edm::InputTag("mtdSoA"));
  desc.add<bool>("useMTDTiming", true);
  desc.add<double>("tkEnergyCut", 2.0)
      ->setComment("Min track energy sqrt(p^2+mpi^2) [GeV] for candidate linking; was hardcoded.");
  desc.add<std::string>("cutTk",
                        "1.48 < abs(eta) < 3.0 && pt > 1. && quality(\"highPurity\") && "
                        "hitPattern().numberOfLostHits(\"MISSING_OUTER_HITS\") < 5");
  descriptions.add("tracksterLinksProducer", desc);
}

DEFINE_FWK_MODULE(TracksterLinksProducer);
