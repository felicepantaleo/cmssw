// Author: Felice Pantaleo (CERN) - felice.pantaleo@cern.ch
// Date: 10/2026
//
// Interpretation stage of TICL. The interpretations give hypotheses on the linked tracksters, the superclusters and
// the tracks; the global arbitration accepts the maximum-weight set of hypotheses with no shared track and no large
// layer-cluster overlap; the charged hadrons claim the layer clusters along their tracks. The products are the final
// tracksters and the per-track assignment maps that TICLCandidateArbitrationProducer reads.

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/MultiSpan.h"
#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/HGCalReco/interface/MtdHostCollection.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "RecoTICL/Common/interface/TICLUtils.h"
#include "RecoTICL/Common/interface/TrackstersPCA.h"
#include "RecoTICL/Inference/interface/TICLONNXGlobalCache.h"
#include "RecoTICL/Inference/interface/TracksterInferenceAlgoFactory.h"
#include "RecoTICL/Interpretation/interface/MaxWeightIndependentSet.h"
#include "RecoTICL/Interpretation/interface/TICLInterpretationAlgoBase.h"
#include "TICLInterpretationPluginFactory.h"
#include "RecoTICL/Interpretation/interface/TrackImpact.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

using namespace ticl;

namespace {
  // A network with input "features" (rows x nFeatures) and one output, the logits (rows x nClasses).
  class ArbitrationModel {
  public:
    ArbitrationModel(const cms::Ort::ONNXRuntime &network, unsigned int nFeatures, unsigned int nClasses)
        : network_(network), nFeatures_(nFeatures), nClasses_(nClasses) {
      // The first output dimension is the row.
      const auto &outputs = network_.getOutputNames();
      int64_t perRow = 1;
      if (outputs.size() == 1) {
        const auto &shape = network_.getOutputShape(outputs[0]);
        for (std::size_t k = 1; k < shape.size(); ++k)
          perRow *= shape[k];
      }
      if (outputs.size() != 1 || perRow != static_cast<int64_t>(nClasses_))
        throw cms::Exception("Configuration")
            << "ArbitrationModel: expected one output with " << nClasses_ << " logits per row";
    }

    std::vector<float> logits(const std::vector<float> &x, unsigned int rows) const {
      if (rows == 0)
        return {};
      cms::Ort::FloatArrays input{x};
      auto result =
          network_.run({"features"}, input, {{static_cast<int64_t>(rows), static_cast<int64_t>(nFeatures_)}}, {}, rows);
      if (result.size() != 1 || result[0].size() != static_cast<std::size_t>(rows) * nClasses_)
        throw cms::Exception("ArbitrationModel") << "expected " << rows * nClasses_ << " logits from the network";
      return std::move(result[0]);
    }

  private:
    const cms::Ort::ONNXRuntime &network_;
    const unsigned int nFeatures_;
    const unsigned int nClasses_;
  };

  // Inference results of the distinct tracksters of one event: the PID, the regressed energy, or both. Two tracksters are
  // the same when they have the same layer clusters with the same multiplicities, in the same order.
  class InferenceCache {
  public:
    using Probabilities = std::remove_cvref_t<decltype(std::declval<const Trackster &>().id_probabilities())>;
    struct Entry {
      std::vector<unsigned int> vertices;
      std::vector<float> multiplicities;
      bool hasPID = false;
      bool hasRegression = false;
      Probabilities probabilities{};
      float regressedEnergy = 0.f;
    };

    // Index of the entry of ts; a new entry when there is none.
    unsigned int entry(const Trackster &ts) {
      const std::size_t h = hash(ts);
      const auto range = index_.equal_range(h);
      for (auto it = range.first; it != range.second; ++it) {
        const auto &e = entries_[it->second];
        if (e.vertices == ts.vertices() && e.multiplicities == ts.vertex_multiplicity())
          return it->second;
      }
      entries_.push_back({ts.vertices(), ts.vertex_multiplicity()});
      index_.emplace(h, entries_.size() - 1);
      return entries_.size() - 1;
    }
    const Entry &operator[](unsigned int i) const { return entries_[i]; }
    void storePID(unsigned int i, const Trackster &ts) {
      std::copy(ts.id_probabilities().begin(), ts.id_probabilities().end(), entries_[i].probabilities.begin());
      entries_[i].hasPID = true;
    }
    void storeRegression(unsigned int i, const Trackster &ts) {
      entries_[i].regressedEnergy = ts.regressed_energy();
      entries_[i].hasRegression = true;
    }
    // The state the inference leaves: zero for what it did not compute.
    void apply(unsigned int i, Trackster &ts) const {
      const auto &e = entries_[i];
      ts.setRegressedEnergy(e.hasRegression ? e.regressedEnergy : 0.f);
      Probabilities probabilities = e.hasPID ? e.probabilities : Probabilities{};
      ts.setProbabilities(probabilities.data());
    }

  private:
    static std::size_t hash(const Trackster &ts) {
      std::size_t h = ts.vertices().size();
      for (auto v : ts.vertices())
        h ^= std::hash<unsigned int>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      return h;
    }
    std::vector<Entry> entries_;
    std::unordered_multimap<std::size_t, unsigned int> index_;
  };

  // A charged hadron from the track alone, with no footprint.
  bool isTrackOnly(const Hypothesis &h) {
    return h.tracksterIdx < 0 && h.type == Hypothesis::Type::RecoveryChargedHadron;
  }
  bool isNeutral(const Hypothesis &h) {
    return h.type == Hypothesis::Type::Photon || h.type == Hypothesis::Type::NeutralHadron;
  }

  // trackMode values of the assignment maps.
  enum TrackMode : int {
    kNotSelected = -1,
    kUnassigned = 0,
    kMuon = 1,
    kChargedHadron = 2,
    kElectron = 3,
    kJetMember = 4,
    kRecovery = 5
  };

  // Number of features of the hypothesis model, and of the neutral species model.
  constexpr unsigned int kNHypothesisFeatures = 30;
  constexpr unsigned int kNNeutralFeatures = 12;
  // Classes of the neutral species model: photon, pi0, neutral hadron.
  constexpr unsigned int kNSpecies = 3;
}  // namespace

class TICLInterpretationProducer
    : public edm::stream::EDProducer<edm::GlobalCache<TICLONNXGlobalCache>, edm::stream::WatchRuns> {
public:
  TICLInterpretationProducer(const edm::ParameterSet &ps, const TICLONNXGlobalCache *cache);
  static std::unique_ptr<TICLONNXGlobalCache> initializeGlobalCache(const edm::ParameterSet &ps);
  static void globalEndJob(const TICLONNXGlobalCache *) {}
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginRun(edm::Run const &, edm::EventSetup const &es) override;
  void produce(edm::Event &evt, const edm::EventSetup &es) override;
  void endStream() override;

private:
  using Inputs = TICLInterpretationAlgoBase<reco::Track>::Inputs;
  // The hypotheses of the interpretations and their footprint tracksters.
  struct Opinions {
    std::vector<Trackster> tracksters;
    std::vector<Hypothesis> hypotheses;
  };

  // Selected tracks: cutTk, energy (pion mass) at least tkEnergyCut, and not a muon without a tracker-muon fit. Among
  // them, the tracks of identified muons.
  void selectTracks(const edm::Handle<std::vector<reco::Track>> &tracks_h,
                    const edm::Handle<std::vector<reco::Muon>> &muons_h,
                    std::vector<bool> &selected,
                    std::vector<bool> &muonTracks) const;
  // Runs the inference once per distinct trackster of the event and copies the result to the others: the PID, and the
  // regressed energy when withRegression is set. Tracksters with a barrel layer cluster are not changed.
  void runInference(const std::vector<reco::CaloCluster> &layerClusters,
                    std::vector<Trackster> &tracksters,
                    InferenceCache &cache,
                    bool withRegression) const;
  // Hypotheses of the interpretations, with PCA and PID on their footprints, plus one track-only hypothesis per
  // selected track.
  Opinions makeOpinions(const Inputs &muonInput,
                        const Inputs &allInput,
                        const Inputs &egammaInput,
                        const edm::Handle<MtdHostCollection> &timing_h,
                        InferenceCache &cache) const;
  // Two hypotheses conflict when they share a track, or layer clusters with more than maxSharedEnergyFraction of the
  // smaller footprint energy. Returns the sorted neighbours of each hypothesis.
  std::vector<std::vector<unsigned int>> conflictGraph(const Opinions &opinions,
                                                       const std::vector<reco::CaloCluster> &layerClusters) const;
  // Platt-calibrated logit of the hypothesis model; trackOnlyWeight for a track-only hypothesis.
  std::vector<float> modelWeights(const Opinions &opinions,
                                  const std::vector<std::vector<unsigned int>> &conflicts,
                                  const std::vector<reco::Track> &tracks) const;
  void putHypothesisDump(edm::Event &evt,
                         const Opinions &opinions,
                         const std::vector<bool> &accepted,
                         const std::vector<float> &weights) const;
  // Leftovers: the input tracksters in descending raw energy, without the layer clusters that the winners and the
  // leftovers before them hold. A leftover that lost a layer cluster is kept when its raw energy is at least
  // claimMinEnergy. held marks the layer clusters of the winners and of the leftovers.
  std::vector<Trackster> selectLeftovers(const edm::MultiSpan<Trackster> &tracksters,
                                         const std::vector<reco::CaloCluster> &layerClusters,
                                         std::vector<bool> &held) const;
  // Track claim: each accepted charged hadron, recovery, track-only hypothesis and jet track takes the pool layer
  // clusters along its propagated track up to its expected deposit minus its footprint. Returns the claiming track of
  // each layer cluster (-1: none).
  std::vector<int> claimAlongTracks(const Opinions &opinions,
                                    const std::vector<bool> &accepted,
                                    const std::vector<reco::Track> &tracks,
                                    const std::vector<TrackImpact> &impacts,
                                    const std::vector<reco::CaloCluster> &layerClusters,
                                    const std::vector<unsigned int> &pool) const;
  // Species of the neutral candidates: photon (22), pi0 (111) or neutral hadron (130). A photon winner is EM and only
  // chooses between 22 and 111.
  void assignNeutralSpecies(const std::vector<Trackster> &tracksters,
                            const std::vector<int> &neutralIdx,
                            std::vector<int> &neutralPdg) const;

  std::vector<edm::EDGetTokenT<std::vector<Trackster>>> tracksters_tokens_;
  std::vector<edm::EDGetTokenT<std::vector<Trackster>>> egamma_tracksters_tokens_;
  // Tracksters whose layer clusters outside the input tracksters join the track-claim pool.
  std::vector<edm::EDGetTokenT<std::vector<Trackster>>> claimUnlinkedTracksters_tokens_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTime_token_;
  const edm::EDGetTokenT<std::vector<reco::Track>> tracks_token_;
  const edm::EDGetTokenT<std::vector<reco::Muon>> muons_token_;
  const bool useMTDTiming_;
  edm::EDGetTokenT<MtdHostCollection> timing_token_;
  const float tkEnergyCut_;
  const StringCutObjectSelector<reco::Track> cutTk_;

  const edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
  const edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
  const edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hdc_token_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bfield_token_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagator_token_;
  ticlgeom::Tools rhtools_;
  edm::ESHandle<MagneticField> bfield_;
  edm::ESHandle<Propagator> propagator_;
  std::array<std::unique_ptr<GeomDet>, 2> frontDisks_;

  std::unique_ptr<TICLInterpretationAlgoBase<reco::Track>> chargedHadronAlgo_;
  std::unique_ptr<TICLInterpretationAlgoBase<reco::Track>> muonAlgo_;
  std::unique_ptr<TICLInterpretationAlgoBase<reco::Track>> egammaAlgo_;
  std::unique_ptr<TICLInterpretationAlgoBase<reco::Track>> jetAlgo_;

  std::unique_ptr<TracksterInferenceAlgoBase> inferenceAlgo_;
  // The same inference with the PID only and with the regression only, when the plugin has doPID and doRegression.
  std::unique_ptr<TracksterInferenceAlgoBase> inferencePIDOnly_;
  std::unique_ptr<TracksterInferenceAlgoBase> inferenceRegressionOnly_;

  // Global arbitration: weight = Platt-calibrated logit of the hypothesis model + trackBonus per track.
  const ArbitrationModel arbitrationModel_;
  const float maxSharedEnergyFraction_;
  const float plattA_;
  const float plattB_;
  const float trackBonus_;
  const float trackOnlyWeight_;
  const unsigned int maxExactComponent_;
  const unsigned long searchBudget_;
  MaxWeightIndependentSetStats solverStats_;
  // Track claim: expected raw deposit (response + responseLogSlope x ln p) x p, search radius radius0 + radiusSlope x
  // depth.
  const float claimResponse_;
  const float claimResponseLogSlope_;
  const float claimRadius0_;
  const float claimRadiusSlope_;
  const float claimMinEnergy_;
  // Depth [cm] from the HGCAL front that the claim search covers.
  static constexpr float kClaimMaxDepth = 200.f;
  // Rank offset that puts every CE-E cluster after every CE-H cluster.
  static constexpr float kClaimCalorimeterOffset = 1.e3f;
  const ArbitrationModel neutralModel_;
  const float neutralModelThreshold_;
  const float emRawEnergyBelow_;
  const bool dumpHypotheses_;
};

std::unique_ptr<TICLONNXGlobalCache> TICLInterpretationProducer::initializeGlobalCache(const edm::ParameterSet &ps) {
  auto cache = TICLONNXGlobalCache::initialize(ps);
  cache->loadModel(ps.getParameter<std::string>("arbitrationModelFile"));
  cache->loadModel(ps.getParameter<std::string>("neutralModelFile"));
  return cache;
}

TICLInterpretationProducer::TICLInterpretationProducer(const edm::ParameterSet &ps, const TICLONNXGlobalCache *cache)
    : clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"))),
      clustersTime_token_(
          consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("layer_clustersTime"))),
      tracks_token_(consumes<std::vector<reco::Track>>(ps.getParameter<edm::InputTag>("tracks"))),
      muons_token_(consumes<std::vector<reco::Muon>>(ps.getParameter<edm::InputTag>("muons"))),
      useMTDTiming_(ps.getParameter<bool>("useMTDTiming")),
      tkEnergyCut_(ps.getParameter<float>("tkEnergyCut")),
      cutTk_(ps.getParameter<std::string>("cutTk")),
      ticlGeomToken_(esConsumes<TICLGeomHost, CaloGeometryRecord, edm::Transition::BeginRun>()),
      ticlGeomLookupToken_(esConsumes<TICLGeomLookupHost, CaloGeometryRecord, edm::Transition::BeginRun>()),
      ticlGeomLayersToken_(esConsumes<TICLGeomLayersHost, CaloGeometryRecord, edm::Transition::BeginRun>()),
      hdc_token_(esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag("", "HGCalEESensitive"))),
      bfield_token_(esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()),
      propagator_token_(esConsumes<Propagator, TrackingComponentsRecord, edm::Transition::BeginRun>(
          edm::ESInputTag("", ps.getParameter<std::string>("propagator")))),
      arbitrationModel_(
          *cache->getByModelPathString(ps.getParameter<std::string>("arbitrationModelFile")), kNHypothesisFeatures, 1),
      maxSharedEnergyFraction_(ps.getParameter<float>("arbitrationMaxSharedEnergyFraction")),
      plattA_(ps.getParameter<float>("arbitrationPlattA")),
      plattB_(ps.getParameter<float>("arbitrationPlattB")),
      trackBonus_(ps.getParameter<float>("arbitrationTrackBonus")),
      trackOnlyWeight_(ps.getParameter<float>("arbitrationTrackOnlyWeight")),
      maxExactComponent_(ps.getParameter<unsigned int>("arbitrationMaxExactComponent")),
      searchBudget_(ps.getParameter<unsigned int>("arbitrationSearchBudget")),
      claimResponse_(ps.getParameter<float>("arbitrationTrackClaimResponse")),
      claimResponseLogSlope_(ps.getParameter<float>("arbitrationTrackClaimResponseLogSlope")),
      claimRadius0_(ps.getParameter<float>("arbitrationTrackClaimRadius0")),
      claimRadiusSlope_(ps.getParameter<float>("arbitrationTrackClaimRadiusSlope")),
      claimMinEnergy_(ps.getParameter<float>("arbitrationTrackClaimMinEnergy")),
      neutralModel_(
          *cache->getByModelPathString(ps.getParameter<std::string>("neutralModelFile")), kNNeutralFeatures, kNSpecies),
      neutralModelThreshold_(ps.getParameter<float>("neutralModelThreshold")),
      emRawEnergyBelow_(ps.getParameter<float>("emRawEnergyBelow")),
      dumpHypotheses_(ps.getParameter<bool>("dumpHypotheses")) {
  for (auto const &tag : ps.getParameter<std::vector<edm::InputTag>>("tracksters_collections"))
    tracksters_tokens_.emplace_back(consumes<std::vector<Trackster>>(tag));
  for (auto const &tag : ps.getParameter<std::vector<edm::InputTag>>("egamma_tracksters_collections"))
    egamma_tracksters_tokens_.emplace_back(consumes<std::vector<Trackster>>(tag));
  for (auto const &tag : ps.getParameter<std::vector<edm::InputTag>>("claimUnlinkedTracksters"))
    claimUnlinkedTracksters_tokens_.emplace_back(consumes<std::vector<Trackster>>(tag));
  if (useMTDTiming_)
    timing_token_ = consumes<MtdHostCollection>(ps.getParameter<edm::InputTag>("timingSoA"));

  if (maxExactComponent_ > 64)
    throw cms::Exception("Configuration") << "arbitrationMaxExactComponent must be at most 64";
  if (!(claimResponse_ > 0.f) || !(claimRadius0_ >= 0.f) || !(claimRadiusSlope_ >= 0.f) || !(claimMinEnergy_ >= 0.f) ||
      !(claimRadius0_ + claimRadiusSlope_ * kClaimMaxDepth < kClaimCalorimeterOffset))
    throw cms::Exception("Configuration")
        << "track claim: response > 0, radius0 >= 0, radiusSlope >= 0, minEnergy >= 0, and the largest radius below "
        << kClaimCalorimeterOffset << " cm are required";
  if (!(neutralModelThreshold_ > 0.f && neutralModelThreshold_ < 1.f))
    throw cms::Exception("Configuration") << "neutralModelThreshold must be in (0, 1)";

  auto makeAlgo = [&](const char *name) {
    const auto pset = ps.getParameter<edm::ParameterSet>(name);
    return TICLGeneralInterpretationPluginFactory::get()->create(
        pset.getParameter<std::string>("type"), pset, consumesCollector());
  };
  chargedHadronAlgo_ = makeAlgo("interpretationDescPSet");
  muonAlgo_ = makeAlgo("muonInterpretationDescPSet");
  egammaAlgo_ = makeAlgo("egammaInterpretationDescPSet");
  jetAlgo_ = makeAlgo("jetInterpretationDescPSet");

  const std::string inferencePlugin = ps.getParameter<std::string>("inferenceAlgo");
  const auto inferencePSet = ps.getParameter<edm::ParameterSet>("pluginInferenceAlgo" + inferencePlugin);
  inferenceAlgo_ = TracksterInferenceAlgoFactory::get()->create(inferencePlugin, inferencePSet, cache);
  if (inferencePSet.existsAs<int>("doPID") && inferencePSet.existsAs<int>("doRegression") &&
      inferencePSet.getParameter<int>("doPID") != 0 && inferencePSet.getParameter<int>("doRegression") != 0) {
    auto only = [&](const char *off) {
      edm::ParameterSet pset;
      pset.copyForModify(inferencePSet);
      pset.eraseSimpleParameter(off);
      pset.addParameter<int>(off, 0);
      return TracksterInferenceAlgoFactory::get()->create(inferencePlugin, pset, cache);
    };
    inferencePIDOnly_ = only("doRegression");
    inferenceRegressionOnly_ = only("doPID");
  }

  produces<std::vector<Trackster>>();
  produces<std::vector<int>>("trackToTrackster");
  produces<std::vector<int>>("trackMode");
  produces<std::vector<int>>("neutralIdx");
  produces<std::vector<int>>("neutralPdg");
  produces<std::vector<int>>("trackToClaimTrackster");
  if (dumpHypotheses_) {
    for (auto const *n :
         {"hypType", "hypTrack", "hypAccepted", "hypLCOffsets", "hypLCs", "hypJetOffsets", "hypJetTracks"})
      produces<std::vector<int>>(n);
    for (auto const *n : {"hypScore", "hypRawE", "hypRawEmE", "hypX", "hypY", "hypZ", "hypPid", "hypWeight"})
      produces<std::vector<float>>(n);
    produces<std::vector<int>>("trackClaimedLC");
  }
}

void TICLInterpretationProducer::beginRun(edm::Run const &, edm::EventSetup const &es) {
  const auto &hgcons = es.getData(hdc_token_);
  rhtools_.setGeometry(es.getData(ticlGeomToken_), es.getData(ticlGeomLookupToken_), es.getData(ticlGeomLayersToken_));
  bfield_ = es.getHandle(bfield_token_);
  propagator_ = es.getHandle(propagator_token_);
  frontDisks_ = ticl::utils::buildHGCalFirstDisks(hgcons);
  for (auto *algo : {chargedHadronAlgo_.get(), muonAlgo_.get(), egammaAlgo_.get(), jetAlgo_.get()})
    algo->initialize(&hgcons, rhtools_, bfield_, propagator_);
}

void TICLInterpretationProducer::endStream() {
  if (solverStats_.budgetExhausted > 0)
    edm::LogWarning("TICLInterpretationProducer")
        << "Global arbitration: the search budget was reached in " << solverStats_.budgetExhausted << " of "
        << solverStats_.exactComponents << " components; the best set found was kept";
  LogDebug("TICLInterpretationProducer") << "Global arbitration: " << solverStats_.exactComponents
                                         << " components solved exactly, " << solverStats_.greedyComponents
                                         << " solved greedily";
}

void TICLInterpretationProducer::selectTracks(const edm::Handle<std::vector<reco::Track>> &tracks_h,
                                              const edm::Handle<std::vector<reco::Muon>> &muons_h,
                                              std::vector<bool> &selected,
                                              std::vector<bool> &muonTracks) const {
  const auto &tracks = *tracks_h;
  selected.assign(tracks.size(), false);
  muonTracks.assign(tracks.size(), false);
  for (unsigned int i = 0; i < tracks.size(); ++i) {
    const auto &tk = tracks[i];
    const int muId = PFMuonAlgo::muAssocToTrack(reco::TrackRef(tracks_h, i), *muons_h);
    const bool isMuon = muId != -1 && PFMuonAlgo::isMuon(reco::MuonRef(muons_h, muId));
    if (!cutTk_(tk) || (isMuon && !(*muons_h)[muId].isTrackerMuon()))
      continue;
    if (std::sqrt(tk.p() * tk.p() + mpion2) < tkEnergyCut_)
      continue;
    selected[i] = true;
    muonTracks[i] = isMuon;
  }
}

void TICLInterpretationProducer::runInference(const std::vector<reco::CaloCluster> &layerClusters,
                                              std::vector<Trackster> &tracksters,
                                              InferenceCache &cache,
                                              bool withRegression) const {
  const bool split = inferencePIDOnly_ && inferenceRegressionOnly_;
  std::vector<int> entryOf(tracksters.size(), -1);
  // Representatives of the distinct tracksters that need the PID and the regression, the PID only, or the regression
  // only.
  std::vector<Trackster> both, pidOnly, regressionOnly;
  std::vector<unsigned int> bothEntry, pidOnlyEntry, regressionOnlyEntry;
  std::vector<bool> queued;
  for (unsigned int i = 0; i < tracksters.size(); ++i) {
    const auto &ts = tracksters[i];
    if (std::any_of(ts.vertices().begin(), ts.vertices().end(), [&](unsigned int v) {
          return rhtools_.isBarrel(layerClusters[v].seed());
        }))
      continue;
    const unsigned int e = cache.entry(ts);
    entryOf[i] = static_cast<int>(e);
    if (queued.size() <= e)
      queued.resize(e + 1, false);
    if (queued[e])
      continue;
    const bool needPID = !cache[e].hasPID;
    const bool needRegression = !cache[e].hasRegression && (withRegression || !split);
    if (needPID && (needRegression || !split)) {
      both.push_back(ts);
      bothEntry.push_back(e);
    } else if (needPID) {
      pidOnly.push_back(ts);
      pidOnlyEntry.push_back(e);
    } else if (needRegression) {
      regressionOnly.push_back(ts);
      regressionOnlyEntry.push_back(e);
    }
    queued[e] = true;
  }
  inferenceAlgo_->runInference(layerClusters, both, rhtools_);
  for (unsigned int k = 0; k < both.size(); ++k) {
    cache.storePID(bothEntry[k], both[k]);
    cache.storeRegression(bothEntry[k], both[k]);
  }
  if (split) {
    inferencePIDOnly_->runInference(layerClusters, pidOnly, rhtools_);
    for (unsigned int k = 0; k < pidOnly.size(); ++k)
      cache.storePID(pidOnlyEntry[k], pidOnly[k]);
    inferenceRegressionOnly_->runInference(layerClusters, regressionOnly, rhtools_);
    for (unsigned int k = 0; k < regressionOnly.size(); ++k)
      cache.storeRegression(regressionOnlyEntry[k], regressionOnly[k]);
  }
  for (unsigned int i = 0; i < tracksters.size(); ++i)
    if (entryOf[i] >= 0)
      cache.apply(entryOf[i], tracksters[i]);
}

TICLInterpretationProducer::Opinions TICLInterpretationProducer::makeOpinions(
    const Inputs &muonInput,
    const Inputs &allInput,
    const Inputs &egammaInput,
    const edm::Handle<MtdHostCollection> &timing_h,
    InferenceCache &cache) const {
  Opinions opinions;
  // The charged-hadron interpretation sees all the selected tracks, muons included: a muon track also gets a hadron
  // hypothesis.
  muonAlgo_->makeOpinions(muonInput, timing_h, opinions.tracksters, opinions.hypotheses);
  chargedHadronAlgo_->makeOpinions(allInput, timing_h, opinions.tracksters, opinions.hypotheses);
  egammaAlgo_->makeOpinions(egammaInput, timing_h, opinions.tracksters, opinions.hypotheses);
  jetAlgo_->makeOpinions(allInput, timing_h, opinions.tracksters, opinions.hypotheses);

  // One PID for all the footprints: the photon and the neutral-hadron hypotheses of the same energy take their scores
  // from it.
  if (!opinions.tracksters.empty()) {
    const auto &layerClusters = allInput.layerClusters;
    assignPCAtoTracksters(opinions.tracksters,
                          layerClusters,
                          allInput.layerClustersTime,
                          rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z(),
                          rhtools_,
                          true);
    runInference(layerClusters, opinions.tracksters, cache, false);
    for (auto &h : opinions.hypotheses) {
      if (h.tracksterIdx < 0)
        continue;
      const auto &ts = opinions.tracksters[h.tracksterIdx];
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

  const auto &selected = allInput.maskedTracks;
  for (size_t iTrack = 0; iTrack < selected.size(); ++iTrack)
    if (selected[iTrack]) {
      Hypothesis h;
      h.type = Hypothesis::Type::RecoveryChargedHadron;
      h.trackIdx = static_cast<int>(iTrack);
      opinions.hypotheses.push_back(h);
    }
  return opinions;
}

std::vector<std::vector<unsigned int>> TICLInterpretationProducer::conflictGraph(
    const Opinions &opinions, const std::vector<reco::CaloCluster> &layerClusters) const {
  const auto &hypotheses = opinions.hypotheses;
  const unsigned int nH = hypotheses.size();
  std::vector<float> footE(nH, 0.f);
  std::unordered_map<unsigned int, std::vector<unsigned int>> lcOwners;
  for (unsigned int i = 0; i < nH; ++i)
    if (hypotheses[i].tracksterIdx >= 0)
      for (auto v : opinions.tracksters[hypotheses[i].tracksterIdx].vertices()) {
        footE[i] += layerClusters[v].energy();
        lcOwners[v].push_back(i);
      }
  std::map<std::pair<unsigned int, unsigned int>, float> sharedE;
  for (auto const &[lc, owners] : lcOwners)
    for (unsigned int a = 0; a < owners.size(); ++a)
      for (unsigned int b = a + 1; b < owners.size(); ++b)
        sharedE[{std::min(owners[a], owners[b]), std::max(owners[a], owners[b])}] += layerClusters[lc].energy();
  std::vector<std::vector<unsigned int>> adj(nH);
  auto addEdge = [&adj](unsigned int a, unsigned int b) {
    adj[a].push_back(b);
    adj[b].push_back(a);
  };
  for (auto const &[ab, e] : sharedE) {
    const float smaller = std::min(footE[ab.first], footE[ab.second]);
    if (smaller > 0.f && e / smaller > maxSharedEnergyFraction_)
      addEdge(ab.first, ab.second);
  }
  std::unordered_map<int, std::vector<unsigned int>> trackOwners;
  for (unsigned int i = 0; i < nH; ++i) {
    if (hypotheses[i].trackIdx >= 0)
      trackOwners[hypotheses[i].trackIdx].push_back(i);
    for (int iTk : hypotheses[i].trackIdxs)
      trackOwners[iTk].push_back(i);
  }
  for (auto const &[t, owners] : trackOwners)
    for (unsigned int a = 0; a < owners.size(); ++a)
      for (unsigned int b = a + 1; b < owners.size(); ++b)
        addEdge(owners[a], owners[b]);
  for (auto &n : adj) {
    std::sort(n.begin(), n.end());
    n.erase(std::unique(n.begin(), n.end()), n.end());
  }
  return adj;
}

std::vector<float> TICLInterpretationProducer::modelWeights(const Opinions &opinions,
                                                            const std::vector<std::vector<unsigned int>> &conflicts,
                                                            const std::vector<reco::Track> &tracks) const {
  const auto &hypotheses = opinions.hypotheses;
  const unsigned int nH = hypotheses.size();
  // Size of the conflict component of each hypothesis, without the track-only hypotheses: the model was trained
  // without them.
  std::vector<unsigned int> compId(nH, nH);
  std::vector<int> compSize(nH, 1);
  for (unsigned int s0 = 0; s0 < nH; ++s0) {
    if (compId[s0] != nH || isTrackOnly(hypotheses[s0]))
      continue;
    std::vector<unsigned int> stack{s0}, members;
    compId[s0] = s0;
    while (!stack.empty()) {
      const unsigned int v = stack.back();
      stack.pop_back();
      members.push_back(v);
      for (unsigned int u : conflicts[v])
        if (compId[u] == nH && !isTrackOnly(hypotheses[u])) {
          compId[u] = s0;
          stack.push_back(u);
        }
    }
    for (unsigned int v : members)
      compSize[v] = members.size();
  }
  // Tracks that carry a muon or an electron hypothesis.
  std::vector<bool> muonTrack(tracks.size(), false), electronTrack(tracks.size(), false);
  for (auto const &h : hypotheses)
    if (h.trackIdx >= 0 && h.type == Hypothesis::Type::Muon)
      muonTrack[h.trackIdx] = true;
    else if (h.trackIdx >= 0 && h.type == Hypothesis::Type::Electron)
      electronTrack[h.trackIdx] = true;

  // The model scores the hypotheses of the interpretations, with the features in the order of training.
  std::vector<unsigned int> rows;
  for (unsigned int i = 0; i < nH; ++i)
    if (!isTrackOnly(hypotheses[i]))
      rows.push_back(i);
  std::vector<float> features(rows.size() * kNHypothesisFeatures, 0.f);
  for (unsigned int r = 0; r < rows.size(); ++r) {
    const auto &h = hypotheses[rows[r]];
    float *x = &features[r * kNHypothesisFeatures];
    static_assert(static_cast<int>(Hypothesis::Type::RecoveryChargedHadron) == 6, "one-hot slots 0-6 are the types");
    x[static_cast<int>(h.type)] = 1.f;
    x[7] = h.score;
    const bool hasTs = h.tracksterIdx >= 0;
    const Trackster *ts = hasTs ? &opinions.tracksters[h.tracksterIdx] : nullptr;
    const float rawE = hasTs ? ts->raw_energy() : -1.f;
    const bool hasE = rawE > 0.f;
    const float tsEta = hasE ? ts->barycenter().eta() : -99.f;
    const float tsPhi = hasE ? ts->barycenter().phi() : -99.f;
    x[8] = hasE ? std::log1p(rawE) : -1.f;
    x[9] = hasE ? ts->raw_em_energy() / std::max(rawE, 1e-6f) : -1.f;
    x[10] = hasE ? std::abs(tsEta) : -1.f;
    for (int k = 0; k < 8; ++k)
      x[11 + k] = hasTs ? ts->id_probabilities(k) : -1.f;
    x[19] = hasTs ? static_cast<float>(ts->vertices().size()) : 0.f;
    x[20] = static_cast<float>(h.trackIdxs.size());
    float trackP = -1.f, trackPt = -1.f, trackEta = -1.f, dR = -1.f;
    if (h.trackIdx >= 0) {
      const auto &tk = tracks[h.trackIdx];
      trackP = tk.p();
      trackPt = tk.pt();
      trackEta = tk.eta();
      const auto dir = tk.outerOk() ? tk.outerMomentum() : tk.momentum();
      dR = hasE ? reco::deltaR(dir.eta(), dir.phi(), tsEta, tsPhi) : -1.f;
    } else if (!h.trackIdxs.empty()) {
      // Jet: scalar sums of p and pt, direction of the summed outer momenta.
      math::XYZVector sum;
      trackP = trackPt = 0.f;
      for (int iTk : h.trackIdxs) {
        const auto &tk = tracks[iTk];
        trackP += tk.p();
        trackPt += tk.pt();
        sum += tk.outerOk() ? tk.outerMomentum() : tk.momentum();
      }
      trackEta = sum.eta();
      dR = hasE ? reco::deltaR(sum.eta(), sum.phi(), tsEta, tsPhi) : -1.f;
    }
    x[21] = trackP > 0.f ? std::log1p(trackP) : -1.f;
    x[22] = trackPt;
    x[23] = trackP > 0.f ? std::abs(trackEta) : -1.f;
    const float eOverP = (hasE && trackP > 0.f) ? rawE / trackP : -1.f;
    x[24] = std::clamp(eOverP, -1.f, 20.f);
    x[25] = dR;
    // x[26]: GSF track flag of the training sample, always 0.
    x[27] = static_cast<float>(compSize[rows[r]]);
    // Tracks of the hypothesis that also have a muon or an electron hypothesis.
    auto countLeptonHyps = [&](int iTk) {
      if (iTk >= 0) {
        x[28] += muonTrack[iTk] ? 1.f : 0.f;
        x[29] += electronTrack[iTk] ? 1.f : 0.f;
      }
    };
    if (h.trackIdxs.empty())
      countLeptonHyps(h.trackIdx);
    else
      for (int iTk : h.trackIdxs)
        countLeptonHyps(iTk);
  }
  const std::vector<float> logits = arbitrationModel_.logits(features, rows.size());
  std::vector<float> weights(nH, trackOnlyWeight_);
  for (unsigned int r = 0; r < rows.size(); ++r)
    weights[rows[r]] = plattA_ * logits[r] + plattB_;
  return weights;
}

void TICLInterpretationProducer::putHypothesisDump(edm::Event &evt,
                                                   const Opinions &opinions,
                                                   const std::vector<bool> &accepted,
                                                   const std::vector<float> &weights) const {
  auto iv = []() { return std::make_unique<std::vector<int>>(); };
  auto fv = []() { return std::make_unique<std::vector<float>>(); };
  auto type = iv(), track = iv(), acc = iv(), lcOff = iv(), lcs = iv(), jetOff = iv(), jetTk = iv();
  auto score = fv(), rawE = fv(), rawEmE = fv(), x = fv(), y = fv(), z = fv(), pid = fv();
  lcOff->push_back(0);
  jetOff->push_back(0);
  for (unsigned int idx = 0; idx < opinions.hypotheses.size(); ++idx) {
    const auto &h = opinions.hypotheses[idx];
    type->push_back(static_cast<int>(h.type));
    track->push_back(h.trackIdx);
    acc->push_back(accepted[idx] ? 1 : 0);
    score->push_back(h.score);
    if (h.tracksterIdx >= 0) {
      const auto &ts = opinions.tracksters[h.tracksterIdx];
      rawE->push_back(ts.raw_energy());
      rawEmE->push_back(ts.raw_em_energy());
      x->push_back(ts.barycenter().x());
      y->push_back(ts.barycenter().y());
      z->push_back(ts.barycenter().z());
      for (int k = 0; k < 8; ++k)
        pid->push_back(ts.id_probabilities(k));
      for (auto v : ts.vertices())
        lcs->push_back(static_cast<int>(v));
    } else {
      for (auto *v : {rawE.get(), rawEmE.get(), x.get(), y.get(), z.get()})
        v->push_back(-1.f);
      for (int k = 0; k < 8; ++k)
        pid->push_back(-1.f);
    }
    lcOff->push_back(static_cast<int>(lcs->size()));
    for (int iTk : h.trackIdxs)
      jetTk->push_back(iTk);
    jetOff->push_back(static_cast<int>(jetTk->size()));
  }
  evt.put(std::move(type), "hypType");
  evt.put(std::move(track), "hypTrack");
  evt.put(std::move(acc), "hypAccepted");
  evt.put(std::move(lcOff), "hypLCOffsets");
  evt.put(std::move(lcs), "hypLCs");
  evt.put(std::move(jetOff), "hypJetOffsets");
  evt.put(std::move(jetTk), "hypJetTracks");
  evt.put(std::move(score), "hypScore");
  evt.put(std::move(rawE), "hypRawE");
  evt.put(std::move(rawEmE), "hypRawEmE");
  evt.put(std::move(x), "hypX");
  evt.put(std::move(y), "hypY");
  evt.put(std::move(z), "hypZ");
  evt.put(std::move(pid), "hypPid");
  evt.put(std::make_unique<std::vector<float>>(weights), "hypWeight");
}

std::vector<Trackster> TICLInterpretationProducer::selectLeftovers(const edm::MultiSpan<Trackster> &tracksters,
                                                                   const std::vector<reco::CaloCluster> &layerClusters,
                                                                   std::vector<bool> &held) const {
  std::vector<unsigned int> order(tracksters.size());
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(), [&](unsigned int a, unsigned int b) {
    return tracksters[a].raw_energy() > tracksters[b].raw_energy();
  });
  std::vector<Trackster> leftovers;
  for (unsigned int iTs : order) {
    const auto &ts = tracksters[iTs];
    Trackster rest(ts);
    rest.vertices().clear();
    rest.vertex_multiplicity().clear();
    float total = 0.f, raw = 0.f;
    for (size_t k = 0; k < ts.vertices().size(); ++k) {
      const float e = layerClusters[ts.vertices()[k]].energy();
      total += e;
      if (!held[ts.vertices()[k]]) {
        rest.vertices().push_back(ts.vertices()[k]);
        rest.vertex_multiplicity().push_back(k < ts.vertex_multiplicity().size() ? ts.vertex_multiplicity()[k] : 1.f);
        raw += e;
      }
    }
    if (!(total > 0.f) || rest.vertices().empty() ||
        (rest.vertices().size() < ts.vertices().size() && raw < claimMinEnergy_))
      continue;
    for (auto v : rest.vertices())
      held[v] = true;
    leftovers.push_back(std::move(rest));
  }
  return leftovers;
}

std::vector<int> TICLInterpretationProducer::claimAlongTracks(const Opinions &opinions,
                                                              const std::vector<bool> &accepted,
                                                              const std::vector<reco::Track> &tracks,
                                                              const std::vector<TrackImpact> &impacts,
                                                              const std::vector<reco::CaloCluster> &layerClusters,
                                                              const std::vector<unsigned int> &pool) const {
  const auto &hypotheses = opinions.hypotheses;
  std::vector<int> claimingTrack(layerClusters.size(), -1);
  // The pool, binned in (|eta|, phi) per endcap. Grid key: (endcap x kSideStride + |eta| bin) x kEtaStride + phi bin.
  constexpr float kBin = 0.05f;
  constexpr int kSideStride = 200;
  constexpr int kEtaStride = 1000;
  auto key = [](float eta, float phi, int side) {
    return (side * kSideStride + static_cast<int>(std::floor(std::abs(eta) / kBin))) * kEtaStride +
           static_cast<int>(std::floor((phi + static_cast<float>(M_PI)) / kBin));
  };
  std::unordered_map<int, std::vector<unsigned int>> grid;
  for (unsigned int v : pool) {
    const auto &lc = layerClusters[v];
    grid[key(lc.eta(), lc.phi(), lc.z() > 0 ? 1 : 0)].push_back(v);
  }
  // Sources: (expected deposit still to claim, track). A single-track hypothesis on an EM footprint does not claim:
  // its candidate takes the energy of the trackster.
  std::vector<std::pair<float, unsigned int>> sources;
  for (unsigned int i = 0; i < hypotheses.size(); ++i) {
    if (!accepted[i])
      continue;
    const auto &h = hypotheses[i];
    if (h.type == Hypothesis::Type::Muon || h.type == Hypothesis::Type::Electron || isNeutral(h))
      continue;
    const Trackster *footprint = h.tracksterIdx >= 0 ? &opinions.tracksters[h.tracksterIdx] : nullptr;
    if (h.type != Hypothesis::Type::Jet && footprint != nullptr && !footprint->isHadronic())
      continue;
    const std::vector<int> tks = h.type == Hypothesis::Type::Jet ? h.trackIdxs : std::vector<int>{h.trackIdx};
    float sumP = 0.f;
    float expected = 0.f;
    for (int t : tks) {
      const float p = static_cast<float>(tracks[t].p());
      sumP += tracks[t].p();
      expected += (claimResponse_ + claimResponseLogSlope_ * std::log(p)) * p;
    }
    const float deficit = expected - (footprint != nullptr ? footprint->raw_energy() : 0.f);
    if (!(deficit > 0.f) || !(sumP > 0.f))
      continue;
    for (int t : tks)
      sources.emplace_back(deficit * static_cast<float>(tracks[t].p()) / sumP, static_cast<unsigned int>(t));
  }
  // Highest momentum first.
  std::sort(sources.begin(), sources.end(), [&](auto const &a, auto const &b) {
    const double pa = tracks[a.second].p(), pb = tracks[b.second].p();
    return pa != pb ? pa > pb : a.second < b.second;
  });
  const int nPhi = static_cast<int>(std::ceil(2.f * static_cast<float>(M_PI) / kBin));
  const float maxRadius = claimRadius0_ + claimRadiusSlope_ * kClaimMaxDepth;
  // Clusters up to kFrontTolerance [cm] in front of the impact are kept.
  constexpr float kFrontTolerance = 1.f;
  std::vector<std::pair<float, unsigned int>> candidates;
  for (auto const &[deficit, t] : sources) {
    const auto &impact = impacts[t];
    if (!impact.valid)
      continue;
    const int side = impact.position.z() > 0.f ? 1 : 0;
    const float zFront = std::abs(impact.position.z());
    // Search window in bins: the line from the impact to kClaimMaxDepth, widened by the largest claim radius.
    const auto dirUnit = impact.direction.unit();
    const GlobalPoint deepest = impact.position + dirUnit * (kClaimMaxDepth / std::abs(dirUnit.z()));
    const float rhoFront = std::max(impact.position.perp(), 1.f);
    const int nWin = std::min(static_cast<int>(std::ceil(maxRadius / rhoFront / kBin)) + 1, nPhi / 2);
    const int ieFront = static_cast<int>(std::floor(std::abs(impact.position.eta()) / kBin));
    const int ieDeep = static_cast<int>(std::floor(std::abs(deepest.eta()) / kBin));
    const int ip0 = static_cast<int>(std::floor((impact.position.barePhi() + static_cast<float>(M_PI)) / kBin));
    const int dpDeep =
        static_cast<int>(std::lround(reco::deltaPhi(deepest.barePhi(), impact.position.barePhi()) / kBin));
    candidates.clear();
    // A phi bin is visited at most once.
    const int dpLow = std::min(0, dpDeep) - nWin;
    const int dpHigh = std::min(std::max(0, dpDeep) + nWin, dpLow + nPhi - 1);
    for (int ie = std::min(ieFront, ieDeep) - nWin; ie <= std::max(ieFront, ieDeep) + nWin; ++ie)
      for (int dp = dpLow; dp <= dpHigh; ++dp) {
        const int ip = ((ip0 + dp) % nPhi + nPhi) % nPhi;
        auto it = grid.find((side * kSideStride + ie) * kEtaStride + ip);
        if (it == grid.end())
          continue;
        for (auto v : it->second) {
          if (claimingTrack[v] >= 0)
            continue;
          const auto &lc = layerClusters[v];
          const float depth = std::abs(static_cast<float>(lc.z())) - zFront;
          if (depth < -kFrontTolerance)
            continue;
          float dist = 0.f;
          if (!impactTransverseDistanceBelow(
                  impact, lc.position(), claimRadius0_ + claimRadiusSlope_ * std::max(depth, 0.f), dist))
            continue;
          // Nearest first, CE-H before CE-E: in CE-E the clusters near a track hold much EM energy.
          const bool inCEE = lc.hitsAndFractions()[0].first.det() == DetId::HGCalEE;
          candidates.emplace_back(inCEE ? dist + kClaimCalorimeterOffset : dist, v);
        }
      }
    std::sort(candidates.begin(), candidates.end());
    float claimed = 0.f;
    for (auto const &[rank, v] : candidates) {
      if (claimed >= deficit)
        break;
      claimingTrack[v] = static_cast<int>(t);
      claimed += layerClusters[v].energy();
    }
  }
  return claimingTrack;
}

void TICLInterpretationProducer::assignNeutralSpecies(const std::vector<Trackster> &tracksters,
                                                      const std::vector<int> &neutralIdx,
                                                      std::vector<int> &neutralPdg) const {
  std::vector<unsigned int> rows;
  std::vector<float> x;
  for (unsigned int k = 0; k < neutralIdx.size(); ++k) {
    const auto &ts = tracksters[neutralIdx[k]];
    if (!(ts.raw_energy() > 0.f))
      continue;
    rows.push_back(k);
    for (int c = 0; c < 8; ++c)
      x.push_back(ts.id_probabilities(c));
    x.push_back(ts.raw_em_energy() / ts.raw_energy());
    x.push_back(std::log1p(ts.raw_energy()));
    x.push_back(std::abs(ts.barycenter().eta()));
    x.push_back(static_cast<float>(ts.vertices().size()));
  }
  const auto logits = neutralModel_.logits(x, rows.size());
  for (unsigned int r = 0; r < rows.size(); ++r) {
    const float *m = &logits[r * kNSpecies];
    const float mMax = std::max({m[0], m[1], m[2]});
    const float pPhoton = std::exp(m[0] - mMax), pPi0 = std::exp(m[1] - mMax), pHadron = std::exp(m[2] - mMax);
    const float pEM = (pPhoton + pPi0) / (pPhoton + pPi0 + pHadron);
    int &pdg = neutralPdg[rows[r]];
    if (pdg == 22 || pEM >= neutralModelThreshold_)
      pdg = pPi0 > pPhoton ? 111 : 22;
    else
      pdg = 130;
  }
}

void TICLInterpretationProducer::produce(edm::Event &evt, const edm::EventSetup &es) {
  const auto &layerClusters = evt.get(clusters_token_);
  const auto &layerClustersTimes = evt.get(clustersTime_token_);
  edm::MultiSpan<Trackster> tracksters, superclusters;
  for (auto const &token : tracksters_tokens_)
    tracksters.add(evt.get(token));
  for (auto const &token : egamma_tracksters_tokens_)
    superclusters.add(evt.get(token));
  const auto tracks_h = evt.getHandle(tracks_token_);
  const auto &tracks = *tracks_h;
  const auto muons_h = evt.getHandle(muons_token_);
  edm::Handle<MtdHostCollection> timing_h;
  if (useMTDTiming_)
    timing_h = evt.getHandle(timing_token_);

  std::vector<bool> selected, muonTracks;
  selectTracks(tracks_h, muons_h, selected, muonTracks);
  std::vector<TrackImpact> impacts(tracks.size());
  for (size_t i = 0; i < tracks.size(); ++i)
    if (selected[i])
      impacts[i] = propagateToHGCalFront(tracks[i], bfield_.product(), *propagator_, frontDisks_);

  // Hypotheses, weights and the accepted set.
  InferenceCache inferenceCache;
  const Inputs muonInput(evt, es, layerClusters, layerClustersTimes, tracksters, tracks_h, muonTracks, &impacts);
  const Inputs allInput(evt, es, layerClusters, layerClustersTimes, tracksters, tracks_h, selected, &impacts);
  const Inputs egammaInput(evt, es, layerClusters, layerClustersTimes, superclusters, tracks_h, selected, &impacts);
  const Opinions opinions = makeOpinions(muonInput, allInput, egammaInput, timing_h, inferenceCache);
  const auto &hypotheses = opinions.hypotheses;
  const auto conflicts = conflictGraph(opinions, layerClusters);
  const auto weights = modelWeights(opinions, conflicts, tracks);
  // With a bonus per track larger than the spread of the model weights, the solver keeps every selected track, and
  // the model chooses between the hypotheses of each track.
  std::vector<float> solverWeights(weights);
  for (unsigned int i = 0; i < hypotheses.size(); ++i)
    solverWeights[i] +=
        trackBonus_ * static_cast<float>((hypotheses[i].trackIdx >= 0 ? 1 : 0) + hypotheses[i].trackIdxs.size());
  const auto accepted =
      maxWeightIndependentSet(solverWeights, conflicts, maxExactComponent_, searchBudget_, solverStats_);
  if (dumpHypotheses_)
    putHypothesisDump(evt, opinions, accepted, weights);

  // Leftovers and the track claim. The pool of the claim: the layer clusters of the leftovers and of the
  // claimUnlinkedTracksters outside the input tracksters, without the winner footprints.
  std::vector<bool> held(layerClusters.size(), false);
  std::vector<bool> inWinner(layerClusters.size(), false);
  for (unsigned int i = 0; i < hypotheses.size(); ++i)
    if (accepted[i] && hypotheses[i].tracksterIdx >= 0)
      for (auto v : opinions.tracksters[hypotheses[i].tracksterIdx].vertices())
        held[v] = inWinner[v] = true;
  const auto leftovers = selectLeftovers(tracksters, layerClusters, held);
  std::vector<unsigned int> pool;
  std::vector<bool> inPool(layerClusters.size(), false);
  auto addToPool = [&](unsigned int v) {
    if (!inWinner[v] && !inPool[v]) {
      inPool[v] = true;
      pool.push_back(v);
    }
  };
  for (auto const &ts : leftovers)
    for (auto v : ts.vertices())
      addToPool(v);
  if (!claimUnlinkedTracksters_tokens_.empty()) {
    std::vector<bool> inInput(layerClusters.size(), false);
    for (unsigned int j = 0; j < tracksters.size(); ++j)
      for (auto v : tracksters[j].vertices())
        inInput[v] = true;
    for (auto const &token : claimUnlinkedTracksters_tokens_)
      for (auto const &ts : evt.get(token))
        for (auto v : ts.vertices())
          if (!inInput[v])
            addToPool(v);
  }
  const auto claimingTrack = claimAlongTracks(opinions, accepted, tracks, impacts, layerClusters, pool);
  if (dumpHypotheses_) {
    auto claimedOut = std::make_unique<std::vector<int>>();
    for (unsigned int v = 0; v < claimingTrack.size(); ++v)
      if (claimingTrack[v] >= 0)
        claimedOut->push_back(static_cast<int>(v));
    evt.put(std::move(claimedOut), "trackClaimedLC");
  }

  // Final tracksters: the winners, the neutrals without the claimed layer clusters, and one claim trackster per
  // claiming track. A neutral is kept when layer clusters are left and, if the claim took one, its raw energy is at
  // least claimMinEnergy.
  auto result = std::make_unique<std::vector<Trackster>>();
  auto withoutClaims = [&](const Trackster &ts) {
    Trackster out(ts);
    out.vertices().clear();
    out.vertex_multiplicity().clear();
    float raw = 0.f;
    for (size_t k = 0; k < ts.vertices().size(); ++k)
      if (claimingTrack[ts.vertices()[k]] < 0) {
        out.vertices().push_back(ts.vertices()[k]);
        out.vertex_multiplicity().push_back(k < ts.vertex_multiplicity().size() ? ts.vertex_multiplicity()[k] : 1.f);
        raw += layerClusters[ts.vertices()[k]].energy();
      }
    const bool keep =
        !out.vertices().empty() && (out.vertices().size() == ts.vertices().size() || raw >= claimMinEnergy_);
    return std::make_pair(std::move(out), keep);
  };
  std::vector<int> winnerResultIdx(hypotheses.size(), -1);
  for (unsigned int idx = 0; idx < hypotheses.size(); ++idx) {
    if (!accepted[idx] || hypotheses[idx].tracksterIdx < 0)
      continue;
    const auto &ts = opinions.tracksters[hypotheses[idx].tracksterIdx];
    if (isNeutral(hypotheses[idx])) {
      auto [pruned, keep] = withoutClaims(ts);
      if (!keep)
        continue;
      winnerResultIdx[idx] = static_cast<int>(result->size());
      result->push_back(std::move(pruned));
      continue;
    }
    winnerResultIdx[idx] = static_cast<int>(result->size());
    result->push_back(ts);
  }
  std::vector<int> leftoverResultIdx;
  for (auto const &leftover : leftovers) {
    auto [pruned, keep] = withoutClaims(leftover);
    if (!keep)
      continue;
    leftoverResultIdx.push_back(static_cast<int>(result->size()));
    result->push_back(std::move(pruned));
  }
  // No candidate uses a claim trackster: the charged candidate takes its energy from the track.
  auto trackToClaimTrackster = std::make_unique<std::vector<int>>(tracks.size(), -1);
  for (unsigned int v = 0; v < claimingTrack.size(); ++v) {
    if (claimingTrack[v] < 0)
      continue;
    int &iClaim = (*trackToClaimTrackster)[claimingTrack[v]];
    if (iClaim < 0) {
      iClaim = static_cast<int>(result->size());
      result->emplace_back();
    }
    (*result)[iClaim].vertices().push_back(v);
    (*result)[iClaim].vertex_multiplicity().push_back(1.f);
  }
  assignPCAtoTracksters(
      *result, layerClusters, layerClustersTimes, rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z(), rhtools_, true);
  runInference(layerClusters, *result, inferenceCache, true);
  // An EM trackster below emRawEnergyBelow keeps its raw energy: the regression is not trained for it.
  for (auto &ts : *result)
    if (!ts.isHadronic() && ts.raw_energy() < emRawEnergyBelow_)
      ts.setRegressedEnergy(ts.raw_energy());

  // Assignment maps.
  auto trackToTrackster = std::make_unique<std::vector<int>>(tracks.size(), -1);
  auto trackMode = std::make_unique<std::vector<int>>(tracks.size(), kNotSelected);
  for (size_t i = 0; i < selected.size(); ++i)
    if (selected[i])
      (*trackMode)[i] = kUnassigned;
  auto neutralIdx = std::make_unique<std::vector<int>>();
  auto neutralPdg = std::make_unique<std::vector<int>>();
  auto assign = [&](int iTrack, TrackMode mode, int iTrackster) {
    (*trackMode)[iTrack] = mode;
    (*trackToTrackster)[iTrack] = iTrackster;
  };
  for (unsigned int idx = 0; idx < hypotheses.size(); ++idx) {
    if (!accepted[idx])
      continue;
    const auto &h = hypotheses[idx];
    const int iResult = winnerResultIdx[idx];
    switch (h.type) {
      case Hypothesis::Type::Muon:
        assign(h.trackIdx, kMuon, iResult);
        break;
      case Hypothesis::Type::Electron:
        assign(h.trackIdx, kElectron, iResult);
        break;
      case Hypothesis::Type::ChargedHadron:
        assign(h.trackIdx, kChargedHadron, iResult);
        break;
      case Hypothesis::Type::Jet:
        for (int iTk : h.trackIdxs)
          assign(iTk, kJetMember, iResult);
        break;
      case Hypothesis::Type::RecoveryChargedHadron:
        assign(h.trackIdx, kRecovery, iResult);
        break;
      case Hypothesis::Type::Photon:
      case Hypothesis::Type::NeutralHadron:
        // A neutral winner the track claim emptied has no final trackster. A photon winner is a photon or a pi0; a
        // neutral-hadron winner is typed by the neutral species model, as a leftover.
        if (iResult >= 0) {
          neutralIdx->push_back(iResult);
          neutralPdg->push_back(h.type == Hypothesis::Type::Photon ? 22 : 0);
        }
        break;
    }
  }
  for (int iTrackster : leftoverResultIdx) {
    neutralIdx->push_back(iTrackster);
    neutralPdg->push_back(0);
  }
  assignNeutralSpecies(*result, *neutralIdx, *neutralPdg);

  evt.put(std::move(result));
  evt.put(std::move(trackToClaimTrackster), "trackToClaimTrackster");
  evt.put(std::move(trackToTrackster), "trackToTrackster");
  evt.put(std::move(trackMode), "trackMode");
  evt.put(std::move(neutralIdx), "neutralIdx");
  evt.put(std::move(neutralPdg), "neutralPdg");
}

void TICLInterpretationProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("tracksters_collections", {edm::InputTag("ticlTracksterLinks")})
      ->setComment("Linked tracksters: the footprints of the charged-hadron, jet and muon hypotheses.");
  desc.add<std::vector<edm::InputTag>>("egamma_tracksters_collections",
                                       {edm::InputTag("ticlTracksterLinksSuperclusteringDNN")})
      ->setComment("Superclusters: the footprints of the electron and photon hypotheses.");
  desc.add<std::vector<edm::InputTag>>("claimUnlinkedTracksters", {edm::InputTag("ticlTrackstersRecovery")})
      ->setComment("Tracksters whose layer clusters outside the input tracksters join the track-claim pool.");
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("layer_clustersTime", edm::InputTag("hgcalMergeLayerClusters", "timeLayerCluster"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons1stStep"));
  desc.add<bool>("useMTDTiming", true)->setComment("Give the MTD track times to the interpretations.");
  desc.add<edm::InputTag>("timingSoA", edm::InputTag("mtdSoA"));
  desc.add<std::string>("propagator", "PropagatorWithMaterial");
  desc.add<float>("tkEnergyCut", 2.0f)->setComment("Min track energy sqrt(p^2 + m_pi^2) [GeV] of a selected track.");
  desc.add<std::string>("cutTk", "1.48 < abs(eta) < 3.0 && pt > 1. && quality(\"highPurity\") && ptError < 0.5 * pt")
      ->setComment("Selection of the tracks.");

  edm::ParameterSetDescription chargedHadronDesc;
  chargedHadronDesc.addNode(edm::PluginDescription<TICLGeneralInterpretationPluginFactory>("type", "General", true));
  desc.add<edm::ParameterSetDescription>("interpretationDescPSet", chargedHadronDesc);
  edm::ParameterSetDescription jetDesc;
  jetDesc.addNode(edm::PluginDescription<TICLGeneralInterpretationPluginFactory>("type", "Jet", true));
  desc.add<edm::ParameterSetDescription>("jetInterpretationDescPSet", jetDesc);
  edm::ParameterSetDescription muonDesc;
  muonDesc.addNode(edm::PluginDescription<TICLGeneralInterpretationPluginFactory>("type", "Muon", true));
  desc.add<edm::ParameterSetDescription>("muonInterpretationDescPSet", muonDesc);
  edm::ParameterSetDescription egammaDesc;
  egammaDesc.addNode(edm::PluginDescription<TICLGeneralInterpretationPluginFactory>("type", "EGamma", true));
  desc.add<edm::ParameterSetDescription>("egammaInterpretationDescPSet", egammaDesc);

  desc.add<std::string>("inferenceAlgo", "TracksterInferenceByPFN")
      ->setComment("PID and energy regression of the footprints and of the final tracksters.");
  edm::ParameterSetDescription inferenceDesc;
  inferenceDesc.addNode(edm::PluginDescription<TracksterInferenceAlgoFactory>("type", "TracksterInferenceByPFN", true));
  desc.add<edm::ParameterSetDescription>("pluginInferenceAlgoTracksterInferenceByPFN", inferenceDesc);

  desc.add<float>("arbitrationMaxSharedEnergyFraction", 0.2f)
      ->setComment("Two hypotheses conflict above this fraction of the smaller footprint energy in shared clusters.");
  desc.add<std::string>("arbitrationModelFile", "RecoTICL/Interpretation/data/arbitration/hypothesis_mlp_v1.onnx")
      ->setComment("ONNX model of the probability that a hypothesis is correct (logit).");
  desc.add<float>("arbitrationPlattA", 0.9908671975135803f)->setComment("Platt calibration: logit' = a x logit + b.");
  desc.add<float>("arbitrationPlattB", -0.02162671647965908f);
  desc.add<float>("arbitrationTrackBonus", 20.0f)->setComment("Weight added per track a hypothesis covers.");
  desc.add<float>("arbitrationTrackOnlyWeight", -1.0f)->setComment("Weight of a track-only hypothesis.");
  desc.add<unsigned int>("arbitrationMaxExactComponent", 64)
      ->setComment("Largest component solved by branch and bound (at most 64); larger ones are greedy.");
  desc.add<unsigned int>("arbitrationSearchBudget", 1000000)
      ->setComment("Max branch-and-bound calls per component; then the best set found is kept.");
  desc.add<float>("arbitrationTrackClaimResponse", 0.67f)
      ->setComment("Track claim: expected raw HGCAL deposit of a charged hadron per GeV of track momentum.");
  desc.add<float>("arbitrationTrackClaimResponseLogSlope", 0.04f)
      ->setComment("Track claim: the expected deposit per GeV is response + this x ln(p / GeV).");
  desc.add<float>("arbitrationTrackClaimRadius0", 3.f)
      ->setComment("Track claim: radius [cm] around the propagated track at the HGCAL front.");
  desc.add<float>("arbitrationTrackClaimRadiusSlope", 0.05f)
      ->setComment("Track claim: growth of the radius per cm of depth.");
  desc.add<float>("arbitrationTrackClaimMinEnergy", 0.5f)
      ->setComment(
          "A neutral that lost a layer cluster to a winner or to the track claim is dropped below this raw "
          "energy [GeV].");
  desc.add<std::string>("neutralModelFile", "RecoTICL/Interpretation/data/arbitration/neutralSpecies_mlp_v1.onnx")
      ->setComment("ONNX 3-class model (photon, pi0, neutral hadron; logits) that types the neutral candidates.");
  desc.add<float>("neutralModelThreshold", 0.7f)->setComment("P(photon) + P(pi0) at and above which a neutral is EM.");
  desc.add<float>("emRawEnergyBelow", 50.f)
      ->setComment("EM final tracksters below this raw energy [GeV] keep the raw energy.");
  desc.add<bool>("dumpHypotheses", false)
      ->setComment("Write the hypotheses as flat products, for the training of the hypothesis model.");
  descriptions.add("ticlInterpretationProducer", desc);
}

DEFINE_FWK_MODULE(TICLInterpretationProducer);
