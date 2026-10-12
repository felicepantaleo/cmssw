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
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <tuple>
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
#include "RecoTICL/Interpretation/interface/AssignmentMaps.h"
#include "RecoTICL/Interpretation/interface/TypeProbabilities.h"
#include "RecoTICL/Interpretation/interface/MaxWeightIndependentSet.h"
#include "RecoTICL/Interpretation/interface/ModelFeatures.h"
#include "RecoTICL/Interpretation/interface/MuonKinematics.h"
#include "RecoTICL/Interpretation/interface/TICLInterpretationAlgoBase.h"
#include "TICLInterpretationPluginFactory.h"
#include "RecoTICL/Interpretation/interface/TrackImpact.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

using namespace ticl;

namespace {
  // A network with input "features" (rows x nFeatures) and one output, the logits (rows x nClasses). A non-finite
  // feature enters as 0; a non-finite logit throws.
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

    std::vector<float> logits(std::vector<float> x, unsigned int rows) const {
      if (rows == 0)
        return {};
      for (auto &v : x)
        if (!std::isfinite(v))
          v = 0.f;
      cms::Ort::FloatArrays input;
      input.push_back(std::move(x));
      auto result =
          network_.run({"features"}, input, {{static_cast<int64_t>(rows), static_cast<int64_t>(nFeatures_)}}, {}, rows);
      if (result.size() != 1 || result[0].size() != static_cast<std::size_t>(rows) * nClasses_)
        throw cms::Exception("LogicError")
            << "ArbitrationModel: expected " << rows * nClasses_ << " logits from the network";
      if (std::any_of(result[0].begin(), result[0].end(), [](float v) { return !std::isfinite(v); }))
        throw cms::Exception("LogicError") << "ArbitrationModel: non-finite logit";
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
    // The entry of ts, or nullptr when there is none.
    const Entry *find(const Trackster &ts) const {
      const auto range = index_.equal_range(hash(ts));
      for (auto it = range.first; it != range.second; ++it) {
        const auto &e = entries_[it->second];
        if (e.vertices == ts.vertices() && e.multiplicities == ts.vertex_multiplicity())
          return &e;
      }
      return nullptr;
    }
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

  bool isNeutral(const Hypothesis &h) {
    return h.type == Hypothesis::Type::Photon || h.type == Hypothesis::Type::NeutralHadron;
  }

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

  // Selected tracks: energy (pion mass) at least tkEnergyCut, cutTk, and not a muon without a tracker-muon fit. Among
  // them, the tracks of identified muons.
  struct TrackSelection {
    std::vector<bool> selected;
    std::vector<bool> muon;
  };
  TrackSelection selectTracks(const edm::Handle<std::vector<reco::Track>> &tracks_h,
                              const edm::Handle<std::vector<reco::Muon>> &muons_h) const;
  // Runs the inference once per distinct trackster of the event and copies the result to the others: the PID, and the
  // regressed energy when withRegression is set or when the PID and the regression cannot run apart. An EM trackster
  // below emRawEnergyBelow with a known PID gets no regression. Tracksters with a barrel layer cluster are not changed.
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
  // For each Jet hypothesis and each Electron hypothesis of one of its tracks: the Jet without that track, on the
  // footprint without the layer clusters of the Electron footprint, scored by the balance of E and the summed p. Two
  // tracks or more left give a Jet hypothesis, one track a Recovery hypothesis.
  static void addJetVariantsWithoutElectrons(Opinions &opinions,
                                             const std::vector<reco::Track> &tracks,
                                             const std::vector<reco::CaloCluster> &layerClusters);
  // Two hypotheses conflict when they share a track, or layer clusters with more than maxSharedEnergyFraction of the
  // smaller footprint energy. Returns the sorted neighbours of each hypothesis.
  std::vector<std::vector<unsigned int>> conflictGraph(const Opinions &opinions,
                                                       const std::vector<reco::CaloCluster> &layerClusters) const;
  // Size of the conflict component of each hypothesis, without the track-only hypotheses: the model was trained
  // without them.
  static std::vector<int> componentSizes(const std::vector<Hypothesis> &hypotheses,
                                         const std::vector<std::vector<unsigned int>> &conflicts);
  // Platt-calibrated logit of the hypothesis model; trackOnlyWeight for a track-only hypothesis; -infinity for a jet
  // on an EM footprint when vetoJetOnEMFootprint is set.
  std::vector<float> modelWeights(const Opinions &opinions,
                                  const std::vector<std::vector<unsigned int>> &conflicts,
                                  const std::vector<reco::Track> &tracks) const;
  // The trackster without the layer clusters that removed marks, when it is kept: layer clusters are left with a
  // positive raw energy, at least claimMinEnergy when one was removed.
  std::optional<Trackster> without(const Trackster &ts,
                                   const std::vector<bool> &removed,
                                   const std::vector<reco::CaloCluster> &layerClusters) const;
  // Leftovers: the input tracksters in descending raw energy, without the layer clusters that the winners and the
  // leftovers before them hold. held marks the layer clusters of the winners and of the leftovers. source is the input
  // trackster of each leftover.
  std::vector<Trackster> selectLeftovers(const edm::MultiSpan<Trackster> &tracksters,
                                         const std::vector<reco::CaloCluster> &layerClusters,
                                         std::vector<bool> &held,
                                         std::vector<unsigned int> &source) const;
  // Leftovers and the pool of the track claim: the layer clusters of the leftovers and of the claimUnlinkedTracksters
  // outside the input tracksters, without the winner footprints. With claimProtectEMThreshold, a leftover whose input
  // trackster has P(photon) + P(electron) at or above it keeps its layer clusters out of the pool.
  struct ClaimPool {
    std::vector<Trackster> leftovers;
    std::vector<unsigned int> pool;
  };
  ClaimPool claimPool(const edm::Event &evt,
                      const Opinions &opinions,
                      const std::vector<bool> &accepted,
                      const InferenceCache &inferenceCache,
                      const edm::MultiSpan<Trackster> &tracksters,
                      const std::vector<reco::CaloCluster> &layerClusters) const;
  // Final tracksters: the winners, each without the layer clusters of the heavier winners; the leftovers without the
  // claimed layer clusters; and one claim trackster per claiming track. A trackster that lost a layer cluster is kept
  // when its raw energy is at least claimMinEnergy. A charged winner that loses its footprint keeps its track.
  struct FinalTracksters {
    std::unique_ptr<std::vector<Trackster>> tracksters;
    std::vector<int> winnerIndex;    // the final trackster of each hypothesis (-1: none)
    std::vector<int> leftoverIndex;  // the final trackster of each kept leftover
  };
  FinalTracksters finalTracksters(const Opinions &opinions,
                                  const std::vector<bool> &accepted,
                                  const std::vector<float> &solverWeights,
                                  const std::vector<Trackster> &leftovers,
                                  const std::vector<int> &claimingTrack,
                                  const std::vector<reco::CaloCluster> &layerClusters,
                                  std::size_t nTracks) const;
  // Assignment maps: the mode and the final trackster of each track, the neutral final tracksters and their pdgId
  // (0 until assignNeutralSpecies types them).
  struct AssignmentProducts {
    std::unique_ptr<std::vector<int>> trackToTrackster;
    std::unique_ptr<std::vector<int>> trackMode;
    std::unique_ptr<std::vector<int>> neutralIdx;
    std::unique_ptr<std::vector<int>> neutralPdg;
  };
  static AssignmentProducts assignmentMaps(const std::vector<Hypothesis> &hypotheses,
                                           const std::vector<bool> &accepted,
                                           const FinalTracksters &finalTs,
                                           std::size_t nTracks);
  // For each track, the final trackster that holds the largest energy of the footprint of the track's Electron
  // hypothesis with the highest model weight (-1: no Electron hypothesis, or no final trackster holds its footprint).
  static std::vector<int> electronTracksters(const Opinions &opinions,
                                             const std::vector<float> &weights,
                                             const std::vector<Trackster> &finalTracksters,
                                             const std::vector<reco::CaloCluster> &layerClusters,
                                             std::size_t nTracks);
  // Track claim: each accepted charged hadron, recovery and track-only hypothesis that is not a single track on an EM
  // footprint, and each jet track, takes the pool layer clusters along its propagated track up to its expected deposit
  // minus its footprint. Returns the claiming track of each layer cluster (-1: none).
  std::vector<int> claimAlongTracks(const Opinions &opinions,
                                    const std::vector<bool> &accepted,
                                    const std::vector<reco::Track> &tracks,
                                    const std::vector<TrackImpact> &impacts,
                                    const std::vector<reco::CaloCluster> &layerClusters,
                                    const std::vector<unsigned int> &pool) const;
  // Species of the neutral candidates: photon (22), pi0 (111) or neutral hadron (130), and their probabilities (photon,
  // pi0, neutral hadron; 0 for a neutral without raw energy). A photon winner is EM and only chooses between 22 and 111.
  void assignNeutralSpecies(const std::vector<Trackster> &tracksters,
                            const std::vector<int> &neutralIdx,
                            std::vector<int> &neutralPdg,
                            std::vector<float> &probabilities) const;
  // Species of the final tracksters of the tracks (0 for the other tracksters), the type of the neutral residuals of
  // TICLCandidateArbitrationProducer, and their probabilities (photon, pi0, neutral hadron per trackster).
  std::vector<int> trackTracksterSpecies(const std::vector<Trackster> &tracksters,
                                         const std::vector<int> &trackToTrackster,
                                         std::vector<float> &probabilities) const;
  // 22, 111 or 130 from the logits of the neutral species model; an EM trackster is 22 or 111. The species
  // probabilities go to p (photon, pi0, neutral hadron).
  int speciesPdg(const float *logits, bool em, float *p) const;

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

  // Global arbitration: weight = Platt-calibrated logit of the hypothesis model + trackBonus per track. The model reads
  // the first modelFeatures_ slots of each row.
  const unsigned int modelFeatures_;
  const ArbitrationModel arbitrationModel_;
  const float maxSharedEnergyFraction_;
  const float plattA_;
  const float plattB_;
  const float trackBonus_;
  const float trackOnlyWeight_;
  const unsigned int maxExactComponent_;
  const unsigned long searchBudget_;
  // A jet hypothesis on an EM footprint is never accepted.
  const bool vetoJetOnEMFootprint_;
  // The track of a Muon hypothesis with a positive score (a MIP-like footprint) is in no Jet or Recovery hypothesis.
  const bool jetVariantsWithoutElectrons_;
  // Without neutral-hadron hypotheses, their footprints only give the PID of the input tracksters to the inference
  // cache.
  const bool neutralHadronHypotheses_;
  // A leftover of an EM input trackster keeps its layer clusters out of the track claim; 0 disables the test.
  const float claimProtectEMThreshold_;
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
  const ArbitrationModel neutralModel_;
  const float neutralModelThreshold_;
  const bool tracksterSpeciesForTracks_;
  const float emRawEnergyBelow_;
};

std::unique_ptr<TICLONNXGlobalCache> TICLInterpretationProducer::initializeGlobalCache(const edm::ParameterSet &ps) {
  auto cache = TICLONNXGlobalCache::initialize(ps);
  cache->loadModel(ps.getParameter<edm::FileInPath>("arbitrationModelFile").relativePath());
  cache->loadModel(ps.getParameter<edm::FileInPath>("neutralModelFile").relativePath());
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
      modelFeatures_(ps.getParameter<unsigned int>("arbitrationModelFeatures")),
      arbitrationModel_(
          *cache->getByModelPathString(ps.getParameter<edm::FileInPath>("arbitrationModelFile").relativePath()),
          modelFeatures_,
          1),
      maxSharedEnergyFraction_(ps.getParameter<float>("arbitrationMaxSharedEnergyFraction")),
      plattA_(ps.getParameter<float>("arbitrationPlattA")),
      plattB_(ps.getParameter<float>("arbitrationPlattB")),
      trackBonus_(ps.getParameter<float>("arbitrationTrackBonus")),
      trackOnlyWeight_(ps.getParameter<float>("arbitrationTrackOnlyWeight")),
      maxExactComponent_(ps.getParameter<unsigned int>("arbitrationMaxExactComponent")),
      searchBudget_(ps.getParameter<unsigned int>("arbitrationSearchBudget")),
      vetoJetOnEMFootprint_(ps.getParameter<bool>("arbitrationVetoJetOnEMFootprint")),
      jetVariantsWithoutElectrons_(ps.getParameter<bool>("arbitrationJetVariantsWithoutElectrons")),
      neutralHadronHypotheses_(ps.getParameter<bool>("arbitrationNeutralHadronHypotheses")),
      claimProtectEMThreshold_(ps.getParameter<float>("arbitrationTrackClaimProtectEMThreshold")),
      claimResponse_(ps.getParameter<float>("arbitrationTrackClaimResponse")),
      claimResponseLogSlope_(ps.getParameter<float>("arbitrationTrackClaimResponseLogSlope")),
      claimRadius0_(ps.getParameter<float>("arbitrationTrackClaimRadius0")),
      claimRadiusSlope_(ps.getParameter<float>("arbitrationTrackClaimRadiusSlope")),
      claimMinEnergy_(ps.getParameter<float>("arbitrationTrackClaimMinEnergy")),
      neutralModel_(*cache->getByModelPathString(ps.getParameter<edm::FileInPath>("neutralModelFile").relativePath()),
                    modelfeatures::neutralSpecies::kN,
                    kNSpecies),
      neutralModelThreshold_(ps.getParameter<float>("neutralModelThreshold")),
      tracksterSpeciesForTracks_(ps.getParameter<bool>("tracksterSpeciesForTracks")),
      emRawEnergyBelow_(ps.getParameter<float>("emRawEnergyBelow")) {
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
  if (modelFeatures_ != modelfeatures::hypothesis::kNv2 && modelFeatures_ != modelfeatures::hypothesis::kN)
    throw cms::Exception("Configuration") << "arbitrationModelFeatures must be " << modelfeatures::hypothesis::kNv2
                                          << " or " << modelfeatures::hypothesis::kN;
  // The track-only hypothesis must have a positive weight, so that every selected track gets an accepted hypothesis.
  if (!(trackBonus_ + trackOnlyWeight_ > 0.f))
    throw cms::Exception("Configuration") << "arbitrationTrackBonus + arbitrationTrackOnlyWeight must be positive";
  if (!(claimResponse_ > 0.f) || !(claimRadius0_ >= 0.f) || !(claimRadiusSlope_ >= 0.f) || !(claimMinEnergy_ >= 0.f))
    throw cms::Exception("Configuration")
        << "track claim: response > 0, radius0 >= 0, radiusSlope >= 0 and minEnergy >= 0 are required";
  if (!(neutralModelThreshold_ > 0.f && neutralModelThreshold_ < 1.f))
    throw cms::Exception("Configuration") << "neutralModelThreshold must be in (0, 1)";
  if (!(claimProtectEMThreshold_ >= 0.f && claimProtectEMThreshold_ <= 1.f))
    throw cms::Exception("Configuration") << "arbitrationTrackClaimProtectEMThreshold must be in [0, 1]";

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
  produces<std::vector<int>>("electronTrackster");
  produces<std::vector<int>>("tracksterSpecies");
  produces<std::vector<float>>("tracksterSpeciesProbabilities");
  produces<std::vector<float>>("trackTypeProbabilities");
  produces<std::vector<float>>("neutralSpeciesProbabilities");
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

TICLInterpretationProducer::TrackSelection TICLInterpretationProducer::selectTracks(
    const edm::Handle<std::vector<reco::Track>> &tracks_h, const edm::Handle<std::vector<reco::Muon>> &muons_h) const {
  const auto &tracks = *tracks_h;
  TrackSelection selection{std::vector<bool>(tracks.size(), false), std::vector<bool>(tracks.size(), false)};
  const auto muonOf = muonOfTrack(tracks_h, *muons_h);
  for (unsigned int i = 0; i < tracks.size(); ++i) {
    const auto &tk = tracks[i];
    if (std::sqrt(tk.p() * tk.p() + mpion2) < tkEnergyCut_ || !cutTk_(tk))
      continue;
    const int muId = muonOf[i];
    const bool isMuon = muId != -1 && PFMuonAlgo::isMuon(reco::MuonRef(muons_h, muId));
    if (isMuon && !(*muons_h)[muId].isTrackerMuon())
      continue;
    selection.selected[i] = true;
    selection.muon[i] = isMuon;
  }
  return selection;
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
    // An EM trackster below emRawEnergyBelow keeps its raw energy: with a known PID, its regression is not needed.
    const auto &p = cache[e].probabilities;
    const bool regressionUnused = !needPID && ts.raw_energy() < emRawEnergyBelow_ &&
                                  !(p[static_cast<int>(Trackster::ParticleType::photon)] +
                                        p[static_cast<int>(Trackster::ParticleType::electron)] <
                                    0.5f);
    const bool needRegression = !cache[e].hasRegression && (withRegression || !split) && !regressionUnused;
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
  if (jetVariantsWithoutElectrons_)
    addJetVariantsWithoutElectrons(opinions, *allInput.tracksHandle, allInput.layerClusters);

  // The PID of each footprint gives the scores of its photon and neutral-hadron hypotheses.
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
  if (!neutralHadronHypotheses_)
    std::erase_if(opinions.hypotheses, [](const Hypothesis &h) { return h.type == Hypothesis::Type::NeutralHadron; });

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

void TICLInterpretationProducer::addJetVariantsWithoutElectrons(Opinions &opinions,
                                                                const std::vector<reco::Track> &tracks,
                                                                const std::vector<reco::CaloCluster> &layerClusters) {
  auto &hypotheses = opinions.hypotheses;
  std::vector<std::vector<unsigned int>> electronsOf(tracks.size());
  for (unsigned int i = 0; i < hypotheses.size(); ++i)
    if (hypotheses[i].type == Hypothesis::Type::Electron && hypotheses[i].tracksterIdx >= 0)
      electronsOf[hypotheses[i].trackIdx].push_back(i);

  std::vector<bool> removed(layerClusters.size(), false);
  const unsigned int nHypotheses = hypotheses.size();
  for (unsigned int j = 0; j < nHypotheses; ++j) {
    if (hypotheses[j].type != Hypothesis::Type::Jet || hypotheses[j].tracksterIdx < 0)
      continue;
    const std::vector<int> jetTracks = hypotheses[j].trackIdxs;
    const int jetTracksterIdx = hypotheses[j].tracksterIdx;
    for (const int iElectronTrack : jetTracks) {
      for (const unsigned int e : electronsOf[iElectronTrack]) {
        const Trackster &electronTs = opinions.tracksters[hypotheses[e].tracksterIdx];
        for (auto v : electronTs.vertices())
          removed[v] = true;
        const Trackster &jetTs = opinions.tracksters[jetTracksterIdx];
        Trackster variant;
        float energy = 0.f;
        for (size_t k = 0; k < jetTs.vertices().size(); ++k) {
          const auto v = jetTs.vertices()[k];
          if (removed[v])
            continue;
          const float multiplicity = k < jetTs.vertex_multiplicity().size() ? jetTs.vertex_multiplicity()[k] : 1.f;
          variant.vertices().push_back(v);
          variant.vertex_multiplicity().push_back(multiplicity);
          energy += layerClusters[v].energy() / multiplicity;
        }
        for (auto v : electronTs.vertices())
          removed[v] = false;

        std::vector<int> left;
        float sumP = 0.f;
        for (const int iTk : jetTracks)
          if (iTk != iElectronTrack) {
            left.push_back(iTk);
            sumP += tracks[iTk].p();
          }
        if (!(energy > 0.f) || left.empty())
          continue;
        const float balance = 1.f - std::abs(energy - sumP) / std::max(energy, sumP);
        if (balance <= 0.f)
          continue;
        Hypothesis h;
        h.score = balance;
        if (left.size() >= 2) {
          h.type = Hypothesis::Type::Jet;
          h.trackIdxs = std::move(left);
        } else {
          h.type = Hypothesis::Type::RecoveryChargedHadron;
          h.trackIdx = left.front();
        }
        h.tracksterIdx = static_cast<int>(opinions.tracksters.size());
        opinions.tracksters.push_back(std::move(variant));
        hypotheses.push_back(std::move(h));
      }
    }
  }
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

std::vector<int> TICLInterpretationProducer::componentSizes(const std::vector<Hypothesis> &hypotheses,
                                                            const std::vector<std::vector<unsigned int>> &conflicts) {
  const unsigned int nH = hypotheses.size();
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
  return compSize;
}

std::vector<float> TICLInterpretationProducer::modelWeights(const Opinions &opinions,
                                                            const std::vector<std::vector<unsigned int>> &conflicts,
                                                            const std::vector<reco::Track> &tracks) const {
  using modelfeatures::hypothesis::kN;
  const auto &hypotheses = opinions.hypotheses;
  const unsigned int nH = hypotheses.size();
  const auto compSize = componentSizes(hypotheses, conflicts);
  // Tracks that carry a muon or an electron hypothesis, and tracks whose Recovery hypothesis has an EM footprint:
  // P(photon) + P(electron) >= 0.5 or a raw EM fraction above 0.9.
  std::vector<bool> muonTrack(tracks.size(), false), electronTrack(tracks.size(), false),
      emRecoveryTrack(tracks.size(), false);
  for (auto const &h : hypotheses)
    if (h.trackIdx >= 0 && h.type == Hypothesis::Type::Muon)
      muonTrack[h.trackIdx] = true;
    else if (h.trackIdx >= 0 && h.type == Hypothesis::Type::Electron)
      electronTrack[h.trackIdx] = true;
    else if (h.trackIdx >= 0 && h.type == Hypothesis::Type::RecoveryChargedHadron && h.tracksterIdx >= 0) {
      const auto &ts = opinions.tracksters[h.tracksterIdx];
      if (!ts.isHadronic() || ts.raw_em_energy() > 0.9f * ts.raw_energy())
        emRecoveryTrack[h.trackIdx] = true;
    }

  // The model scores the hypotheses of the interpretations; a track-only hypothesis takes trackOnlyWeight. A model
  // with arbitrationModelFeatures inputs reads the first slots of each row.
  std::vector<unsigned int> rows;
  for (unsigned int i = 0; i < nH; ++i)
    if (!isTrackOnly(hypotheses[i]))
      rows.push_back(i);
  std::vector<float> features(rows.size() * modelFeatures_, 0.f);
  std::array<float, kN> x;
  for (unsigned int r = 0; r < rows.size(); ++r) {
    const auto &h = hypotheses[rows[r]];
    const Trackster *footprint = h.tracksterIdx >= 0 ? &opinions.tracksters[h.tracksterIdx] : nullptr;
    x.fill(0.f);
    modelfeatures::fillHypothesisFeatures(
        h, footprint, tracks, {compSize[rows[r]], muonTrack, electronTrack, emRecoveryTrack}, x);
    std::copy_n(x.begin(), modelFeatures_, features.begin() + r * modelFeatures_);
  }
  const std::vector<float> logits = arbitrationModel_.logits(std::move(features), rows.size());
  std::vector<float> weights(nH, trackOnlyWeight_);
  for (unsigned int r = 0; r < rows.size(); ++r)
    weights[rows[r]] = plattA_ * logits[r] + plattB_;
  // The solver selects only positive weights, whatever the track bonus.
  if (vetoJetOnEMFootprint_)
    for (unsigned int i = 0; i < nH; ++i) {
      const auto &h = hypotheses[i];
      if (h.type == Hypothesis::Type::Jet && h.tracksterIdx >= 0 && !opinions.tracksters[h.tracksterIdx].isHadronic())
        weights[i] = -std::numeric_limits<float>::infinity();
    }
  return weights;
}

std::optional<Trackster> TICLInterpretationProducer::without(
    const Trackster &ts, const std::vector<bool> &removed, const std::vector<reco::CaloCluster> &layerClusters) const {
  Trackster out(ts);
  out.vertices().clear();
  out.vertex_multiplicity().clear();
  float raw = 0.f;
  for (size_t k = 0; k < ts.vertices().size(); ++k)
    if (!removed[ts.vertices()[k]]) {
      out.vertices().push_back(ts.vertices()[k]);
      out.vertex_multiplicity().push_back(k < ts.vertex_multiplicity().size() ? ts.vertex_multiplicity()[k] : 1.f);
      raw += layerClusters[ts.vertices()[k]].energy();
    }
  if (!(raw > 0.f && (out.vertices().size() == ts.vertices().size() || raw >= claimMinEnergy_)))
    return std::nullopt;
  return out;
}

std::vector<Trackster> TICLInterpretationProducer::selectLeftovers(const edm::MultiSpan<Trackster> &tracksters,
                                                                   const std::vector<reco::CaloCluster> &layerClusters,
                                                                   std::vector<bool> &held,
                                                                   std::vector<unsigned int> &source) const {
  std::vector<unsigned int> order(tracksters.size());
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(), [&](unsigned int a, unsigned int b) {
    return tracksters[a].raw_energy() > tracksters[b].raw_energy();
  });
  std::vector<Trackster> leftovers;
  source.clear();
  for (unsigned int iTs : order) {
    auto rest = without(tracksters[iTs], held, layerClusters);
    if (!rest)
      continue;
    for (auto v : rest->vertices())
      held[v] = true;
    leftovers.push_back(std::move(*rest));
    source.push_back(iTs);
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
  // The pool, binned in (|eta|, phi) per endcap with bins of kBin: the clusters of bin b are
  // binned[binStart[b]] to binned[binStart[b + 1] - 1], in pool order, with the fields that the search reads.
  constexpr float kBin = 0.05f;
  const int nPhi = static_cast<int>(std::ceil(2.f * static_cast<float>(M_PI) / kBin));
  auto etaBin = [](float eta) { return static_cast<int>(std::floor(std::abs(eta) / kBin)); };
  auto phiBin = [](float phi) { return static_cast<int>(std::floor((phi + static_cast<float>(M_PI)) / kBin)); };
  struct PoolCluster {
    math::XYZPoint position;
    float absZ;
    bool inCEE;
    unsigned int index;
  };
  std::vector<int> etaBinOf(pool.size()), phiBinOf(pool.size());
  int nEta = 1;
  for (std::size_t k = 0; k < pool.size(); ++k) {
    const auto &lc = layerClusters[pool[k]];
    etaBinOf[k] = etaBin(lc.eta());
    phiBinOf[k] = phiBin(lc.phi());
    nEta = std::max(nEta, etaBinOf[k] + 1);
  }
  auto bin = [&](int side, int ie, int ip) { return (side * nEta + ie) * nPhi + ip; };
  std::vector<int> binStart(2 * nEta * nPhi + 1, 0);
  auto inGrid = [&](std::size_t k) { return etaBinOf[k] >= 0 && phiBinOf[k] >= 0 && phiBinOf[k] < nPhi; };
  for (std::size_t k = 0; k < pool.size(); ++k)
    if (inGrid(k))
      ++binStart[bin(layerClusters[pool[k]].z() > 0 ? 1 : 0, etaBinOf[k], phiBinOf[k]) + 1];
  std::partial_sum(binStart.begin(), binStart.end(), binStart.begin());
  std::vector<PoolCluster> binned(binStart.back());
  {
    std::vector<int> next(binStart.begin(), binStart.end() - 1);
    for (std::size_t k = 0; k < pool.size(); ++k) {
      if (!inGrid(k))
        continue;
      const unsigned int v = pool[k];
      const auto &lc = layerClusters[v];
      binned[next[bin(lc.z() > 0 ? 1 : 0, etaBinOf[k], phiBinOf[k])]++] = {
          lc.position(), std::abs(static_cast<float>(lc.z())), lc.seed().det() == DetId::HGCalEE, v};
    }
  }
  // Sources: (expected deposit still to claim, track). A single-track hypothesis on an EM footprint does not claim:
  // its candidate combines the track and the trackster energies.
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
  const float maxRadius = claimRadius0_ + claimRadiusSlope_ * kClaimMaxDepth;
  // Clusters up to kFrontTolerance [cm] in front of the impact are kept.
  constexpr float kFrontTolerance = 1.f;
  // (in CE-E, distance, layer cluster): nearest first, CE-H before CE-E. In CE-E the clusters near a track hold much
  // EM energy.
  std::vector<std::tuple<bool, float, unsigned int>> candidates;
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
    for (int ie = std::max(std::min(ieFront, ieDeep) - nWin, 0);
         ie <= std::min(std::max(ieFront, ieDeep) + nWin, nEta - 1);
         ++ie)
      for (int dp = dpLow; dp <= dpHigh; ++dp) {
        const int b = bin(side, ie, ((ip0 + dp) % nPhi + nPhi) % nPhi);
        for (int k = binStart[b]; k < binStart[b + 1]; ++k) {
          const auto &c = binned[k];
          if (claimingTrack[c.index] >= 0)
            continue;
          const float depth = c.absZ - zFront;
          if (depth < -kFrontTolerance)
            continue;
          float dist = 0.f;
          if (!impactTransverseDistanceBelow(
                  impact, c.position, claimRadius0_ + claimRadiusSlope_ * std::max(depth, 0.f), dist))
            continue;
          candidates.emplace_back(c.inCEE, dist, c.index);
        }
      }
    std::sort(candidates.begin(), candidates.end());
    float claimed = 0.f;
    for (auto const &[inCEE, dist, v] : candidates) {
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
                                                      std::vector<int> &neutralPdg,
                                                      std::vector<float> &probabilities) const {
  using modelfeatures::neutralSpecies::kN;
  probabilities.assign(kNSpecies * neutralIdx.size(), 0.f);
  std::vector<unsigned int> rows;
  std::vector<float> x;
  for (unsigned int k = 0; k < neutralIdx.size(); ++k) {
    const auto &ts = tracksters[neutralIdx[k]];
    if (!(ts.raw_energy() > 0.f))
      continue;
    rows.push_back(k);
    x.resize(x.size() + kN, 0.f);
    modelfeatures::fillNeutralSpeciesFeatures(ts, std::span<float, kN>(x.data() + x.size() - kN, kN));
  }
  const auto logits = neutralModel_.logits(std::move(x), rows.size());
  for (unsigned int r = 0; r < rows.size(); ++r) {
    int &pdg = neutralPdg[rows[r]];
    pdg = speciesPdg(&logits[r * kNSpecies], pdg == 22, &probabilities[kNSpecies * rows[r]]);
  }
}

int TICLInterpretationProducer::speciesPdg(const float *m, bool em, float *p) const {
  const float mMax = std::max({m[0], m[1], m[2]});
  const float pPhoton = std::exp(m[0] - mMax), pPi0 = std::exp(m[1] - mMax), pHadron = std::exp(m[2] - mMax);
  const float sum = pPhoton + pPi0 + pHadron;
  const float pEM = (pPhoton + pPi0) / sum;
  p[0] = pPhoton / sum;
  p[1] = pPi0 / sum;
  p[2] = pHadron / sum;
  if (em || pEM >= neutralModelThreshold_)
    return pPi0 > pPhoton ? 111 : 22;
  return 130;
}

std::vector<int> TICLInterpretationProducer::trackTracksterSpecies(const std::vector<Trackster> &tracksters,
                                                                   const std::vector<int> &trackToTrackster,
                                                                   std::vector<float> &probabilities) const {
  using modelfeatures::neutralSpecies::kN;
  std::vector<int> species(tracksters.size(), 0);
  probabilities.assign(kNSpecies * tracksters.size(), 0.f);
  std::vector<bool> seen(tracksters.size(), false);
  std::vector<unsigned int> rows;
  std::vector<float> x;
  for (int iTs : trackToTrackster) {
    if (iTs < 0 || seen[iTs] || !(tracksters[iTs].raw_energy() > 0.f))
      continue;
    seen[iTs] = true;
    rows.push_back(iTs);
    x.resize(x.size() + kN, 0.f);
    modelfeatures::fillNeutralSpeciesFeatures(tracksters[iTs], std::span<float, kN>(x.data() + x.size() - kN, kN));
  }
  const auto logits = neutralModel_.logits(std::move(x), rows.size());
  for (unsigned int r = 0; r < rows.size(); ++r)
    species[rows[r]] = speciesPdg(&logits[r * kNSpecies], false, &probabilities[kNSpecies * rows[r]]);
  return species;
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

  const auto [selected, muonTracks] = selectTracks(tracks_h, muons_h);
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

  auto [leftovers, pool] = claimPool(evt, opinions, accepted, inferenceCache, tracksters, layerClusters);
  const auto claimingTrack = claimAlongTracks(opinions, accepted, tracks, impacts, layerClusters, pool);
  auto finalTs =
      finalTracksters(opinions, accepted, solverWeights, leftovers, claimingTrack, layerClusters, tracks.size());
  auto &result = *finalTs.tracksters;
  assignPCAtoTracksters(
      result, layerClusters, layerClustersTimes, rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z(), rhtools_, true);
  runInference(layerClusters, result, inferenceCache, true);
  // An EM trackster below emRawEnergyBelow keeps its raw energy: the regression is not trained for it.
  for (auto &ts : result)
    if (!ts.isHadronic() && ts.raw_energy() < emRawEnergyBelow_)
      ts.setRegressedEnergy(ts.raw_energy());

  auto maps = assignmentMaps(hypotheses, accepted, finalTs, tracks.size());
  auto neutralProbabilities = std::make_unique<std::vector<float>>();
  assignNeutralSpecies(result, *maps.neutralIdx, *maps.neutralPdg, *neutralProbabilities);
  auto trackProbabilities =
      std::make_unique<std::vector<float>>(trackTypeProbabilities(hypotheses, weights, tracks.size()));
  auto electronTrackster =
      std::make_unique<std::vector<int>>(electronTracksters(opinions, weights, result, layerClusters, tracks.size()));
  auto tracksterSpecies = std::make_unique<std::vector<int>>();
  auto tracksterProbabilities = std::make_unique<std::vector<float>>();
  if (tracksterSpeciesForTracks_)
    *tracksterSpecies = trackTracksterSpecies(result, *maps.trackToTrackster, *tracksterProbabilities);

  evt.put(std::move(finalTs.tracksters));
  evt.put(std::move(maps.trackToTrackster), "trackToTrackster");
  evt.put(std::move(maps.trackMode), "trackMode");
  evt.put(std::move(maps.neutralIdx), "neutralIdx");
  evt.put(std::move(maps.neutralPdg), "neutralPdg");
  evt.put(std::move(electronTrackster), "electronTrackster");
  evt.put(std::move(tracksterSpecies), "tracksterSpecies");
  evt.put(std::move(tracksterProbabilities), "tracksterSpeciesProbabilities");
  evt.put(std::move(trackProbabilities), "trackTypeProbabilities");
  evt.put(std::move(neutralProbabilities), "neutralSpeciesProbabilities");
}

TICLInterpretationProducer::ClaimPool TICLInterpretationProducer::claimPool(
    const edm::Event &evt,
    const Opinions &opinions,
    const std::vector<bool> &accepted,
    const InferenceCache &inferenceCache,
    const edm::MultiSpan<Trackster> &tracksters,
    const std::vector<reco::CaloCluster> &layerClusters) const {
  const auto &hypotheses = opinions.hypotheses;
  std::vector<bool> held(layerClusters.size(), false);
  std::vector<bool> inWinner(layerClusters.size(), false);
  for (unsigned int i = 0; i < hypotheses.size(); ++i)
    if (accepted[i] && hypotheses[i].tracksterIdx >= 0)
      for (auto v : opinions.tracksters[hypotheses[i].tracksterIdx].vertices())
        held[v] = inWinner[v] = true;
  std::vector<unsigned int> leftoverSource;
  ClaimPool claim{selectLeftovers(tracksters, layerClusters, held, leftoverSource), {}};
  std::vector<bool> inPool(layerClusters.size(), false);
  auto addToPool = [&](unsigned int v) {
    if (!inWinner[v] && !inPool[v]) {
      inPool[v] = true;
      claim.pool.push_back(v);
    }
  };
  // The PID of an input trackster is in the inference cache when a hypothesis has it as its footprint.
  auto isEM = [&](const Trackster &ts) {
    const auto *e = inferenceCache.find(ts);
    return e != nullptr && e->hasPID &&
           e->probabilities[static_cast<int>(Trackster::ParticleType::photon)] +
                   e->probabilities[static_cast<int>(Trackster::ParticleType::electron)] >=
               claimProtectEMThreshold_;
  };
  for (std::size_t k = 0; k < claim.leftovers.size(); ++k) {
    if (claimProtectEMThreshold_ > 0.f && isEM(tracksters[leftoverSource[k]]))
      continue;
    for (auto v : claim.leftovers[k].vertices())
      addToPool(v);
  }
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
  return claim;
}

TICLInterpretationProducer::FinalTracksters TICLInterpretationProducer::finalTracksters(
    const Opinions &opinions,
    const std::vector<bool> &accepted,
    const std::vector<float> &solverWeights,
    const std::vector<Trackster> &leftovers,
    const std::vector<int> &claimingTrack,
    const std::vector<reco::CaloCluster> &layerClusters,
    std::size_t nTracks) const {
  const auto &hypotheses = opinions.hypotheses;
  FinalTracksters finalTs{std::make_unique<std::vector<Trackster>>(), std::vector<int>(hypotheses.size(), -1), {}};
  auto &result = *finalTs.tracksters;
  std::vector<bool> claimed(layerClusters.size());
  for (unsigned int v = 0; v < claimingTrack.size(); ++v)
    claimed[v] = claimingTrack[v] >= 0;
  std::vector<unsigned int> winners;
  for (unsigned int idx = 0; idx < hypotheses.size(); ++idx)
    if (accepted[idx] && hypotheses[idx].tracksterIdx >= 0)
      winners.push_back(idx);
  std::stable_sort(winners.begin(), winners.end(), [&](unsigned int a, unsigned int b) {
    return solverWeights[a] > solverWeights[b];
  });
  std::vector<bool> heldByWinner(layerClusters.size(), false);
  std::vector<std::optional<Trackster>> winnerTrackster(hypotheses.size());
  for (unsigned int idx : winners) {
    const auto &ts = opinions.tracksters[hypotheses[idx].tracksterIdx];
    if (std::none_of(ts.vertices().begin(), ts.vertices().end(), [&](unsigned int v) { return heldByWinner[v]; })) {
      winnerTrackster[idx] = ts;
    } else {
      winnerTrackster[idx] = without(ts, heldByWinner, layerClusters);
      if (!winnerTrackster[idx])
        continue;
    }
    for (auto v : winnerTrackster[idx]->vertices())
      heldByWinner[v] = true;
  }
  for (unsigned int idx = 0; idx < hypotheses.size(); ++idx) {
    if (!winnerTrackster[idx])
      continue;
    finalTs.winnerIndex[idx] = static_cast<int>(result.size());
    result.push_back(std::move(*winnerTrackster[idx]));
  }
  for (auto const &leftover : leftovers) {
    auto pruned = without(leftover, claimed, layerClusters);
    if (!pruned)
      continue;
    finalTs.leftoverIndex.push_back(static_cast<int>(result.size()));
    result.push_back(std::move(*pruned));
  }
  // One claim trackster per claiming track. No candidate uses it: the charged candidate takes its energy from the track.
  std::vector<int> claimTracksterOfTrack(nTracks, -1);
  for (unsigned int v = 0; v < claimingTrack.size(); ++v) {
    if (claimingTrack[v] < 0)
      continue;
    int &iClaim = claimTracksterOfTrack[claimingTrack[v]];
    if (iClaim < 0) {
      iClaim = static_cast<int>(result.size());
      result.emplace_back();
    }
    result[iClaim].vertices().push_back(v);
    result[iClaim].vertex_multiplicity().push_back(1.f);
  }
  return finalTs;
}

TICLInterpretationProducer::AssignmentProducts TICLInterpretationProducer::assignmentMaps(
    const std::vector<Hypothesis> &hypotheses,
    const std::vector<bool> &accepted,
    const FinalTracksters &finalTs,
    std::size_t nTracks) {
  // The accepted set is maximal, so every selected track has an accepted hypothesis.
  AssignmentProducts maps{std::make_unique<std::vector<int>>(nTracks, -1),
                          std::make_unique<std::vector<int>>(nTracks, static_cast<int>(TrackMode::kNotSelected)),
                          std::make_unique<std::vector<int>>(),
                          std::make_unique<std::vector<int>>()};
  auto assign = [&](int iTrack, TrackMode mode, int iTrackster) {
    (*maps.trackMode)[iTrack] = static_cast<int>(mode);
    (*maps.trackToTrackster)[iTrack] = iTrackster;
  };
  for (unsigned int idx = 0; idx < hypotheses.size(); ++idx) {
    if (!accepted[idx])
      continue;
    const auto &h = hypotheses[idx];
    const int iResult = finalTs.winnerIndex[idx];
    switch (h.type) {
      case Hypothesis::Type::Muon:
        assign(h.trackIdx, TrackMode::kMuon, iResult);
        break;
      case Hypothesis::Type::Electron:
        assign(h.trackIdx, TrackMode::kElectron, iResult);
        break;
      case Hypothesis::Type::ChargedHadron:
        assign(h.trackIdx, TrackMode::kChargedHadron, iResult);
        break;
      case Hypothesis::Type::Jet:
        for (int iTk : h.trackIdxs)
          assign(iTk, TrackMode::kJetMember, iResult);
        break;
      case Hypothesis::Type::RecoveryChargedHadron:
        assign(h.trackIdx, TrackMode::kRecovery, iResult);
        break;
      case Hypothesis::Type::Photon:
      case Hypothesis::Type::NeutralHadron:
        // A neutral winner the track claim emptied has no final trackster. A photon winner is a photon or a pi0; a
        // neutral-hadron winner is typed by the neutral species model, as a leftover.
        if (iResult >= 0) {
          maps.neutralIdx->push_back(iResult);
          maps.neutralPdg->push_back(h.type == Hypothesis::Type::Photon ? 22 : 0);
        }
        break;
    }
  }
  for (int iTrackster : finalTs.leftoverIndex) {
    maps.neutralIdx->push_back(iTrackster);
    maps.neutralPdg->push_back(0);
  }
  return maps;
}

std::vector<int> TICLInterpretationProducer::electronTracksters(const Opinions &opinions,
                                                                const std::vector<float> &weights,
                                                                const std::vector<Trackster> &finalTracksters,
                                                                const std::vector<reco::CaloCluster> &layerClusters,
                                                                std::size_t nTracks) {
  const auto &hypotheses = opinions.hypotheses;
  std::vector<int> best(nTracks, -1);
  for (unsigned int i = 0; i < hypotheses.size(); ++i) {
    const auto &h = hypotheses[i];
    if (h.type != Hypothesis::Type::Electron || h.trackIdx < 0 || h.tracksterIdx < 0)
      continue;
    int &b = best[h.trackIdx];
    if (b < 0 || weights[i] > weights[b])
      b = static_cast<int>(i);
  }
  std::vector<int> owner(layerClusters.size(), -1);
  for (unsigned int k = 0; k < finalTracksters.size(); ++k)
    for (auto v : finalTracksters[k].vertices())
      if (owner[v] < 0)
        owner[v] = static_cast<int>(k);
  std::vector<int> result(nTracks, -1);
  std::vector<std::pair<int, float>> shares;
  for (std::size_t iTrack = 0; iTrack < nTracks; ++iTrack) {
    if (best[iTrack] < 0)
      continue;
    shares.clear();
    for (auto v : opinions.tracksters[hypotheses[best[iTrack]].tracksterIdx].vertices()) {
      if (owner[v] < 0)
        continue;
      auto it = std::find_if(shares.begin(), shares.end(), [&](const auto &s) { return s.first == owner[v]; });
      if (it == shares.end())
        shares.emplace_back(owner[v], layerClusters[v].energy());
      else
        it->second += layerClusters[v].energy();
    }
    const auto it = std::max_element(
        shares.begin(), shares.end(), [](const auto &a, const auto &b) { return a.second < b.second; });
    if (it != shares.end())
      result[iTrack] = it->first;
  }
  return result;
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

  edm::ParameterSetDescription inferenceDesc;
  inferenceDesc.addNode(edm::PluginDescription<TracksterInferenceAlgoFactory>("type", "TracksterInferenceByPFN", true));
  // The plugin and its parameter set pluginInferenceAlgo<plugin>.
  desc.ifValue(edm::ParameterDescription<std::string>(
                   "inferenceAlgo",
                   "TracksterInferenceByPFN",
                   true,
                   edm::Comment("PID and energy regression of the footprints and of the final tracksters.")),
               "TracksterInferenceByPFN" >> edm::ParameterDescription<edm::ParameterSetDescription>(
                                                "pluginInferenceAlgoTracksterInferenceByPFN", inferenceDesc, true));

  desc.add<float>("arbitrationMaxSharedEnergyFraction", 0.2f)
      ->setComment("Two hypotheses conflict above this fraction of the smaller footprint energy in shared clusters.");
  desc.add<edm::FileInPath>("arbitrationModelFile",
                            edm::FileInPath("RecoTICL/Interpretation/data/arbitration/hypothesis_mlp_v2.onnx"))
      ->setComment("ONNX model of the probability that a hypothesis is correct (logit).");
  desc.add<unsigned int>("arbitrationModelFeatures", modelfeatures::hypothesis::kNv2)
      ->setComment(
          "Inputs of the hypothesis model per row: 30 (hypothesis_mlp_v2), or 32 with the tracks whose Recovery "
          "hypothesis has an EM footprint and the tracks with a missing inner hit.");
  desc.add<float>("arbitrationPlattA", 0.9958741664886475f)->setComment("Platt calibration: logit' = a x logit + b.");
  desc.add<float>("arbitrationPlattB", -0.012220372445881367f);
  desc.add<float>("arbitrationTrackBonus", 20.0f)->setComment("Weight added per track a hypothesis covers.");
  desc.add<float>("arbitrationTrackOnlyWeight", -1.0f)->setComment("Weight of a track-only hypothesis.");
  desc.add<unsigned int>("arbitrationMaxExactComponent", 64)
      ->setComment("Largest component solved by branch and bound (at most 64); larger ones are greedy.");
  desc.add<unsigned int>("arbitrationSearchBudget", 1000000)
      ->setComment("Max branch-and-bound calls per component; then the best set found is kept.");
  desc.add<bool>("arbitrationVetoJetOnEMFootprint", true)
      ->setComment("A jet hypothesis on an EM footprint (P(photon) + P(electron) >= 0.5) is never accepted.");
  desc.add<bool>("arbitrationJetVariantsWithoutElectrons", true)
      ->setComment(
          "For each Jet hypothesis and each Electron hypothesis of one of its tracks, add the Jet without that track, "
          "on the footprint without the layer clusters of the Electron footprint (one track left: a Recovery "
          "hypothesis).");
  desc.add<bool>("arbitrationNeutralHadronHypotheses", false)
      ->setComment(
          "Neutral-hadron hypotheses take part in the arbitration. When false, their footprints only give the PID of "
          "the input tracksters to the inference cache.");
  desc.add<float>("arbitrationTrackClaimProtectEMThreshold", 0.5f)
      ->setComment(
          "A leftover whose input trackster has P(photon) + P(electron) at or above this value keeps its layer "
          "clusters out of the track claim. 0 disables the test.");
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
  desc.add<edm::FileInPath>("neutralModelFile",
                            edm::FileInPath("RecoTICL/Interpretation/data/arbitration/neutralSpecies_mlp_v2.onnx"))
      ->setComment("ONNX 3-class model (photon, pi0, neutral hadron; logits) that types the neutral candidates.");
  desc.add<float>("neutralModelThreshold", 0.7f)->setComment("P(photon) + P(pi0) at and above which a neutral is EM.");
  desc.add<bool>("tracksterSpeciesForTracks", true)
      ->setComment(
          "Put the species of the final tracksters of the tracks and their probabilities (instances "
          "tracksterSpecies, tracksterSpeciesProbabilities); empty if false.");
  desc.add<float>("emRawEnergyBelow", 50.f)
      ->setComment("EM final tracksters below this raw energy [GeV] keep the raw energy.");
  descriptions.add("ticlInterpretationProducer", desc);
}

DEFINE_FWK_MODULE(TICLInterpretationProducer);
