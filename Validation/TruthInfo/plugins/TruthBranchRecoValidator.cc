// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// Truth-branch validation, one plugin template for every reco domain. It books only
// num/denom histograms; DQMGenericClient forms the ratios. Reco-driven metrics have one
// folder per (collection, working point). Truth-driven metrics have one folder per
// (collection, graph level), because the level fixes the truth target and the working
// point does not change it.
//
// RecoValidationTraits binds the association map type and the kinematics of a reco
// object to the reco type.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <mutex>
#include <type_traits>
#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "PhysicsTools/TruthInfo/interface/SubgraphHitView.h"
#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/VertexData.h"

#include "Validation/TruthInfo/interface/TruthBranchHistoProducerAlgo.h"

namespace {
  using Kinematics = truth::TruthBranchHistoProducerAlgo::Kinematics;

  // The association value of an element, whatever its payload calls it: shared
  // energy for the calorimetric and hit-count payloads, fraction for composite ones.
  template <typename E>
  float payloadValue(E const& element) {
    if constexpr (requires { element.sharedEnergy(); }) {
      return element.sharedEnergy();
    } else {
      return element.fraction();
    }
  }

  // The bin of the reason axis is the vertex reason itself, whatever side it came from.
  unsigned int reasonBin(truth::VertexData const& vertex) { return static_cast<unsigned int>(vertex.reason); }

  template <typename RECO>
  struct RecoValidationTraits;

  template <>
  struct RecoValidationTraits<reco::Track> {
    using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
    static constexpr const char* cfiName = "truthBranchTrackValidator";
    static constexpr const char* defaultAssociator = "allTrackToTruthBranchAssociators";
    static constexpr const char* defaultDir = "TruthInfo/Tracking/";
    // The channel whose footprint is the truth analogue of this object's own hits.
    static constexpr truth::HitChannel hitChannel = truth::HitChannel::Tracker;

    static Kinematics kinematics(reco::Track const& track) {
      Kinematics kin;
      kin.pt = track.pt();
      kin.eta = track.eta();
      kin.phi = track.phi();
      kin.nhits = track.numberOfValidHits();
      kin.vertpos = std::sqrt(track.vx() * track.vx() + track.vy() * track.vy());
      kin.zpos = track.vz();
      kin.dxy = track.dxy();
      kin.dz = track.dz();
      return kin;
    }
    static bool hasDirection(reco::Track const&) { return true; }
    // The truth side of a hit-based domain iterates branch roots, which are particles.
    static constexpr bool truthIsVertex = false;
    // A calorimetric domain is judged by the HGCalValidator criteria on shared energy.
    // Other domains are judged on shared components.
    static constexpr bool calorimetric = false;
    // A prefix: the capitalized level name is appended, one product per level.
    static constexpr const char* denominatorInstance = "truthToRecoTargets";
  };

  // A vertex has no momentum, so only its position and its track multiplicity are defined.
  template <>
  struct RecoValidationTraits<reco::Vertex> {
    using MapType = ticl::TICLAssociationMap<ticl::mapWithFractionAndScore>;
    static constexpr const char* cfiName = "truthBranchVertexValidator";
    static constexpr const char* defaultAssociator = "allVertexToTruthBranchAssociators";
    static constexpr const char* defaultDir = "TruthInfo/Vertexing/";
    // No hitChannel: the truth object of this domain is a vertex, not a particle branch,
    // so it has no hit footprint of its own.

    static Kinematics kinematics(reco::Vertex const& vertex) {
      Kinematics kin;
      // The number of tracks of the vertex, the vertex analogue of a track hit count.
      kin.nhits = vertex.tracksSize();
      kin.vertpos = std::sqrt(vertex.x() * vertex.x() + vertex.y() * vertex.y());
      kin.zpos = vertex.z();
      return kin;
    }
    static bool hasDirection(reco::Vertex const&) { return false; }
    // A composite object is associated to a truth VERTEX, so the truth side iterates
    // vertices and the denominator is the set of reconstructable ones.
    static constexpr bool truthIsVertex = true;
    static constexpr bool calorimetric = false;
    static constexpr const char* denominatorInstance = "truthToRecoTargets";
  };

  // A trackster carries calorimeter energy through its layer clusters. Its momentum
  // direction is the barycentre, and its hit count is the number of layer clusters.
  template <>
  struct RecoValidationTraits<ticl::Trackster> {
    using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
    static constexpr const char* cfiName = "truthBranchTracksterValidator";
    static constexpr const char* defaultAssociator = "truthBranchTracksterAssociators";
    static constexpr const char* defaultDir = "TruthInfo/Calorimetry/";
    static constexpr truth::HitChannel hitChannel = truth::HitChannel::Calo;

    static Kinematics kinematics(ticl::Trackster const& trackster) {
      Kinematics kin;
      auto const& bary = trackster.barycenter();
      const double rho = std::sqrt(bary.x() * bary.x() + bary.y() * bary.y());
      const double mag = std::sqrt(rho * rho + bary.z() * bary.z());
      // A trackster has no track: its pt is the raw energy projected on the transverse
      // plane along the barycentre direction.
      kin.pt = (mag > 0.) ? trackster.raw_energy() * rho / mag : 0.;
      kin.eta = bary.eta();
      kin.phi = bary.phi();
      kin.nhits = trackster.vertices().size();
      kin.vertpos = rho;
      kin.zpos = bary.z();
      return kin;
    }
    static bool hasDirection(ticl::Trackster const&) { return true; }
    static constexpr bool truthIsVertex = false;
    static constexpr bool calorimetric = true;
    static constexpr const char* denominatorInstance = "truthToRecoTargets";
  };

  // A particle-flow cluster carries calorimeter energy through its own cells. Its
  // direction is the cluster position, and its hit count is the number of cells.
  template <>
  struct RecoValidationTraits<reco::PFCluster> {
    using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
    static constexpr const char* cfiName = "truthBranchPFClusterValidator";
    static constexpr const char* defaultAssociator = "truthBranchPFClusterAssociators";
    static constexpr const char* defaultDir = "TruthInfo/PFClusters/";
    static constexpr truth::HitChannel hitChannel = truth::HitChannel::Calo;

    static Kinematics kinematics(reco::PFCluster const& cluster) {
      Kinematics kin;
      auto const& pos = cluster.position();
      const double rho = std::sqrt(pos.x() * pos.x() + pos.y() * pos.y());
      const double mag = std::sqrt(rho * rho + pos.z() * pos.z());
      // A cluster has no track: its pt is the energy projected on the transverse plane
      // along the cluster position.
      kin.pt = (mag > 0.) ? cluster.energy() * rho / mag : 0.;
      kin.eta = pos.eta();
      kin.phi = pos.phi();
      kin.nhits = cluster.hitsAndFractions().size();
      kin.vertpos = rho;
      kin.zpos = pos.z();
      return kin;
    }
    static bool hasDirection(reco::PFCluster const&) { return true; }
    static constexpr bool truthIsVertex = false;
    static constexpr bool calorimetric = true;
    static constexpr const char* denominatorInstance = "truthToRecoTargets";
  };
}  // namespace

template <typename RECO>
class TruthBranchRecoValidator : public DQMGlobalEDAnalyzer<truth::TruthBranchHistograms> {
public:
  using Histograms = truth::TruthBranchHistograms;
  using Traits = RecoValidationTraits<RECO>;
  using MapType = typename Traits::MapType;

  explicit TruthBranchRecoValidator(edm::ParameterSet const&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, Histograms&) const override;
  void dqmAnalyze(edm::Event const&, edm::EventSetup const&, Histograms const&) const override;

  // One entry per (collection, working point), in booking order: the reco-driven
  // monitor elements.
  struct WpEntry {
    std::string folder;
    edm::EDGetTokenT<std::vector<RECO>> recoToken;
    // Reco-driven, one per working point: score is 1 - reco purity.
    edm::EDGetTokenT<MapType> recoToTruthToken;
    // The map of the first working point, the WP-free hit-sharing measure. It is the only
    // map that carries every candidate branch: an adaptive working point inserts only the
    // branch it climbs to. Dominance is read from this map for every working point.
    edm::EDGetTokenT<MapType> allCandidatesToken;
  };

  // The antichain of the dominance measure: the targets of one level, so the candidates
  // are distinct particles. The full set of selected roots is not an antichain: a tau and
  // its daughter pion have nested subgraphs with nearly the same shared energy. With that
  // set, 99.9% of tracksters on no-PU TenTau have a leading-to-runner-up ratio near one.
  edm::EDGetTokenT<std::vector<unsigned int>> dominanceTargetsToken_;
  bool hasDominanceTargets_ = false;

  // The truth-driven entries, in booking order. A hit-based domain has one entry per
  // (collection, graph level) plus the signal entries. A composite domain has one per
  // collection.
  struct TruthEntry {
    std::string folder;
    // The level's denominator: the target set the efficiency is measured over.
    edm::EDGetTokenT<std::vector<unsigned int>> targetsToken;
    // Parallel to the targets: the mask of plotted-axis cuts that each target fails, so
    // the efficiency against pt keeps the targets that fail only the pt cut. Not consumed
    // for the signal entries.
    edm::EDGetTokenT<std::vector<unsigned int>> eligibilityToken;
    bool hasEligibility = false;
    // Truth-driven, one product per collection: score is 1 - truth purity.
    edm::EDGetTokenT<MapType> truthToRecoToken;
    // The reco-driven map of the first working point (Fixed), the WP-free hit-sharing
    // measure. Read only for the loose reco-purity gate on Individual.
    edm::EDGetTokenT<MapType> firstWpRecoToTruthToken;
  };

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const std::string dirName_;
  // Shared-component domains (tracks, vertices): one reco object reconstructs a truth
  // object when it covers enough of it and is not mostly something else. The second is
  // the loose reco-purity cut of QuickTrackAssociatorByHits and HGVHistoProducerAlgo.
  const double minTruthPurityForIndividual_;
  const double minRecoPurityLoose_;
  // Calorimetric domains use the three HGCalValidator cuts: efficiency on the shared
  // energy fraction, duplicate on the simToReco score, strict match on the recoToSim
  // score (Validation/HGCalValidation/src/HGVHistoProducerAlgo.cc:2819-2820, 2897-2899).
  const double minSharedEnergyFractionForIndividual_;
  const double maxSimToRecoScoreForDuplicate_;
  const double maxRecoToSimScore_;
  const double minCollectiveCoverage_;
  // A reco object is not a fake when one branch of the dominance antichain owns at least
  // this share of the shared quantity that all the antichain members contribute.
  const double minLeadingTruthShare_;
  std::vector<WpEntry> wpEntries_;
  // Working points per collection; wpEntries_ is collection-major with this stride.
  std::size_t nWorkingPoints_ = 1;
  std::vector<TruthEntry> truthEntries_;
  // A missing input leaves a folder booked and empty, which reads as a measurement of
  // zero rather than as a configuration error. Warn once per job per entry.
  mutable std::vector<std::once_flag> wpMissingWarned_;
  mutable std::vector<std::once_flag> truthMissingWarned_;
  const truth::TruthBranchHistoProducerAlgo algo_;
};

template <typename RECO>
TruthBranchRecoValidator<RECO>::TruthBranchRecoValidator(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      dirName_(cfg.getParameter<std::string>("dirName")),
      // Each domain declares only the thresholds it is judged by, so a parameter that
      // does not apply cannot be set to a value that silently does nothing.
      minTruthPurityForIndividual_(Traits::calorimetric ? 0. : cfg.getParameter<double>("minTruthPurityForIndividual")),
      minRecoPurityLoose_(Traits::calorimetric ? 0. : cfg.getParameter<double>("minRecoPurityLoose")),
      minSharedEnergyFractionForIndividual_(
          Traits::calorimetric ? cfg.getParameter<double>("minSharedEnergyFractionForIndividual") : 0.),
      maxSimToRecoScoreForDuplicate_(Traits::calorimetric ? cfg.getParameter<double>("maxSimToRecoScoreForDuplicate")
                                                          : 0.),
      maxRecoToSimScore_(Traits::calorimetric ? cfg.getParameter<double>("maxRecoToSimScore") : 0.),
      minCollectiveCoverage_(cfg.getParameter<double>("minCollectiveCoverage")),
      minLeadingTruthShare_(Traits::truthIsVertex ? 0. : cfg.getParameter<double>("minLeadingTruthShare")),
      algo_(cfg.getParameter<edm::ParameterSet>("histoProducerAlgoBlock")) {
  const auto associator = cfg.getParameter<std::string>("associator");
  const auto targetsProducer = cfg.getParameter<std::string>("targetsProducer");
  const auto workingPoints = cfg.getParameter<std::vector<std::string>>("workingPoints");
  nWorkingPoints_ = std::max<std::size_t>(1, workingPoints.size());

  // The truth-driven folder suffixes and the denominator instance each consumes. A
  // hit-based domain has one target set per graph level; a composite one has a single
  // target set, named by the domain's vertex resolution.
  std::vector<std::pair<std::string, std::string>> truthTargets;
  if constexpr (Traits::truthIsVertex) {
    truthTargets.emplace_back(cfg.getParameter<std::string>("vertexResolution"), Traits::denominatorInstance);
  } else {
    for (auto const& level : cfg.getParameter<std::vector<std::string>>("truthLevels")) {
      std::string capitalized = level;
      capitalized[0] = std::toupper(static_cast<unsigned char>(capitalized[0]));
      truthTargets.emplace_back(level, std::string(Traits::denominatorInstance) + capitalized);
    }
    // The signal entry: its denominator is the preset seed objects among the selected
    // roots (the tau, not its decay legs). Not booked when the seeds name no resonance,
    // because an empty folder reads as an efficiency of zero.
    if (truth::seedsNameAResonance(cfg.getParameter<std::vector<int>>("signalSeedPdgIds"),
                                   cfg.getParameter<std::vector<int>>("signalSeedHadronFlavors"))) {
      truthTargets.emplace_back("signal", "signalSeeds");
      // The same seed objects with no selector cut. The difference to the signal folder
      // is what the selection removes.
      truthTargets.emplace_back("signalNoSelection", "signalSeedsNoSelection");
    }
    // Every denominator is an antichain. The set of all selected roots is not offered: it
    // can contain a particle and its ancestor, which counts the same object twice.
  }

  // Dominance is measured against ONE level, so the candidates are distinct particles.
  if constexpr (!Traits::truthIsVertex) {
    auto const& levels = cfg.getParameter<std::vector<std::string>>("truthLevels");
    const std::string wanted = cfg.getParameter<std::string>("dominanceLevel");
    if (std::find(levels.begin(), levels.end(), wanted) != levels.end()) {
      std::string capitalized = wanted;
      capitalized[0] = std::toupper(static_cast<unsigned char>(capitalized[0]));
      dominanceTargetsToken_ = consumes<std::vector<unsigned int>>(
          edm::InputTag(targetsProducer, std::string(Traits::denominatorInstance) + capitalized));
      hasDominanceTargets_ = true;
    }
  }

  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("recoCollections")) {
    // "label_instance" is the key of the product instance labels and of the folder name.
    std::string key = tag.label();
    if (!tag.instance().empty()) {
      key += "_" + tag.instance();
    }
    for (auto const& wp : workingPoints) {
      WpEntry entry;
      entry.folder = key + "_" + wp;
      entry.recoToken = consumes<std::vector<RECO>>(tag);
      entry.recoToTruthToken = consumes<MapType>(edm::InputTag(associator, key + "RecoToTruth" + wp));
      entry.allCandidatesToken =
          consumes<MapType>(edm::InputTag(associator, key + "RecoToTruth" + workingPoints.front()));
      wpEntries_.push_back(std::move(entry));
    }
    for (auto const& [suffix, instance] : truthTargets) {
      TruthEntry entry;
      entry.folder = key + "_" + suffix;
      // A hit-based domain reads its targets from the targets producer, a composite domain
      // from its associator. The association maps come from the associator.
      const std::string targetsLabel = Traits::truthIsVertex ? associator : targetsProducer;
      entry.targetsToken = consumes<std::vector<unsigned int>>(edm::InputTag(targetsLabel, instance));
      // Only the per-level denominators carry it: the signal entries are seed lists with
      // no kinematic selection to suppress.
      entry.hasEligibility = (instance.rfind(Traits::denominatorInstance, 0) == 0);
      if (entry.hasEligibility) {
        entry.eligibilityToken =
            consumes<std::vector<unsigned int>>(edm::InputTag(targetsLabel, instance + "Eligibility"));
      }
      entry.truthToRecoToken = consumes<MapType>(edm::InputTag(associator, key + "TruthToReco"));
      // The first working point (Fixed) is the WP-free hit-sharing measure, so its map
      // supplies the loose reco-purity gate for every level.
      entry.firstWpRecoToTruthToken =
          consumes<MapType>(edm::InputTag(associator, key + "RecoToTruth" + workingPoints.front()));
      truthEntries_.push_back(std::move(entry));
    }
  }
  wpMissingWarned_ = std::vector<std::once_flag>(wpEntries_.size());
  truthMissingWarned_ = std::vector<std::once_flag>(truthEntries_.size());
}

template <typename RECO>
void TruthBranchRecoValidator<RECO>::bookHistograms(DQMStore::IBooker& booker,
                                                    edm::Run const&,
                                                    edm::EventSetup const&,
                                                    Histograms& histograms) const {
  // Each list is booked in its own entry order; the fill side indexes each list by the
  // same order, so the two must stay in lockstep per list.
  for (auto const& entry : wpEntries_) {
    booker.setCurrentFolder(dirName_ + entry.folder);
    algo_.bookRecoHistos(booker, histograms, Traits::calorimetric);
  }
  for (auto const& entry : truthEntries_) {
    booker.setCurrentFolder(dirName_ + entry.folder);
    // A calorimetric domain books the shared energy fraction and no duplicate histograms.
    algo_.bookTruthHistos(booker, histograms, Traits::calorimetric);
  }
}

template <typename RECO>
void TruthBranchRecoValidator<RECO>::dqmAnalyze(edm::Event const& event,
                                                edm::EventSetup const&,
                                                Histograms const& histograms) const {
  auto const& graph = event.get(graphToken_);
  auto const& hitIndexProduct = event.get(hitIndexToken_);
  truth::SubgraphHitView hitIndex(hitIndexProduct);

  // The eta where the branch enters the calorimeter. The eta of the root does not tell
  // which detector saw the branch: a top at eta 0 deposits in both endcaps. The value is
  // the boundary crossing (checkpoint 0) of the most energetic branch particle that reaches
  // the calorimeter, propagated up to every ancestor. It is computed once per event,
  // because per-branch descendant walks are quadratic at PU200.
  std::vector<double> caloEntryEta(graph.nParticles(), truth::kNoCaloEntry);
  if constexpr (!Traits::truthIsVertex) {
    const uint32_t nParticles = graph.nParticles();
    std::vector<double> caloEntryEnergy(nParticles, -1.);
    std::vector<uint32_t> worklist;
    worklist.reserve(nParticles);

    for (uint32_t p = 0; p < nParticles; ++p) {
      // Back-scattered tracks crossed the boundary INWARD; that is the same particle
      // coming back, not an entry, and truth::atLevel excludes them for the same reason.
      if (graph.particles()[p].backscattered) {
        continue;
      }
      const auto crossing = truth::Particle(&graph, p).checkpoint(0);
      if (!crossing.has_value()) {
        continue;
      }
      caloEntryEnergy[p] = crossing->momentum.energy();
      caloEntryEta[p] = crossing->position.eta();
      worklist.push_back(p);
    }

    // Carry the most energetic crossing up to every ancestor. Values only ever increase
    // and are bounded by the largest crossing in the event, so this terminates even if
    // the graph contains a cycle.
    for (std::size_t head = 0; head < worklist.size(); ++head) {
      const uint32_t p = worklist[head];
      for (const uint32_t vertexId : graph.productionVertices(p)) {
        if (vertexId >= graph.nVertices()) {
          continue;
        }
        for (const uint32_t parent : graph.incomingParticles(vertexId)) {
          if (parent < nParticles && caloEntryEnergy[p] > caloEntryEnergy[parent]) {
            caloEntryEnergy[parent] = caloEntryEnergy[p];
            caloEntryEta[parent] = caloEntryEta[p];
            worklist.push_back(parent);
          }
        }
      }
    }
  }

  // Projection onto the dominance antichain: for each particle, the antichain member it
  // descends from, or itself if it is a member. A branch and its descendants then add up
  // as one contributor. On the five validation samples no rate changes at four decimals,
  // because the associators insert no ancestor roots of the level.
  // Empty means the criterion does not apply: the domain configures no antichain or the
  // product is missing. The fake rate is then "matched to nothing".
  constexpr uint32_t kNoDominanceRoot = std::numeric_limits<uint32_t>::max();
  std::vector<uint32_t> dominanceRoot;
  if (hasDominanceTargets_) {
    edm::Handle<std::vector<unsigned int>> targets;
    event.getByToken(dominanceTargetsToken_, targets);
    if (targets.isValid()) {
      dominanceRoot.assign(graph.nParticles(), kNoDominanceRoot);
      std::vector<uint32_t> worklist;
      worklist.reserve(graph.nParticles());
      for (const unsigned int b : *targets) {
        if (b < dominanceRoot.size() && dominanceRoot[b] == kNoDominanceRoot) {
          dominanceRoot[b] = b;
          worklist.push_back(b);
        }
      }
      // First writer wins, so a particle reachable from two members is attributed once
      // and is never pushed twice. That also bounds the walk on a graph with a cycle.
      for (std::size_t head = 0; head < worklist.size(); ++head) {
        const uint32_t p = worklist[head];
        for (const uint32_t vertexId : graph.decayVertices(p)) {
          if (vertexId >= graph.nVertices()) {
            continue;
          }
          for (const uint32_t child : graph.outgoingParticles(vertexId)) {
            if (child < dominanceRoot.size() && dominanceRoot[child] == kNoDominanceRoot) {
              dominanceRoot[child] = dominanceRoot[p];
              worklist.push_back(child);
            }
          }
        }
      }
    }
  }

  // Scratch for the per-object projected contributions, reused across objects.
  std::vector<std::pair<uint32_t, double>> contributions;

  // Working-point-independent per-object quantities: the kinematics and the dominance
  // measure are identical at every working point by construction, so they are computed
  // once per collection and reused by its other working points.
  struct WpIndependent {
    Kinematics kin;
    double leadingShare = -1.;
    double dominanceRatio = -1.;
  };
  std::vector<WpIndependent> perObject;
  std::size_t cachedCollection = std::numeric_limits<std::size_t>::max();

  // Reco-driven side, one pass per (collection, working point).
  for (std::size_t i = 0; i < wpEntries_.size(); ++i) {
    auto const& entry = wpEntries_[i];

    edm::Handle<std::vector<RECO>> recoHandle;
    event.getByToken(entry.recoToken, recoHandle);
    edm::Handle<MapType> recoToTruthHandle;
    event.getByToken(entry.recoToTruthToken, recoToTruthHandle);
    if (!recoHandle.isValid() || !recoToTruthHandle.isValid()) {
      std::call_once(wpMissingWarned_[i], [&entry]() {
        edm::LogWarning("TruthBranchRecoValidator") << "no reco collection or no association map for " << entry.folder
                                                    << "; that folder stays booked and empty for the whole job.";
      });
      continue;
    }

    auto const& recoToTruth = recoToTruthHandle->getMap();

    // Every candidate branch of every object, from the first working point's map. Used
    // only for the dominance measure below.
    edm::Handle<MapType> allCandidatesHandle;
    event.getByToken(entry.allCandidatesToken, allCandidatesHandle);
    MapType const* allCandidates = allCandidatesHandle.isValid() ? allCandidatesHandle.product() : nullptr;

    if (const std::size_t collection = i / nWorkingPoints_; collection != cachedCollection) {
      perObject.assign(recoHandle->size(), WpIndependent{});
      for (std::size_t r = 0; r < recoHandle->size(); ++r) {
        auto& cached = perObject[r];
        cached.kin = Traits::kinematics((*recoHandle)[r]);

        // Dominance uses the shared quantity of each candidate, not the score. The score
        // penalises every contamination quadratically, so it rejects an object that one
        // branch dominates. The row is not sorted by shared quantity.
        if (allCandidates == nullptr) {
          continue;
        }
        auto const& row = allCandidates->getMap()[r];
        // Shared quantity per antichain member. A candidate above the antichain projects
        // nowhere and is dropped: its subgraph contains the members, so counting it counts
        // the same energy twice.
        contributions.clear();
        double total = 0.;
        for (auto const& cand : row) {
          const uint32_t member = dominanceRoot.empty()                   ? cand.index()
                                  : (cand.index() < dominanceRoot.size()) ? dominanceRoot[cand.index()]
                                                                          : kNoDominanceRoot;
          if (member == kNoDominanceRoot) {
            continue;
          }
          const double e = payloadValue(cand);
          total += e;
          auto it = std::find_if(
              contributions.begin(), contributions.end(), [member](auto const& c) { return c.first == member; });
          if (it == contributions.end()) {
            contributions.emplace_back(member, e);
          } else {
            it->second += e;
          }
        }
        if (total > 0.) {
          double best = 0., second = 0.;
          for (auto const& [member, e] : contributions) {
            if (e > best) {
              second = best;
              best = e;
            } else if (e > second) {
              second = e;
            }
          }
          cached.leadingShare = best / total;
          // Capped so the one-contributor case and a runaway ratio share a top bin.
          cached.dominanceRatio = (second > 0.) ? std::min(best / second, 20.) : 20.;
        }
      }
      cachedCollection = collection;
    }

    // Reco side: every object, whether it is matched, and whether the match is pileup.
    for (std::size_t r = 0; r < recoHandle->size(); ++r) {
      // [0] is the best match. The calorimetric recoToSim score does not enter
      // "associated": it is normalised to the total truth energy of the cell, so at PU200
      // it is near 1 also for a matched object. It enters strictMatch only.
      const bool associated = r < recoToTruth.size() && !recoToTruth[r].empty();
      bool strictMatch = associated;
      if constexpr (Traits::calorimetric) {
        strictMatch = associated && recoToTruth[r][0].score() < maxRecoToSimScore_;
      }
      const Kinematics& kin = perObject[r].kin;

      bool pileup = false;
      if (associated) {
        // The match index is a truth vertex for a composite domain and a particle for a
        // hit-based domain.
        const unsigned int matched = recoToTruth[r][0].index();
        if constexpr (Traits::truthIsVertex) {
          if (matched < graph.nVertices()) {
            pileup = graph.vertices()[matched].isFromPileup();
          }
        } else {
          if (matched < graph.nParticles()) {
            pileup = graph.particles()[matched].isFromPileup();
          }
        }
      }
      // Reco purity. For a composite object it is the pt^2 share of the constituents
      // from the leading truth vertex.
      const double recoPurity = associated ? 1. - static_cast<double>(recoToTruth[r][0].score()) : 0.;

      const double leadingShare = perObject[r].leadingShare;
      const double dominanceRatio = perObject[r].dominanceRatio;
      algo_.fill_dominance(histograms, i, leadingShare, dominanceRatio);

      // A fake is an object matched to nothing, or one where no antichain member owns
      // minLeadingTruthShare of the contributions.
      // An object with no candidate at the dominance level is not a fake: that measures
      // level coverage, not reconstruction. On no-PU ttbar it is 32.5% of tracksters and
      // 36.8% of tracks, against 0.3% of tracks matched to nothing. With the tracking level
      // at stableDecayProducts, it is 27.7% of tracks.
      // The dominance measure comes from the first working point, so it is the same at
      // every working point.
      const bool hasLevelCandidate = leadingShare >= 0.;
      const bool dominated = dominanceRoot.empty()
                                 ? associated
                                 : (associated && (!hasLevelCandidate || leadingShare >= minLeadingTruthShare_));
      algo_.fill_reco(histograms,
                      i,
                      kin,
                      {.dominated = dominated,
                       .associated = associated,
                       .hasLevelCandidate = hasLevelCandidate,
                       .pileup = pileup,
                       .strictMatch = strictMatch,
                       .matchQuality = recoPurity});
      if (associated) {
        algo_.fill_match(histograms, i, recoToTruth[r][0].score(), payloadValue(recoToTruth[r][0]), recoPurity);
        // Resolution against the truth object that this working point matches. A
        // composite truth object is a vertex with no direction, so it has no residual.
        if constexpr (!Traits::truthIsVertex) {
          const unsigned int matched = recoToTruth[r][0].index();
          if (matched < graph.nParticles() && Traits::hasDirection((*recoHandle)[r])) {
            auto const& p4 = graph.particles()[matched].momentum;
            if (p4.pt() > 0.) {
              Kinematics truthKin;
              truthKin.pt = p4.pt();
              truthKin.eta = p4.eta();
              truthKin.phi = p4.phi();
              algo_.fill_resolution(histograms, i, truthKin, kin.pt, kin.eta, kin.phi);
            }
          }
        }
      }
    }
  }

  // Truth-driven side, one pass per (collection, level). The denominator is the target
  // set of the level.
  for (std::size_t i = 0; i < truthEntries_.size(); ++i) {
    auto const& entry = truthEntries_[i];

    edm::Handle<std::vector<unsigned int>> targetsHandle;
    event.getByToken(entry.targetsToken, targetsHandle);
    edm::Handle<MapType> truthToRecoHandle;
    event.getByToken(entry.truthToRecoToken, truthToRecoHandle);
    edm::Handle<MapType> firstWpHandle;
    event.getByToken(entry.firstWpRecoToTruthToken, firstWpHandle);
    if (!targetsHandle.isValid() || !truthToRecoHandle.isValid() || !firstWpHandle.isValid()) {
      std::call_once(truthMissingWarned_[i], [&entry]() {
        edm::LogWarning("TruthBranchRecoValidator") << "no truth targets or no truth-driven map for " << entry.folder
                                                    << "; that folder stays booked and empty for the whole job.";
      });
      continue;
    }

    auto const& truthToReco = truthToRecoHandle->getMap();
    auto const& recoToTruth = firstWpHandle->getMap();

    // Reco-normalised score of a (truth, reco) pair, from the reco-driven map of the first
    // working point. An absent pair scores 1, the worst value.
    auto recoScoreOf = [&recoToTruth](unsigned int recoIndex, unsigned int truthIndex) {
      if (recoIndex >= recoToTruth.size()) {
        return 1.;
      }
      for (auto const& match : recoToTruth[recoIndex]) {
        if (match.index() == truthIndex) {
          return static_cast<double>(match.score());
        }
      }
      return 1.;
    };

    // The association map can be shorter than the number of particles. A target beyond
    // its end has no matches and counts as lost; it stays in the denominator.
    using MatchList = std::decay_t<decltype(truthToReco[0])>;
    static const MatchList kNoMatches{};

    // Parallel to the target list. Absent for the signal entries, which have no kinematic
    // selection.
    edm::Handle<std::vector<unsigned int>> eligibilityHandle;
    if (entry.hasEligibility) {
      event.getByToken(entry.eligibilityToken, eligibilityHandle);
    }
    const bool haveEligibility = eligibilityHandle.isValid() && eligibilityHandle->size() == targetsHandle->size();

    for (std::size_t t = 0; t < targetsHandle->size(); ++t) {
      const unsigned int b = (*targetsHandle)[t];
      // Mask of the plotted-axis cuts that this target fails. 0 when no eligibility
      // product is present.
      const unsigned int failedCuts = haveEligibility ? (*eligibilityHandle)[t] : 0u;
      auto const& matches = (b < truthToReco.size()) ? truthToReco[b] : kNoMatches;
      Kinematics kin;
      auto reason = static_cast<unsigned int>(truth::VertexReason::Unknown);

      if constexpr (Traits::truthIsVertex) {
        // The truth object is a vertex: its position, its number of outgoing particles and
        // its creation process.
        if (b >= graph.nVertices()) {
          continue;
        }
        const truth::Vertex vertex(&graph, b);
        auto const& vdata = vertex.data();
        auto const& pos = vertex.position();
        kin.vertpos = std::sqrt(pos.x() * pos.x() + pos.y() * pos.y());
        kin.zpos = pos.z();
        kin.nhits = vertex.outgoingParticles().size();
        reason = reasonBin(vdata);
      } else {
        if (b >= graph.nParticles()) {
          continue;
        }
        auto const& particle = graph.particles()[b];
        const auto& p4 = particle.momentum;
        // A resonance in its pre-ISR copy has pt exactly 0 and no direction. It stays in
        // the denominator; only the undefined quantities go to the underflow.
        const bool hasDirection = p4.pt() > 0.;
        kin.pt = p4.pt();
        kin.eta = hasDirection ? p4.eta() : truth::kNoCaloEntry;
        kin.phi = hasDirection ? p4.phi() : truth::kNoCaloEntry;
        kin.caloeta = caloEntryEta[b];
        // The branch footprint, the truth analogue of a track hit count.
        const auto subgraph = hitIndex.subgraphHits(Traits::hitChannel, b);
        kin.nhits = subgraph.size();
        const truth::Particle branchRoot(&graph, b);
        // The number of ancestors of the branch root.
        kin.depth = branchRoot.ancestorCount();
        // The fraction of the branch footprint that the root particle deposits itself.
        const auto direct = hitIndex.directHits(Traits::hitChannel, b);
        kin.root_footprint_fraction = subgraph.empty() ? 0. : static_cast<double>(direct.size()) / subgraph.size();
        // A root that is not a quark or a gluon fills the Other bin.
        kin.flavour = truth::flavourBin(particle.pdgId);

        const auto vertices = branchRoot.productionVertices();
        if (!vertices.empty()) {
          auto const& vdata = vertices.front().data();
          reason = reasonBin(vdata);
          const auto& pos = vertices.front().position();
          kin.vertpos = std::sqrt(pos.x() * pos.x() + pos.y() * pos.y());
          kin.zpos = pos.z();
          // Transverse and longitudinal impact parameters with respect to the origin, the
          // truth counterpart of the track dxy and dz. Undefined at pt 0.
          if (hasDirection) {
            kin.dxy = (-pos.x() * p4.py() + pos.y() * p4.px()) / p4.pt();
            kin.dz = pos.z() - (pos.x() * p4.px() + pos.y() * p4.py()) / p4.pt() * (p4.pz() / p4.pt());
          } else {
            kin.dxy = truth::kNoCaloEntry;
            kin.dz = truth::kNoCaloEntry;
          }
        } else {
          // No production vertex: position and impact parameters go to the underflow, not
          // to a false peak at 0.
          kin.vertpos = truth::kNoCaloEntry;
          kin.zpos = truth::kNoCaloEntry;
          kin.dxy = truth::kNoCaloEntry;
          kin.dz = truth::kNoCaloEntry;
        }
      }

      // Classify the truth object from the truth-driven map. Individual: one reco object
      // covers it. Duplicate: more than one does. Split: only several together cover it.
      using Outcome = truth::TruthBranchHistoProducerAlgo::TruthOutcome;
      unsigned int nIndividual = 0;
      unsigned int nPure = 0;
      double collectiveCoverage = 0.;
      double leadingTruthPurity = 0.;
      double leadingSharedEnergyFraction = 0.;
      for (auto const& match : matches) {
        const double truthPurity = 1. - static_cast<double>(match.score());
        leadingTruthPurity = std::max(leadingTruthPurity, truthPurity);
        if constexpr (Traits::calorimetric) {
          // Calorimetric payload: the shared energy over the branch energy in the
          // detectors of this collection. The score is the simToReco score.
          const double sharedEnergyFraction = payloadValue(match);
          leadingSharedEnergyFraction = std::max(leadingSharedEnergyFraction, sharedEnergyFraction);
          collectiveCoverage += sharedEnergyFraction;
          if (sharedEnergyFraction > minSharedEnergyFractionForIndividual_ &&
              recoScoreOf(match.index(), b) < maxRecoToSimScore_) {
            ++nIndividual;
          }
          if (match.score() < maxSimToRecoScoreForDuplicate_) {
            ++nPure;
          }
        } else {
          collectiveCoverage += truthPurity;
          if (truthPurity >= minTruthPurityForIndividual_ &&
              1. - recoScoreOf(match.index(), b) >= minRecoPurityLoose_) {
            ++nIndividual;
          }
        }
      }
      const bool collective = collectiveCoverage >= minCollectiveCoverage_ && !matches.empty();
      Outcome outcome = Outcome::Lost;
      if constexpr (Traits::calorimetric) {
        // Duplicate is a subset of Individual, so efficiency is exactly the shared energy
        // fraction cut.
        outcome = (nIndividual >= 1) ? (nPure > 1 ? Outcome::Duplicate : Outcome::Individual)
                  : collective       ? Outcome::Split
                                     : Outcome::Lost;
      } else {
        outcome = (nIndividual == 1)  ? Outcome::Individual
                  : (nIndividual > 1) ? Outcome::Duplicate
                  : collective        ? Outcome::Split
                                      : Outcome::Lost;
      }
      // Cumulative: one reco object or several together cover the truth object.
      const bool cumulative = nIndividual >= 1 || collective;

      algo_.fill_simul(histograms, i, kin, outcome, cumulative, failedCuts);
      algo_.fill_reason(histograms, i, reason, outcome);
      if (!matches.empty()) {
        algo_.fill_truth_purity(histograms, i, leadingTruthPurity);
        if constexpr (Traits::calorimetric) {
          algo_.fill_shared_energy_fraction(histograms, i, leadingSharedEnergyFraction);
        }
      }
    }
  }
}

template <typename RECO>
void TruthBranchRecoValidator<RECO>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<std::string>("dirName", Traits::defaultDir);
  desc.add<std::string>("associator", Traits::defaultAssociator);
  desc.add<std::string>("targetsProducer", "truthBranchTargets")
      ->setComment("Module holding the level denominators and signal seeds for the hit-based domains");
  desc.add<std::vector<edm::InputTag>>("recoCollections", {});
  desc.add<std::vector<std::string>>("workingPoints", {"Fixed"});
  if constexpr (Traits::truthIsVertex) {
    desc.add<std::string>("vertexResolution", "interaction")
        ->setComment(
            "Names the one truth-driven folder of a composite domain, matching the associator's resolution: "
            "'interaction' for primary vertices, 'immediate' for secondary vertices");
  } else {
    desc.add<double>("minLeadingTruthShare", 0.5)
        ->setComment(
            "Fake criterion: one branch of dominanceLevel must own at least this share of the shared quantity all "
            "candidates at that level contribute. An object below it, or with no candidate there, is a fake");
    desc.add<std::string>("dominanceLevel", "caloBoundary")
        ->setComment(
            "The level whose targets the leading-truth-contributor measure is computed over. It must be an "
            "ANTICHAIN of distinct particles: the full selected-root set is not one, and using it compares a branch "
            "against its own descendants");
    desc.add<std::vector<std::string>>("truthLevels", {"caloBoundary"})
        ->setComment(
            "Graph levels the truth-driven metrics are measured at, one folder per level. Must match the "
            "associator's truthLevels: each level consumes its own denominator product");
    desc.add<std::vector<int>>("signalSeedPdgIds", {})
        ->setComment(
            "The selection preset's seed species, the SAME values the associators get. Empty, or the full-graph "
            "escape hatch {0}, means the sample has no resonance, and then the signal and signalNoSelection "
            "folders are NOT BOOKED: the question has no meaning where the configuration names no signal");
    desc.add<std::vector<int>>("signalSeedHadronFlavors", {})
        ->setComment(
            "The preset's heavy-flavour hadron seeds, the SAME values the associators get; flavours alone also "
            "name a resonance and book the signal folders");
  }
  if constexpr (Traits::calorimetric) {
    desc.add<double>("minSharedEnergyFractionForIndividual", 0.5)
        ->setComment(
            "Efficiency gate: the shared energy over the truth branch energy. HGCalValidator's "
            "minTSTSharedEneFracEfficiency (Validation/HGCalValidation/python/HGVHistoProducerAlgoBlock_cfi.py:82, "
            "applied src/HGVHistoProducerAlgo.cc:2897). This is an ENERGY FRACTION, not a score");
    desc.add<double>("maxSimToRecoScoreForDuplicate", 0.2)
        ->setComment(
            "More than one reco object below this simToReco score makes the truth object a duplicate. "
            "HGCalValidator's maxSimToRecoScoreForPurity/Duplicate (HGVHistoProducerAlgoBlock_cfi.py:72-73, "
            "applied HGVHistoProducerAlgo.cc:2898-2899)");
    desc.add<double>("maxRecoToSimScore", 0.6)
        ->setComment(
            "A reco object below this recoToSim score is not a fake. HGCalValidator's "
            "maxRecoToSimScoreForNonFake/Merge (HGVHistoProducerAlgoBlock_cfi.py:70-71, applied "
            "HGVHistoProducerAlgo.cc:2819-2820)");
  } else {
    desc.add<double>("minTruthPurityForIndividual", 0.5)
        ->setComment(
            "A single reco object must cover at least this much of the truth object to have reconstructed it. "
            "truthBranchValidation_cff sets both purity cuts per domain to the corresponding standard "
            "validation's thresholds");
    desc.add<double>("minRecoPurityLoose", 0.25)
        ->setComment("Loose cut in the other direction: that object must not be mostly something else");
  }
  desc.add<double>("minCollectiveCoverage", 0.5)
      ->setComment("Several objects together must cover at least this much of the truth object to count as split");

  edm::ParameterSetDescription algo;
  // Every axis is declared here. The two variable lists select the axes a domain books.
  const std::vector<std::tuple<std::string, int, double, double>> axes = {{"pt", 50, 0., 100.},
                                                                          {"eta", 50, -4., 4.},
                                                                          {"phi", 36, -3.2, 3.2},
                                                                          {"nhits", 40, 0., 40.},
                                                                          {"vertpos", 40, 0., 60.},
                                                                          {"zpos", 40, -30., 30.},
                                                                          {"dxy", 40, -5., 5.},
                                                                          {"dz", 40, -20., 20.},
                                                                          {"depth", 15, 0., 15.},
                                                                          {"root_footprint_fraction", 20, 0., 1.},
                                                                          {"caloeta", 50, -4., 4.},
                                                                          {"flavour", 8, 0., 8.}};
  for (auto const& [name, nbins, lo, hi] : axes) {
    algo.add<int>("nint_" + name, nbins);
    algo.add<double>("min_" + name, lo);
    algo.add<double>("max_" + name, hi);
    // 0 keeps the axis uniform. A positive value selects symlog binning: one linear bin up
    // to the value, then log-spaced bins. The linear bin keeps the entries at exactly 0.
    algo.add<double>("linthresh_" + name, 0.);
    // Optional reco-side override of the axis. When unset, the reco side uses the shared
    // range.
    algo.addOptional<int>("nint_reco_" + name);
    algo.addOptional<double>("min_reco_" + name);
    algo.addOptional<double>("max_reco_" + name);
    algo.addOptional<double>("linthresh_reco_" + name);
  }
  // The absolute-pseudorapidity bands that have a folder, in addition to the inclusive
  // folder. Each domain lists only the bands that its detector covers.
  algo.add<std::vector<std::string>>("etaRegions", {"etaLt15", "eta15to30", "eta30to45"});
  algo.add<std::vector<std::string>>("truthVariables", {"pt", "eta", "phi"});
  algo.add<std::vector<std::string>>("recoVariables", {"pt", "eta", "phi"});
  algo.add<int>("nintScore", 50);
  algo.add<double>("minScore", 0.);
  algo.add<double>("maxScore", 1.);
  algo.add<int>("nintShared", 50);
  algo.add<double>("minShared", 0.);
  algo.add<double>("maxShared", 50.);
  algo.add<int>("nintRes", 120);
  algo.add<double>("minRes", -1.5);
  algo.add<double>("maxRes", 1.5);
  algo.add<int>("nint_res_eta", 20);
  algo.add<double>("min_res_eta", -4.);
  algo.add<double>("max_res_eta", 4.);
  algo.add<int>("nint_res_pt", 15);
  algo.add<double>("min_res_pt", 0.);
  algo.add<double>("max_res_pt", 100.);
  desc.add<edm::ParameterSetDescription>("histoProducerAlgoBlock", algo);

  descriptions.add(Traits::cfiName, desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using TruthBranchTrackValidator = TruthBranchRecoValidator<reco::Track>;
DEFINE_FWK_MODULE(TruthBranchTrackValidator);
using TruthBranchVertexValidator = TruthBranchRecoValidator<reco::Vertex>;
DEFINE_FWK_MODULE(TruthBranchVertexValidator);
using TruthBranchTracksterValidator = TruthBranchRecoValidator<ticl::Trackster>;
DEFINE_FWK_MODULE(TruthBranchTracksterValidator);
using TruthBranchPFClusterValidator = TruthBranchRecoValidator<reco::PFCluster>;
DEFINE_FWK_MODULE(TruthBranchPFClusterValidator);
