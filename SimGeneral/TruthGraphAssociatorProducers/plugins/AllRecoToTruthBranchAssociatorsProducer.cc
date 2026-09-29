// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

// Associates every configured reco collection of one type to the truth graph, at every
// working point. For each collection it produces one RecoToTruth map per working point
// and one TruthToReco map, with instance labels derived from the input tag.
//
// A hit-based reco type needs a truth::recoHits overload. A concept selects the overload.
//
// Working points differ only in the arguments to bestAdaptiveBranch. The associator and
// the candidate list of each object are built once and shared by every working point.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <concepts>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <string_view>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

#include "PhysicsTools/TruthInfo/interface/Branch.h"
#include "PhysicsTools/TruthInfo/interface/AssignableTarget.h"
#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"
#include "PhysicsTools/TruthInfo/interface/Interactions.h"
#include "PhysicsTools/TruthInfo/interface/BranchSelector.h"
#include "PhysicsTools/TruthInfo/interface/RecoHitAdapters.h"
#include "PhysicsTools/TruthInfo/interface/TrackerCells.h"
#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

// Not in the anonymous namespace: the producer, which has external linkage, holds a
// VertexResolution member, and a member cannot have a type with internal linkage.
namespace truthassociation {
  // The truth vertex at which a constituent counts.
  //   Immediate    the production vertex of the matched particle. For a secondary
  //                vertex, where the tracks are produced.
  //   Interaction  the vertex of the interaction that the particle belongs to. For a
  //                primary vertex: a track from a downstream decay counts at the
  //                interaction.
  enum class VertexResolution { Immediate, Interaction };
}  // namespace truthassociation

namespace {
  using truth::byAscendingScore;
  using truthassociation::VertexResolution;

  // DetId::Detector by name. An unknown name is a configuration error.
  DetId::Detector detectorFromName(std::string const& name) {
    static constexpr std::pair<std::string_view, DetId::Detector> kDetectors[] = {
        {"Tracker", DetId::Tracker},
        {"Muon", DetId::Muon},
        {"Ecal", DetId::Ecal},
        {"Hcal", DetId::Hcal},
        {"Calo", DetId::Calo},
        {"Forward", DetId::Forward},
        {"VeryForward", DetId::VeryForward},
        {"HGCalEE", DetId::HGCalEE},
        {"HGCalHSi", DetId::HGCalHSi},
        {"HGCalHSc", DetId::HGCalHSc},
        {"HGCalTrigger", DetId::HGCalTrigger}};
    for (auto const& [known, detector] : kDetectors) {
      if (known == name) {
        return detector;
      }
    }
    throw cms::Exception("Configuration") << "denominatorDetectors: unknown DetId::Detector name '" << name << "'";
  }

  // A reco type that yields its own hits needs nothing but itself.
  template <typename RECO>
  concept SelfContainedRecoHits = requires(RECO const& r) {
    { truth::recoHits(r) } -> std::same_as<std::vector<truth::RecoHit>>;
  };

  // A reco type built out of layer clusters needs the layer-cluster collection too.
  template <typename RECO>
  concept LayerClusterBackedRecoHits = requires(RECO const& r, std::vector<reco::CaloCluster> const& lcs) {
    { truth::recoHits(r, lcs) } -> std::same_as<std::vector<truth::RecoHit>>;
  };

  template <typename RECO>
  concept AdaptableToTruthHits = SelfContainedRecoHits<RECO> || LayerClusterBackedRecoHits<RECO>;

  // How a reco type reaches the truth.
  //   HitBased         the object owns detector hits and is matched on them (tracks by
  //                    shared hits, tracksters by shared energy).
  //   ConstituentBased the object is built from objects that are already associated,
  //                    and its truth is aggregated from their maps. A vertex uses the
  //                    track maps, as VertexAssociatorByPositionAndTracks does.
  enum class AssociationStrategy { HitBased, ConstituentBased };

  template <typename RECO>
  struct TruthAssociationTraits;

  template <>
  struct TruthAssociationTraits<reco::Track> {
    static constexpr auto strategy = AssociationStrategy::HitBased;
    using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
    static constexpr truth::HitChannel channel = truth::HitChannel::Tracker;
    static constexpr auto metric = truth::BranchHitAssociator::Metric::SharedHits;
    static constexpr const char* cfiName = "allTrackToTruthBranchAssociators";
  };

  // A vertex has no hits: its truth is aggregated from its tracks. The payload is a
  // pt^2-weighted fraction of the vertex tracks.
  template <>
  struct TruthAssociationTraits<reco::Vertex> {
    static constexpr auto strategy = AssociationStrategy::ConstituentBased;
    using ConstituentType = reco::Track;
    using MapType = ticl::TICLAssociationMap<ticl::mapWithFractionAndScore>;
    static constexpr const char* cfiName = "allVertexToTruthBranchAssociators";

    // Visits (constituent index, weight). The index is the Ref key, which is the row of
    // the constituent association map.
    // The weight is pt^2, as in the sharedPt2Fraction of calculateVertexSharedTracks
    // (SimTracker/VertexAssociation/src/calculateVertexSharedTracks.cc). The vertex fit
    // weight is not used: it measures the constraint on the fit, not the momentum share.
    template <typename F>
    static void forEachConstituent(reco::Vertex const& vertex, F&& visit) {
      for (auto it = vertex.tracks_begin(); it != vertex.tracks_end(); ++it) {
        const float pt = (*it)->pt();
        visit(static_cast<unsigned int>(it->key()), pt * pt);
      }
    }

    static float totalWeight(reco::Vertex const& vertex) {
      float total = 0.f;
      forEachConstituent(vertex, [&total](unsigned int, float w) { total += w; });
      return total;
    }
  };

  // A particle-flow cluster owns its calorimeter cells. reco::PFCluster derives from
  // reco::CaloCluster, so the reco::CaloCluster adapter applies and the cluster is matched
  // on shared energy. Use one module per subdetector, with its own denominatorDetectors:
  // against a joint ECAL+HCAL denominator, an ECAL cluster alone never reaches the
  // individual threshold for a hadron branch.
  template <>
  struct TruthAssociationTraits<reco::PFCluster> {
    static constexpr auto strategy = AssociationStrategy::HitBased;
    using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
    static constexpr truth::HitChannel channel = truth::HitChannel::Calo;
    static constexpr auto metric = truth::BranchHitAssociator::Metric::SharedEnergy;
    static constexpr const char* cfiName = "truthBranchPFClusterAssociators";
  };

  // A trackster owns calorimeter energy through its layer clusters. It is matched on
  // shared energy in the calorimeter channel, the metric of the TICL trackster validation.
  template <>
  struct TruthAssociationTraits<ticl::Trackster> {
    static constexpr auto strategy = AssociationStrategy::HitBased;
    using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
    static constexpr truth::HitChannel channel = truth::HitChannel::Calo;
    static constexpr auto metric = truth::BranchHitAssociator::Metric::SharedEnergy;
    static constexpr const char* cfiName = "truthBranchTracksterAssociators";
  };

  [[nodiscard]] inline std::optional<uint32_t> countingVertex(
      truth::Graph const& graph,
      uint32_t particleId,
      VertexResolution resolution,
      std::unordered_map<uint64_t, uint32_t> const& interactionVertex) {
    if (resolution == VertexResolution::Interaction) {
      const auto it = interactionVertex.find(graph.particles()[particleId].eventId);
      if (it == interactionVertex.end()) {
        return std::nullopt;
      }
      return it->second;
    }
    const auto production = truth::Particle(&graph, particleId).productionVertices();
    if (production.empty()) {
      return std::nullopt;
    }
    return production.front().id();
  }

  template <typename RECO>
  concept HitBasedDomain = TruthAssociationTraits<RECO>::strategy == AssociationStrategy::HitBased;

  template <typename RECO>
  concept ConstituentBasedDomain = TruthAssociationTraits<RECO>::strategy == AssociationStrategy::ConstituentBased;
}  // namespace

template <typename RECO>
  requires(AdaptableToTruthHits<RECO> || ConstituentBasedDomain<RECO>)
class AllRecoToTruthBranchAssociatorsProducer : public edm::global::EDProducer<> {
public:
  explicit AllRecoToTruthBranchAssociatorsProducer(edm::ParameterSet const&);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  struct WorkingPoint {
    std::string name;
    float reverseWeight;
    float maxReverseScore;
    bool adaptive;
  };

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClustersToken_;
  // Calorimetric domains only: the rechit collections whose energies weight every
  // cell of the shared-energy metric, as the TICL associators weight them. With no
  // collection configured the metric weights the cells by their sim energy.
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> hgcalRecHitTokens_;
  std::vector<edm::EDGetTokenT<reco::PFRecHitCollection>> pfRecHitTokens_;
  // One warning per job for each input condition that the maps cannot repair.
  mutable std::once_flag moduleKeyedWarned_;
  mutable std::once_flag placeholderVertexWarned_;

  std::vector<std::pair<std::string, edm::EDGetTokenT<std::vector<RECO>>>> recoTokens_;
  // One warning per collection per job when none of its hits is in denominatorDetectors:
  // every shared-energy fraction is then zero.
  mutable std::vector<std::once_flag> outOfScopeWarned_;
  std::vector<WorkingPoint> workingPoints_;
  // Bit mask of the detectors that the sim-normalised shared-energy fraction is
  // normalised to.
  uint32_t denominatorDetectors_ = truth::BranchHitAssociator::kAllDetectors;
  const bool truthToRecoSignalOnly_;
  const bool heavyFlavorOnly_;
  // Composite domains only: the worst score a constituent's best match may have and
  // still place the constituent at a truth vertex.
  float maxConstituentScore_ = 1.f;
  // The selector-passing candidate roots, from TruthBranchTargetsProducer.
  edm::EDGetTokenT<std::vector<unsigned int>> targetsToken_;
  // The subset of those roots that an adaptive working point may answer with.
  edm::EDGetTokenT<std::vector<unsigned int>> assignableTargetsToken_;

  using Traits = TruthAssociationTraits<RECO>;
  using MapType = typename Traits::MapType;

  // Composite domains read the association maps of their constituents, one per
  // collection and working point. The constituents are tracks.
  using ConstituentMapType = TruthAssociationTraits<reco::Track>::MapType;
  std::vector<std::vector<edm::EDGetTokenT<ConstituentMapType>>> constituentMapTokens_;
  VertexResolution vertexResolution_ = VertexResolution::Immediate;
};

template <typename RECO>
  requires(AdaptableToTruthHits<RECO> || ConstituentBasedDomain<RECO>)
AllRecoToTruthBranchAssociatorsProducer<RECO>::AllRecoToTruthBranchAssociatorsProducer(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      truthToRecoSignalOnly_(cfg.getParameter<bool>("truthToRecoSignalOnly")),
      heavyFlavorOnly_(cfg.getParameter<bool>("heavyFlavorOnly")) {
  if constexpr (LayerClusterBackedRecoHits<RECO>) {
    layerClustersToken_ = consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"));
    for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("hgcalRecHits"))
      hgcalRecHitTokens_.push_back(consumes<HGCRecHitCollection>(tag));
    for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("pfRecHits"))
      pfRecHitTokens_.push_back(consumes<reco::PFRecHitCollection>(tag));
  }

  targetsToken_ = consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("targetsSrc"));
  assignableTargetsToken_ =
      consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("assignableTargetsSrc"));

  if constexpr (ConstituentBasedDomain<RECO>) {
    maxConstituentScore_ = cfg.getParameter<double>("maxConstituentScore");
  }

  if (auto const detectors = cfg.getParameter<std::vector<std::string>>("denominatorDetectors"); !detectors.empty()) {
    denominatorDetectors_ = 0u;
    for (auto const& name : detectors) {
      denominatorDetectors_ |= 1u << static_cast<uint32_t>(detectorFromName(name));
    }
  }

  const auto names = cfg.getParameter<std::vector<std::string>>("workingPointNames");
  const auto weights = cfg.getParameter<std::vector<float>>("adaptiveReverseWeight");
  const auto ceilings = cfg.getParameter<std::vector<float>>("adaptiveMaxReverseScore");
  if (names.size() != weights.size() || names.size() != ceilings.size()) {
    throw cms::Exception("Configuration")
        << "workingPointNames, adaptiveReverseWeight and adaptiveMaxReverseScore must have the same length";
  }
  if (names.empty()) {
    throw cms::Exception("Configuration")
        << "workingPointNames is empty: the truth-driven maps are filled inside the working-point loop, so an empty "
           "list would silently produce empty TruthToReco products";
  }
  for (std::size_t i = 0; i < names.size(); ++i) {
    // "Fixed" means the plain per-root match; every other point drives the climb.
    workingPoints_.push_back({names[i], weights[i], ceilings[i], names[i] != "Fixed"});
  }

  if constexpr (ConstituentBasedDomain<RECO>) {
    // The truth-driven direction reads the constituent map of the first working point,
    // so that point must be the plain per-root match.
    if (names.front() != "Fixed") {
      throw cms::Exception("Configuration")
          << "workingPointNames starts with '" << names.front()
          << "': a composite domain reads the first point's constituent map, so the first point must be 'Fixed'";
    }
    // The truth target of a composite object is a vertex, so there is one denominator.
    produces<std::vector<unsigned int>>("truthToRecoTargets");
    // Every selected truth vertex, before the signal-only restriction.
    produces<std::vector<unsigned int>>("selectedTruthVertices");
    const auto resolution = cfg.getParameter<std::string>("vertexResolution");
    if (resolution == "interaction") {
      vertexResolution_ = VertexResolution::Interaction;
    } else if (resolution == "immediate") {
      vertexResolution_ = VertexResolution::Immediate;
    } else {
      throw cms::Exception("Configuration")
          << "vertexResolution must be 'immediate' or 'interaction', got '" << resolution << "'";
    }
  }

  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("recoCollections")) {
    // "label_instance", the same key that names the DQM folder.
    std::string key = tag.label();
    if (!tag.instance().empty()) {
      key += "_" + tag.instance();
    }
    recoTokens_.emplace_back(key, consumes<std::vector<RECO>>(tag));

    if constexpr (ConstituentBasedDomain<RECO>) {
      // One constituent map per working point, in the same order as workingPoints_.
      const auto upstream = cfg.getParameter<std::string>("constituentAssociator");
      const auto constituentKey = cfg.getParameter<std::string>("constituentCollection");
      std::vector<edm::EDGetTokenT<ConstituentMapType>> perWp;
      perWp.reserve(workingPoints_.size());
      for (auto const& wp : workingPoints_) {
        perWp.push_back(
            consumes<ConstituentMapType>(edm::InputTag(upstream, constituentKey + "RecoToTruth" + wp.name)));
      }
      constituentMapTokens_.push_back(std::move(perWp));
    }

    // The two directions are not transposes of each other.
    // RecoToTruth is reco-driven, one product per working point. Score: 1 - reco purity.
    // TruthToReco is truth-driven, one product, with no adaptive climb. Score: 1 - truth
    // purity.
    for (auto const& wp : workingPoints_) {
      produces<MapType>(key + "RecoToTruth" + wp.name);
    }
    produces<MapType>(key + "TruthToReco");
  }
  outOfScopeWarned_ = std::vector<std::once_flag>(recoTokens_.size());
}

template <typename RECO>
  requires(AdaptableToTruthHits<RECO> || ConstituentBasedDomain<RECO>)
void AllRecoToTruthBranchAssociatorsProducer<RECO>::produce(edm::StreamID,
                                                            edm::Event& event,
                                                            edm::EventSetup const&) const {
  auto const& graph = event.get(graphToken_);
  auto const& hitIndex = event.get(hitIndexToken_);
  if constexpr (!ConstituentBasedDomain<RECO>) {
    if (Traits::channel == truth::HitChannel::Tracker && truth::isModuleKeyedTracker(hitIndex)) {
      std::call_once(moduleKeyedWarned_, [] {
        edm::LogWarning("AllRecoToTruthBranchAssociatorsProducer")
            << "an input event has tracker truth with no cells, so no track of that event matches it. Its input was "
               "digitised before the tracker truth was keyed by cell; reprocess it from the DIGI step.";
      });
    }
  }

  std::vector<reco::CaloCluster> const* layerClusters = nullptr;
  // The rechit energy of every cell, the weight of the shared-energy metric. Every
  // configured collection is required. The barrel PFRecHits (DetId detectors 3 and 4) go
  // in before the HGCAL rechits (8 to 10), so the table is in DetId order when each
  // collection is sorted.
  truth::CellEnergyTable recHitEnergies;
  truth::CellEnergyTable const* recHitEnergiesPtr = nullptr;
  if constexpr (LayerClusterBackedRecoHits<RECO>) {
    for (auto const& token : pfRecHitTokens_) {
      auto const& hits = event.get(token);
      recHitEnergies.reserve(recHitEnergies.size() + hits.size());
      for (auto const& hit : hits)
        recHitEnergies.add(hit.detId(), hit.energy());
    }
    for (auto const& token : hgcalRecHitTokens_) {
      auto const& hits = event.get(token);
      recHitEnergies.reserve(recHitEnergies.size() + hits.size());
      for (auto const& hit : hits)
        recHitEnergies.add(hit.id().rawId(), hit.energy());
    }
    recHitEnergies.finalize();
    if (!pfRecHitTokens_.empty() || !hgcalRecHitTokens_.empty())
      recHitEnergiesPtr = &recHitEnergies;
    layerClusters = &event.get(layerClustersToken_);
  }

  const unsigned int nBranches = graph.nParticles();

  auto const& selectedRoots = event.get(targetsToken_);
  // Membership mask of the assignable roots. A composite domain answers with a vertex,
  // so it needs no mask.
  std::vector<uint8_t> isAssignable;
  if constexpr (!ConstituentBasedDomain<RECO>) {
    isAssignable.assign(nBranches, 0);
    for (const unsigned int id : event.get(assignableTargetsToken_)) {
      if (id < isAssignable.size()) {
        isAssignable[id] = 1;
      }
    }
  }

  // The denominator of the truth-side fraction: what each truth vertex itself produced,
  // in the same pt^2 weighting the numerator uses.
  std::unordered_map<unsigned int, float> truthWeightPerVertex;

  unsigned int placeholderCount = 0;
  std::unordered_map<uint64_t, uint32_t> interactionVertex;
  if (ConstituentBasedDomain<RECO> && vertexResolution_ == VertexResolution::Interaction) {
    for (auto const& interaction : truth::interactions(graph)) {
      interactionVertex.emplace(interaction.eventId(), interaction.vertexId());
      placeholderCount += interaction.isPlaceholder() ? 1 : 0;
    }
  }
  if (placeholderCount > 0) {
    std::call_once(placeholderVertexWarned_, [placeholderCount] {
      edm::LogWarning("AllRecoToTruthBranchAssociatorsProducer")
          << placeholderCount
          << " interactions resolve only to a logical vertex that did not merge with a SimVertex and whose "
             "position is indistinguishable from a default-constructed one. Their constituents are counted "
             "there, so a vertex efficiency or purity for those interactions carries no position. This is what "
             "a pileup sub-event looks like when all of its GenToSim links were dropped. Reported once per job.";
    });
  }

  if constexpr (ConstituentBasedDomain<RECO>) {
    // The findable truth vertices: those that produce at least two findable tracks, since
    // one track cannot make a vertex. The count, the signal count and the pt^2 weight use
    // one population: in-time, charged particles that own tracker hits. A charged
    // particle that decays before the tracker (a D+ from a B, after about 300 um) has no
    // track. The tracker-hit requirement is dropped when the input has no tracker truth.
    // With Interaction resolution, only generator particles count, as for a
    // TrackingVertex, and only the earliest member of each chain, so a particle that
    // interacts counts once.
    const bool trackerTruthPresent = hitIndex.hasChannel(truth::HitChannel::Tracker);
    std::unordered_map<unsigned int, unsigned int> rootsPerVertex;
    std::unordered_map<unsigned int, unsigned int> signalRootsPerVertex;
    {
      std::vector<uint32_t> counted;
      for (uint32_t root : selectedRoots) {
        // In-time only, as in the reference vertex validation
        // (Validation/RecoVertex/src/PrimaryVertexAnalyzer4PUSlimmed.cc:877-883).
        if (!truth::Branch(&graph, root).isInTime()) {
          continue;
        }
        if (graph.particle(root).charge() == 0.) {
          continue;
        }
        if (trackerTruthPresent && hitIndex.directHits(truth::HitChannel::Tracker, root).empty()) {
          continue;
        }
        if (vertexResolution_ == VertexResolution::Interaction && !graph.particles()[root].hasGen()) {
          continue;
        }
        counted.push_back(root);
      }
      if (vertexResolution_ == VertexResolution::Interaction) {
        truth::dropCoveredMembers(graph, counted, /*keepDeepest=*/false);
      }
      for (uint32_t root : counted) {
        // The same resolution as the numerator.
        if (const auto vertexId = countingVertex(graph, root, vertexResolution_, interactionVertex)) {
          ++rootsPerVertex[*vertexId];
          if (graph.particles()[root].isSignal()) {
            ++signalRootsPerVertex[*vertexId];
          }
          const float rootPt = static_cast<float>(graph.particles()[root].momentum.pt());
          truthWeightPerVertex[*vertexId] += rootPt * rootPt;
        }
      }
    }

    // Decay vertices of heavy-flavour hadrons, which inclusiveSecondaryVertices
    // reconstructs. Each level is an antichain, so a B* that radiates down to a B gives one
    // vertex. Beauty and charm are separate levels: a B decays to a D, and one combined
    // level drops every charm vertex.
    const std::unordered_set<unsigned int> heavyFlavorDecayVertices = [&graph, heavyFlavorOnly = heavyFlavorOnly_] {
      std::unordered_set<unsigned int> vertices;
      // Only the secondary-vertex flavour of this producer reads the set.
      if (!heavyFlavorOnly)
        return vertices;
      for (const truth::Level level : {truth::Level::BHadrons, truth::Level::CHadrons}) {
        for (const uint32_t id : truth::levelAntichain(graph, level)) {
          for (const uint32_t vertexId : graph.decayVertices(id)) {
            vertices.insert(vertexId);
          }
        }
      }
      return vertices;
    }();

    auto selectedVertices = std::make_unique<std::vector<unsigned int>>();
    auto targets = std::make_unique<std::vector<unsigned int>>();
    for (auto const& [vertexId, count] : rootsPerVertex) {
      if (count < 2u) {
        continue;
      }
      // A simulated vertex beyond |z| of 1000 cm is not counted, as in the reference
      // vertex validation (Validation/RecoVertex/src/PrimaryVertexAnalyzer4PUSlimmed.cc:885-886).
      if (std::abs(graph.vertices()[vertexId].position.z()) > 1000.) {
        continue;
      }
      if (heavyFlavorOnly_ && heavyFlavorDecayVertices.count(vertexId) == 0u) {
        continue;
      }
      selectedVertices->push_back(vertexId);
      // Signal comes from the particles produced at the vertex, not from the vertex
      // eventId: a collapsed GEN vertex has eventId 0 also when its particles are pileup.
      if (!truthToRecoSignalOnly_ || signalRootsPerVertex[vertexId] > 0u) {
        targets->push_back(vertexId);
      }
    }
    std::sort(selectedVertices->begin(), selectedVertices->end());
    std::sort(targets->begin(), targets->end());
    event.put(std::move(selectedVertices), "selectedTruthVertices");
    event.put(std::move(targets), "truthToRecoTargets");
  }

  // Orders a particle before an ancestor that owns the same cells. Built once per event,
  // and only where a hit associator runs; declared first, so it outlives the associators
  // that read it.
  std::vector<uint32_t> generations;
  if constexpr (!ConstituentBasedDomain<RECO>)
    generations = truth::particleGenerations(graph);
  // Associator cache shared by all collections of this domain, keyed by mask.
  std::vector<std::pair<uint32_t, std::unique_ptr<truth::BranchHitAssociator>>> associatorPerMask;

  for (std::size_t collectionIndex = 0; collectionIndex < recoTokens_.size(); ++collectionIndex) {
    auto const& [key, token] = recoTokens_[collectionIndex];
    auto const& collection = event.get(token);
    const unsigned int nReco = collection.size();

    // Truth-driven direction, one map for all working points. Score: 1 - truth purity.
    const unsigned int nTruthRows = ConstituentBasedDomain<RECO> ? graph.nVertices() : nBranches;
    auto truthToReco = std::make_unique<MapType>(nTruthRows);

    // Hit-based domains: the hits of each object, shared by every working point.
    std::vector<std::vector<truth::RecoHit>> recoHitsPerObject;
    if constexpr (!ConstituentBasedDomain<RECO>) {
      recoHitsPerObject.resize(nReco);
      for (unsigned int i = 0; i < nReco; ++i) {
        if constexpr (LayerClusterBackedRecoHits<RECO>) {
          recoHitsPerObject[i] = truth::recoHits(collection[i], *layerClusters);
        } else {
          recoHitsPerObject[i] = truth::recoHits(collection[i]);
        }
      }
      if constexpr (Traits::metric == truth::BranchHitAssociator::Metric::SharedEnergy) {
        uint32_t seen = 0;
        for (auto const& hits : recoHitsPerObject) {
          for (auto const& hit : hits) {
            seen |= truth::BranchHitAssociator::detectorBit(hit.detId);
          }
        }
        if (seen != 0u && (seen & denominatorDetectors_) == 0u) {
          std::call_once(outOfScopeWarned_[collectionIndex], [&key] {
            edm::LogWarning("AllRecoToTruthBranchAssociatorsProducer")
                << "collection '" << key
                << "' has no hit in the configured denominatorDetectors; every shared-energy fraction will be zero";
          });
        }
      }
    }

    // Composite domains only: (reco index, shared weight) per truth vertex. The truth
    // purity is formed after every reco object of the collection contributes.
    std::unordered_map<unsigned int, std::vector<std::pair<unsigned int, float>>> sharedWeightPerTruthVertex;

    // Hit-based domains: the associator does not depend on the reco collection, so it is
    // cached per detector mask.
    truth::BranchHitAssociator const* hitAssociator = nullptr;
    if constexpr (!ConstituentBasedDomain<RECO>) {
      for (auto const& [mask, cached] : associatorPerMask) {
        if (mask == denominatorDetectors_) {
          hitAssociator = cached.get();
          break;
        }
      }
      if (hitAssociator == nullptr) {
        associatorPerMask.emplace_back(denominatorDetectors_,
                                       std::make_unique<truth::BranchHitAssociator>(hitIndex,
                                                                                    selectedRoots,
                                                                                    Traits::metric,
                                                                                    Traits::channel,
                                                                                    /*emptyRootsMeansAll=*/false,
                                                                                    denominatorDetectors_,
                                                                                    recHitEnergiesPtr,
                                                                                    generations));
        hitAssociator = associatorPerMask.back().second.get();
      }
    }

    if constexpr (ConstituentBasedDomain<RECO>) {
      for (std::size_t wpIndex = 0; wpIndex < workingPoints_.size(); ++wpIndex) {
        auto const& wp = workingPoints_[wpIndex];
        auto recoToTruth = std::make_unique<MapType>(nReco);

        // A composite object is associated to a truth vertex. Constituents from another
        // truth vertex are contamination, and the share of the leading vertex is the purity.
        auto const& constituentMap = event.get(constituentMapTokens_[collectionIndex][wpIndex]);
        for (unsigned int i = 0; i < nReco; ++i) {
          auto const& object = collection[i];
          // A separate pass, not fused into the scan below: fusion changes the rounding of
          // the float pt^2 sums and moves the scores in the last ulp.
          const float total = Traits::totalWeight(object);
          if (total <= 0.f) {
            continue;
          }
          std::unordered_map<unsigned int, float> weightPerVertex;
          Traits::forEachConstituent(object, [&](unsigned int constituentIndex, float weight) {
            if (constituentIndex >= constituentMap.size()) {
              return;
            }
            // Rows are sorted by ascending score, so [0] is the best match.
            for (auto const& match : constituentMap[constituentIndex]) {
              // A constituent gives its whole weight to one vertex, so a weak match gives
              // it to none. 1 - score is the reco purity of the constituent.
              if (match.score() > maxConstituentScore_) {
                break;
              }
              const unsigned int particle = match.index();
              if (particle < nBranches) {
                if (const auto vertexId = countingVertex(graph, particle, vertexResolution_, interactionVertex)) {
                  weightPerVertex[*vertexId] += weight;
                }
              }
              break;
            }
          });
          // The denominator is all constituents, as in CMSSW: an unmatched track lowers
          // the shared fraction.
          for (auto const& [vertexId, weight] : weightPerVertex) {
            // Reco purity: the share of the pt^2 of this reco object from this truth vertex.
            const float recoPurity = weight / total;
            recoToTruth->insert(i, vertexId, recoPurity, 1.f - recoPurity);
            // Truth purity: the shared weight over the weight of the truth vertex,
            // formed below after the whole collection.
            if (wpIndex == 0) {
              sharedWeightPerTruthVertex[vertexId].emplace_back(i, weight);
            }
          }
        }

        if (wpIndex == 0) {
          for (auto const& [vertexId, entries] : sharedWeightPerTruthVertex) {
            auto const denominatorIt = truthWeightPerVertex.find(vertexId);
            if (denominatorIt == truthWeightPerVertex.end() || denominatorIt->second <= 0.f) {
              continue;
            }
            const float denominator = denominatorIt->second;
            for (auto const& [recoIndex, weight] : entries) {
              // A reco vertex can hold a track the truth vertex did not produce, so the
              // ratio is clamped.
              const float truthPurity = std::min(1.f, weight / denominator);
              truthToReco->insert(vertexId, recoIndex, truthPurity, 1.f - truthPurity);
            }
          }
        }

        // Ascending score, so [0] is the best match. The map's own sort(true) orders by
        // descending score.
        recoToTruth->sort(byAscendingScore);
        event.put(std::move(recoToTruth), key + "RecoToTruth" + wp.name);
      }
    } else {
      // One map per working point, filled together: the candidate list of each object is
      // computed once and every working point re-ranks it.
      std::vector<std::unique_ptr<MapType>> recoToTruthPerWp;
      recoToTruthPerWp.reserve(workingPoints_.size());
      for (std::size_t wpIndex = 0; wpIndex < workingPoints_.size(); ++wpIndex) {
        recoToTruthPerWp.push_back(std::make_unique<MapType>(nReco));
      }

      // The candidates of one reco object that an adaptive point may answer with, and the
      // row of a fixed point. Reused across objects.
      std::vector<truth::BranchMatch> assignableMatches;
      std::vector<truth::BranchMatch> fixedRow;

      for (unsigned int i = 0; i < nReco; ++i) {
        if (recoHitsPerObject[i].empty()) {
          continue;
        }
        const std::span<const truth::RecoHit> span(recoHitsPerObject[i]);
        const auto matches = hitAssociator->bestBranches(span);

        // A parton, a beam particle or an invented node covers the reco object entirely
        // through its subgraph, and wins on score. An adaptive point may not answer with
        // one. A fixed point keeps these barred roots after every assignable root, so its
        // row [0] is a detector particle when one matches.
        assignableMatches.clear();
        for (auto const& match : matches) {
          if (match.rootParticleId < isAssignable.size() && isAssignable[match.rootParticleId] != 0) {
            assignableMatches.push_back(match);
          }
        }
        fixedRow = assignableMatches;
        for (auto const& match : matches) {
          if (match.rootParticleId >= isAssignable.size() || isAssignable[match.rootParticleId] == 0) {
            fixedRow.push_back(match);
          }
        }
        // The filter keeps the ascending-score order of bestBranches, so the climb starts
        // from the best candidate.
        const std::span<const truth::BranchMatch> assignableSpan(assignableMatches);

        // Reco to truth: the working point drives the search. 1 - score is the reco purity.
        for (std::size_t wpIndex = 0; wpIndex < workingPoints_.size(); ++wpIndex) {
          auto const& wp = workingPoints_[wpIndex];
          if (wp.adaptive) {
            const auto match =
                truth::BranchHitAssociator::bestAdaptiveBranch(assignableSpan, wp.reverseWeight, wp.maxReverseScore);
            if (match.rootParticleId != truth::BranchMatch::kInvalidRoot) {
              recoToTruthPerWp[wpIndex]->insert(i, match.rootParticleId, match.sharedEnergy, match.score);
            }
          } else {
            for (auto const& match : fixedRow) {
              recoToTruthPerWp[wpIndex]->insert(i, match.rootParticleId, match.sharedEnergy, match.score);
            }
          }
        }

        // Truth to reco, with no adaptive climb. Both payloads are truth-normalised: the
        // shared energy fraction and the reverse score. A shared-hits domain reports the
        // shared hit count.
        constexpr bool sharedEnergyMetric = Traits::metric == truth::BranchHitAssociator::Metric::SharedEnergy;
        for (auto const& match : matches) {
          // The row index comes from the hit index and the map is sized from the graph.
          // A candidate outside the graph means the two products were built from
          // different events, which is a configuration error.
          if (match.rootParticleId >= nTruthRows) {
            throw cms::Exception("Configuration")
                << "the hit index and the truth graph disagree on the particle count for '" << key
                << "'. Both products must come from the same event.";
          }
          const float truthValue = sharedEnergyMetric ? match.sharedEnergyFraction : match.sharedEnergy;
          truthToReco->insert(match.rootParticleId, i, truthValue, match.reverseScore);
        }
      }

      for (std::size_t wpIndex = 0; wpIndex < workingPoints_.size(); ++wpIndex) {
        // The rows keep the fill order: assignable roots first, each group by ascending
        // score, so [0] is the best detector particle. Do not sort them here.
        event.put(std::move(recoToTruthPerWp[wpIndex]), key + "RecoToTruth" + workingPoints_[wpIndex].name);
      }
    }
    truthToReco->sort(byAscendingScore);
    event.put(std::move(truthToReco), key + "TruthToReco");
  }
}

template <typename RECO>
  requires(AdaptableToTruthHits<RECO> || ConstituentBasedDomain<RECO>)
void AllRecoToTruthBranchAssociatorsProducer<RECO>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<std::vector<edm::InputTag>>("recoCollections", {});
  desc.add<edm::InputTag>("assignableTargetsSrc", edm::InputTag("truthBranchTargets", "assignableRoots"))
      ->setComment(
          "The roots an adaptive working point may answer with. The barred ones stay in targetsSrc and in the "
          "first working point's map, because they are the members of the hard-process and parton-jet "
          "denominators and the truth-driven direction reads its pair scores from that map");
  desc.add<edm::InputTag>("targetsSrc", edm::InputTag("truthBranchTargets", "selectedRoots"))
      ->setComment(
          "The selector-passing candidate roots from TruthBranchTargetsProducer, which also emits the level "
          "denominators and the signal seeds every associator module shares");
  desc.add<std::vector<std::string>>("denominatorDetectors", {})
      ->setComment(
          "DetId::Detector names the sim-normalised shared-energy denominator covers. Empty means the whole hit "
          "channel. One channel spans several detectors: HitChannel::Calo carries the barrel ECAL and HCAL "
          "deposits next to the HGCAL ones, and PCaloHit energies are sampling energies, so a branch that "
          "showered in the barrel has a channel-wide energy no endcap trackster can cover half of. Measured on "
          "200 no-PU ttbar events: 0.5% to 10% of a top branch's channel energy is in HGCAL, so the fraction was "
          "zero for every top");
  desc.add<bool>("truthToRecoSignalOnly", true)
      ->setComment(
          "Composite domains only: restrict the TruthToReco vertex targets to the ones that produced a signal "
          "particle. Efficiency, duplicate and split are meaningless averaged over the overlaid pileup "
          "interactions. Hit-based domains take their denominators from the targets producer. A pileup particle "
          "is matchable when it passes the candidate selection of that producer, which has a pt and an eta cut, "
          "so a reco object from a softer pileup particle finds no truth");
  desc.add<double>("maxConstituentScore", 0.25)
      ->setComment(
          "Composite domains only. A constituent whose best match scores worse than this places the constituent "
          "at no truth vertex, because it would otherwise donate its whole pt^2 to a vertex it barely belongs to. "
          "1 - score is the constituent's reco purity, so the default is the 0.75 shared fraction the tracker "
          "association applies as Cut_RecoToSim, above which a reco-to-sim pair enters its map at all "
          "(SimTracker/TrackAssociatorProducers/python/quickTrackAssociatorByHits_cfi.py). The denominator stays "
          "over ALL constituents, as the reference vertex association does: an unmatched track legitimately "
          "lowers the shared fraction");
  desc.add<bool>("heavyFlavorOnly", false)
      ->setComment(
          "Composite domains only. Keep in the denominator only the vertices where a b or c hadron DECAYED, which "
          "is what inclusiveSecondaryVertices reconstructs: 4 and 5 per no-PU ttbar event against 4.1 "
          "reconstructed. Off by default; the secondary-vertex associator turns it on. Without it the denominator "
          "is every graph vertex with two selected roots, 45.9 per event, and the efficiency is capped by the "
          "denominator.");
  desc.add<std::vector<std::string>>("workingPointNames", {"Fixed"})
      ->setComment(
          "One RecoToTruth product per name. The name \"Fixed\" selects the plain per-root match; any other name "
          "makes the point adaptive, driven by its adaptiveReverseWeight and adaptiveMaxReverseScore entries. The "
          "FIRST listed point is the reference: the single TruthToReco product is computed at it, so list Fixed "
          "first");
  desc.add<std::vector<float>>("adaptiveReverseWeight", {0.f});
  desc.add<std::vector<float>>("adaptiveMaxReverseScore", {0.f});
  if constexpr (LayerClusterBackedRecoHits<RECO>) {
    desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
    desc.add<std::vector<edm::InputTag>>("hgcalRecHits",
                                         {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEBRecHits")})
        ->setComment(
            "HGCAL rechits whose energies weight the cells of the shared-energy metric. Every listed collection is "
            "required. With this list and pfRecHits both empty the cells are weighted by their sim energy");
    desc.add<std::vector<edm::InputTag>>(
            "pfRecHits", {edm::InputTag("particleFlowRecHitECAL"), edm::InputTag("particleFlowRecHitHBHE")})
        ->setComment(
            "Barrel PFRecHits whose energies weight the cells of the shared-energy metric; the collections the "
            "barrel layer clusters are built from, not the Cleaned instances. Every listed collection is required");
  }
  if constexpr (ConstituentBasedDomain<RECO>) {
    desc.add<std::string>("constituentAssociator", "allTrackToTruthBranchAssociators")
        ->setComment(
            "Module that produced the constituents' association maps. Its map rows must be sorted ascending by "
            "score, best first, as this package's producers emit them; the constituent scan reads row [0] as the "
            "best match and stops at the first row above maxConstituentScore");
    desc.add<std::string>("constituentCollection", "generalTracks")
        ->setComment("Constituent collection key, used to rebuild the instance labels");
    desc.add<std::string>("vertexResolution", "immediate")
        ->setComment(
            "Which truth vertex a constituent counts at: 'immediate' is the production vertex of its matched "
            "particle, right for a secondary vertex; 'interaction' is the production vertex of that particle's "
            "topmost ancestor, right for a primary vertex, where a track from a downstream decay still belongs "
            "to the interaction the chain started from");
  }
  descriptions.add(Traits::cfiName, desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using AllTrackToTruthBranchAssociatorsProducer = AllRecoToTruthBranchAssociatorsProducer<reco::Track>;
DEFINE_FWK_MODULE(AllTrackToTruthBranchAssociatorsProducer);
using AllVertexToTruthBranchAssociatorsProducer = AllRecoToTruthBranchAssociatorsProducer<reco::Vertex>;
DEFINE_FWK_MODULE(AllVertexToTruthBranchAssociatorsProducer);
// Not named AllTracksterToTruthBranchAssociatorsProducer: the NanoAOD training branch has
// a different producer of that name, and a duplicate plugin name breaks an area that has
// both.
using TruthBranchTracksterAssociatorsProducer = AllRecoToTruthBranchAssociatorsProducer<ticl::Trackster>;
DEFINE_FWK_MODULE(TruthBranchTracksterAssociatorsProducer);
using TruthBranchPFClusterAssociatorsProducer = AllRecoToTruthBranchAssociatorsProducer<reco::PFCluster>;
DEFINE_FWK_MODULE(TruthBranchPFClusterAssociatorsProducer);
