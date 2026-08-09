// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

// One producer, every configured reco collection of one type, every branch-association
// working point. Follows the All* pattern of
// SimCalorimetry/HGCalAssociatorProducers: the module takes a VInputTag of reco
// collections and emits one pair of association maps per (collection, working point),
// with instance labels derived from the input tags.
//
// The reco type only has to be adaptable to (DetId, fraction) hits. Which adapter
// applies is decided by a concept rather than by a per-type producer, so a new domain
// is a truth::recoHits overload plus a label in truthGraphAssociationLabels_cff, not a
// new plugin.
//
// Working points differ only in the arguments passed to bestAdaptiveBranch, not in the
// associator itself, so the inverted DetId index is built ONCE per event and reused
// across every working point.

#include <algorithm>
#include <cctype>
#include <concepts>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

#include "PhysicsTools/TruthInfo/interface/Branch.h"
#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"
#include "PhysicsTools/TruthInfo/interface/BranchSelector.h"
#include "PhysicsTools/TruthInfo/interface/RecoHitAdapters.h"
#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace {
  // Lower score is better everywhere in this file. The comparator is shared by every
  // map sort so the [0]-is-best contract cannot drift between products.
  constexpr auto byAscendingScore = [](const auto& a, const auto& b) {
    if (a.score() != b.score())
      return a.score() < b.score();
    return a.index() < b.index();
  };

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

  // How a domain reaches the truth is a property of its reco type, not of runtime
  // configuration. Two strategies cover everything:
  //
  //   HitBased         the object owns detector hits, so it is matched directly
  //                    (tracks by shared hits, tracksters by shared energy).
  //   ConstituentBased the object is BUILT from objects that are already associated,
  //                    so its truth is aggregated from theirs rather than recomputed
  //                    from hits. A vertex shares tracks, a jet shares constituents,
  //                    a candidate shares a track and clusters. This is the layering
  //                    CMSSW already uses: VertexAssociatorByPositionAndTracks
  //                    consumes the track maps, it does not revisit hits.
  //
  // Binding payload and strategy to the type means the declared product type and the
  // produced one cannot drift apart, which they did when the metric was a config string.
  enum class AssociationStrategy { HitBased, ConstituentBased };

  template <typename RECO>
  struct TruthAssociationTraits;

  template <>
  struct TruthAssociationTraits<reco::Track> {
    static constexpr auto strategy = AssociationStrategy::HitBased;
    using MapType = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>;
    static constexpr truth::HitChannel channel = truth::HitChannel::Tracker;
    static constexpr auto metric = truth::BranchHitAssociator::Metric::SharedHits;
    static constexpr const char* cfiName = "allTrackToTruthBranchAssociators";
  };

  // A vertex carries no hits of its own: its truth is whatever its tracks point to.
  // The payload is therefore a FRACTION of the vertex's tracks, weighted the way
  // calculateVertexSharedTracks weights them, not an energy.
  template <>
  struct TruthAssociationTraits<reco::Vertex> {
    static constexpr auto strategy = AssociationStrategy::ConstituentBased;
    using ConstituentType = reco::Track;
    using MapType = ticl::AssociationMap<ticl::mapWithFractionAndScore>;
    static constexpr const char* cfiName = "allVertexToTruthBranchAssociators";

    // Visit (constituent index into its own collection, weight). The index is the Ref
    // key, which is exactly the row the constituent's association map is indexed by.
    //
    // The weight is pt SQUARED, which is what CMSSW's own vertex association uses:
    // calculateVertexSharedTracks returns sharedPt2Fraction as
    // sum(pt^2 of shared tracks) / sum(pt^2 of ALL the vertex's tracks)
    // (SimTracker/VertexAssociation/src/calculateVertexSharedTracks.cc). The vertex FIT
    // weight answers a different question: it says how strongly a track constrained the
    // fit, not how much of the vertex's momentum it carries, and it gives a soft pileup
    // track the same standing as a hard signal one.
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

  // A trackster owns calorimeter energy through its layer clusters, so it is matched
  // directly like a track, but on SHARED ENERGY in the calorimeter channel rather than
  // on a hit count in the tracker. This is the same metric the TICL trackster
  // validation scores against, so the two are comparable.
  template <>
  struct TruthAssociationTraits<ticl::Trackster> {
    static constexpr auto strategy = AssociationStrategy::HitBased;
    using MapType = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>;
    static constexpr truth::HitChannel channel = truth::HitChannel::Calo;
    static constexpr auto metric = truth::BranchHitAssociator::Metric::SharedEnergy;
    static constexpr const char* cfiName = "truthBranchTracksterAssociators";
  };

  // Which truth vertex a constituent should be counted at.
  //
  //   Immediate    the production vertex of the matched particle itself. Right for a
  //                secondary vertex, which IS a decay or interaction vertex: the tracks
  //                that belong to it were produced there.
  //   Interaction  the one vertex representing the interaction the particle belongs to,
  //                so a track from a decay downstream of the vertex is counted at the
  //                vertex the chain started from. Right for a primary vertex, where the
  //                question is which interaction a track came from, not which decay.
  enum class VertexResolution { Immediate, Interaction };

  // One representative vertex per interaction, for Interaction resolution.
  //
  // No VertexRole::Interaction node is materialised unless a selection preset builds one:
  // measured on ttbar, all 534 vertices of an event are Normal. eventId IS the
  // interaction instead, 0 being the signal and anything else an overlaid pileup
  // interaction, so every particle of one interaction must count at a single vertex.
  //
  // That vertex is the lowest-numbered production vertex of the interaction that is
  // usable, ids being handed out in build order so the lowest is where the interaction
  // started. Usable excludes a vertex that neither merged with a SimVertex nor carries a
  // position: a pileup sub-event built with collapsePileupGen has a single synthetic GEN
  // vertex, and when its GenToSim links are all dropped it never merges and keeps a
  // default position, so electing it would count that whole interaction at the origin,
  // where any reco vertex near the beamspot absorbs it.
  //
  // Position alone does not identify the right vertex: after VtxSmeared every shower and
  // hadronisation vertex of a Pythia record sits at the same smeared point, so "it came
  // out at the beamspot" would be true of almost any choice. The build order is what
  // picks it; the usability test only rejects the placeholder.
  //
  // The placeholder this rejects is a default-constructed position, an in-band value: a
  // genuine unsmeared vertex at the exact origin is indistinguishable and gets demoted
  // too. Harmless for the association itself, since on such a sample every candidate
  // shares the position anyway, but the elected id can differ from the plain build-order
  // choice there. Time is part of the test so a real origin vertex with nonzero time is
  // kept.
  [[nodiscard]] inline bool usableAsInteractionVertex(truth::VertexData const& vertex) {
    if (vertex.hasSim()) {
      return true;
    }
    auto const& position = vertex.position;
    return position.x() != 0. || position.y() != 0. || position.z() != 0. || position.t() != 0.;
  }

  [[nodiscard]] inline std::unordered_map<uint64_t, uint32_t> interactionVertices(truth::Graph const& graph) {
    std::unordered_map<uint64_t, uint32_t> representative;

    // An interaction the graph actually models gets a VertexRole::Interaction node, built
    // by the selection preset, and THAT is the primary vertex: it is the interaction
    // point, not a vertex elected to stand for it. Only these enter the primary-vertex
    // plots, so what is drawn is the interaction rather than whichever production vertex
    // happened to be built first and whichever position that carries.
    for (uint32_t v = 0; v < graph.nVertices(); ++v) {
      auto const& data = graph.vertices()[v];
      if (data.vertexRole() == truth::VertexRole::Interaction) {
        representative.emplace(data.eventId, v);
      }
    }
    if (!representative.empty()) {
      return representative;
    }

    // No preset ran, so no interaction node exists and there is nothing to plot but an
    // elected stand-in. Measured on ttbar without a preset, all 534 vertices are Normal.
    // The election below is kept for that case, and it is the reason a primary-vertex
    // position is only as good as the preset: with one, the node is the interaction.
    std::unordered_map<uint64_t, uint32_t> placeholderOnly;
    const uint32_t nParticles = graph.nParticles();
    for (uint32_t id = 0; id < nParticles; ++id) {
      const auto production = truth::Particle(&graph, id).productionVertices();
      if (production.empty()) {
        continue;
      }
      const uint32_t vertexId = production.front().id();
      const uint64_t eventId = graph.particles()[id].eventId;
      auto& target = usableAsInteractionVertex(graph.vertices()[vertexId]) ? representative : placeholderOnly;
      auto [it, inserted] = target.emplace(eventId, vertexId);
      if (!inserted) {
        it->second = std::min(it->second, vertexId);
      }
    }

    // An interaction with nothing but placeholders still has to resolve, or every
    // composite object built from its constituents silently matches nothing. Take the
    // placeholder and say that its position is not to be trusted.
    for (auto const& [eventId, vertexId] : placeholderOnly) {
      if (representative.emplace(eventId, vertexId).second) {
        edm::LogWarning("AllRecoToTruthBranchAssociators")
            << "interaction " << eventId << " resolves only to logical vertex " << vertexId
            << ", which did not merge with a SimVertex and whose position is "
               "indistinguishable from a default-constructed one. Its constituents are "
               "counted there, so any vertex efficiency or purity for that interaction "
               "is positional nonsense. This is what a pileup sub-event looks like when "
               "all of its GenToSim links were dropped.";
      }
    }
    return representative;
  }

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

  std::vector<std::pair<std::string, edm::EDGetTokenT<std::vector<RECO>>>> recoTokens_;
  // One warning per collection per job when its input is absent: a silently empty map
  // is indistinguishable from a perfectly working associator on a bad label, and the
  // downstream symptom is a fully booked, zero-entry DQM folder.
  mutable std::vector<std::once_flag> missingWarned_;
  std::vector<WorkingPoint> workingPoints_;
  truth::BranchSelector branchSelector_;
  const bool truthToRecoSignalOnly_;
  const bool heavyFlavorOnly_;
  // The selection preset's seed species. signalSeeds is the subset of the selected
  // roots with one of these pdgIds, so the _signal efficiency denominator is the
  // preset's signal object itself and not every selected root; signalSeedsNoSelection
  // is the same species with no selector cut at all. Empty means no preset ran and both
  // fall back to all selected roots.
  std::vector<int> signalSeedPdgIds_;
  std::vector<int> signalSeedHadronFlavors_;
  // Which levels of the graph the truth-driven direction asks about, each with its
  // product instance label. NOT working points: the working points are the reco-driven
  // adaptive search, and the truth targets must be fixed before any reco object is
  // looked at. One denominator product per level.
  std::vector<std::pair<truth::Level, std::string>> truthLevels_;

  using Traits = TruthAssociationTraits<RECO>;
  using MapType = typename Traits::MapType;

  // Composite domains read their constituents' association maps instead of hits. The
  // upstream module is named by a cms.string and the instance labels are rebuilt here,
  // the same way the HGCal All* producers reach allHitToTracksterAssociations.
  using ConstituentMapType = typename std::conditional_t<ConstituentBasedDomain<RECO>,
                                                         TruthAssociationTraits<reco::Track>,
                                                         TruthAssociationTraits<reco::Track>>::MapType;
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
  }

  {
    // Restrict which branches are candidates at all. Without this the maps and the
    // efficiency denominators are dominated by soft particles that no reconstruction
    // was ever going to find, exactly as CaloParticleSelector and the TrackingParticle
    // selectors guard their own denominators.
    auto const& sel = cfg.getParameter<edm::ParameterSet>("branchSelector");
    truth::BranchSelector::Config selectorConfig;
    selectorConfig.ptMin = sel.getParameter<float>("ptMin");
    selectorConfig.ptMax = sel.getParameter<float>("ptMax");
    selectorConfig.etaMin = sel.getParameter<float>("etaMin");
    selectorConfig.etaMax = sel.getParameter<float>("etaMax");
    selectorConfig.pdgIds = sel.getParameter<std::vector<int>>("pdgIds");
    selectorConfig.signalOnly = sel.getParameter<bool>("signalOnly");
    selectorConfig.intimeOnly = sel.getParameter<bool>("intimeOnly");
    selectorConfig.chargedOnly = sel.getParameter<bool>("chargedOnly");
    selectorConfig.invertEta = sel.getParameter<bool>("invertEta");
    selectorConfig.kinematicsOnStableOnly = sel.getParameter<bool>("kinematicsOnStableOnly");
    branchSelector_ = truth::BranchSelector(std::move(selectorConfig));
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
    workingPoints_.push_back(
        {names[i], static_cast<float>(weights[i]), static_cast<float>(ceilings[i]), names[i] != "Fixed"});
  }

  // The selected candidate roots are published so a consumer can use exactly the same
  // set as the efficiency DENOMINATOR. Without it a validator counts every particle in
  // the graph, including those the selector rejected, and every efficiency comes out
  // low by the rejection factor.
  // The preset seed objects among them, the denominator of the signal efficiency.
  signalSeedPdgIds_ = cfg.getParameter<std::vector<int>>("signalSeedPdgIds");
  signalSeedHadronFlavors_ = cfg.getParameter<std::vector<int>>("signalSeedHadronFlavors");
  produces<std::vector<unsigned int>>("signalSeeds");
  // The same seed species without any selector cut, so an efficiency can be quoted
  // against EVERY seed in the event and not only against those the kinematic selection
  // kept. The two denominators together separate "not reconstructed" from "never
  // offered": on 200 no-PU ttbar events the selector keeps 390 of the 400 tops.
  produces<std::vector<unsigned int>>("signalSeedsNoSelection");
  // The TruthToReco denominators, which are NOT the same set as the associator's
  // candidates. Efficiency, duplicate rate and split rate ask what fraction of the
  // truth was reconstructed, and in a pileup sample that question is only meaningful
  // for the signal interaction: averaging it over 200 overlaid ones measures how well
  // pileup is reconstructed. This is why MTV puts signalOnly on the TrackingParticle
  // selector that guards its efficiency denominator.
  //
  // The candidate set stays complete on purpose. RecoToTruth metrics, fake rate above
  // all, need pileup branches to remain matchable: a reco object built from a pileup
  // particle is not a fake, and it would become one if the candidates were signal-only.
  if constexpr (!ConstituentBasedDomain<RECO>) {
    // One denominator product per configured level, labelled
    // "truthToRecoTargets" + the level name with its first letter capitalized.
    for (auto const& name : cfg.getParameter<std::vector<std::string>>("truthLevels")) {
      if (name.empty()) {
        throw cms::Exception("Configuration") << "empty entry in truthLevels";
      }
      std::string capitalized = name;
      capitalized[0] = std::toupper(static_cast<unsigned char>(capitalized[0]));
      truthLevels_.emplace_back(truth::levelFromName(name), "truthToRecoTargets" + capitalized);
      produces<std::vector<unsigned int>>(truthLevels_.back().second);
      // Parallel to the denominator: which plotted-axis cut each target fails.
      produces<std::vector<unsigned int>>(truthLevels_.back().second + "Eligibility");
    }
  }
  if constexpr (ConstituentBasedDomain<RECO>) {
    // A composite object's truth target is a vertex, not a branch at some level, so
    // there is a single denominator.
    produces<std::vector<unsigned int>>("truthToRecoTargets");
    // A composite object is associated to a truth VERTEX, so its efficiency denominator
    // is a set of vertices, not of branch roots.
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
    // Key rule for this package: label and instance joined by an underscore, the same
    // string that names the DQM folder. HGCal concatenates for products and
    // underscores for folders; keeping one rule avoids that asymmetry.
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

    // The two directions are NOT transposes of each other and are deliberately not
    // named as if they were.
    //
    // RecoToTruth is reco-driven: given a reco object, the adaptive search picks the
    // graph level that best matches it, so there is one product per working point. Its
    // score is 1 - RECO purity, the reco object being the denominator.
    //
    // TruthToReco is truth-driven: the truth target is fixed A PRIORI by the domain's
    // resolution, so the working point does not enter and there is ONE product. Its
    // score is 1 - TRUTH purity, the truth object being the denominator.
    for (auto const& wp : workingPoints_) {
      produces<MapType>(key + "RecoToTruth" + wp.name);
    }
    produces<MapType>(key + "TruthToReco");
  }
  missingWarned_ = std::vector<std::once_flag>(recoTokens_.size());
}

template <typename RECO>
  requires(AdaptableToTruthHits<RECO> || ConstituentBasedDomain<RECO>)
void AllRecoToTruthBranchAssociatorsProducer<RECO>::produce(edm::StreamID,
                                                            edm::Event& event,
                                                            edm::EventSetup const&) const {
  auto const& graph = event.get(graphToken_);
  auto const& hitIndex = event.get(hitIndexToken_);

  std::vector<reco::CaloCluster> const* layerClusters = nullptr;
  if constexpr (LayerClusterBackedRecoHits<RECO>) {
    // Tolerant like the reco collections below: the HLT twin runs in jobs whose input
    // may carry no HLT reconstruction at all, and then it must produce empty maps
    // rather than throw. Trackster collections cannot be adapted without the clusters,
    // so they are skipped when the clusters are absent.
    const edm::Handle<std::vector<reco::CaloCluster>> handle = event.getHandle(layerClustersToken_);
    if (handle.isValid()) {
      layerClusters = &(*handle);
    } else {
      edm::LogWarning("AllRecoToTruthBranchAssociatorsProducer")
          << "layer clusters absent; trackster collections will produce empty maps this event";
    }
  }

  const unsigned int nBranches = graph.nParticles();

  // Selected candidate roots, computed once. emptyRootsMeansAll is false on purpose:
  // if the selection accepts nothing the answer is "no candidates", not "every
  // particle", which would silently undo the selection.
  std::vector<uint32_t> selectedRoots;
  selectedRoots.reserve(nBranches);
  for (uint32_t id = 0; id < nBranches; ++id) {
    if (branchSelector_(truth::Branch(&graph, id))) {
      selectedRoots.push_back(id);
    }
  }

  // selectedRoots is NOT emitted. It is every particle passing the selector, so it can
  // hold a particle together with its own ancestor and any efficiency over it counts the
  // same object twice: 1.3% of its members have an ancestor in the same set. It stays a
  // local, because the signal seeds below are the subset of it carrying a seed species.

  // The preset seed objects: with a tau preset the tau roots alone, so the signal
  // efficiency is the tau's own, not its decay legs'. No preset means every selected
  // root is the signal.
  {
    auto const isSeedSpecies = [this, &graph](uint32_t id) {
      const int32_t pdgId = graph.particles()[id].pdgId;
      if (std::find(signalSeedPdgIds_.begin(), signalSeedPdgIds_.end(), pdgId) != signalSeedPdgIds_.end()) {
        return true;
      }
      for (const int flavor : signalSeedHadronFlavors_) {
        if (truth::hadronHasQuark(pdgId, flavor)) {
          return true;
        }
      }
      return false;
    };
    auto signalSeeds = std::make_unique<std::vector<unsigned int>>();
    // Same species, every one of them: the selector's kinematic cuts are what this
    // denominator exists to leave out. Without a preset there is no seed species to
    // look for, so there is nothing to un-select and it repeats signalSeeds rather
    // than promoting every particle in the graph to an efficiency denominator.
    auto signalSeedsNoSelection = std::make_unique<std::vector<unsigned int>>();
    // With no seed species there is no resonance in this sample, so BOTH products stay
    // EMPTY. Every selected root is not a substitute: that set holds particles together
    // with their own ancestors, so it is not an antichain and an efficiency over it
    // counts the same energy twice (on QCD it is 518.89 per event against 164
    // generator-stable particles).
    if (truth::seedsNameAResonance(signalSeedPdgIds_, signalSeedHadronFlavors_)) {
      for (uint32_t id : selectedRoots) {
        if (isSeedSpecies(id)) {
          signalSeeds->push_back(id);
        }
      }
      for (uint32_t id = 0; id < nBranches; ++id) {
        if (isSeedSpecies(id)) {
          signalSeedsNoSelection->push_back(id);
        }
      }
    }
    event.put(std::move(signalSeeds), "signalSeeds");
    event.put(std::move(signalSeedsNoSelection), "signalSeedsNoSelection");
  }

  // eventId 0 is the signal interaction; anything else is overlaid pileup.
  auto isSignalParticle = [&graph](uint32_t particleId) { return graph.particles()[particleId].eventId == 0; };
  if constexpr (!ConstituentBasedDomain<RECO>) {
    // One denominator per level. The level antichain, then the signal restriction, then
    // the kinematic selector. Order matters: taking the antichain of an already
    // kinematically-selected set would promote a soft particle to a level it does not
    // belong to just because its parent failed the pt cut.
    for (auto const& [level, instance] : truthLevels_) {
      auto targets = std::make_unique<std::vector<unsigned int>>();
      // Parallel to targets: which plotted-axis cut each one FAILS, 0 for those passing
      // both. An efficiency against pt must not have the pt cut applied to its own
      // denominator, so a target failing only the pt cut is kept and enters the pt plot
      // alone. Measured on no-PU ttbar, the caloBoundary denominator in the first pt bin
      // is 10024 with the cut and 144529 without, a factor 14.4; the second bin moves by
      // 1.05. The cut was deforming the turn-on it exists to show.
      auto eligibility = std::make_unique<std::vector<unsigned int>>();
      for (uint32_t id : truth::levelAntichain(graph, level)) {
        if (truthToRecoSignalOnly_ && !isSignalParticle(id)) {
          continue;
        }
        const truth::Branch branch(&graph, id);
        if (!branchSelector_.passesNonKinematic(branch)) {
          continue;
        }
        // No plot can suppress two cuts at once, so a branch failing more than one enters
        // none of them and is dropped here rather than carried and filtered everywhere.
        const uint32_t failed = branchSelector_.failedKinematicCuts(branch);
        if ((failed & (failed - 1u)) != 0u) {
          continue;
        }
        targets->push_back(id);
        eligibility->push_back(failed);
      }
      event.put(std::move(targets), instance);
      event.put(std::move(eligibility), instance + "Eligibility");
    }
  }

  [[maybe_unused]] const auto interactionVertex =
      ConstituentBasedDomain<RECO> && vertexResolution_ == VertexResolution::Interaction
          ? interactionVertices(graph)
          : std::unordered_map<uint64_t, uint32_t>{};

  if constexpr (ConstituentBasedDomain<RECO>) {
    // The vertices a composite object could have been reconstructed at: those where at
    // least two selected branch roots were produced. One track cannot make a vertex, so
    // a one-particle vertex in the denominator is a guaranteed miss that scales every
    // efficiency down without measuring anything, the same reason the branch selector
    // guards the particle denominator.
    std::unordered_map<unsigned int, unsigned int> rootsPerVertex;
    std::unordered_map<unsigned int, unsigned int> signalRootsPerVertex;
    for (uint32_t root : selectedRoots) {
      // In-time only, as the reference vertex validation counts only bunch-crossing-0
      // simulated vertices in its denominator
      // (Validation/RecoVertex/src/PrimaryVertexAnalyzer4PUSlimmed.cc:877-883).
      if (!truth::Branch(&graph, root).isInTime()) {
        continue;
      }
      // Same resolution the numerator uses: counting the denominator at a different set
      // of vertices than the numerator is the denominator bug all over again.
      if (const auto vertexId = countingVertex(graph, root, vertexResolution_, interactionVertex)) {
        ++rootsPerVertex[*vertexId];
        if (isSignalParticle(root)) {
          ++signalRootsPerVertex[*vertexId];
        }
      }
    }
    // Restrict to what the collection is actually for. inclusiveSecondaryVertices
    // reconstructs DISPLACED HEAVY-FLAVOUR vertices, about 4 per ttbar event, while every
    // graph vertex with two selected roots sweeps in every nuclear interaction, conversion
    // and decay in flight: 45.9 per event, an 11x excess that caps the efficiency near 9%
    // however good the reconstruction is. The graph answers the question directly.
    // WHERE THE HEAVY-FLAVOUR HADRON DECAYED, which is what a secondary vertex is. Asking
    // instead whether the incoming particle's subgraph contains a b or c hadron anywhere
    // is true at every vertex along the chain above and below it: measured on no-PU ttbar
    // it selects 12 and 16 vertices per event against the 4 and 5 the hadrons actually
    // decay at, and 4.1 reconstructed, so the denominator is inflated 3x and caps the
    // efficiency near a third however good the reconstruction is.
    //
    // The levels are antichains, so a B* radiating down to a B contributes ONE vertex
    // rather than one per generator copy. Beauty and charm are asked separately because a
    // B decays to a D and a combined level would drop every charm vertex.
    const std::unordered_set<unsigned int> heavyFlavorDecayVertices = [&graph] {
      std::unordered_set<unsigned int> vertices;
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
      // Junk-vertex guard of the reference vertex validation: a simulated vertex
      // beyond |z| of 1000 cm is not counted
      // (Validation/RecoVertex/src/PrimaryVertexAnalyzer4PUSlimmed.cc:885-886).
      if (std::abs(graph.vertices()[vertexId].position.z()) > 1000.) {
        continue;
      }
      if (heavyFlavorOnly_ && heavyFlavorDecayVertices.count(vertexId) == 0u) {
        continue;
      }
      selectedVertices->push_back(vertexId);
      // Signal is decided from the PARTICLES produced there, not from the vertex's own
      // eventId: a collapsed GEN vertex carries 0 even when everything it produced
      // belongs to a pileup interaction.
      if (!truthToRecoSignalOnly_ || signalRootsPerVertex[vertexId] > 0u) {
        targets->push_back(vertexId);
      }
    }
    std::sort(selectedVertices->begin(), selectedVertices->end());
    std::sort(targets->begin(), targets->end());
    event.put(std::move(selectedVertices), "selectedTruthVertices");
    event.put(std::move(targets), "truthToRecoTargets");
  }

  for (std::size_t collectionIndex = 0; collectionIndex < recoTokens_.size(); ++collectionIndex) {
    auto const& [key, token] = recoTokens_[collectionIndex];
    edm::Handle<std::vector<RECO>> handle;
    event.getByToken(token, handle);
    // A trackster collection without its layer clusters cannot be adapted to hits, so
    // it is treated exactly like an absent collection: valid empty maps.
    bool valid = handle.isValid();
    if constexpr (LayerClusterBackedRecoHits<RECO>) {
      valid = valid && layerClusters != nullptr;
    }
    const unsigned int nReco = valid ? handle->size() : 0u;
    if (!valid) {
      std::call_once(missingWarned_[collectionIndex], [&key] {
        edm::LogWarning("AllRecoToTruthBranchAssociatorsProducer")
            << "input collection '" << key << "' absent; its association maps will be empty for this job";
      });
    }

    // Truth-driven direction, built ONCE: the truth target is fixed a priori, so the
    // reco-driven working point plays no part in it. Its score is 1 - truth purity.
    const unsigned int nTruthRows = ConstituentBasedDomain<RECO> ? graph.nVertices() : nBranches;
    auto truthToReco = std::make_unique<MapType>(nTruthRows);

    // The detectors this collection reconstructs, which is what the sim-normalised
    // shared-energy fraction is normalised to. One hit channel spans several
    // detectors: HitChannel::Calo carries the barrel ECAL and HCAL deposits next to
    // the HGCAL ones, and PCaloHit energies are sampling energies, so a branch that
    // showered in the barrel has a channel-wide energy no endcap trackster can cover
    // half of. Measured on 200 no-PU ttbar events: 0.5% to 10% of a top branch's
    // channel energy is in HGCAL, so the fraction was zero for every top.
    // Hit-based domains: each object's hit adaptation is independent of the working
    // point, so it is built ONCE per collection and shared by the detector scan and
    // every working point below.
    std::vector<std::vector<truth::RecoHit>> recoHitsPerObject;
    if constexpr (!ConstituentBasedDomain<RECO>) {
      recoHitsPerObject.resize(nReco);
      for (unsigned int i = 0; i < nReco; ++i) {
        if constexpr (LayerClusterBackedRecoHits<RECO>) {
          recoHitsPerObject[i] = truth::recoHits((*handle)[i], *layerClusters);
        } else {
          recoHitsPerObject[i] = truth::recoHits((*handle)[i]);
        }
      }
    }

    uint32_t denominatorDetectors = truth::BranchHitAssociator::kAllDetectors;
    if constexpr (!ConstituentBasedDomain<RECO>) {
      if constexpr (Traits::metric == truth::BranchHitAssociator::Metric::SharedEnergy) {
        uint32_t seen = 0;
        for (auto const& hits : recoHitsPerObject) {
          for (auto const& hit : hits) {
            seen |= truth::BranchHitAssociator::detectorBit(hit.detId);
          }
        }
        // An empty collection produces no match at all, so which detectors its
        // denominator would have covered never enters a number.
        if (seen != 0u) {
          denominatorDetectors = seen;
        }
      }
    }

    // Composite domains only: (reco index, shared weight) per truth vertex and the
    // per-truth-vertex total, so the truth-normalised fraction can be formed once every
    // reco object of the collection has contributed.
    std::unordered_map<unsigned int, std::vector<std::pair<unsigned int, float>>> sharedWeightPerTruthVertex;
    std::unordered_map<unsigned int, float> truthWeightTotal;

    // Hit-based domains: the inverted DetId index and the per-cell denominators are
    // independent of the working point, so the associator is built ONCE per
    // collection and shared by every working point below.
    std::optional<truth::BranchHitAssociator> hitAssociator;
    if constexpr (!ConstituentBasedDomain<RECO>) {
      hitAssociator.emplace(hitIndex,
                            selectedRoots,
                            Traits::metric,
                            Traits::channel,
                            /*emptyRootsMeansAll=*/false,
                            denominatorDetectors);
    }

    for (std::size_t wpIndex = 0; wpIndex < workingPoints_.size(); ++wpIndex) {
      auto const& wp = workingPoints_[wpIndex];
      auto recoToTruth = std::make_unique<MapType>(nReco);

      if constexpr (ConstituentBasedDomain<RECO>) {
        // A composite object is associated to a truth VERTEX, not to a particle branch.
        // Keying the aggregation by the branch a constituent points at cannot disagree
        // with itself, so every object matched something and the purity was 1 by
        // construction. Keying it by the PRODUCTION VERTEX of that branch is what makes
        // the number mean anything: constituents whose particles were produced at an
        // unrelated vertex are contamination, and the leading vertex's share is the
        // purity.
        auto const& constituentMap = event.get(constituentMapTokens_[collectionIndex][wpIndex]);
        for (unsigned int i = 0; i < nReco; ++i) {
          auto const& object = (*handle)[i];
          const float total = Traits::totalWeight(object);
          if (total <= 0.f) {
            continue;
          }
          std::unordered_map<unsigned int, float> weightPerVertex;
          Traits::forEachConstituent(object, [&](unsigned int constituentIndex, float weight) {
            if (constituentIndex >= constituentMap.size()) {
              return;
            }
            // maps are score-sorted, so [0] is the constituent's best match
            for (auto const& match : constituentMap[constituentIndex]) {
              const unsigned int particle = match.index();
              if (particle < nBranches) {
                if (const auto vertexId = countingVertex(graph, particle, vertexResolution_, interactionVertex)) {
                  weightPerVertex[*vertexId] += weight;
                }
              }
              break;
            }
          });
          // Denominator over ALL constituents, the CMSSW convention: a track with no
          // truth match legitimately lowers the shared fraction. With pt^2 weighting
          // that dilution is small, because the tracks that go unmatched are the soft
          // ones, which is exactly why the standard weighting is pt^2 and not a count.
          for (auto const& [vertexId, weight] : weightPerVertex) {
            // RECO purity: the leading truth vertex's share of THIS reco object's pt^2.
            const float recoPurity = weight / total;
            recoToTruth->insert(i, vertexId, recoPurity, 1.f - recoPurity);
            // TRUTH purity: the same shared weight over everything the truth vertex
            // produced that was reconstructed at all, accumulated below once the whole
            // collection has been seen.
            if (wpIndex == 0) {
              sharedWeightPerTruthVertex[vertexId].emplace_back(i, weight);
              truthWeightTotal[vertexId] += weight;
            }
          }
        }
      } else {
        auto const& associator = *hitAssociator;
        for (unsigned int i = 0; i < nReco; ++i) {
          if (recoHitsPerObject[i].empty()) {
            continue;
          }
          const std::span<const truth::RecoHit> span(recoHitsPerObject[i]);

          // One candidate list serves both the fixed working point and the
          // truth-driven fill below.
          std::vector<truth::BranchMatch> matches;
          if (!wp.adaptive || wpIndex == 0) {
            matches = associator.bestBranches(span);
          }

          // RECO to TRUTH: the working point drives the search, and the score is
          // reco-normalised, so 1 - score is the RECO purity.
          if (wp.adaptive) {
            const auto match = associator.bestAdaptiveBranch(span, wp.reverseWeight, wp.maxReverseScore);
            if (match.rootParticleId != truth::BranchMatch::kInvalidRoot) {
              recoToTruth->insert(i, match.rootParticleId, match.sharedEnergy, match.score);
            }
          } else {
            for (auto const& match : matches) {
              recoToTruth->insert(i, match.rootParticleId, match.sharedEnergy, match.score);
            }
          }

          // TRUTH to RECO, filled only once. NO adaptive climb: the climb chooses a
          // graph level to suit the reco object, which is meaningless when the truth
          // target is the thing being asked about. Both payloads of this direction are
          // TRUTH-normalised: the sim-normalised shared energy fraction, which is the
          // axis HGCalValidator gates efficiency on, and the truth-normalised score,
          // which gates purity and duplicate. A shared-hits domain has no energy, so it
          // keeps reporting the shared hit count.
          if (wpIndex == 0) {
            constexpr bool sharedEnergyMetric = Traits::metric == truth::BranchHitAssociator::Metric::SharedEnergy;
            for (auto const& match : matches) {
              const float truthValue = sharedEnergyMetric ? match.sharedEnergyFraction : match.sharedEnergy;
              truthToReco->insert(match.rootParticleId, i, truthValue, match.reverseScore);
            }
          }
        }
      }

      if constexpr (ConstituentBasedDomain<RECO>) {
        if (wpIndex == 0) {
          for (auto const& [vertexId, entries] : sharedWeightPerTruthVertex) {
            const float denominator = truthWeightTotal[vertexId];
            if (denominator <= 0.f) {
              continue;
            }
            for (auto const& [recoIndex, weight] : entries) {
              const float truthPurity = weight / denominator;
              truthToReco->insert(vertexId, recoIndex, truthPurity, 1.f - truthPurity);
            }
          }
        }
      }

      // Ascending score, so [0] is the best match; consumers rely on this. An explicit
      // comparator: the map's own sort(true) orders DESCENDING by score, worst first.
      recoToTruth->sort(byAscendingScore);
      // Every declared instance label must be put on every path, including the one
      // where the reco collection was absent: a missing put is a framework error.
      event.put(std::move(recoToTruth), key + "RecoToTruth" + wp.name);
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

  edm::ParameterSetDescription selector;
  selector.add<float>("ptMin", 1.f)->setComment("Reject branches whose root is softer than this");
  selector.add<float>("ptMax", std::numeric_limits<float>::max());
  selector.add<float>("etaMin", -4.f);
  selector.add<float>("etaMax", 4.f);
  selector.add<std::vector<int>>("pdgIds", {})->setComment("Empty accepts every species");
  selector.add<bool>("signalOnly", false);
  selector.add<bool>("intimeOnly", false);
  selector.add<bool>("chargedOnly", false);
  selector.add<bool>("invertEta", false);
  selector.add<bool>("kinematicsOnStableOnly", true)
      ->setComment(
          "Apply ptMin/ptMax/etaMin/etaMax only to a root that decayed nowhere. The momentum of a root "
          "that decayed is not a detector observable: a resonance at rest has pt about 0 and |eta| "
          "unbounded, so a track-shaped cut rejects it while its decay products fill the calorimeter.");
  desc.add<edm::ParameterSetDescription>("branchSelector", selector);
  if constexpr (!ConstituentBasedDomain<RECO>) {
    desc.add<std::vector<std::string>>("truthLevels", {"caloBoundary"})
        ->setComment(
            "Which levels of the graph the TruthToReco direction asks about, one denominator product per level: "
            "stableLegsFromUpstream, caloBoundary, stableDecayProducts, hardProcess. Each is an antichain, so "
            "every physical object at that level is counted once. A kinematic selection alone is NOT a level: a "
            "tau, its decay products and their calorimeter-crossing descendants would all enter the denominator "
            "at the same time");
  }
  desc.add<std::vector<int>>("signalSeedHadronFlavors", {})
      ->setComment(
          "Heavy-flavour hadron content (5 = b, 4 = c) the selection preset seeds with, for presets that name "
          "their signal by flavour rather than by pdg id. Matched with the same digit rule the preset uses");
  desc.add<std::vector<int>>("signalSeedPdgIds", {})
      ->setComment(
          "The selection preset's seed pdgIds (truthGraphSelections seedPdgIdsForPreset). signalSeeds is the "
          "subset of the selected roots with one of these pdgIds, signalSeedsNoSelection every particle with "
          "one of them whatever the branch selector says. Empty, with no seed flavours either, means the sample "
          "has no resonance and both products stay empty");
  desc.add<bool>("truthToRecoSignalOnly", true)
      ->setComment(
          "Restrict the TruthToReco denominator to the signal interaction. Efficiency, duplicate and split are "
          "meaningless averaged over the overlaid pileup interactions. The associator's candidate set is NOT "
          "restricted, so pileup branches stay matchable and a pileup-matched reco object is not counted a fake");
  desc.add<bool>("heavyFlavorOnly", false)
      ->setComment(
          "Composite domains only. Keep in the denominator only the vertices where a b or c hadron DECAYED, which "
          "is what inclusiveSecondaryVertices reconstructs: 4 and 5 per no-PU ttbar event against 4.1 "
          "reconstructed. Off by default; the secondary-vertex associator turns it on. Without it the denominator "
          "is every graph vertex with two selected roots, 45.9 per event, and the efficiency is capped by the "
          "denominator.");
  desc.add<std::vector<std::string>>("workingPointNames", {"Fixed"});
  desc.add<std::vector<float>>("adaptiveReverseWeight", {0.f});
  desc.add<std::vector<float>>("adaptiveMaxReverseScore", {0.f});
  if constexpr (LayerClusterBackedRecoHits<RECO>) {
    desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
  }
  if constexpr (ConstituentBasedDomain<RECO>) {
    desc.add<std::string>("constituentAssociator", "allTrackToTruthBranchAssociators")
        ->setComment("Module that produced the constituents' association maps");
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
// NOT named AllTracksterToTruthBranchAssociatorsProducer: that class already exists in
// PhysicsTools/TruthInfo. The two are not interchangeable. The older one keys its
// product instances by label+instance with no separator, offers one adaptive point next
// to the fixed match, and takes its candidate roots from an external product; this one
// keys by label_instance, carries the whole working-point list, and publishes
// the per-level target lists, every one of them an antichain, so no efficiency can count
// one object twice.
// Consolidating the two is follow-up work; until then a duplicate class name would make
// the framework pick one of them at random.
using TruthBranchTracksterAssociatorsProducer = AllRecoToTruthBranchAssociatorsProducer<ticl::Trackster>;
DEFINE_FWK_MODULE(TruthBranchTracksterAssociatorsProducer);
