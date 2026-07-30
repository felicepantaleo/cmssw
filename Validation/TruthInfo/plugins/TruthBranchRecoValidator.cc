// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// Truth-branch validation, one plugin template covering every reco domain, booking
// only num/denom so all harvesting stays DQMGenericClient string config. Two folder
// families: the reco-driven metrics get one folder per (collection, working point),
// the truth-driven ones one folder per (collection, graph level), because the truth
// target is fixed a priori by the level and the working point never enters it.
//
// DQMGlobalEDAnalyzer, not DQMEDAnalyzer: booking and filling are both const and the
// MonitorElements live in a per-run cache, which is the modern convention shared by
// MultiTrackValidator and HGCalValidator.
//
// What differs between domains is only (a) which association map type the associator
// wrote and (b) how to read kinematics off a reco object. Both are bound to the reco
// type by RecoValidationTraits, mirroring TruthAssociationTraits on the producer side,
// so the declared product type and the consumed one cannot drift apart.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "PhysicsTools/TruthInfo/interface/SubgraphHitView.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/VertexData.h"

#include "Validation/TruthInfo/interface/TruthBranchHistoProducerAlgo.h"

namespace {
  using Kinematics = truth::TruthBranchHistoProducerAlgo::Kinematics;

  template <typename RECO>
  struct RecoValidationTraits;

  template <>
  struct RecoValidationTraits<reco::Track> {
    using MapType = ticl::AssociationMap<ticl::mapWithSharedHitsAndScore>;
    static constexpr const char* cfiName = "truthBranchTrackValidator";
    static constexpr const char* defaultAssociator = "allTrackToTruthBranchAssociators";
    static constexpr const char* defaultDir = "TruthInfo/Tracking/";

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
    // The denominator instance is a PREFIX: the capitalized level name is appended, one
    // product per configured level.
    static constexpr bool truthIsVertex = false;
    // A domain matched on shared ENERGY in the calorimeter channel, which is judged by
    // the HGCal validation criteria; everything else is judged on shared components.
    static constexpr bool calorimetric = false;
    static constexpr const char* denominatorInstance = "truthToRecoTargets";
  };

  // A vertex has no momentum, so only its position and its track multiplicity are
  // meaningful; the configuration books exactly those and nothing else.
  template <>
  struct RecoValidationTraits<reco::Vertex> {
    using MapType = ticl::AssociationMap<ticl::mapWithFractionAndScore>;
    static constexpr const char* cfiName = "truthBranchVertexValidator";
    static constexpr const char* defaultAssociator = "allVertexToTruthBranchAssociators";
    static constexpr const char* defaultDir = "TruthInfo/Vertexing/";

    static Kinematics kinematics(reco::Vertex const& vertex) {
      Kinematics kin;
      // The number of tracks the vertex was built from, which is the vertex analogue of
      // a track's hit count: the constituents its truth was aggregated from.
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
    using MapType = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>;
    static constexpr const char* cfiName = "truthBranchTracksterValidator";
    static constexpr const char* defaultAssociator = "truthBranchTracksterAssociators";
    static constexpr const char* defaultDir = "TruthInfo/Calorimetry/";

    static Kinematics kinematics(ticl::Trackster const& trackster) {
      Kinematics kin;
      auto const& bary = trackster.barycenter();
      const double rho = std::sqrt(bary.x() * bary.x() + bary.y() * bary.y());
      const double mag = std::sqrt(rho * rho + bary.z() * bary.z());
      // Energy shared out along the barycentre direction: a trackster has no track, so
      // its transverse momentum is the raw energy projected transversally.
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
  };

  // One entry per (collection, graph level) for a hit-based domain, per collection for
  // a composite one, in booking order: the truth-driven monitor elements.
  struct TruthEntry {
    std::string folder;
    // The level's denominator: the target set the efficiency is measured over.
    edm::EDGetTokenT<std::vector<unsigned int>> targetsToken;
    // Truth-driven, one product per collection because the truth target is fixed a
    // priori: score is 1 - truth purity.
    edm::EDGetTokenT<MapType> truthToRecoToken;
    // The FIRST working point's reco-driven map (Fixed), which is the WP-free
    // hit-sharing measure, read only for the loose reco-purity gate on Individual.
    edm::EDGetTokenT<MapType> firstWpRecoToTruthToken;
  };

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const std::string dirName_;
  // A truth object counts as reconstructed by one reco object when that object covers
  // enough of it AND is not mostly something else. The second is the loose cut in the
  // other direction that both QuickTrackAssociatorByHits and HGVHistoProducerAlgo use.
  // Shared-component domains (tracks, vertices) are judged on these two.
  const double minTruthPurityForIndividual_;
  const double minRecoPurityLoose_;
  // Calorimetric domains are judged on the three HGCalValidator quantities instead,
  // which are NOT the same axis: efficiency is a shared-energy-fraction cut, purity and
  // duplicate are simToReco score cuts, fake and merge recoToSim score cuts
  // (Validation/HGCalValidation/src/HGVHistoProducerAlgo.cc:2819-2820 and 2897-2899).
  const double minSharedEnergyFractionForIndividual_;
  const double maxSimToRecoScoreForDuplicate_;
  const double maxRecoToSimScore_;
  const double minCollectiveCoverage_;
  std::vector<WpEntry> wpEntries_;
  std::vector<TruthEntry> truthEntries_;
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
      algo_(cfg.getParameter<edm::ParameterSet>("histoProducerAlgoBlock")) {
  const auto associator = cfg.getParameter<std::string>("associator");
  const auto workingPoints = cfg.getParameter<std::vector<std::string>>("workingPoints");

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
    // The overall signal entry: its denominator is the preset SEED objects among the
    // selected roots (the tau, not its decay legs), so the folder measures the signal
    // object's own efficiency. Without a preset every selected root is a seed.
    truthTargets.emplace_back("signal", "signalSeeds");
    // The same seed objects with NO selector cut, so the efficiency is quoted against
    // every seed in the event rather than against the ones the kinematic selection
    // kept. The gap to the signal folder is what the selection removed.
    truthTargets.emplace_back("signalNoSelection", "signalSeedsNoSelection");
    // Every selected root, whatever its level or species: the widest truth denominator.
    truthTargets.emplace_back("allSelectedRoots", "selectedBranchRoots");
  }

  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("recoCollections")) {
    // The one key rule of this package: label and instance joined by an underscore,
    // used for the product instance labels AND for the folder name.
    std::string key = tag.label();
    if (!tag.instance().empty()) {
      key += "_" + tag.instance();
    }
    for (auto const& wp : workingPoints) {
      WpEntry entry;
      entry.folder = key + "_" + wp;
      entry.recoToken = consumes<std::vector<RECO>>(tag);
      entry.recoToTruthToken = consumes<MapType>(edm::InputTag(associator, key + "RecoToTruth" + wp));
      wpEntries_.push_back(std::move(entry));
    }
    for (auto const& [suffix, instance] : truthTargets) {
      TruthEntry entry;
      entry.folder = key + "_" + suffix;
      entry.targetsToken = consumes<std::vector<unsigned int>>(edm::InputTag(associator, instance));
      entry.truthToRecoToken = consumes<MapType>(edm::InputTag(associator, key + "TruthToReco"));
      // The first working point (Fixed) is the WP-free hit-sharing measure, so its map
      // supplies the loose reco-purity gate for every level.
      entry.firstWpRecoToTruthToken =
          consumes<MapType>(edm::InputTag(associator, key + "RecoToTruth" + workingPoints.front()));
      truthEntries_.push_back(std::move(entry));
    }
  }
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
    algo_.bookRecoHistos(booker, histograms);
  }
  for (auto const& entry : truthEntries_) {
    booker.setCurrentFolder(dirName_ + entry.folder);
    // The shared energy fraction is the axis the calorimetric efficiency cut acts on,
    // so it is booked exactly where that cut is applied and nowhere else.
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

  // Reco-driven side, one pass per (collection, working point).
  for (std::size_t i = 0; i < wpEntries_.size(); ++i) {
    auto const& entry = wpEntries_[i];

    edm::Handle<std::vector<RECO>> recoHandle;
    event.getByToken(entry.recoToken, recoHandle);
    edm::Handle<MapType> recoToTruthHandle;
    event.getByToken(entry.recoToTruthToken, recoToTruthHandle);
    if (!recoHandle.isValid() || !recoToTruthHandle.isValid()) {
      continue;
    }

    auto const& recoToTruth = recoToTruthHandle->getMap();

    // Reco side: every object, whether it found a branch, and whether that branch came
    // from a pileup interaction rather than the signal one.
    for (std::size_t r = 0; r < recoHandle->size(); ++r) {
      // The maps are score-sorted, so [0] is the best match. A calorimetric object
      // additionally has to pass the recoToSim score cut to count as anything but a
      // fake, which is HGCalValidator's non-fake criterion.
      bool associated = r < recoToTruth.size() && !recoToTruth[r].empty();
      if constexpr (Traits::calorimetric) {
        associated = associated && recoToTruth[r][0].score() < maxRecoToSimScore_;
      }
      const Kinematics kin = Traits::kinematics((*recoHandle)[r]);

      bool pileup = false;
      if (associated) {
        // eventId 0 is the signal interaction; anything else is overlaid pileup. The
        // row index means a truth vertex for a composite domain and a particle for a
        // hit-based one, so the lookup follows the same split.
        const unsigned int matched = recoToTruth[r][0].index();
        if constexpr (Traits::truthIsVertex) {
          if (matched < graph.nVertices()) {
            pileup = graph.vertices()[matched].eventId != 0;
          }
        } else {
          if (matched < graph.nParticles()) {
            pileup = graph.particles()[matched].eventId != 0;
          }
        }
      }
      // For a composite object the association always finds something, so counting the
      // match tells nothing; what it is worth is the leading truth vertex's share of the
      // object's constituents. Constituents whose particles were produced at an
      // unrelated vertex are the remainder.
      // Reco purity, the reco-normalised quantity this direction exists to measure.
      const double recoPurity = associated ? 1. - static_cast<double>(recoToTruth[r][0].score()) : 0.;
      algo_.fill_reco(histograms, i, kin, associated, pileup, recoPurity);
      if (associated) {
        algo_.fill_match(histograms, i, recoToTruth[r][0].score(), recoToTruth[r][0].value(), recoPurity);
        // Resolution against the truth object THIS working point matched, so the
        // residuals follow the working point like every other reco-driven metric. A
        // composite truth object is a vertex with no direction, so no residual there.
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

  // Truth-driven side, one pass per (collection, level). The denominator is the
  // level's target set: iterating every particle instead would put objects outside the
  // level in the denominator as guaranteed misses.
  for (std::size_t i = 0; i < truthEntries_.size(); ++i) {
    auto const& entry = truthEntries_[i];

    edm::Handle<std::vector<unsigned int>> targetsHandle;
    event.getByToken(entry.targetsToken, targetsHandle);
    edm::Handle<MapType> truthToRecoHandle;
    event.getByToken(entry.truthToRecoToken, truthToRecoHandle);
    edm::Handle<MapType> firstWpHandle;
    event.getByToken(entry.firstWpRecoToTruthToken, firstWpHandle);
    if (!targetsHandle.isValid() || !truthToRecoHandle.isValid() || !firstWpHandle.isValid()) {
      continue;
    }

    auto const& truthToReco = truthToRecoHandle->getMap();
    auto const& recoToTruth = firstWpHandle->getMap();

    // Reco-normalised score of a (truth, reco) pair, read from the FIRST working point's
    // reco-driven product, the WP-free hit-sharing measure. This is the loose cut in
    // the other direction, so it is looked up rather than recomputed. A pair that is
    // absent scores the worst possible value.
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

    for (unsigned int b : *targetsHandle) {
      if (b >= truthToReco.size()) {
        continue;
      }
      Kinematics kin;
      auto reason = static_cast<unsigned int>(truth::VertexReason::Unknown);

      if constexpr (Traits::truthIsVertex) {
        // The truth object IS a vertex: its position, how many selected particles were
        // produced there, and the Geant4 process that made it. depth and root_footprint_fraction are
        // properties of a particle branch and are not booked for this domain.
        if (b >= graph.nVertices()) {
          continue;
        }
        const truth::Vertex vertex(&graph, b);
        auto const& vdata = vertex.data();
        auto const& pos = vertex.position();
        kin.vertpos = std::sqrt(pos.x() * pos.x() + pos.y() * pos.y());
        kin.zpos = pos.z();
        kin.nhits = vertex.outgoingParticles().size();
        reason = vdata.hasSim() ? static_cast<unsigned int>(vdata.reason)
                                : static_cast<unsigned int>(truth::VertexReason::Other) + 1;
      } else {
        if (b >= graph.nParticles()) {
          continue;
        }
        auto const& particle = graph.particles()[b];
        const auto& p4 = particle.momentum;
        if (p4.pt() <= 0.) {
          continue;
        }
        kin.pt = p4.pt();
        kin.eta = p4.eta();
        kin.phi = p4.phi();
        // The branch's own detector footprint, which is the truth analogue of a track's
        // hit count.
        const auto subgraph = hitIndex.subgraphHits(truth::HitChannel::Tracker, b);
        kin.nhits = subgraph.size();
        const truth::Particle branchRoot(&graph, b);
        // How deep in the graph the branch root sits. A frozen truth object has one
        // fixed level and no such axis.
        kin.depth = branchRoot.ancestors().size();
        // How much of the branch footprint is the root particle's own hits rather than
        // its descendants'. Near 1 is a clean single particle, near 0 a branch whose
        // hits all come from what it produced.
        const auto direct = hitIndex.directHits(truth::HitChannel::Tracker, b);
        kin.root_footprint_fraction = subgraph.empty() ? 0. : static_cast<double>(direct.size()) / subgraph.size();

        const auto vertices = branchRoot.productionVertices();
        if (!vertices.empty()) {
          // A GEN-only production vertex has no Geant4 creator process, so its reason is
          // Unknown by construction rather than by failure to classify. It gets its own
          // bin, one past the enum, so the two do not get read as the same thing.
          auto const& vdata = vertices.front().data();
          reason = vdata.hasSim() ? static_cast<unsigned int>(vdata.reason)
                                  : static_cast<unsigned int>(truth::VertexReason::Other) + 1;
          const auto& pos = vertices.front().position();
          kin.vertpos = std::sqrt(pos.x() * pos.x() + pos.y() * pos.y());
          kin.zpos = pos.z();
          // Transverse and longitudinal impact parameter of the branch direction with
          // respect to the origin, the truth counterpart of the track dxy and dz.
          kin.dxy = (-pos.x() * p4.py() + pos.y() * p4.px()) / p4.pt();
          kin.dz = pos.z() - (pos.x() * p4.px() + pos.y() * p4.py()) / p4.pt() * (p4.pz() / p4.pt());
        }
      }

      // Classify how this truth object was reconstructed, from the TRUTH-driven
      // product. Individual means one reco object covered it; duplicate means more than
      // one did; split means none did alone but together they cover it.
      using Outcome = truth::TruthBranchHistoProducerAlgo::TruthOutcome;
      unsigned int nIndividual = 0;
      unsigned int nPure = 0;
      double collectiveCoverage = 0.;
      double leadingTruthPurity = 0.;
      double leadingSharedEnergyFraction = 0.;
      for (auto const& match : truthToReco[b]) {
        const double truthPurity = 1. - static_cast<double>(match.score());
        leadingTruthPurity = std::max(leadingTruthPurity, truthPurity);
        if constexpr (Traits::calorimetric) {
          // The truth-to-reco payload of a calorimetric domain is sim-normalised: the
          // value is the shared energy over the branch energy in the detectors this
          // collection reconstructs, the score the simToReco one. Efficiency gates on
          // the fraction, duplicate on the score.
          const double sharedEnergyFraction = match.value();
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
      const bool collective = collectiveCoverage >= minCollectiveCoverage_ && !truthToReco[b].empty();
      Outcome outcome = Outcome::Lost;
      if constexpr (Traits::calorimetric) {
        // Duplicate refines Individual rather than competing with it, so the four
        // outcomes stay mutually exclusive and efficiency stays exactly the shared
        // energy fraction cut.
        outcome = (nIndividual >= 1) ? (nPure > 1 ? Outcome::Duplicate : Outcome::Individual)
                  : collective       ? Outcome::Split
                                     : Outcome::Lost;
      } else {
        outcome = (nIndividual == 1)  ? Outcome::Individual
                  : (nIndividual > 1) ? Outcome::Duplicate
                  : collective        ? Outcome::Split
                                      : Outcome::Lost;
      }
      // Cumulative: the collection as a whole covers the truth object, by one reco
      // object or by several together, so it is a superset of individual.
      const bool cumulative = nIndividual >= 1 || collective;

      algo_.fill_simul(histograms, i, kin, outcome, cumulative);
      algo_.fill_reason(histograms, i, reason, outcome);
      if (!truthToReco[b].empty()) {
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
  desc.add<std::vector<edm::InputTag>>("recoCollections", {});
  desc.add<std::vector<std::string>>("workingPoints", {"Fixed"});
  if constexpr (Traits::truthIsVertex) {
    desc.add<std::string>("vertexResolution", "interaction")
        ->setComment(
            "Names the one truth-driven folder of a composite domain, matching the associator's resolution: "
            "'interaction' for primary vertices, 'immediate' for secondary vertices");
  } else {
    desc.add<std::vector<std::string>>("truthLevels", {"caloBoundary"})
        ->setComment(
            "Graph levels the truth-driven metrics are measured at, one folder per level. Must match the "
            "associator's truthLevels: each level consumes its own denominator product");
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
  // Every axis is declared here; which of them a domain books is chosen by the two
  // variable lists, so adding a domain needs no new axis parameter.
  const std::vector<std::tuple<std::string, int, double, double>> axes = {{"pt", 50, 0., 100.},
                                                                          {"eta", 50, -4., 4.},
                                                                          {"phi", 36, -3.2, 3.2},
                                                                          {"nhits", 40, 0., 40.},
                                                                          {"vertpos", 40, 0., 60.},
                                                                          {"zpos", 40, -30., 30.},
                                                                          {"dxy", 40, -5., 5.},
                                                                          {"dz", 40, -20., 20.},
                                                                          {"depth", 15, 0., 15.},
                                                                          {"root_footprint_fraction", 20, 0., 1.}};
  for (auto const& [name, nbins, lo, hi] : axes) {
    algo.add<int>("nint_" + name, nbins);
    algo.add<double>("min_" + name, lo);
    algo.add<double>("max_" + name, hi);
  }
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
