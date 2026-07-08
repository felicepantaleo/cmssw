// NanoAOD tables for the trackster <-> truth-branch associations: one TruthBranch
// table (the union of matched branch roots, with pdgId/kinematics/provenance) and
// one pair table per configured association (trackster index, local branch index,
// shared energy, score). Together with the trackster tables this is a per-trackster
// training dataset: continuous truth labels (shared energy, purity, completeness
// against every matching branch) including the principled "unknown" (a trackster
// with no pair rows matches no branch).

#include <algorithm>
#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

namespace {
  using BranchAssociationMap = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>;
}

class TracksterTruthBranchTableProducer : public edm::stream::EDProducer<> {
public:
  explicit TracksterTruthBranchTableProducer(edm::ParameterSet const&);
  void produce(edm::Event&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  std::vector<edm::EDGetTokenT<BranchAssociationMap>> mapTokens_;
  std::vector<edm::EDGetTokenT<std::vector<ticl::Trackster>>> tracksterTokens_;
  std::vector<std::string> tableNames_;
  std::vector<std::string> tracksterTableNames_;
  const std::string branchTableName_;
  // Hierarchical label thresholds (see the label table columns).
  const double labelPurityMin_;
  const double contributorMinFraction_;
  const double minSharedEnergy_;
  const bool computeLabels_;
};

TracksterTruthBranchTableProducer::TracksterTruthBranchTableProducer(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      branchTableName_(cfg.getParameter<std::string>("branchTableName")),
      labelPurityMin_(cfg.getParameter<double>("labelPurityMin")),
      contributorMinFraction_(cfg.getParameter<double>("contributorMinFraction")),
      minSharedEnergy_(cfg.getParameter<double>("minSharedEnergy")),
      computeLabels_(cfg.getParameter<bool>("computeLabels")) {
  auto tags = cfg.getParameter<std::vector<edm::InputTag>>("associations");
  auto tsTags = cfg.getParameter<std::vector<edm::InputTag>>("tracksterCollections");
  auto names = cfg.getParameter<std::vector<std::string>>("tableNames");
  auto tsNames = cfg.getParameter<std::vector<std::string>>("tracksterTableNames");
  if (tags.size() != names.size() || tags.size() != tsTags.size() || tags.size() != tsNames.size()) {
    throw cms::Exception("Configuration")
        << "associations, tracksterCollections, tableNames and tracksterTableNames must have the same length";
  }
  for (size_t i = 0; i < tags.size(); ++i) {
    mapTokens_.emplace_back(consumes<BranchAssociationMap>(tags[i]));
    tracksterTokens_.emplace_back(consumes<std::vector<ticl::Trackster>>(tsTags[i]));
    tableNames_.push_back(names[i]);
    tracksterTableNames_.push_back(tsNames[i]);
    produces<nanoaod::FlatTable>(names[i]);
    if (computeLabels_)
      produces<nanoaod::FlatTable>(names[i] + "Labels");
  }
  produces<nanoaod::FlatTable>(branchTableName_);
}

void TracksterTruthBranchTableProducer::produce(edm::Event& event, edm::EventSetup const&) {
  auto const& graph = event.get(graphToken_);

  // Union of the branch roots referenced by any association, with a local index.
  std::vector<edm::Handle<BranchAssociationMap>> maps(mapTokens_.size());
  std::vector<uint32_t> roots;
  for (size_t i = 0; i < mapTokens_.size(); ++i) {
    event.getByToken(mapTokens_[i], maps[i]);
    for (unsigned int obj = 0; obj < maps[i]->size(); ++obj)
      for (auto const& el : (*maps[i])[obj])
        roots.push_back(el.index());
  }
  std::sort(roots.begin(), roots.end());
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
  std::vector<int> localIndex(graph.nParticles(), -1);
  for (size_t i = 0; i < roots.size(); ++i)
    localIndex[roots[i]] = static_cast<int>(i);

  // TruthBranch table.
  std::vector<int> pdgId;
  std::vector<float> energy, pt, eta, phi;
  std::vector<uint8_t> hasGen, hasSim, backscattered;
  std::vector<int> rootId;
  for (uint32_t r : roots) {
    auto const& p = graph.particles()[r];
    pdgId.push_back(p.pdgId);
    energy.push_back(p.momentum.energy());
    pt.push_back(p.momentum.pt());
    eta.push_back(p.momentum.eta());
    phi.push_back(p.momentum.phi());
    hasGen.push_back(p.hasGen());
    hasSim.push_back(p.hasSim());
    backscattered.push_back(p.backscattered);
    rootId.push_back(static_cast<int>(r));
  }
  auto branchTable = std::make_unique<nanoaod::FlatTable>(roots.size(), branchTableName_, false);
  branchTable->addColumn<int>("pdgId", pdgId, "branch root PDG id");
  branchTable->addColumn<float>("energy", energy, "branch root energy [GeV]");
  branchTable->addColumn<float>("pt", pt, "branch root pt [GeV]");
  branchTable->addColumn<float>("eta", eta, "branch root eta");
  branchTable->addColumn<float>("phi", phi, "branch root phi");
  branchTable->addColumn<uint8_t>("hasGen", hasGen, "root has a GEN node");
  branchTable->addColumn<uint8_t>("hasSim", hasSim, "root has a SIM node");
  branchTable->addColumn<uint8_t>("backscattered", backscattered, "root flagged as back-scattered");
  branchTable->addColumn<int>("rootId", rootId, "root particle index in the truth graph");
  event.put(std::move(branchTable), branchTableName_);

  // Pair tables.
  for (size_t i = 0; i < mapTokens_.size(); ++i) {
    std::vector<uint16_t> tsIdx, brIdx;
    std::vector<float> sharedEnergy, score;
    for (unsigned int obj = 0; obj < maps[i]->size(); ++obj) {
      for (auto const& el : (*maps[i])[obj]) {
        tsIdx.push_back(static_cast<uint16_t>(obj));
        brIdx.push_back(static_cast<uint16_t>(localIndex[el.index()]));
        sharedEnergy.push_back(el.sharedEnergy());
        score.push_back(el.score());
      }
    }
    auto table = std::make_unique<nanoaod::FlatTable>(tsIdx.size(), tableNames_[i], false);
    table->addColumn<uint16_t>("tracksterIdx", tsIdx, "index in the trackster collection");
    table->addColumn<uint16_t>("branchIdx", brIdx, "index in the " + branchTableName_ + " table");
    table->addColumn<float>("sharedEnergy", sharedEnergy, "shared HGCAL rechit energy [GeV]");
    table->addColumn<float>("score", score, "association score (lower is better)");
    event.put(std::move(table), tableNames_[i]);

    if (!computeLabels_)
      continue;

    // Hierarchical labels, one row per trackster (extension of the trackster
    // feature table): the label is the LOWEST truth-graph node whose branch
    // contains the trackster with purity >= labelPurityMin.
    //   class 0 (clean): a single calo-entering particle dominates; its pdgId.
    //   class 1 (ambiguous): only an ANCESTOR is pure, i.e. the trackster mixes
    //     energy from different legs of the same decay/interaction (e.g. the two
    //     photons of a pi0: the label pdgId is then 111); which leaf PID to
    //     assign is genuinely unclear.
    //   class 2 (unknown): the significant contributors share no physical
    //     ancestor (or nothing matches above threshold): the trackster mixes
    //     unrelated particles, i.e. it is fake.
    auto const& tracksters = event.get(tracksterTokens_[i]);
    const size_t nTs = tracksters.size();
    std::vector<uint8_t> labelClass(nTs, 2);
    std::vector<int> labelPdgId(nTs, 0), labelRootId(nTs, -1);
    std::vector<float> labelPurity(nTs, 0.f), leafPurity(nTs, 0.f), matchedFraction(nTs, 0.f);
    for (unsigned int t = 0; t < nTs; ++t) {
      auto const& row = (*maps[i])[t];
      const float raw = tracksters[t].raw_energy();
      if (row.empty() || raw <= 0.f)
        continue;
      float total = 0.f, best = 0.f;
      uint32_t bestRoot = 0;
      for (auto const& el : row) {
        total += el.sharedEnergy();
        if (el.sharedEnergy() > best) {
          best = el.sharedEnergy();
          bestRoot = el.index();
        }
      }
      matchedFraction[t] = total / raw;
      leafPurity[t] = best / raw;
      if (total < minSharedEnergy_)
        continue;  // stays unknown: nothing meaningful matched
      if (leafPurity[t] >= labelPurityMin_) {
        labelClass[t] = 0;
        labelPdgId[t] = graph.particles()[bestRoot].pdgId;
        labelRootId[t] = static_cast<int>(bestRoot);
        labelPurity[t] = leafPurity[t];
        continue;
      }
      // Significant contributors and their lowest common ancestor.
      std::vector<truth::Particle> contributors;
      float contribSum = 0.f;
      for (auto const& el : row) {
        if (el.sharedEnergy() >= contributorMinFraction_ * total) {
          contributors.emplace_back(&graph, el.index());
          contribSum += el.sharedEnergy();
        }
      }
      if (contributors.size() < 2 || contribSum / raw < labelPurityMin_)
        continue;  // unknown: no ancestor can be pure either
      auto lca = graph.lowestCommonAncestor(contributors);
      // The ambiguous class means "different legs of the same physical decay or
      // interaction" (a pi0's photons, a D0's or phi's products): the ancestor must
      // be a real hadron, lepton or photon. A partonic ancestor (quark or gluon)
      // means the contributors only share the jet, i.e. they are unrelated
      // particles, which is the unknown/fake class; so is the event level (an
      // ancestor with no parents).
      auto isPartonic = [](int pdg) {
        const int a = std::abs(pdg);
        return a == 0 || a <= 8 || a == 21;
      };
      if (lca && !lca->parents().empty() && !isPartonic(lca->pdgId())) {
        labelClass[t] = 1;
        labelPdgId[t] = lca->pdgId();
        labelRootId[t] = static_cast<int>(lca->id());
        labelPurity[t] = contribSum / raw;
      }
    }
    auto labels = std::make_unique<nanoaod::FlatTable>(nTs, tracksterTableNames_[i], false, true);
    labels->addColumn<uint8_t>("labelClass", labelClass, "0 clean, 1 ambiguous (same-ancestor legs), 2 unknown/fake");
    labels->addColumn<int>("labelPdgId", labelPdgId, "pdgId of the labelling truth node (leaf or ancestor)");
    labels->addColumn<int>("labelRootId", labelRootId, "truth-graph particle index of the label (-1: none)");
    labels->addColumn<float>("labelPurity", labelPurity, "shared energy under the label / trackster raw energy");
    labels->addColumn<float>("leafPurity", leafPurity, "best single-branch shared energy / raw energy");
    labels->addColumn<float>("matchedFraction", matchedFraction, "total branch-matched energy / raw energy");
    event.put(std::move(labels), tableNames_[i] + "Labels");
  }
}

void TracksterTruthBranchTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<std::vector<edm::InputTag>>(
      "associations",
      {edm::InputTag("allTrackstersToTruthBranchAssociations", "ticlTrackstersCLUE3DHighToTruthBranch"),
       edm::InputTag("allTrackstersToTruthBranchAssociations", "ticlTracksterInterpretationsToTruthBranch")});
  desc.add<std::vector<edm::InputTag>>(
      "tracksterCollections",
      {edm::InputTag("ticlTrackstersCLUE3DHigh"), edm::InputTag("ticlTracksterInterpretations")});
  desc.add<std::vector<std::string>>(
      "tableNames", {"ticlTrackstersCLUE3DHighToTruthBranch", "ticlTracksterInterpretationsToTruthBranch"});
  desc.add<std::vector<std::string>>("tracksterTableNames",
                                     {"ticlTrackstersCLUE3DHigh", "ticlTracksterInterpretations"});
  desc.add<std::string>("branchTableName", "TruthBranch");
  desc.add<bool>("computeLabels", true)
      ->setComment("Emit the hierarchical label extension tables (meaningful for leaf-level associations only).");
  desc.add<double>("labelPurityMin", 0.75)->setComment("Min purity (shared/raw) for a clean or ambiguous label.");
  desc.add<double>("contributorMinFraction", 0.1)
      ->setComment("Min fraction of the matched energy for a contributor to enter the ancestor search.");
  desc.add<double>("minSharedEnergy", 0.5)
      ->setComment("Min total matched energy [GeV]; below it the label is unknown.");
  descriptions.add("hgcalTruthBranchTables", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TracksterTruthBranchTableProducer);
