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
  std::vector<std::string> tableNames_;
  const std::string branchTableName_;
};

TracksterTruthBranchTableProducer::TracksterTruthBranchTableProducer(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      branchTableName_(cfg.getParameter<std::string>("branchTableName")) {
  auto tags = cfg.getParameter<std::vector<edm::InputTag>>("associations");
  auto names = cfg.getParameter<std::vector<std::string>>("tableNames");
  if (tags.size() != names.size()) {
    throw cms::Exception("Configuration") << "associations and tableNames must have the same length";
  }
  for (size_t i = 0; i < tags.size(); ++i) {
    mapTokens_.emplace_back(consumes<BranchAssociationMap>(tags[i]));
    tableNames_.push_back(names[i]);
    produces<nanoaod::FlatTable>(names[i]);
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
  }
}

void TracksterTruthBranchTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<std::vector<edm::InputTag>>(
      "associations",
      {edm::InputTag("allTrackstersToTruthBranchAssociations", "ticlTrackstersCLUE3DHighToTruthBranch"),
       edm::InputTag("allTrackstersToTruthBranchAssociations", "ticlTracksterInterpretationsToTruthBranch")});
  desc.add<std::vector<std::string>>("tableNames",
                                     {"ticlTrackstersCLUE3DHighToTruthBranch",
                                      "ticlTracksterInterpretationsToTruthBranch"});
  desc.add<std::string>("branchTableName", "TruthBranch");
  descriptions.add("hgcalTruthBranchTables", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TracksterTruthBranchTableProducer);
