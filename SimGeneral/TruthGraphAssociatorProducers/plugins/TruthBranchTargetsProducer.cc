// The truth-side targets that every associator and validator consumes: the candidate
// roots, the subset that a reco object may be assigned to, the signal-seed denominators,
// and one TruthToReco denominator per graph level with its eligibility mask. They depend
// only on the graph and the selection configuration, so all consumers share one copy.

#include <cctype>
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/TruthInfo/interface/AssignableTarget.h"
#include "PhysicsTools/TruthInfo/interface/Branch.h"
#include "PhysicsTools/TruthInfo/interface/BranchSelector.h"
#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"

class TruthBranchTargetsProducer : public edm::global::EDProducer<> {
public:
  explicit TruthBranchTargetsProducer(edm::ParameterSet const&);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  truth::BranchSelector branchSelector_;
  // (level, product instance) pairs, instance = "truthToRecoTargets" + capitalized name.
  std::vector<std::pair<truth::Level, std::string>> truthLevels_;
  truth::AssignableTargetConfig assignableConfig_;
  const std::vector<int> signalSeedPdgIds_;
  const std::vector<int> signalSeedHadronFlavors_;
  const bool truthToRecoSignalOnly_;
};

TruthBranchTargetsProducer::TruthBranchTargetsProducer(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      signalSeedPdgIds_(cfg.getParameter<std::vector<int>>("signalSeedPdgIds")),
      signalSeedHadronFlavors_(cfg.getParameter<std::vector<int>>("signalSeedHadronFlavors")),
      truthToRecoSignalOnly_(cfg.getParameter<bool>("truthToRecoSignalOnly")) {
  {
    // Restrict the candidate branches, as CaloParticleSelector and the TrackingParticle
    // selectors do, so soft particles that no reconstruction finds do not dominate.
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

  {
    auto const& assignable = cfg.getParameter<edm::ParameterSet>("assignableTargets");
    assignableConfig_.excludeSynthetic = assignable.getParameter<bool>("excludeSynthetic");
    assignableConfig_.excludeBeamParticles = assignable.getParameter<bool>("excludeBeamParticles");
    assignableConfig_.excludePartons = assignable.getParameter<bool>("excludePartons");
    assignableConfig_.excludeElectroweakBosons = assignable.getParameter<bool>("excludeElectroweakBosons");
    for (const int pdgId : assignable.getParameter<std::vector<int>>("extraBarredPdgIds")) {
      assignableConfig_.extraBarredPdgIds.push_back(static_cast<int32_t>(std::abs(pdgId)));
    }
  }

  // The associators' candidate roots. NOT an efficiency denominator: the set can hold a
  // particle together with its own ancestor, so it is not an antichain.
  produces<std::vector<unsigned int>>("selectedRoots");
  // The subset of selectedRoots that a reco object may be assigned to: the detector
  // particles. The barred roots stay candidates, because they are members of the
  // hard-process and parton-jet denominators.
  produces<std::vector<unsigned int>>("assignableRoots");
  produces<std::vector<unsigned int>>("signalSeeds");
  // The signal seeds without the selector cuts. The two denominators separate "not
  // reconstructed" from "not selected": on 200 no-PU ttbar events the selector keeps 390
  // of the 400 tops.
  produces<std::vector<unsigned int>>("signalSeedsNoSelection");
  // One denominator per level, "truthToRecoTargets" + the capitalized level name.
  for (auto const& name : cfg.getParameter<std::vector<std::string>>("truthLevels")) {
    if (name.empty()) {
      throw cms::Exception("Configuration") << "empty entry in truthLevels";
    }
    std::string capitalized = name;
    capitalized[0] = std::toupper(static_cast<unsigned char>(capitalized[0]));
    truthLevels_.emplace_back(truth::levelFromName(name), "truthToRecoTargets" + capitalized);
    produces<std::vector<unsigned int>>(truthLevels_.back().second);
    // Parallel to the denominator: the mask of plotted-axis cuts that each target fails.
    produces<std::vector<unsigned int>>(truthLevels_.back().second + "Eligibility");
  }
}

void TruthBranchTargetsProducer::produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const {
  auto const& graph = event.get(graphToken_);
  const unsigned int nBranches = graph.nParticles();

  // Selected candidate roots. When the selection accepts nothing, there is no candidate.
  auto selectedRoots = std::make_unique<std::vector<unsigned int>>();
  selectedRoots->reserve(nBranches);
  std::vector<bool> isCandidate(nBranches, false);
  for (uint32_t id = 0; id < nBranches; ++id) {
    // Skip shower bookkeeping (partons, diquarks, strings): on one ttbar event that is
    // 176 partons and 2 diquarks. partonJets and hardProcess add their own members below.
    // The top decays before it hadronizes, so it stays a candidate.
    if (truth::hadronizes(graph.particles()[id].pdgId)) {
      continue;
    }
    if (branchSelector_(truth::Branch(&graph, id))) {
      selectedRoots->push_back(id);
      isCandidate[id] = true;
    }
  }

  // The preset seed objects: with a tau preset, the taus and not their decay legs.
  {
    // The species alone is not enough: it also occurs in pileup, among Geant4
    // secondaries, and along a heavy-flavour chain (B**, B*, B). LevelFlag::Signal marks
    // the most upstream seed-species particle of the signal interaction.
    auto const isSignalSeed = [this, &graph](uint32_t id) {
      if (!graph.particles()[id].isAtLevel(truth::LevelFlag::Signal)) {
        return false;
      }
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
    auto signalSeedsNoSelection = std::make_unique<std::vector<unsigned int>>();
    // NoSelection drops the selector, not the signal requirement. Without seed species,
    // or without a Signal flag, both products are empty. The selected roots are not a
    // substitute: they are not an antichain (on QCD, 518.89 per event against 164
    // generator-stable particles).
    if (truth::seedsNameAResonance(signalSeedPdgIds_, signalSeedHadronFlavors_)) {
      for (uint32_t id : *selectedRoots) {
        if (isSignalSeed(id)) {
          signalSeeds->push_back(id);
        }
      }
      for (uint32_t id = 0; id < nBranches; ++id) {
        if (isSignalSeed(id)) {
          signalSeedsNoSelection->push_back(id);
        }
      }
    }
    event.put(std::move(signalSeeds), "signalSeeds");
    event.put(std::move(signalSeedsNoSelection), "signalSeedsNoSelection");
  }

  // One denominator per level: the level antichain, then the selector, then the signal
  // restriction. The antichain comes first: an antichain of a selected set promotes a
  // soft particle whose parent fails the pt cut.
  std::vector<unsigned int> extraCandidates;
  for (auto const& [level, instance] : truthLevels_) {
    auto targets = std::make_unique<std::vector<unsigned int>>();
    // Parallel to targets: the mask of plotted-axis cuts that each target fails. A target
    // that fails only the pt cut is kept and enters the pt plot only.
    auto eligibility = std::make_unique<std::vector<unsigned int>>();
    for (uint32_t id : truth::levelAntichain(graph, level)) {
      const truth::Branch branch(&graph, id);
      if (!branchSelector_.passesNonKinematic(branch)) {
        continue;
      }
      // A branch that fails more than one kinematic cut enters no plot, so drop it here.
      const uint32_t failed = branchSelector_.failedKinematicCuts(branch);
      if ((failed & (failed - 1u)) != 0u) {
        continue;
      }
      if (truthToRecoSignalOnly_ && !graph.particles()[id].isSignal()) {
        continue;
      }
      if (!isCandidate[id]) {
        isCandidate[id] = true;
        extraCandidates.push_back(id);
      }
      targets->push_back(id);
      eligibility->push_back(failed);
    }
    event.put(std::move(targets), instance);
    event.put(std::move(eligibility), instance + "Eligibility");
  }

  // Every target must be a candidate, or it can never be matched. Only the emitted
  // targets join, not every particle that fails one cut: that costs 128% more time per
  // PU200 ttbar event in the track associator alone.
  selectedRoots->insert(selectedRoots->end(), extraCandidates.begin(), extraCandidates.end());
  std::sort(selectedRoots->begin(), selectedRoots->end());

  auto assignableRoots = std::make_unique<std::vector<unsigned int>>();
  assignableRoots->reserve(selectedRoots->size());
  for (const unsigned int id : *selectedRoots) {
    if (truth::isAssignableTarget(graph, id, assignableConfig_)) {
      assignableRoots->push_back(id);
    }
  }

  event.put(std::move(selectedRoots), "selectedRoots");
  event.put(std::move(assignableRoots), "assignableRoots");
}

void TruthBranchTargetsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));

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

  // Which candidate roots a reco object may be assigned to. Each clause bars one class of
  // particle that no detector sees.
  edm::ParameterSetDescription assignable;
  assignable.add<bool>("excludeSynthetic", true)->setComment("Bar the connector and signal stand-in nodes");
  assignable.add<bool>("excludeBeamParticles", true)->setComment("Bar a particle with no production vertex");
  assignable.add<bool>("excludePartons", true)->setComment("Bar quarks and gluons");
  assignable.add<bool>("excludeElectroweakBosons", true)->setComment("Bar the W, the Z and the Higgs");
  assignable.add<std::vector<int>>("extraBarredPdgIds", {})
      ->setComment("Further species to bar, matched on the absolute value");
  desc.add<edm::ParameterSetDescription>("assignableTargets", assignable);

  desc.add<std::vector<std::string>>("truthLevels", {"caloBoundary"})
      ->setComment("Graph levels to emit a TruthToReco denominator for, one product per level");
  desc.add<std::vector<int>>("signalSeedPdgIds", {})
      ->setComment("The selection preset's seed species; empty or {0} means no resonance and empty signal products");
  desc.add<std::vector<int>>("signalSeedHadronFlavors", {})
      ->setComment("Heavy-flavour hadron seeds; flavours alone also name a resonance");
  desc.add<bool>("truthToRecoSignalOnly", true)
      ->setComment("Restrict the level denominators to the signal interaction");
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TruthBranchTargetsProducer);
