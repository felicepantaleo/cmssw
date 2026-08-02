// Minimal reader for the trackster-to-truth-branch association maps produced by
// AllTracksterToTruthBranchAssociatorsProducer: prints, per event, how many
// tracksters got a fixed-level and an adaptive-level match, with the branch key,
// shared energy and score. The maps are ticl::AssociationMap templates, which cannot
// be instantiated from bare FWLite/cppyy, so reading them needs compiled code.

#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

namespace {
  using BranchAssociationMap = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>;
}

class AdaptiveAssociationDumper : public edm::one::EDAnalyzer<> {
public:
  explicit AdaptiveAssociationDumper(edm::ParameterSet const&);
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void dump(edm::Event const&, edm::EDGetTokenT<BranchAssociationMap> const&, std::string const&) const;

  const edm::InputTag fixedTag_;
  const edm::InputTag adaptiveTag_;
  const edm::EDGetTokenT<BranchAssociationMap> fixedToken_;
  const edm::EDGetTokenT<BranchAssociationMap> adaptiveToken_;
};

AdaptiveAssociationDumper::AdaptiveAssociationDumper(edm::ParameterSet const& cfg)
    : fixedTag_(cfg.getParameter<edm::InputTag>("fixed")),
      adaptiveTag_(cfg.getParameter<edm::InputTag>("adaptive")),
      fixedToken_(consumes<BranchAssociationMap>(fixedTag_)),
      adaptiveToken_(consumes<BranchAssociationMap>(adaptiveTag_)) {}

void AdaptiveAssociationDumper::dump(edm::Event const& event,
                                     edm::EDGetTokenT<BranchAssociationMap> const& token,
                                     std::string const& what) const {
  edm::Handle<BranchAssociationMap> handle;
  event.getByToken(token, handle);
  if (!handle.isValid()) {
    edm::LogPrint("AdaptiveAssoc") << what << ": PRODUCT NOT FOUND";
    return;
  }
  auto const& map = handle->getMap();
  std::size_t matched = 0;
  for (auto const& entries : map)
    if (!entries.empty())
      ++matched;
  edm::LogPrint("AdaptiveAssoc") << what << ": " << map.size() << " tracksters, " << matched << " with >=1 match";
  for (std::size_t i = 0; i < map.size(); ++i) {
    for (auto const& e : map[i]) {
      edm::LogPrint("AdaptiveAssoc") << "    trackster " << i << " -> branch " << e.index()
                                     << "  sharedEnergy=" << e.sharedEnergy() << "  score=" << e.score();
    }
  }
}

void AdaptiveAssociationDumper::analyze(edm::Event const& event, edm::EventSetup const&) {
  edm::LogPrint("AdaptiveAssoc") << "=== event " << event.id().event() << " ===";
  dump(event, fixedToken_, "FIXED   [" + fixedTag_.instance() + "]");
  dump(event, adaptiveToken_, "ADAPTIVE[" + adaptiveTag_.instance() + "]");
}

void AdaptiveAssociationDumper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("fixed", edm::InputTag("tracksterToTruthBranch", "ticlTrackstersCLUE3DHighToTruthBranch"));
  desc.add<edm::InputTag>("adaptive",
                          edm::InputTag("tracksterToTruthBranch", "ticlTrackstersCLUE3DHighToTruthBranchAdaptive"));
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AdaptiveAssociationDumper);
