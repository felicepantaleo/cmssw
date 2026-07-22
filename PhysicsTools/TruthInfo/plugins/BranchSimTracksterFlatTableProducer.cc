// Sim-side flat table for the adaptive-branch HGCAL validation: one row per truth
// branch (branchSimTracksters), with its kinematics and its best reco-trackster match
// from the reverse adaptive association (TruthBranchTo<coll>Adaptive). This is the
// efficiency/duplicate denominator+numerator that the reco-side feature table lacks:
// efficiency = matched branches / selected branches, duplicate = branches matched >1.
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

#include <vector>
#include <string>
#include <cstring>

namespace {
  // hard-scatter primary: eventId bunchCrossing == 0 and event == 0 (matches the
  // convention in TracksterFeatureFlatTableProducer).
  bool branchIsPrimary(truth::Particle const& p) {
    uint32_t raw = 0;
    const uint64_t packed = p.eventId();
    std::memcpy(&raw, &packed, sizeof(raw));
    const EncodedEventId id(raw);
    return id.bunchCrossing() == 0 && id.event() == 0;
  }
}  // namespace

class BranchSimTracksterFlatTableProducer : public edm::stream::EDProducer<> {
public:
  using BranchAssociationMap = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>;

  explicit BranchSimTracksterFlatTableProducer(edm::ParameterSet const& p)
      : name_(p.getParameter<std::string>("name")),
        branchToken_(consumes<std::vector<ticl::Trackster>>(p.getParameter<edm::InputTag>("branches"))),
        levelToken_(consumes<std::vector<int>>(p.getParameter<edm::InputTag>("level"))),
        rootIdToken_(consumes<std::vector<int>>(p.getParameter<edm::InputTag>("rootId"))),
        pdgIdToken_(consumes<std::vector<int>>(p.getParameter<edm::InputTag>("pdgId"))),
        recoToken_(consumes<std::vector<ticl::Trackster>>(p.getParameter<edm::InputTag>("recoCollection"))),
        revAssocToken_(consumes<BranchAssociationMap>(p.getParameter<edm::InputTag>("reverseAssociation"))),
        graphToken_(consumes<truth::Graph>(p.getParameter<edm::InputTag>("graph"))),
        minSharedEnergy_(p.getParameter<double>("minSharedEnergy")),
        maxScore_(p.getParameter<double>("maxScore")) {
    produces<nanoaod::FlatTable>();
  }

  void produce(edm::Event& ev, edm::EventSetup const&) override {
    auto const& br = ev.get(branchToken_);
    auto const& lvl = ev.get(levelToken_);
    auto const& rid = ev.get(rootIdToken_);
    auto const& pdg = ev.get(pdgIdToken_);
    auto const& reco = ev.get(recoToken_);
    auto const& rev = ev.get(revAssocToken_);
    auto const& graph = ev.get(graphToken_);
    const unsigned n = br.size();

    std::vector<float> energy(n), gen_energy(n), eta(n), pt(n), best_score(n), best_shared(n), reco_energy(n);
    std::vector<int> pdgId(n), level(n), n_matched(n), is_primary(n);

    for (unsigned i = 0; i < n; ++i) {
      auto const& t = br[i];
      energy[i] = t.raw_energy();
      gen_energy[i] = t.regressed_energy();
      eta[i] = t.barycenter().eta();
      pt[i] = t.raw_pt();
      pdgId[i] = pdg[i];
      level[i] = lvl[i];
      const int node = rid[i];  // graph particle index == reverse-association key
      is_primary[i] = (node >= 0 && static_cast<unsigned>(node) < graph.nParticles() &&
                       branchIsPrimary(graph.particle(static_cast<uint32_t>(node))))
                          ? 1
                          : 0;
      float bs = 2.f, bsh = 0.f, re = 0.f;
      int nm = 0;
      if (node >= 0 && node < static_cast<int>(rev.size())) {
        for (auto const& el : rev[node]) {
          if (el.sharedEnergy() >= minSharedEnergy_ && el.score() <= maxScore_) {
            ++nm;
            if (el.score() < bs) {
              bs = el.score();
              bsh = el.sharedEnergy();
              re = (el.index() < reco.size()) ? reco[el.index()].raw_energy() : 0.f;
            }
          }
        }
      }
      n_matched[i] = nm;
      best_score[i] = (bs <= 1.f) ? bs : -1.f;
      best_shared[i] = bsh;
      reco_energy[i] = re;
    }

    auto tab = std::make_unique<nanoaod::FlatTable>(n, name_, false, false);
    tab->addColumn<float>("energy", energy, "branch shared HGCAL energy (GeV)");
    tab->addColumn<float>("gen_energy", gen_energy, "branch gen (momentum) energy (GeV)");
    tab->addColumn<float>("eta", eta, "branch barycenter eta");
    tab->addColumn<float>("pt", pt, "branch raw pt (GeV)");
    tab->addColumn<int>("pdgId", pdgId, "branch pdgId");
    tab->addColumn<int>("level", level, "branch level in the truth graph (0 = leaf)");
    tab->addColumn<int>("is_primary", is_primary, "1 if the branch root is a hard-scatter primary");
    tab->addColumn<int>("n_matched", n_matched, "reco tracksters matched (sharedE>=min, score<=max)");
    tab->addColumn<float>("best_score", best_score, "best (lowest) match score; -1 = unmatched");
    tab->addColumn<float>("best_sharedE", best_shared, "shared energy of the best match");
    tab->addColumn<float>("reco_energy_best", reco_energy, "energy of the best-matched reco trackster (response)");
    ev.put(std::move(tab));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& d) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("name", "SimBranch");
    desc.add<edm::InputTag>("branches", edm::InputTag("branchSimTracksters"));
    desc.add<edm::InputTag>("level", edm::InputTag("branchSimTracksters", "level"));
    desc.add<edm::InputTag>("rootId", edm::InputTag("branchSimTracksters", "rootId"));
    desc.add<edm::InputTag>("pdgId", edm::InputTag("branchSimTracksters", "pdgId"));
    desc.add<edm::InputTag>("recoCollection", edm::InputTag("ticlTrackstersCLUE3DHigh"));
    desc.add<edm::InputTag>("reverseAssociation",
                            edm::InputTag("allTrackstersToTruthBranchAssociations",
                                          "TruthBranchToticlTrackstersCLUE3DHighAdaptive"));
    desc.add<edm::InputTag>("graph", edm::InputTag("truthLogicalGraphProducer"));
    desc.add<double>("minSharedEnergy", 0.5);
    desc.add<double>("maxScore", 0.75);
    d.addWithDefaultLabel(desc);
  }

private:
  const std::string name_;
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> branchToken_;
  const edm::EDGetTokenT<std::vector<int>> levelToken_, rootIdToken_, pdgIdToken_;
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> recoToken_;
  const edm::EDGetTokenT<BranchAssociationMap> revAssocToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const double minSharedEnergy_, maxScore_;
};

DEFINE_FWK_MODULE(BranchSimTracksterFlatTableProducer);
