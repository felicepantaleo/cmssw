// Per-layer-cluster truth table for the adaptive-branch HGCAL validation: one row per
// reco layer cluster with its truth-branch match from the LC-to-branch association
// (AllLayerClustersToTruthBranchAssociatorsProducer). This uses the FIXED (leaf/
// antichain) association, not the adaptive one: a layer cluster is the finest calo
// unit and belongs to a single calo-crossing particle, so its truth is that particle's
// single-shower class (em/mip/hadronic/fake) - there is no shower merging at LC
// granularity. Consuming the adaptive map instead would hand back a merge/ancestor node
// (e.g. a pi0 or tau) whose pdgId does NOT map to a single-shower class. The LC is
// assigned the branch contributing the most shared energy. This gives the LC
// composition underneath a trackster and lets the framework validate layer clustering
// directly against the truth graph.
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

#include <vector>
#include <string>
#include <cstring>
#include <cstdint>

namespace {
  enum { kEM = 0, kMIP = 1, kHAD = 2, kFake = 5 };  // single-shower classes (no merging at LC level)

  bool isEMpdg(int pdg) {
    const int a = std::abs(pdg);
    return a == 11 || a == 22;
  }
  bool isMIPpdg(int pdg) { return std::abs(pdg) == 13; }
  bool isHadronPdg(int pdg) {
    const int a = std::abs(pdg);
    if (a < 100)
      return false;
    if (a >= 1000 && (a / 100) % 10 == 0)
      return false;  // diquarks
    return true;
  }
  int singleClass(int pdg) {
    if (isEMpdg(pdg))
      return kEM;
    if (isMIPpdg(pdg))
      return kMIP;
    if (isHadronPdg(pdg))
      return kHAD;
    return kFake;
  }
  bool isPrimary(truth::Particle const& p) {
    uint32_t raw = 0;
    const uint64_t packed = p.eventId();
    std::memcpy(&raw, &packed, sizeof(raw));
    const EncodedEventId id(raw);
    return id.bunchCrossing() == 0 && id.event() == 0;
  }
}  // namespace

class LayerClusterTruthFlatTableProducer : public edm::stream::EDProducer<> {
public:
  using BranchAssociationMap = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>;

  explicit LayerClusterTruthFlatTableProducer(edm::ParameterSet const& p)
      : name_(p.getParameter<std::string>("name")),
        lcToken_(consumes<std::vector<reco::CaloCluster>>(p.getParameter<edm::InputTag>("layerClusters"))),
        assocToken_(consumes<BranchAssociationMap>(p.getParameter<edm::InputTag>("association"))),
        graphToken_(consumes<truth::Graph>(p.getParameter<edm::InputTag>("graph"))) {
    produces<nanoaod::FlatTable>();
  }

  void produce(edm::Event& ev, edm::EventSetup const&) override {
    auto const& lcs = ev.get(lcToken_);
    auto const& assoc = ev.get(assocToken_);
    auto const& graph = ev.get(graphToken_);
    const unsigned n = lcs.size();

    std::vector<float> energy(n), eta(n), phi(n), truth_score(n), truth_shared(n);
    std::vector<int> n_hits(n), label(n), truth_pdg(n), is_primary(n);

    for (unsigned l = 0; l < n; ++l) {
      auto const& lc = lcs[l];
      energy[l] = lc.energy();
      eta[l] = lc.eta();
      phi[l] = lc.phi();
      n_hits[l] = static_cast<int>(lc.hitsAndFractions().size());

      // The LC belongs to the leaf branch contributing the most shared energy; every
      // fixed-map entry is a calo-crossing single particle (the antichain), so its
      // pdgId maps to a single-shower class.
      int bnode = -1;
      float bscore = -1.f, bshared = 0.f, mostShared = 0.f;
      if (l < assoc.size()) {
        for (auto const& el : assoc[l]) {
          if (el.sharedEnergy() > mostShared) {
            mostShared = el.sharedEnergy();
            bnode = static_cast<int>(el.index());
            bscore = el.score();
            bshared = el.sharedEnergy();
          }
        }
      }
      if (bnode >= 0 && static_cast<unsigned>(bnode) < graph.nParticles()) {
        auto const& bp = graph.particle(static_cast<uint32_t>(bnode));
        truth_pdg[l] = bp.pdgId();
        label[l] = singleClass(bp.pdgId());
        is_primary[l] = isPrimary(bp) ? 1 : 0;
        truth_score[l] = bscore;
        truth_shared[l] = bshared;
      } else {
        truth_pdg[l] = 0;
        label[l] = kFake;  // no truth branch: an unmatched (fake/noise) LC
        is_primary[l] = 0;
        truth_score[l] = -1.f;
        truth_shared[l] = 0.f;
      }
    }

    auto tab = std::make_unique<nanoaod::FlatTable>(n, name_, false, false);
    tab->addColumn<float>("energy", energy, "layer-cluster energy (GeV)");
    tab->addColumn<float>("eta", eta, "layer-cluster eta");
    tab->addColumn<float>("phi", phi, "layer-cluster phi");
    tab->addColumn<int>("n_hits", n_hits, "number of rechits in the layer cluster");
    tab->addColumn<int>("label", label,
                        "single-shower class of the matched leaf particle (0 em, 1 mip, 2 hadronic, 5 fake)");
    tab->addColumn<int>("truth_pdg", truth_pdg, "pdgId of the matched leaf particle (0 = unmatched)");
    tab->addColumn<float>("truth_score", truth_score, "match score to the leaf particle; -1 = unmatched");
    tab->addColumn<float>("truth_sharedE", truth_shared, "shared energy with the matched leaf particle (GeV)");
    tab->addColumn<int>("is_primary", is_primary, "1 if the matched particle is hard-scatter, 0 if pileup/unmatched");
    ev.put(std::move(tab));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& d) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("name", "LayerClusterTruth");
    desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
    desc.add<edm::InputTag>("association",
                            edm::InputTag("allLayerClustersToTruthBranchAssociations",
                                          "hgcalMergeLayerClustersToTruthBranch"));
    desc.add<edm::InputTag>("graph", edm::InputTag("truthLogicalGraphProducer"));
    d.addWithDefaultLabel(desc);
  }

private:
  const std::string name_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const edm::EDGetTokenT<BranchAssociationMap> assocToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
};

DEFINE_FWK_MODULE(LayerClusterTruthFlatTableProducer);
