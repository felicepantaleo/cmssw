// Sim-side flat table for the adaptive-branch HGCAL validation: one row per truth
// branch (branchSimTracksters), with its kinematics and, from the logical-graph hit
// index, whether a single reco trackster captured it.
//
// Efficiency criterion (deposited-sim-energy fraction): the branch's deposited sim
// energy is the sum of the energy of the RECHITS associated to the branch's sim hits
// in HGCAL. The LogicalGraphHitIndex gives the branch subtree's sim-hit cells
// (subgraphHits); the rechit energy per cell comes from the HGCAL rechit collections
// (the same reco detId space). Cells whose sim hit has no associated rechit contribute
// nothing. A reco trackster's shared energy with the branch is that branch rechit
// energy summed over the cells the trackster reconstructs (weighted by the
// layer-cluster cell fraction). The branch is FOUND if a single reco trackster shares
// at least minSharedFraction (0.5) of the branch's deposited energy. This uses the
// branch's own rechit energy, not the association's cellTotalEnergy (which sums all
// sim contributors and is pileup-inflated).
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "PhysicsTools/TruthInfo/interface/RecoHitAdapters.h"

#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <limits>
#include <unordered_map>

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
  explicit BranchSimTracksterFlatTableProducer(edm::ParameterSet const& p)
      : name_(p.getParameter<std::string>("name")),
        branchToken_(consumes<std::vector<ticl::Trackster>>(p.getParameter<edm::InputTag>("branches"))),
        levelToken_(consumes<std::vector<int>>(p.getParameter<edm::InputTag>("level"))),
        rootIdToken_(consumes<std::vector<int>>(p.getParameter<edm::InputTag>("rootId"))),
        pdgIdToken_(consumes<std::vector<int>>(p.getParameter<edm::InputTag>("pdgId"))),
        recoToken_(consumes<std::vector<ticl::Trackster>>(p.getParameter<edm::InputTag>("recoCollection"))),
        layerClustersToken_(
            consumes<std::vector<reco::CaloCluster>>(p.getParameter<edm::InputTag>("layerClusters"))),
        hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(p.getParameter<edm::InputTag>("hitIndex"))),
        graphToken_(consumes<truth::Graph>(p.getParameter<edm::InputTag>("graph"))),
        minSharedFraction_(p.getParameter<double>("minSharedFraction")),
        minContribFraction_(p.getParameter<double>("minContribFraction")) {
    for (auto const& tag : p.getParameter<std::vector<edm::InputTag>>("recHits"))
      recHitTokens_.push_back(consumes<HGCRecHitCollection>(tag));
    produces<nanoaod::FlatTable>();
  }

  void produce(edm::Event& ev, edm::EventSetup const&) override {
    auto const& br = ev.get(branchToken_);
    auto const& lvl = ev.get(levelToken_);
    auto const& rid = ev.get(rootIdToken_);
    auto const& pdg = ev.get(pdgIdToken_);
    auto const& reco = ev.get(recoToken_);
    auto const& lcs = ev.get(layerClustersToken_);
    auto const& hitIndex = ev.get(hitIndexToken_);
    auto const& graph = ev.get(graphToken_);
    const unsigned n = br.size();
    if (lvl.size() != n || rid.size() != n || pdg.size() != n)
      throw cms::Exception("SizeMismatch")
          << "branchSimTracksters aligned vectors disagree with the branch collection: "
          << "branches=" << n << " level=" << lvl.size() << " rootId=" << rid.size() << " pdgId=" << pdg.size();

    // Rechit energy per reco detId, from the HGCAL rechit collections (same detId
    // space as the branch hits and the layer clusters). This is the "energy of the
    // rechits associated to the sim hits": a branch cell with no rechit is absent here
    // and contributes nothing to the deposit.
    std::unordered_map<uint32_t, float> rechitEnergy;
    for (auto const& token : recHitTokens_) {
      auto const& hits = ev.get(token);
      rechitEnergy.reserve(rechitEnergy.size() + hits.size());
      for (auto const& rh : hits)
        rechitEnergy[rh.detid().rawId()] = rh.energy();
    }

    // Invert the reco tracksters into detId -> (reco index, cell fraction). A cell can
    // be shared across tracksters through layer-cluster fractions, so keep a list.
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>> cellToReco;
    for (unsigned t = 0; t < reco.size(); ++t)
      for (auto const& h : truth::recoHits(reco[t], lcs))
        cellToReco[h.detId].emplace_back(t, h.fraction);

    std::vector<float> deposit(n), gen_energy(n), eta(n), pt(n), shared_frac(n), reco_energy(n), reco_regressed(n);
    std::vector<int> pdgId(n), level(n), is_primary(n), is_found(n), n_contrib(n);

    std::unordered_map<uint32_t, double> sharedByReco;  // reused per branch
    for (unsigned i = 0; i < n; ++i) {
      auto const& t = br[i];
      gen_energy[i] = t.regressed_energy();
      eta[i] = t.barycenter().eta();
      pt[i] = t.raw_pt();
      pdgId[i] = pdg[i];
      level[i] = lvl[i];
      const int node = rid[i];  // graph particle index == hit-index particle id
      is_primary[i] = (node >= 0 && static_cast<unsigned>(node) < graph.nParticles() &&
                       branchIsPrimary(graph.particle(static_cast<uint32_t>(node))))
                          ? 1
                          : 0;

      // Branch deposited energy = sum over the branch subtree cells of the associated
      // rechit energy (cells with no rechit are absent from rechitEnergy). Shared
      // energy per reco = that rechit energy on the cells the reco reconstructs,
      // cell-fraction weighted.
      double dep = 0., max_shared = 0.;
      float re = 0.f, rreg = 0.f;
      int nc = 0;
      sharedByReco.clear();
      if (node >= 0 && static_cast<unsigned>(node) < hitIndex.nParticles()) {
        for (auto const& h : hitIndex.subgraphHits(truth::HitChannel::Calo, static_cast<uint32_t>(node))) {
          auto reh = rechitEnergy.find(h.detId);
          if (reh == rechitEnergy.end())
            continue;  // sim hit with no associated rechit: not deposited rechit energy
          const double e = reh->second;
          dep += e;
          auto it = cellToReco.find(h.detId);
          if (it != cellToReco.end())
            for (auto const& [recoIdx, frac] : it->second)
              sharedByReco[recoIdx] += e * frac;
        }
        uint32_t best = std::numeric_limits<uint32_t>::max();
        for (auto const& [recoIdx, sh] : sharedByReco) {
          if (dep > 0. && sh >= minContribFraction_ * dep)
            ++nc;
          // strict-greater with a lowest-index tie-break, so the choice does not
          // depend on unordered_map iteration order.
          if (sh > max_shared || (sh == max_shared && recoIdx < best)) {
            max_shared = sh;
            best = recoIdx;
          }
        }
        if (max_shared > 0. && best < reco.size()) {
          re = reco[best].raw_energy();
          rreg = reco[best].regressed_energy();
        }
      }
      deposit[i] = dep;
      shared_frac[i] = (dep > 0.) ? static_cast<float>(max_shared / dep) : 0.f;
      is_found[i] = (dep > 0. && max_shared >= minSharedFraction_ * dep) ? 1 : 0;
      n_contrib[i] = nc;
      reco_energy[i] = re;
      reco_regressed[i] = rreg;
    }

    auto tab = std::make_unique<nanoaod::FlatTable>(n, name_, false, false);
    tab->addColumn<float>("deposit", deposit, "branch deposited energy (sum of the rechits associated to its sim hits, GeV)");
    tab->addColumn<float>("gen_energy", gen_energy, "branch gen (momentum) energy (GeV)");
    tab->addColumn<float>("eta", eta, "branch barycenter eta");
    tab->addColumn<float>("pt", pt, "branch raw pt (GeV)");
    tab->addColumn<int>("pdgId", pdgId, "branch pdgId");
    tab->addColumn<int>("level", level, "branch level in the truth graph (0 = leaf)");
    tab->addColumn<int>("is_primary", is_primary, "1 if the branch root is a hard-scatter primary");
    tab->addColumn<int>("is_found", is_found,
                        "1 if a single reco trackster shares >= minSharedFraction of the branch deposited energy");
    tab->addColumn<float>("shared_frac", shared_frac,
                          "fraction of branch deposited energy captured by the dominant reco trackster");
    tab->addColumn<int>("n_contrib", n_contrib,
                        "reco tracksters each sharing >= minContribFraction of the branch deposited energy");
    tab->addColumn<float>("reco_energy_best", reco_energy, "raw energy of the dominant reco trackster (GeV)");
    tab->addColumn<float>("reco_regressed_best", reco_regressed,
                          "regressed (calibrated) energy of the dominant reco trackster (GeV)");
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
    desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
    desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
    desc.add<edm::InputTag>("graph", edm::InputTag("truthLogicalGraphProducer"));
    desc.add<std::vector<edm::InputTag>>("recHits",
                                         {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
    desc.add<double>("minSharedFraction", 0.5);
    desc.add<double>("minContribFraction", 0.1);
    d.addWithDefaultLabel(desc);
  }

private:
  const std::string name_;
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> branchToken_;
  const edm::EDGetTokenT<std::vector<int>> levelToken_, rootIdToken_, pdgIdToken_;
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> recoToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClustersToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> recHitTokens_;
  const double minSharedFraction_, minContribFraction_;
};

DEFINE_FWK_MODULE(BranchSimTracksterFlatTableProducer);
