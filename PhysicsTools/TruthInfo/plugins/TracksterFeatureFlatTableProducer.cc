// NanoAOD feature dumper for the trackster-transformer PID training set.
//
// Emits two FlatTables per event:
//   <name>       (one row per trackster): the truth-branch label + global features
//                (barycenter R/z/eta, log raw energy, n_lc, PCA), plus the offset and
//                count into the layer-cluster table so a reader can gather a trackster's
//                layer clusters.
//   <name>LC     (one row per (trackster, layer cluster)): the barycenter-frame
//                (u, v, layer, energy, size) that the ttpid feature builder smears into
//                the per-layer grid. u = R - R_bary, v = R_bary * wrap(phi - phi_bary),
//                layer = RecHitTools per-endcap layer (both endcaps map identically;
//                the signed z/eta in the trackster row carry the endcap).
//
// Label comes from the trackster -> truth-branch association: best branch pdgId ->
// class; a second branch sharing above ambiguousFraction of the best -> ambiguous; no
// branch above minSharedEnergy -> unknown (fake). Class ids match ttpid/config.py.
#include <cmath>
#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"

namespace {
  // |pdgId| -> ttpid class. ambiguous(8)/unknown(9) come from the association, not pdg.
  int classFromPdg(int pdg) {
    switch (std::abs(pdg)) {
      case 22: return 0;    // photon
      case 11: return 1;    // electron
      case 13: return 2;    // muon
      case 111: return 3;   // neutral_pion
      case 211: return 4;   // charged_pion
      case 321: return 5;   // charged_kaon
      case 2212: return 6;  // proton
      case 130:
      case 310:
      case 2112: return 7;  // neutral_hadron (K0L, K0S, neutron)
      default: return 9;    // unknown
    }
  }
  constexpr int kAmbiguous = 8;
  constexpr int kUnknown = 9;

  float wrapPi(float d) {
    while (d > M_PI)
      d -= 2.f * M_PI;
    while (d <= -M_PI)
      d += 2.f * M_PI;
    return d;
  }
}  // namespace

class TracksterFeatureFlatTableProducer : public edm::stream::EDProducer<edm::stream::WatchRuns> {
public:
  using BranchAssociationMap = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>;

  explicit TracksterFeatureFlatTableProducer(edm::ParameterSet const& p)
      : name_(p.getParameter<std::string>("name")),
        tracksterToken_(consumes<std::vector<ticl::Trackster>>(p.getParameter<edm::InputTag>("tracksters"))),
        lcToken_(consumes<std::vector<reco::CaloCluster>>(p.getParameter<edm::InputTag>("layerClusters"))),
        assocToken_(consumes<BranchAssociationMap>(p.getParameter<edm::InputTag>("association"))),
        graphToken_(consumes<truth::Graph>(p.getParameter<edm::InputTag>("graph"))),
        geomToken_(esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>()),
        minSharedEnergy_(p.getParameter<double>("minSharedEnergy")),
        ambiguousFraction_(p.getParameter<double>("ambiguousFraction")) {
    produces<nanoaod::FlatTable>();          // per-trackster
    produces<nanoaod::FlatTable>("LC");      // per layer cluster
  }

  void beginRun(edm::Run const&, edm::EventSetup const& es) override { rhtools_.setGeometry(es.getData(geomToken_)); }
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

  void produce(edm::Event& ev, edm::EventSetup const&) override {
    auto const& tracksters = ev.get(tracksterToken_);
    auto const& lcs = ev.get(lcToken_);
    auto const& assoc = ev.get(assocToken_);
    auto const& graph = ev.get(graphToken_);
    const unsigned nP = graph.nParticles();

    // per-trackster
    std::vector<int> label, n_lc, lc_offset;
    std::vector<float> R_bary, z_bary, eta_bary, log_e, best_shared;
    std::vector<int> best_pdg;
    // per-LC
    std::vector<int> lc_trkIdx, lc_layer;
    std::vector<float> lc_u, lc_v, lc_energy, lc_size;

    for (unsigned t = 0; t < tracksters.size(); ++t) {
      auto const& trk = tracksters[t];
      const auto b = trk.barycenter();
      const float Rb = std::hypot(b.x(), b.y());
      const float phib = std::atan2(b.y(), b.x());

      // label from the best-matching truth branch
      float be = -1.f, se = -1.f;
      int bidx = -1;
      if (t < assoc.size()) {
        for (auto const& el : assoc[t]) {
          const float e = el.sharedEnergy();
          if (e > be) {
            se = be;
            be = e;
            bidx = static_cast<int>(el.index());
          } else if (e > se) {
            se = e;
          }
        }
      }
      int lab, pdg = 0;
      if (be < minSharedEnergy_) {
        lab = kUnknown;
      } else if (se > ambiguousFraction_ * be) {
        lab = kAmbiguous;
      } else {
        pdg = (bidx >= 0 && bidx < static_cast<int>(nP)) ? graph.particles()[bidx].pdgId : 0;
        lab = classFromPdg(pdg);
      }

      lc_offset.push_back(static_cast<int>(lc_trkIdx.size()));
      for (unsigned iv = 0; iv < trk.vertices().size(); ++iv) {
        const unsigned li = trk.vertices()[iv];
        if (li >= lcs.size())
          continue;
        auto const& lc = lcs[li];
        const float R = std::hypot(lc.x(), lc.y());
        const float phi = std::atan2(lc.y(), lc.x());
        // energy-weighted transverse RMS of the LC's rechits = its "size" (cm)
        float sw = 0.f, swr2 = 0.f;
        for (auto const& hf : lc.hitsAndFractions()) {
          const auto pos = rhtools_.getPosition(hf.first);
          const float dx = pos.x() - lc.x(), dy = pos.y() - lc.y();
          sw += hf.second;
          swr2 += hf.second * (dx * dx + dy * dy);
        }
        const float size = (sw > 0.f) ? std::sqrt(swr2 / sw) : 0.f;
        lc_trkIdx.push_back(static_cast<int>(label.size()));
        lc_u.push_back(R - Rb);
        lc_v.push_back(Rb * wrapPi(phi - phib));
        lc_layer.push_back(static_cast<int>(rhtools_.getLayerWithOffset(lc.seed())) - 1);
        lc_energy.push_back(lc.energy());
        lc_size.push_back(size);
      }

      label.push_back(lab);
      best_pdg.push_back(pdg);
      best_shared.push_back(be > 0 ? be : 0.f);
      n_lc.push_back(static_cast<int>(lc_trkIdx.size()) - lc_offset.back());
      R_bary.push_back(Rb);
      z_bary.push_back(b.z());
      eta_bary.push_back(trk.barycenter().eta());
      log_e.push_back(std::log1p(trk.raw_energy()));
    }

    auto trkTab = std::make_unique<nanoaod::FlatTable>(label.size(), name_, false, false);
    trkTab->addColumn<int>("label", label, "ttpid class (0..9); 8=ambiguous, 9=unknown/fake");
    trkTab->addColumn<int>("best_pdg", best_pdg, "pdgId of the best-matching truth branch");
    trkTab->addColumn<float>("best_sharedE", best_shared, "shared energy with the best branch");
    trkTab->addColumn<int>("n_lc", n_lc, "number of layer clusters");
    trkTab->addColumn<int>("lc_offset", lc_offset, "offset into the LC table");
    trkTab->addColumn<float>("R_bary", R_bary, "barycenter cylindrical radius (cm)");
    trkTab->addColumn<float>("z_bary", z_bary, "barycenter z (cm, signed = endcap)");
    trkTab->addColumn<float>("eta_bary", eta_bary, "barycenter eta");
    trkTab->addColumn<float>("log_raw_energy", log_e, "log1p(raw energy)");

    auto lcTab = std::make_unique<nanoaod::FlatTable>(lc_trkIdx.size(), name_ + "LC", false, false);
    lcTab->addColumn<int>("trkIdx", lc_trkIdx, "index into the trackster table");
    lcTab->addColumn<float>("u", lc_u, "R - R_bary (cm)");
    lcTab->addColumn<float>("v", lc_v, "R_bary * wrap(phi - phi_bary) (cm)");
    lcTab->addColumn<int>("layer", lc_layer, "per-endcap layer index (0-based)");
    lcTab->addColumn<float>("energy", lc_energy, "LC energy");
    lcTab->addColumn<float>("size", lc_size, "LC transverse RMS (cm)");

    ev.put(std::move(trkTab));
    ev.put(std::move(lcTab), "LC");
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("name", "TICLTrackster");
    desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersCLUE3DHigh"));
    desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
    desc.add<edm::InputTag>("association",
                            edm::InputTag("allTrackstersToTruthBranchAssociations",
                                          "ticlTrackstersCLUE3DHighToTruthBranch"));
    desc.add<edm::InputTag>("graph", edm::InputTag("truthLogicalGraphProducer"));
    desc.add<double>("minSharedEnergy", 0.5);
    desc.add<double>("ambiguousFraction", 0.5);
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const std::string name_;
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksterToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const edm::EDGetTokenT<BranchAssociationMap> assocToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  const double minSharedEnergy_;
  const double ambiguousFraction_;
  hgcal::RecHitTools rhtools_;
};

DEFINE_FWK_MODULE(TracksterFeatureFlatTableProducer);
