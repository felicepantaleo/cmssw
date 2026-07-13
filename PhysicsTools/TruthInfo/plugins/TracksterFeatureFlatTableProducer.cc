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
// Per-trackster CLUE3D-stage labels from the trackster -> truth-branch association:
//   label          (leaf/fixed level): class of the best calo-crossing branch, a
//                  diagnostic single-particle label.
//   label_adaptive (scored adaptive level): the branch chosen by the associator's
//                  adaptive search, which climbs the truth graph (under the
//                  hadronization ceiling) to the level balancing completeness against
//                  branch spread. If that level is a single calo-crossing particle ->
//                  em/mip/hadronic; if it is an ancestor merging several -> merged_em
//                  or merged_hadron. This is the primary training target.
//   is_primary     provenance: 1 if the matched particle is from the hard scatter,
//                  0 if pileup. Orthogonal to type (pileup particles carry their real
//                  type); trained as a separate head, meaningful only with PU.
// No track exists at pattern-recognition time, so charge/species are deliberately not
// labeled here. Class ids match ttpid/config.py.
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
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

#include <cstring>

namespace {
  // CLUE3D-stage PID classes. There is no track at pattern-recognition time, so the
  // label is what the calorimeter alone separates: charge and hadron species are NOT
  // resolved (electron==photon=="em", charged==neutral hadron=="hadronic", muon=="mip").
  // merged_em / merged_hadron mark two overlapping showers from one decay (pi0->gg,
  // rho->pipi), which shape distinguishes without a track. Ids match ttpid/config.py.
  enum {
    kEM = 0,
    kMIP = 1,
    kHAD = 2,
    kMergedEM = 3,
    kMergedHad = 4,
    kFake = 5,
  };

  bool isEMpdg(int pdg) {
    const int a = std::abs(pdg);
    return a == 11 || a == 22;  // electron, photon
  }
  bool isMIPpdg(int pdg) { return std::abs(pdg) == 13; }  // muon
  bool isHadronPdg(int pdg) {
    const int a = std::abs(pdg);
    if (a < 100)
      return false;  // leptons, photon, bosons
    if (a >= 1000 && (a / 100) % 10 == 0)
      return false;  // diquarks
    return true;     // mesons and baryons
  }
  // Class of a single calo-crossing particle from its pdgId (no merging).
  int singleClass(int pdg) {
    if (isEMpdg(pdg))
      return kEM;
    if (isMIPpdg(pdg))
      return kMIP;
    if (isHadronPdg(pdg))
      return kHAD;
    return kFake;
  }

  // A logical particle physically entered the calorimeter if it has the tracker-calo
  // boundary checkpoint (id 0). The adaptive level is either such a crossing particle
  // (a single shower) or an ancestor of several (a merged shower).
  bool caloCrossing(truth::Particle const& p) {
    if (!p.hasCheckpoints())
      return false;
    for (auto const& cp : p.checkpoints())
      if (cp.checkpointId == 0)
        return true;
    return false;
  }

  // EncodedEventId is packed into the low word of the 64-bit eventId (same helper as
  // truth::Branch). Signal (hard scatter) is bunchCrossing 0 and event 0; anything
  // else is pileup.
  bool isPrimary(truth::Particle const& p) {
    uint32_t raw = 0;
    const uint64_t packed = p.eventId();
    std::memcpy(&raw, &packed, sizeof(raw));
    const EncodedEventId id(raw);
    return id.bunchCrossing() == 0 && id.event() == 0;
  }

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
        assocAdaptiveToken_(consumes<BranchAssociationMap>(p.getParameter<edm::InputTag>("associationAdaptive"))),
        graphToken_(consumes<truth::Graph>(p.getParameter<edm::InputTag>("graph"))),
        geomToken_(esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>()),
        minSharedEnergy_(p.getParameter<double>("minSharedEnergy")) {
    produces<nanoaod::FlatTable>();          // per-trackster
    produces<nanoaod::FlatTable>("LC");      // per layer cluster
  }

  void beginRun(edm::Run const&, edm::EventSetup const& es) override { rhtools_.setGeometry(es.getData(geomToken_)); }
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

  void produce(edm::Event& ev, edm::EventSetup const&) override {
    auto const& tracksters = ev.get(tracksterToken_);
    auto const& lcs = ev.get(lcToken_);
    auto const& assoc = ev.get(assocToken_);
    auto const& assocA = ev.get(assocAdaptiveToken_);
    auto const& graph = ev.get(graphToken_);
    const unsigned nP = graph.nParticles();

    // per-trackster
    std::vector<int> label, label_adaptive, is_primary, n_lc, lc_offset;
    std::vector<float> R_bary, z_bary, eta_bary, log_e, best_shared, adaptive_shared, adaptive_score;
    std::vector<int> best_pdg, adaptive_pdg;
    std::vector<float> signal_energy_fraction;
    // per-LC
    std::vector<int> lc_trkIdx, lc_layer;
    std::vector<float> lc_u, lc_v, lc_energy, lc_size;

    for (unsigned t = 0; t < tracksters.size(); ++t) {
      auto const& trk = tracksters[t];
      const auto b = trk.barycenter();
      const float Rb = std::hypot(b.x(), b.y());
      const float phib = std::atan2(b.y(), b.x());

      // Leaf (fixed-level) label: best calo-crossing branch by shared energy. Leaves
      // are single particles, so this is always a single-class label (diagnostic).
      // Accumulate signal vs total shared energy over ALL leaf branches: a trackster
      // contains signal if any of its energy comes from a hard-scatter/gun (isPrimary)
      // particle. signal_energy_fraction lets training drop pure-PU tracksters (== 0) and
      // keep signal or signal-contaminated ones.
      float be = -1.f, sigE = 0.f, totE = 0.f;
      int bidx = -1;
      if (t < assoc.size()) {
        for (auto const& el : assoc[t]) {
          const float e = el.sharedEnergy();
          totE += e;
          if (el.index() < nP && isPrimary(graph.particle(static_cast<uint32_t>(el.index()))))
            sigE += e;
          if (e > be) {
            be = e;
            bidx = static_cast<int>(el.index());
          }
        }
      }
      const float signal_frac = totE > 0.f ? sigE / totE : 0.f;
      int lab = kFake, pdg = 0;
      if (be >= minSharedEnergy_ && bidx >= 0 && bidx < static_cast<int>(nP)) {
        pdg = graph.particles()[bidx].pdgId;
        lab = singleClass(pdg);
      }

      // Adaptive-level label: the branch the adaptive search picked (the graph level
      // balancing completeness against branch spread). A calo-crossing particle -> a
      // single-shower class; an ancestor merging several children -> merged_em or
      // merged_hadron by the nature of its children.
      float ae = -1.f, ascore = -1.f;
      int aidx = -1;
      if (t < assocA.size()) {
        for (auto const& el : assocA[t]) {
          if (el.sharedEnergy() > ae) {
            ae = el.sharedEnergy();
            ascore = el.score();
            aidx = static_cast<int>(el.index());
          }
        }
      }
      int labA = kFake, pdgA = 0, prim = 0;
      if (ae >= minSharedEnergy_ && aidx >= 0 && aidx < static_cast<int>(nP)) {
        const auto ap = graph.particle(static_cast<uint32_t>(aidx));
        pdgA = ap.pdgId();
        prim = isPrimary(ap) ? 1 : 0;
        if (caloCrossing(ap) && singleClass(pdgA) != kFake) {
          labA = singleClass(pdgA);
        } else {
          // A merge node (ancestor of several showers), or a particle that has no direct
          // calo class because it decays before showering (a tau). Classify by its
          // calo-crossing descendant leaves, the actual shower-makers, not its direct
          // children, which for a tau are neutrinos and intermediate resonances (rho/a1).
          bool anyHad = false, anyEM = false, anyMIP = false;
          for (auto const& d : ap.descendants()) {
            if (!caloCrossing(d))
              continue;
            switch (singleClass(d.pdgId())) {
              case kEM:
                anyEM = true;
                break;
              case kMIP:
                anyMIP = true;
                break;
              case kHAD:
                anyHad = true;
                break;
              default:
                break;
            }
          }
          labA = anyHad ? kMergedHad : (anyEM ? kMergedEM : (anyMIP ? kMIP : singleClass(pdgA)));
        }
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
      label_adaptive.push_back(labA);
      is_primary.push_back(prim);
      signal_energy_fraction.push_back(signal_frac);
      best_pdg.push_back(pdg);
      adaptive_pdg.push_back(pdgA);
      best_shared.push_back(be > 0 ? be : 0.f);
      adaptive_shared.push_back(ae > 0 ? ae : 0.f);
      adaptive_score.push_back(ascore >= 0 ? ascore : -1.f);
      n_lc.push_back(static_cast<int>(lc_trkIdx.size()) - lc_offset.back());
      R_bary.push_back(Rb);
      z_bary.push_back(b.z());
      eta_bary.push_back(trk.barycenter().eta());
      log_e.push_back(std::log1p(trk.raw_energy()));
    }

    auto trkTab = std::make_unique<nanoaod::FlatTable>(label.size(), name_, false, false);
    trkTab->addColumn<int>("label", label, "leaf single-particle class (0=em,1=mip,2=hadronic,5=fake)");
    trkTab->addColumn<int>("label_adaptive", label_adaptive,
                           "adaptive class: 0=em,1=mip,2=hadronic,3=merged_em,4=merged_hadron,5=fake");
    trkTab->addColumn<int>("is_primary", is_primary, "1 if matched particle is hard-scatter, 0 if pileup");
    trkTab->addColumn<float>("signal_energy_fraction", signal_energy_fraction,
                             "fraction of shared energy from hard-scatter/gun (signal) particles; 0 = pure pileup");
    trkTab->addColumn<int>("best_pdg", best_pdg, "pdgId of the best-matching leaf branch");
    trkTab->addColumn<int>("adaptive_pdg", adaptive_pdg, "pdgId of the adaptive-level branch");
    trkTab->addColumn<float>("best_sharedE", best_shared, "shared energy with the best leaf branch");
    trkTab->addColumn<float>("adaptive_sharedE", adaptive_shared, "shared energy with the adaptive branch");
    trkTab->addColumn<float>("adaptive_score", adaptive_score,
                             "reco-normalized contamination score of the adaptive match (lower is better)");
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
    desc.add<edm::InputTag>("associationAdaptive",
                            edm::InputTag("allTrackstersToTruthBranchAssociations",
                                          "ticlTrackstersCLUE3DHighToTruthBranchAdaptive"));
    desc.add<edm::InputTag>("graph", edm::InputTag("truthLogicalGraphProducer"));
    desc.add<double>("minSharedEnergy", 0.5);
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const std::string name_;
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksterToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const edm::EDGetTokenT<BranchAssociationMap> assocToken_;
  const edm::EDGetTokenT<BranchAssociationMap> assocAdaptiveToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  const double minSharedEnergy_;
  hgcal::RecHitTools rhtools_;
};

DEFINE_FWK_MODULE(TracksterFeatureFlatTableProducer);
