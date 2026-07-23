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
        minSharedEnergy_(p.getParameter<double>("minSharedEnergy")),
        minSharedByHits_(p.getParameter<double>("minSharedByHits")),
        hasByHits_(!p.getParameter<edm::InputTag>("associationByHitsAdaptive").label().empty()) {
    if (hasByHits_)
      assocByHitsToken_ = consumes<BranchAssociationMap>(p.getParameter<edm::InputTag>("associationByHitsAdaptive"));
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
    BranchAssociationMap const* assocBH = hasByHits_ ? &ev.get(assocByHitsToken_) : nullptr;

    // Adaptive label from an adaptive association map: the max-shared branch, mapped to a
    // single-shower class, or to merged_em/merged_hadron by descendant scan for an
    // ancestor/merge node. Reused for the default (BranchHitAssociator) and the parallel
    // by-hits (composition) associations. Returns {label, pdg, is_primary, sharedE, score}.
    struct AdaptiveResult {
      int lab = kFake, pdg = 0, prim = 0;
      float shared = -1.f, score = -1.f;
    };
    auto classifyAdaptive = [&graph, nP](BranchAssociationMap const& amap, unsigned t, double minShared) {
      AdaptiveResult r;
      float ae = -1.f, ascore = -1.f;
      int aidx = -1;
      if (t < amap.size())
        for (auto const& el : amap[t])
          if (el.sharedEnergy() > ae) {
            ae = el.sharedEnergy();
            ascore = el.score();
            aidx = static_cast<int>(el.index());
          }
      r.shared = ae;
      r.score = ascore;
      if (ae >= minShared && aidx >= 0 && aidx < static_cast<int>(nP)) {
        const auto ap = graph.particle(static_cast<uint32_t>(aidx));
        r.pdg = ap.pdgId();
        r.prim = isPrimary(ap) ? 1 : 0;
        if (caloCrossing(ap) && singleClass(r.pdg) != kFake) {
          r.lab = singleClass(r.pdg);
        } else {
          bool anyHad = false, anyEM = false, anyMIP = false;
          for (auto const& d : ap.descendants()) {
            if (!caloCrossing(d))
              continue;
            switch (singleClass(d.pdgId())) {
              case kEM: anyEM = true; break;
              case kMIP: anyMIP = true; break;
              case kHAD: anyHad = true; break;
              default: break;
            }
          }
          r.lab = anyHad ? kMergedHad : (anyEM ? kMergedEM : (anyMIP ? kMIP : singleClass(r.pdg)));
        }
      }
      return r;
    };

    // per-trackster
    std::vector<int> label_adaptive_byhits, adaptive_pdg_byhits;
    std::vector<float> adaptive_shared_byhits, adaptive_score_byhits;
    std::vector<int> label, label_adaptive, is_primary, n_lc, lc_offset;
    std::vector<float> R_bary, z_bary, eta_bary, log_e, best_shared, adaptive_shared, adaptive_score;
    std::vector<int> best_pdg, adaptive_pdg;
    std::vector<float> signal_energy_fraction;
    std::vector<float> local_density_e, event_lc_e;
    std::vector<int> local_density_n, event_lc_n;
    std::vector<float> t_time, t_timeError, t_em_frac, t_spca0, t_spca1, t_spca2;
    std::vector<float> t_regr_e, t_raw_pt, t_raw_em_pt, t_pca_cospt;
    std::vector<int> t_seed_layer, t_seed_scint;
    std::vector<float> t_seed_energy, t_seed_thick, t_seed_eta;
    // per-LC
    std::vector<int> lc_trkIdx, lc_layer, lc_nrh, lc_scint;
    std::vector<float> lc_u, lc_v, lc_energy, lc_size, lc_vmult, lc_thick;

    // Per-event (eta, phi) layer-cluster occupancy histogram, built ONCE over all LCs
    // (O(N_LC)). Each trackster then reads its local pileup density as an O(1) window
    // lookup around its barycenter, instead of a per-object nearest-neighbor search:
    // the same idea as the CLUE tiles, rebuilt here because the tiles are transient.
    constexpr float kEtaMin = -3.3f, kDEta = 0.05f, kDPhi = 0.05f;
    constexpr int kNEta = 132;                          // (3.3-(-3.3))/0.05
    const int kNPhi = static_cast<int>(2.0 * M_PI / kDPhi) + 1;
    std::vector<float> tileE(kNEta * kNPhi, 0.f);
    std::vector<int> tileN(kNEta * kNPhi, 0);
    double evtE = 0.0;
    int evtN = 0;
    auto phiBin = [&](float phi) {
      while (phi < -M_PI) phi += 2.f * M_PI;
      while (phi >= M_PI) phi -= 2.f * M_PI;
      int ip = static_cast<int>((phi + M_PI) / kDPhi);
      return std::min(std::max(ip, 0), kNPhi - 1);
    };
    for (auto const& lc : lcs) {
      const int ie = static_cast<int>((lc.eta() - kEtaMin) / kDEta);
      if (ie >= 0 && ie < kNEta) {
        tileE[ie * kNPhi + phiBin(lc.phi())] += lc.energy();
        tileN[ie * kNPhi + phiBin(lc.phi())] += 1;
      }
      evtE += lc.energy();
      ++evtN;
    }

    for (unsigned t = 0; t < tracksters.size(); ++t) {
      auto const& trk = tracksters[t];
      const auto b = trk.barycenter();
      const float Rb = std::hypot(b.x(), b.y());
      const float phib = std::atan2(b.y(), b.x());

      // Local crowding: sum tile energy/count in a +/-W (~0.1 in eta-phi) window around
      // the barycenter - the pileup density the object sits in. O(W^2) per trackster.
      float locE = 0.f;
      int locN = 0;
      {
        constexpr int W = 2;
        const int ie0 = static_cast<int>((trk.barycenter().eta() - kEtaMin) / kDEta);
        const int ip0 = phiBin(phib);
        for (int de = -W; de <= W; ++de) {
          const int ie = ie0 + de;
          if (ie < 0 || ie >= kNEta)
            continue;
          for (int dp = -W; dp <= W; ++dp) {
            const int ip = ((ip0 + dp) % kNPhi + kNPhi) % kNPhi;  // phi wraparound
            locE += tileE[ie * kNPhi + ip];
            locN += tileN[ie * kNPhi + ip];
          }
        }
      }

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
      const AdaptiveResult A = classifyAdaptive(assocA, t, minSharedEnergy_);
      const int labA = A.lab, pdgA = A.pdg, prim = A.prim;
      const float ae = A.shared, ascore = A.score;
      // Parallel by-hits (composition) adaptive label, when that association is present.
      // Uses its own threshold: the composition shared energy is branch-owned (no PU
      // cellTotal inflation) and runs ~10-25x lower than BranchHitAssociator's, so the
      // 0.5 GeV default cut is far too strict for it; minSharedByHits sets the matched
      // operating point.
      if (assocBH) {
        const AdaptiveResult B = classifyAdaptive(*assocBH, t, minSharedByHits_);
        label_adaptive_byhits.push_back(B.lab);
        adaptive_pdg_byhits.push_back(B.pdg);
        adaptive_shared_byhits.push_back(B.shared > 0 ? B.shared : 0.f);
        adaptive_score_byhits.push_back(B.score >= 0 ? B.score : -1.f);
      }

      lc_offset.push_back(static_cast<int>(lc_trkIdx.size()));
      int seedLayer = -1, seedScint = 0;
      float seedE = -1.f, seedThick = 0.f, seedEta = 0.f;
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
        // detector properties from the LC seed hit: sensor thickness (um; -1 for
        // scintillator) and a silicon/scintillator flag.
        const auto seedDet = lc.seed();
        const int isScint = rhtools_.isScintillator(seedDet) ? 1 : 0;
        const float thick = rhtools_.getSiThickness(seedDet);
        // layer from the LC's first hit, matching the deployed inference builder's
        // getLayerWithOffset(hitsAndFractions()[0].first) convention (was lc.seed()).
        const int layer = static_cast<int>(rhtools_.getLayerWithOffset(lc.hitsAndFractions()[0].first)) - 1;
        lc_trkIdx.push_back(static_cast<int>(label.size()));
        lc_u.push_back(R - Rb);
        lc_v.push_back(Rb * wrapPi(phi - phib));
        lc_layer.push_back(layer);
        lc_energy.push_back(lc.energy());
        lc_size.push_back(size);
        // the grid splats energy/vertex_multiplicity per LC; dump the multiplicity so the
        // offline grid is bit-faithful to the C++ builder. nRecHits = LC hit multiplicity.
        lc_vmult.push_back(trk.vertex_multiplicity(iv));
        lc_nrh.push_back(static_cast<int>(lc.hitsAndFractions().size()));
        lc_thick.push_back(thick);
        lc_scint.push_back(isScint);
        // seed = highest-energy (core) LC of the trackster; keep its detector properties.
        if (lc.energy() > seedE) {
          seedE = lc.energy();
          seedLayer = layer;
          seedThick = thick;
          seedScint = isScint;
          seedEta = rhtools_.getPosition(seedDet).eta();
        }
      }
      t_seed_layer.push_back(seedLayer);
      t_seed_energy.push_back(seedE > 0.f ? seedE : 0.f);
      t_seed_thick.push_back(seedThick);
      t_seed_scint.push_back(seedScint);
      t_seed_eta.push_back(seedEta);

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
      local_density_e.push_back(locE);
      local_density_n.push_back(locN);
      event_lc_e.push_back(static_cast<float>(evtE));
      event_lc_n.push_back(evtN);
      // Physics discriminators: timing (out-of-time = pileup/noise/fake), EM fraction
      // (raw_em/raw), and PCA shower spread (compactness/elongation).
      t_time.push_back(trk.time());
      t_timeError.push_back(trk.timeError());
      t_em_frac.push_back(trk.raw_energy() > 0.f ? trk.raw_em_energy() / trk.raw_energy() : 0.f);
      auto const& _sp = trk.sigmasPCA();
      t_spca0.push_back(_sp[0]);
      t_spca1.push_back(_sp[1]);
      t_spca2.push_back(_sp[2]);
      // Energy-regression estimate + transverse scales, and the shower-axis pointing:
      // cos of the angle between the principal PCA axis and the barycenter direction.
      t_regr_e.push_back(trk.regressed_energy());
      t_raw_pt.push_back(trk.raw_pt());
      t_raw_em_pt.push_back(trk.raw_em_pt());
      {
        auto const& _ax = trk.eigenvectors()[0];
        auto const& _bc = trk.barycenter();
        const float _an = std::sqrt(_ax.x() * _ax.x() + _ax.y() * _ax.y() + _ax.z() * _ax.z());
        const float _bn = std::sqrt(_bc.x() * _bc.x() + _bc.y() * _bc.y() + _bc.z() * _bc.z());
        t_pca_cospt.push_back((_an > 0.f && _bn > 0.f)
                                  ? (_ax.x() * _bc.x() + _ax.y() * _bc.y() + _ax.z() * _bc.z()) / (_an * _bn)
                                  : 0.f);
      }
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
    if (hasByHits_) {
      // Parallel labels from the by-hits (composition) association, for comparison at
      // training time against the default (BranchHitAssociator) labels above.
      trkTab->addColumn<int>("label_adaptive_byhits", label_adaptive_byhits,
                             "adaptive class from the by-hits (composition) association");
      trkTab->addColumn<int>("adaptive_pdg_byhits", adaptive_pdg_byhits, "pdgId of the by-hits adaptive branch");
      trkTab->addColumn<float>("adaptive_sharedE_byhits", adaptive_shared_byhits,
                               "shared energy with the by-hits adaptive branch");
      trkTab->addColumn<float>("adaptive_score_byhits", adaptive_score_byhits,
                               "reco-normalized score of the by-hits adaptive match");
    }
    trkTab->addColumn<int>("n_lc", n_lc, "number of layer clusters");
    trkTab->addColumn<int>("lc_offset", lc_offset, "offset into the LC table");
    trkTab->addColumn<float>("R_bary", R_bary, "barycenter cylindrical radius (cm)");
    trkTab->addColumn<float>("z_bary", z_bary, "barycenter z (cm, signed = endcap)");
    trkTab->addColumn<float>("eta_bary", eta_bary, "barycenter eta");
    trkTab->addColumn<float>("log_raw_energy", log_e, "log1p(raw energy)");
    trkTab->addColumn<float>("local_density_e", local_density_e,
                             "LC energy in a +/-0.1 eta-phi window around the barycenter (local pileup crowding)");
    trkTab->addColumn<int>("local_density_n", local_density_n, "LC count in that window");
    trkTab->addColumn<float>("event_lc_e", event_lc_e, "total event LC energy (global PU proxy)");
    trkTab->addColumn<int>("event_lc_n", event_lc_n, "total event LC count (global PU proxy)");
    trkTab->addColumn<float>("time", t_time, "trackster time (ns); pileup/noise is out-of-time");
    trkTab->addColumn<float>("timeError", t_timeError, "trackster time uncertainty (ns); -1/large = untimed");
    trkTab->addColumn<float>("em_energy_fraction", t_em_frac, "raw_em_energy / raw_energy (EM vs hadronic)");
    trkTab->addColumn<float>("sigmaPCA_long", t_spca0, "RMS spread along principal (longitudinal) axis (cm)");
    trkTab->addColumn<float>("sigmaPCA_tr1", t_spca1, "RMS spread along 1st transverse PCA axis (cm)");
    trkTab->addColumn<float>("sigmaPCA_tr2", t_spca2, "RMS spread along 2nd transverse PCA axis (cm)");
    trkTab->addColumn<float>("regressed_energy", t_regr_e, "energy-regression estimate (GeV)");
    trkTab->addColumn<float>("raw_pt", t_raw_pt, "raw transverse momentum (GeV)");
    trkTab->addColumn<float>("raw_em_pt", t_raw_em_pt, "raw EM transverse momentum (GeV)");
    trkTab->addColumn<float>("pca_cos_pointing", t_pca_cospt,
                             "cos angle between principal PCA axis and barycenter direction (shower pointing)");
    trkTab->addColumn<int>("seed_lc_layer", t_seed_layer, "layer of the highest-energy (seed/core) LC");
    trkTab->addColumn<float>("seed_lc_energy", t_seed_energy, "energy of the highest-energy (seed/core) LC (GeV)");
    trkTab->addColumn<float>("seed_lc_thickness", t_seed_thick,
                             "sensor thickness at the seed LC (um; -1 = scintillator)");
    trkTab->addColumn<int>("seed_lc_isScint", t_seed_scint, "1 if the seed LC is in the scintillator");
    trkTab->addColumn<float>("seed_lc_eta", t_seed_eta, "eta of the seed LC seed-hit position");

    auto lcTab = std::make_unique<nanoaod::FlatTable>(lc_trkIdx.size(), name_ + "LC", false, false);
    lcTab->addColumn<int>("trkIdx", lc_trkIdx, "index into the trackster table");
    lcTab->addColumn<float>("u", lc_u, "R - R_bary (cm)");
    lcTab->addColumn<float>("v", lc_v, "R_bary * wrap(phi - phi_bary) (cm)");
    lcTab->addColumn<int>("layer", lc_layer, "per-endcap layer index (0-based)");
    lcTab->addColumn<float>("energy", lc_energy, "LC energy");
    lcTab->addColumn<float>("size", lc_size, "LC transverse RMS (cm)");
    lcTab->addColumn<float>("vertex_multiplicity", lc_vmult,
                            "trackster vertex_multiplicity for this LC; grid energy weight = energy/vertex_multiplicity");
    lcTab->addColumn<int>("nRecHits", lc_nrh, "number of rechits in the layer cluster");
    lcTab->addColumn<float>("si_thickness", lc_thick, "sensor thickness at the LC seed hit (um; -1 = scintillator)");
    lcTab->addColumn<int>("isScintillator", lc_scint, "1 if the LC seed cell is scintillator, 0 if silicon");

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
    desc.add<edm::InputTag>("associationByHitsAdaptive", edm::InputTag(""))
        ->setComment("Optional parallel by-hits (composition) adaptive association; empty disables the byhits columns.");
    desc.add<edm::InputTag>("graph", edm::InputTag("truthLogicalGraphProducer"));
    desc.add<double>("minSharedEnergy", 0.5);
    desc.add<double>("minSharedByHits", 0.5)
        ->setComment("Shared-energy cut for the by-hits label; the composition shared energy runs "
                     "~10-25x lower than BranchHitAssociator's, so this needs a much smaller value "
                     "(~0.02 GeV) to reach a comparable matched fraction.");
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
  const double minSharedByHits_;
  const bool hasByHits_;
  edm::EDGetTokenT<BranchAssociationMap> assocByHitsToken_;
  hgcal::RecHitTools rhtools_;
};

DEFINE_FWK_MODULE(TracksterFeatureFlatTableProducer);
