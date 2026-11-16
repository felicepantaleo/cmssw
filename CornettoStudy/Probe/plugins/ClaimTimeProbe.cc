// Study probe, not for integration. For charged candidates with a matched truth particle, one row per layer cluster
// inside the claim cone of the propagated track (radius0 + slope x depth): the share of the cluster's simulated energy
// from the track's own particle (and descendants) and from other interactions, the cluster time and error, and the
// track time (MTD t0, sigma) and truth production time. All signal tracks, one pileup track in pileupPrescale.

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/MtdHostCollection.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTICL/Common/interface/TICLUtils.h"
#include "RecoTICL/Interpretation/interface/TrackImpact.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/Vertex.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "TTree.h"

namespace {
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
  bool isHGCal(uint32_t rawId) {
    const auto det = DetId(rawId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }
}  // namespace

class ClaimTimeProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit ClaimTimeProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<std::vector<TICLCandidate>> candToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> lcTimeToken_;
  const edm::EDGetTokenT<std::vector<reco::Track>> trackToken_;
  const edm::EDGetTokenT<edm::ValueMap<float>> t0Token_, sigmat0Token_;
  const edm::EDGetTokenT<MtdHostCollection> mtdToken_;
  const edm::EDGetTokenT<MapType> r2tToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propToken_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hgconsToken_;
  const float radius0_, slope_;
  const unsigned int pileupPrescale_;
  TTree* tree_;
  int event_, track_, signal_, pdg_, nHits_;
  float p_, pt_, t0Truth_, t0_, sigmat0_, e_, dist_, depth_, r_, own_, other_, lcT_, lcTErr_, tMtd_, tMtdErr_, dMtd_;
};

ClaimTimeProbe::ClaimTimeProbe(const edm::ParameterSet& cfg)
    : candToken_(consumes<std::vector<TICLCandidate>>(cfg.getParameter<edm::InputTag>("candidates"))),
      lcToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))),
      lcTimeToken_(
          consumes<edm::ValueMap<std::pair<float, float>>>(cfg.getParameter<edm::InputTag>("layerClustersTime"))),
      trackToken_(consumes<std::vector<reco::Track>>(cfg.getParameter<edm::InputTag>("tracks"))),
      t0Token_(consumes<edm::ValueMap<float>>(cfg.getParameter<edm::InputTag>("t0"))),
      sigmat0Token_(consumes<edm::ValueMap<float>>(cfg.getParameter<edm::InputTag>("sigmat0"))),
      mtdToken_(consumes<MtdHostCollection>(cfg.getParameter<edm::InputTag>("mtd"))),
      r2tToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("trackRecoToTruth"))),
      graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      fieldToken_(esConsumes()),
      propToken_(esConsumes(edm::ESInputTag("", cfg.getParameter<std::string>("propagator")))),
      hgconsToken_(esConsumes(edm::ESInputTag("", "HGCalEESensitive"))),
      radius0_(cfg.getParameter<float>("radius0")),
      slope_(cfg.getParameter<float>("slope")),
      pileupPrescale_(cfg.getParameter<unsigned int>("pileupPrescale")) {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("lcs", "lcs");
  for (auto [n, v] : std::vector<std::pair<const char*, int*>>{
           {"event", &event_}, {"track", &track_}, {"signal", &signal_}, {"pdg", &pdg_}, {"nHits", &nHits_}})
    tree_->Branch(n, v);
  for (auto [n, v] : std::vector<std::pair<const char*, float*>>{{"p", &p_},
                                                                 {"pt", &pt_},
                                                                 {"t0Truth", &t0Truth_},
                                                                 {"t0", &t0_},
                                                                 {"sigmat0", &sigmat0_},
                                                                 {"E", &e_},
                                                                 {"dist", &dist_},
                                                                 {"depth", &depth_},
                                                                 {"r", &r_},
                                                                 {"own", &own_},
                                                                 {"other", &other_},
                                                                 {"lcT", &lcT_},
                                                                 {"lcTErr", &lcTErr_},
                                                                 {"tMtd", &tMtd_},
                                                                 {"tMtdErr", &tMtdErr_},
                                                                 {"dMtd", &dMtd_}})
    tree_->Branch(n, v);
}

void ClaimTimeProbe::analyze(const edm::Event& event, const edm::EventSetup& es) {
  auto const& cands = event.get(candToken_);
  const auto lcH = event.getHandle(lcToken_);
  auto const& lcs = *lcH;
  auto const& lcTime = event.get(lcTimeToken_);
  const auto tkH = event.getHandle(trackToken_);
  auto const& t0Map = event.get(t0Token_);
  auto const& sigmat0Map = event.get(sigmat0Token_);
  const auto mtd = event.get(mtdToken_).const_view();
  auto const& r2t = event.get(r2tToken_).getMap();
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  auto const* field = &es.getData(fieldToken_);
  auto const& prop = es.getData(propToken_);
  const auto frontDisks = ticl::utils::buildHGCalFirstDisks(es.getData(hgconsToken_));
  event_ = event.id().event();
  const unsigned int nP = graph.particles().size();

  // Simulated energy per cell and calo-boundary particle, and the interaction of each particle.
  std::unordered_map<uint32_t, std::vector<std::pair<unsigned int, float>>> cellTruth;
  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  for (unsigned int p = 0; p < nP; ++p) {
    if (!graph.particles()[p].isAtLevel(truth::LevelFlag::CaloBoundary))
      continue;
    hits.clear();
    index.appendSubgraphHits(truth::HitChannel::Calo, p, hits);
    for (auto const& h : hits)
      if (isHGCal(h.detId) && h.energy > 0.f)
        cellTruth[h.detId].emplace_back(p, h.energy);
  }

  std::vector<char> mine(nP, 0);
  unsigned int nPileup = 0;
  track_ = 0;
  for (auto const& c : cands) {
    if (c.charge() == 0 || c.trackPtrs().empty() || c.trackPtrs()[0].isNull() || c.trackPtrs()[0].id() != tkH.id())
      continue;
    const size_t t = c.trackPtrs()[0].key();
    if (t >= r2t.size() || r2t[t].empty())
      continue;
    auto best = std::min_element(
        r2t[t].begin(), r2t[t].end(), [](auto const& a, auto const& b) { return a.score() < b.score(); });
    const unsigned int u = best->index();
    if (u >= nP)
      continue;
    auto const& pd = graph.particles()[u];
    signal_ = pd.eventId == 0;
    if (!signal_ && (nPileup++ % pileupPrescale_) != 0)
      continue;
    auto const& tk = (*tkH)[t];
    const auto impact = ticl::propagateToHGCalFront(tk, field, prop, frontDisks);
    if (!impact.valid)
      continue;
    std::fill(mine.begin(), mine.end(), 0);
    mine[u] = 1;
    for (auto const& d : truth::Particle(&graph, u).descendants())
      mine[d.id()] = 1;
    const reco::TrackRef ref(tkH, t);
    t0_ = t0Map[ref];
    sigmat0_ = sigmat0Map[ref];
    // Time and position of the MTD hit matched to the track (time error < 0: no hit).
    const bool hasMtd = static_cast<int>(t) < mtd.metadata().size();
    tMtd_ = hasMtd ? mtd.time()[t] : -999.f;
    tMtdErr_ = hasMtd ? mtd.timeErr()[t] : -1.f;
    const float mx = hasMtd ? mtd.posInMTD_x()[t] : 0.f, my = hasMtd ? mtd.posInMTD_y()[t] : 0.f,
                mz = hasMtd ? mtd.posInMTD_z()[t] : 0.f;
    const auto pv = truth::Particle(&graph, u).productionVertices();
    t0Truth_ = pv.empty() ? -999.f : pv.front().position().t();
    pdg_ = pd.pdgId;
    p_ = tk.p();
    pt_ = tk.pt();
    ++track_;
    const float zFront = std::abs(impact.position.z());
    for (size_t k = 0; k < lcs.size(); ++k) {
      auto const& lc = lcs[k];
      if (lc.z() * impact.position.z() <= 0.)
        continue;
      depth_ = std::abs(static_cast<float>(lc.z())) - zFront;
      if (depth_ < -1.f)
        continue;
      dist_ = ticl::impactTransverseDistance(impact, lc.position());
      if (!(dist_ < radius0_ + slope_ * std::max(depth_, 0.f)))
        continue;
      float tot = 0.f, own = 0.f, other = 0.f;
      for (auto const& [id, f] : lc.hitsAndFractions()) {
        auto it = cellTruth.find(id.rawId());
        if (it == cellTruth.end())
          continue;
        for (auto const& [p, e] : it->second) {
          tot += f * e;
          if (mine[p])
            own += f * e;
          else
            other += f * e;
        }
      }
      e_ = lc.energy();
      r_ = std::sqrt(lc.position().mag2());
      nHits_ = lc.hitsAndFractions().size();
      own_ = tot > 0.f ? own / tot : -1.f;
      other_ = tot > 0.f ? other / tot : -1.f;
      const auto tm = lcTime.get(lcH.id(), k);
      lcT_ = tm.first;
      lcTErr_ = tm.second;
      dMtd_ = std::sqrt((lc.x() - mx) * (lc.x() - mx) + (lc.y() - my) * (lc.y() - my) + (lc.z() - mz) * (lc.z() - mz));
      tree_->Fill();
    }
  }
}

DEFINE_FWK_MODULE(ClaimTimeProbe);
