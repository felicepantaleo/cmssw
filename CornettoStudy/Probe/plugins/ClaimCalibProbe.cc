// Study probe, not for integration. Calibration of the track claim from the truth graph.
//   tracks: one row per charged candidate whose track matches a truth particle: track p, pt, eta, the particle species
//           and origin, and its own reconstructed layer-cluster energy (cluster energy x share of the cluster's
//           simulated energy from the particle and its descendants), in all layer clusters.
//   own:    one row per (track, layer cluster with an own share): the own energy, the transverse distance to the
//           track propagated to the HGCAL front and extended along its momentum there, and the depth from the front.

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
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
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "TTree.h"

namespace {
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
  bool isHGCal(uint32_t rawId) {
    const auto det = DetId(rawId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }
}  // namespace

class ClaimCalibProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit ClaimCalibProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<std::vector<TICLCandidate>> candToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const edm::EDGetTokenT<std::vector<reco::Track>> trackToken_;
  const edm::EDGetTokenT<MapType> r2tToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propToken_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hgconsToken_;
  TTree* tracks_;
  TTree* own_;
  int event_, row_, pdg_, signal_, reachesCalo_;
  float p_, pt_, eta_, ownTotal_, oE_, oDist_, oDepth_;
};

ClaimCalibProbe::ClaimCalibProbe(const edm::ParameterSet& cfg)
    : candToken_(consumes<std::vector<TICLCandidate>>(cfg.getParameter<edm::InputTag>("candidates"))),
      lcToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))),
      trackToken_(consumes<std::vector<reco::Track>>(cfg.getParameter<edm::InputTag>("tracks"))),
      r2tToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("trackRecoToTruth"))),
      graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      fieldToken_(esConsumes()),
      propToken_(esConsumes(edm::ESInputTag("", cfg.getParameter<std::string>("propagator")))),
      hgconsToken_(esConsumes(edm::ESInputTag("", "HGCalEESensitive"))) {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  tracks_ = fs->make<TTree>("tracks", "tracks");
  for (auto [n, v] : std::vector<std::pair<const char*, int*>>{
           {"event", &event_}, {"row", &row_}, {"pdg", &pdg_}, {"signal", &signal_}, {"reachesCalo", &reachesCalo_}})
    tracks_->Branch(n, v);
  for (auto [n, v] :
       std::vector<std::pair<const char*, float*>>{{"p", &p_}, {"pt", &pt_}, {"eta", &eta_}, {"ownTotal", &ownTotal_}})
    tracks_->Branch(n, v);
  own_ = fs->make<TTree>("own", "own");
  own_->Branch("event", &event_);
  own_->Branch("row", &row_);
  own_->Branch("E", &oE_);
  own_->Branch("dist", &oDist_);
  own_->Branch("depth", &oDepth_);
}

void ClaimCalibProbe::analyze(const edm::Event& event, const edm::EventSetup& es) {
  auto const& cands = event.get(candToken_);
  auto const& lcs = event.get(lcToken_);
  const auto tkH = event.getHandle(trackToken_);
  auto const& r2t = event.get(r2tToken_).getMap();
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  auto const* field = &es.getData(fieldToken_);
  auto const& prop = es.getData(propToken_);
  const auto frontDisks = ticl::utils::buildHGCalFirstDisks(es.getData(hgconsToken_));
  event_ = event.id().event();
  const unsigned int nP = graph.particles().size();

  // Tracked particles (one row each) and the owner row of every particle in their subtrees.
  std::vector<std::pair<size_t, unsigned int>> rows;  // (track, particle)
  std::vector<int> owner(nP, -1);
  for (auto const& c : cands) {
    if (c.charge() == 0 || c.trackPtrs().empty() || c.trackPtrs()[0].isNull() || c.trackPtrs()[0].id() != tkH.id())
      continue;
    const size_t t = c.trackPtrs()[0].key();
    if (t >= r2t.size() || r2t[t].empty())
      continue;
    auto best = std::min_element(
        r2t[t].begin(), r2t[t].end(), [](auto const& a, auto const& b) { return a.score() < b.score(); });
    const unsigned int u = best->index();
    if (u >= nP || owner[u] >= 0)
      continue;
    const int row = static_cast<int>(rows.size());
    rows.emplace_back(t, u);
    owner[u] = row;
    for (auto const& d : truth::Particle(&graph, u).descendants())
      if (owner[d.id()] < 0)
        owner[d.id()] = row;
  }
  // Simulated energy per cell: owner row (or -1) and energy.
  std::unordered_map<uint32_t, std::vector<std::pair<int, float>>> cellTruth;
  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  for (unsigned int p = 0; p < nP; ++p) {
    if (!graph.particles()[p].isAtLevel(truth::LevelFlag::CaloBoundary))
      continue;
    hits.clear();
    index.appendSubgraphHits(truth::HitChannel::Calo, p, hits);
    for (auto const& h : hits)
      if (isHGCal(h.detId) && h.energy > 0.f)
        cellTruth[h.detId].emplace_back(owner[p], h.energy);
  }
  std::vector<ticl::TrackImpact> impacts(rows.size());
  for (size_t r = 0; r < rows.size(); ++r)
    impacts[r] = ticl::propagateToHGCalFront((*tkH)[rows[r].first], field, prop, frontDisks);
  std::vector<float> ownTotal(rows.size(), 0.f);
  // Each layer cluster: its simulated energy by owner row; the own energy is the cluster energy times the share.
  std::unordered_map<int, float> byRow;
  for (auto const& lc : lcs) {
    byRow.clear();
    float tot = 0.f;
    for (auto const& [id, f] : lc.hitsAndFractions()) {
      auto it = cellTruth.find(id.rawId());
      if (it == cellTruth.end())
        continue;
      for (auto const& [row, e] : it->second) {
        tot += f * e;
        if (row >= 0)
          byRow[row] += f * e;
      }
    }
    if (tot <= 0.f)
      continue;
    for (auto const& [row, e] : byRow) {
      const float ownE = lc.energy() * e / tot;
      ownTotal[row] += ownE;
      auto const& imp = impacts[row];
      if (!imp.valid)
        continue;
      row_ = row;
      oE_ = ownE;
      oDist_ = ticl::impactTransverseDistance(imp, lc.position());
      oDepth_ = std::abs(static_cast<float>(lc.z())) - std::abs(imp.position.z());
      own_->Fill();
    }
  }
  for (size_t r = 0; r < rows.size(); ++r) {
    auto const& tk = (*tkH)[rows[r].first];
    auto const& pd = graph.particles()[rows[r].second];
    row_ = static_cast<int>(r);
    pdg_ = pd.pdgId;
    signal_ = pd.eventId == 0;
    reachesCalo_ = pd.isAtLevel(truth::LevelFlag::CaloBoundary);
    p_ = tk.p();
    pt_ = tk.pt();
    eta_ = tk.eta();
    ownTotal_ = ownTotal[r];
    tracks_->Fill();
  }
}

DEFINE_FWK_MODULE(ClaimCalibProbe);
