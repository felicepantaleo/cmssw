// Study probe, not for integration. Truth performance of every TICL stage, against the calo-boundary particles of the
// truth graph (an antichain). A truth object is the signal leg a particle descends from, else the particle itself.
// Trees:
//   lcs     one row per layer cluster (all above lcAllAbove GeV, one in lcPrescale below it, with weight): region, layer,
//           thickness, energy, truth share of the dominant object, the mask bits and the stages that use it.
//   objreg  one row per truth object (signal, or pileup with HGCAL deposit above minObjDeposit) and region: the deposit,
//           the share in layer clusters, after the filter, in the CLUE3D tracksters, and the splitting per layer.
//   ts      one row per trackster per stage: energies, PID, dominant object, purity, completeness, merging.
//   objs    one row per truth object per stage: best completeness, purity of that trackster, fragmentation.
//   cands   one row per candidate: pdg, energy, truth species and purity of its tracksters, track truth.
// Deposit shares use the simulated HGCAL energy; responses use the energy of the object at the calo boundary.

#include <algorithm>
#include <array>
#include <span>
#include <cmath>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoTICL/Interpretation/interface/ModelFeatures.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"

#include "TTree.h"

namespace {
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
  // Region: 0 CE-E silicon, 1 CE-H silicon, 2 CE-H scintillator, -1 not HGCAL.
  int regionOf(DetId id) {
    switch (id.det()) {
      case DetId::HGCalEE:
        return 0;
      case DetId::HGCalHSi:
        return 1;
      case DetId::HGCalHSc:
        return 2;
      default:
        return -1;
    }
  }
  int layerOf(DetId id) {
    const int r = regionOf(id);
    if (r == 0 || r == 1)
      return HGCSiliconDetId(id).layer();
    if (r == 2)
      return HGCScintillatorDetId(id).layer();
    return -1;
  }
  int thicknessOf(DetId id) {
    const int r = regionOf(id);
    return (r == 0 || r == 1) ? HGCSiliconDetId(id).type() : -1;
  }
  // Species: 0 EM (photon, pi0, electron), 1 charged hadron, 2 neutral hadron, 3 muon, 4 other.
  int speciesOfPdg(int pdg) {
    const int a = std::abs(pdg);
    if (a == 22 || a == 111 || a == 11)
      return 0;
    if (a == 211 || a == 321 || a == 2212 || a == 3222 || a == 3112 || a == 3312 || a == 3334)
      return 1;
    if (a == 130 || a == 2112 || a == 310 || a == 3122 || a == 3212 || a == 3322)
      return 2;
    if (a == 13)
      return 3;
    return 4;
  }
}  // namespace

class StageTruthProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit StageTruthProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  std::vector<edm::EDGetTokenT<std::vector<float>>> maskTokens_;
  std::vector<edm::EDGetTokenT<std::vector<ticl::Trackster>>> tsTokens_;
  const edm::EDGetTokenT<std::vector<TICLCandidate>> candToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<unsigned int>> legsToken_;
  const edm::EDGetTokenT<MapType> tkR2tToken_;
  const double lcAllAbove_, minObjDeposit_;
  const unsigned int lcPrescale_;

  TTree *lcT_, *objregT_, *tsT_, *objsT_, *candT_;
  int event_;
  // lcs
  int lcRegion_, lcLayer_, lcThick_, lcNHits_, lcMask_, lcUsed_, lcSpecies_, lcSignal_;
  float lcE_, lcEta_, lcTruth_, lcPurity_, lcWeight_;
  // objreg
  int orObj_, orRegion_, orSpecies_, orSignal_, orPdg_, orSplitLayers_;
  float orDeposit_, orInLC_, orAfterFilter_, orInCLUE3D_, orInRecovery_, orSplit_, orObjE_, orEta_;
  // ts
  int tsStage_, tsNLC_, tsDomSpecies_, tsDomSignal_, tsDomPdg_, tsNObj10_;
  // Truth deposit fractions by species (photon, electron, pi0, charged hadron, neutral hadron, muon, other) and the
  // layer-cluster profile of the neutral share model (largest share, core shares at 2 and 5 cm, rms, min |z|, |z| extent).
  float tsFrac_[7], tsLcProf_[6];
  float tsRawE_, tsRawEmE_, tsRegE_, tsEta_, tsProb_[8], tsPurity_, tsCompleteness_, tsDomE_, tsTruthDeposit_,
      tsInLinks_, tsInFinal_;
  // objs
  int osStage_, osSpecies_, osSignal_, osPdg_, osNTs10_;
  float osObjE_, osEta_, osBestCompl_, osBestPurity_, osBestRegE_, osBestRawE_, osAttribRegE_;
  // cands
  int cPdg_, cCharge_, cNTs_, cDomSpecies_, cDomSignal_, cTrackSpecies_;
  float cE_, cEta_, cPurity_, cDomE_;
};

StageTruthProbe::StageTruthProbe(const edm::ParameterSet& cfg)
    : lcToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))),
      candToken_(consumes<std::vector<TICLCandidate>>(cfg.getParameter<edm::InputTag>("candidates"))),
      graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      legsToken_(consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("legs"))),
      tkR2tToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("trackRecoToTruth"))),
      lcAllAbove_(cfg.getParameter<double>("lcAllAbove")),
      minObjDeposit_(cfg.getParameter<double>("minObjDeposit")),
      lcPrescale_(cfg.getParameter<unsigned int>("lcPrescale")) {
  usesResource(TFileService::kSharedResource);
  for (auto const& t : cfg.getParameter<std::vector<edm::InputTag>>("masks"))
    maskTokens_.push_back(consumes<std::vector<float>>(t));
  for (auto const& t : cfg.getParameter<std::vector<edm::InputTag>>("tracksters"))
    tsTokens_.push_back(consumes<std::vector<ticl::Trackster>>(t));

  edm::Service<TFileService> fs;
  auto ints = [](TTree* t, std::vector<std::pair<const char*, int*>> v) {
    for (auto [n, p] : v)
      t->Branch(n, p);
  };
  auto floats = [](TTree* t, std::vector<std::pair<const char*, float*>> v) {
    for (auto [n, p] : v)
      t->Branch(n, p);
  };
  lcT_ = fs->make<TTree>("lcs", "lcs");
  ints(lcT_,
       {{"event", &event_},
        {"region", &lcRegion_},
        {"layer", &lcLayer_},
        {"thick", &lcThick_},
        {"nHits", &lcNHits_},
        {"mask", &lcMask_},
        {"used", &lcUsed_},
        {"species", &lcSpecies_},
        {"signal", &lcSignal_}});
  floats(lcT_, {{"E", &lcE_}, {"eta", &lcEta_}, {"truth", &lcTruth_}, {"purity", &lcPurity_}, {"weight", &lcWeight_}});
  objregT_ = fs->make<TTree>("objreg", "objreg");
  ints(objregT_,
       {{"event", &event_},
        {"obj", &orObj_},
        {"region", &orRegion_},
        {"species", &orSpecies_},
        {"signal", &orSignal_},
        {"pdg", &orPdg_},
        {"splitLayers", &orSplitLayers_}});
  floats(objregT_,
         {{"deposit", &orDeposit_},
          {"inLC", &orInLC_},
          {"afterFilter", &orAfterFilter_},
          {"inCLUE3D", &orInCLUE3D_},
          {"inRecovery", &orInRecovery_},
          {"split", &orSplit_},
          {"objE", &orObjE_},
          {"eta", &orEta_}});
  tsT_ = fs->make<TTree>("ts", "ts");
  ints(tsT_,
       {{"event", &event_},
        {"stage", &tsStage_},
        {"nLC", &tsNLC_},
        {"domSpecies", &tsDomSpecies_},
        {"domSignal", &tsDomSignal_},
        {"domPdg", &tsDomPdg_},
        {"nObj10", &tsNObj10_}});
  floats(tsT_,
         {{"rawE", &tsRawE_},
          {"rawEmE", &tsRawEmE_},
          {"regE", &tsRegE_},
          {"eta", &tsEta_},
          {"purity", &tsPurity_},
          {"completeness", &tsCompleteness_},
          {"inLinks", &tsInLinks_},
          {"inFinal", &tsInFinal_},
          {"domE", &tsDomE_},
          {"truthDeposit", &tsTruthDeposit_}});
  tsT_->Branch("prob", tsProb_, "prob[8]/F");
  tsT_->Branch("frac", tsFrac_, "frac[7]/F");
  tsT_->Branch("lcProf", tsLcProf_, "lcProf[6]/F");
  objsT_ = fs->make<TTree>("objs", "objs");
  ints(objsT_,
       {{"event", &event_},
        {"stage", &osStage_},
        {"species", &osSpecies_},
        {"signal", &osSignal_},
        {"pdg", &osPdg_},
        {"nTs10", &osNTs10_}});
  floats(objsT_,
         {{"objE", &osObjE_},
          {"eta", &osEta_},
          {"bestCompl", &osBestCompl_},
          {"bestPurity", &osBestPurity_},
          {"bestRegE", &osBestRegE_},
          {"bestRawE", &osBestRawE_},
          {"attribRegE", &osAttribRegE_}});
  candT_ = fs->make<TTree>("cands", "cands");
  ints(candT_,
       {{"event", &event_},
        {"pdg", &cPdg_},
        {"charge", &cCharge_},
        {"nTs", &cNTs_},
        {"domSpecies", &cDomSpecies_},
        {"domSignal", &cDomSignal_},
        {"trackSpecies", &cTrackSpecies_}});
  floats(candT_, {{"E", &cE_}, {"eta", &cEta_}, {"purity", &cPurity_}, {"domE", &cDomE_}});
}

void StageTruthProbe::analyze(const edm::Event& event, const edm::EventSetup&) {
  auto const& lcs = event.get(lcToken_);
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  auto const& legs = event.get(legsToken_);
  auto const& tkR2t = event.get(tkR2tToken_).getMap();
  auto const& cands = event.get(candToken_);
  event_ = event.id().event();
  const unsigned int nP = graph.particles().size();
  const unsigned int nLC = lcs.size();

  // Truth objects.
  std::vector<int> objectOf(nP, -1);
  for (unsigned int leg : legs) {
    if (leg >= nP || !graph.particles()[leg].isSignal())
      continue;
    objectOf[leg] = static_cast<int>(leg);
    for (auto const& d : truth::Particle(&graph, leg).descendants())
      objectOf[d.id()] = static_cast<int>(leg);
  }
  auto objectFor = [&](unsigned int p) { return objectOf[p] >= 0 ? objectOf[p] : static_cast<int>(p); };
  // Species of an object: the leg species; for a photon or electron, EM; else from the pdg id.
  auto speciesOfObj = [&](int obj) { return speciesOfPdg(graph.particles()[obj].pdgId); };
  auto isSignalObj = [&](int obj) { return graph.particles()[obj].eventId == 0 ? 1 : 0; };

  // Cell truth and object totals.
  std::unordered_map<uint32_t, std::vector<std::pair<int, float>>> cellTruth;
  std::unordered_map<int, double> objDeposit, objE;
  std::unordered_map<int, std::array<double, 3>> objRegDeposit;
  std::unordered_map<int, double> objEtaW;
  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  for (unsigned int p = 0; p < nP; ++p) {
    auto const& pd = graph.particles()[p];
    if (!pd.isAtLevel(truth::LevelFlag::CaloBoundary))
      continue;
    hits.clear();
    index.appendSubgraphHits(truth::HitChannel::Calo, p, hits);
    const int obj = objectFor(p);
    bool any = false;
    for (auto const& h : hits) {
      const DetId id(h.detId);
      const int r = regionOf(id);
      if (r < 0 || !(h.energy > 0.f))
        continue;
      cellTruth[h.detId].emplace_back(obj, h.energy);
      objDeposit[obj] += h.energy;
      objRegDeposit[obj][r] += h.energy;
      any = true;
    }
    if (any) {
      objE[obj] += pd.momentum.E();
      objEtaW[obj] += pd.momentum.E() * pd.momentum.eta();
    }
  }

  // Masks and stage usage per layer cluster.
  std::vector<int> maskBits(nLC, 0), usedBits(nLC, 0);
  for (size_t m = 0; m < maskTokens_.size(); ++m) {
    auto const& mask = event.get(maskTokens_[m]);
    for (unsigned int k = 0; k < nLC && k < mask.size(); ++k)
      if (mask[k] > 0.f)
        maskBits[k] |= 1 << m;
  }
  std::vector<std::vector<ticl::Trackster> const*> stages;
  for (size_t s = 0; s < tsTokens_.size(); ++s) {
    stages.push_back(&event.get(tsTokens_[s]));
    for (auto const& t : *stages.back())
      for (auto v : t.vertices())
        if (v < nLC)
          usedBits[v] |= 1 << s;
  }

  // Truth content of each layer cluster, by object.
  std::vector<std::vector<std::pair<int, float>>> lcComp(nLC);
  std::vector<float> lcTruth(nLC, 0.f);
  for (unsigned int k = 0; k < nLC; ++k) {
    std::unordered_map<int, float> c;
    for (auto const& [id, f] : lcs[k].hitsAndFractions()) {
      auto it = cellTruth.find(id.rawId());
      if (it == cellTruth.end())
        continue;
      for (auto const& [obj, e] : it->second)
        c[obj] += f * e;
    }
    for (auto const& [obj, e] : c) {
      lcComp[k].emplace_back(obj, e);
      lcTruth[k] += e;
    }
  }

  // Tree lcs.
  for (unsigned int k = 0; k < nLC; ++k) {
    auto const& lc = lcs[k];
    const bool all = lc.energy() >= lcAllAbove_;
    if (!all && (event_ * 7919u + k) % lcPrescale_ != 0)
      continue;
    lcWeight_ = all ? 1.f : static_cast<float>(lcPrescale_);
    const DetId seed = lc.seed();
    lcRegion_ = regionOf(seed);
    lcLayer_ = layerOf(seed);
    lcThick_ = thicknessOf(seed);
    lcNHits_ = lc.hitsAndFractions().size();
    lcMask_ = maskBits[k];
    lcUsed_ = usedBits[k];
    lcE_ = lc.energy();
    lcEta_ = lc.eta();
    lcTruth_ = lcTruth[k];
    int dom = -1;
    float domE = 0.f;
    for (auto const& [obj, e] : lcComp[k])
      if (e > domE) {
        domE = e;
        dom = obj;
      }
    lcPurity_ = lcTruth[k] > 0.f ? domE / lcTruth[k] : -1.f;
    lcSpecies_ = dom >= 0 ? speciesOfObj(dom) : -1;
    lcSignal_ = dom >= 0 ? isSignalObj(dom) : -1;
    lcT_->Fill();
  }

  // Tree objreg: deposit shares by region, after the filter (mask bit 1) and in CLUE3D / Recovery (stage bits 0, 1).
  std::unordered_map<int, std::array<double, 3>> inLC, afterFilter, inC3, inRec;
  std::unordered_map<int, std::map<int, std::vector<float>>> perLayerLC;  // obj -> (region*100+layer) -> LC shares
  for (unsigned int k = 0; k < nLC; ++k) {
    const DetId seed = lcs[k].seed();
    const int r = regionOf(seed);
    if (r < 0)
      continue;
    const int key = r * 100 + layerOf(seed);
    for (auto const& [obj, e] : lcComp[k]) {
      inLC[obj][r] += e;
      if (maskBits[k] & 2)
        afterFilter[obj][r] += e;
      if (usedBits[k] & 1)
        inC3[obj][r] += e;
      if (usedBits[k] & 2)
        inRec[obj][r] += e;
      perLayerLC[obj][key].push_back(e);
    }
  }
  for (auto const& [obj, dep] : objDeposit) {
    const bool sig = isSignalObj(obj);
    if (!sig && dep < minObjDeposit_)
      continue;
    orObj_ = obj;
    orSpecies_ = speciesOfObj(obj);
    orSignal_ = sig;
    orPdg_ = graph.particles()[obj].pdgId;
    orObjE_ = objE[obj];
    orEta_ = objE[obj] > 0. ? objEtaW[obj] / objE[obj] : 0.f;
    for (int r = 0; r < 3; ++r) {
      const double d = objRegDeposit[obj][r];
      if (d <= 0.)
        continue;
      orRegion_ = r;
      orDeposit_ = d;
      orInLC_ = inLC.count(obj) ? inLC[obj][r] : 0.;
      orAfterFilter_ = afterFilter.count(obj) ? afterFilter[obj][r] : 0.;
      orInCLUE3D_ = inC3.count(obj) ? inC3[obj][r] : 0.;
      orInRecovery_ = inRec.count(obj) ? inRec[obj][r] : 0.;
      // Splitting: mean number of layer clusters holding >= 10% of the object's energy in a layer, over the layers
      // where its captured energy is >= 2% of its deposit in the region.
      double sumN = 0.;
      int nLay = 0;
      if (auto it = perLayerLC.find(obj); it != perLayerLC.end())
        for (auto const& [key, shares] : it->second) {
          if (key / 100 != r)
            continue;
          double layerE = 0.;
          for (float e : shares)
            layerE += e;
          if (layerE < 0.02 * d)
            continue;
          int n = 0;
          for (float e : shares)
            n += e >= 0.1 * layerE ? 1 : 0;
          sumN += n;
          ++nLay;
        }
      orSplitLayers_ = nLay;
      orSplit_ = nLay > 0 ? sumN / nLay : 0.f;
      objregT_->Fill();
    }
  }

  // Trees ts and objs.
  for (size_t s = 0; s < stages.size(); ++s) {
    struct Best {
      float compl_ = 0.f, purity = 0.f, regE = 0.f, rawE = 0.f, attrib = 0.f;
      int n10 = 0;
    };
    std::unordered_map<int, Best> best;
    for (auto const& t : *stages[s]) {
      std::unordered_map<int, double> c;
      double tot = 0.;
      for (size_t i = 0; i < t.vertices().size(); ++i) {
        const auto v = t.vertices()[i];
        if (v >= nLC)
          continue;
        const float m = t.vertex_multiplicity().size() > i && t.vertex_multiplicity()[i] > 0.f
                            ? 1.f / t.vertex_multiplicity()[i]
                            : 1.f;
        for (auto const& [obj, e] : lcComp[v]) {
          c[obj] += m * e;
          tot += m * e;
        }
      }
      int dom = -1;
      double domE = 0.;
      int n10 = 0;
      for (auto const& [obj, e] : c) {
        if (e > domE) {
          domE = e;
          dom = obj;
        }
        if (tot > 0. && e >= 0.1 * tot)
          ++n10;
      }
      // Share of the trackster LC energy used by the linking output (stage 2) and by the final tracksters (last stage).
      double eAll = 0., eLinks = 0., eFinal = 0.;
      for (auto v : t.vertices())
        if (v < nLC) {
          const double e = lcs[v].energy();
          eAll += e;
          if (usedBits[v] & (1 << 2))
            eLinks += e;
          if (usedBits[v] & (1 << (stages.size() - 1)))
            eFinal += e;
        }
      tsInLinks_ = eAll > 0. ? eLinks / eAll : 0.f;
      tsInFinal_ = eAll > 0. ? eFinal / eAll : 0.f;
      tsStage_ = s;
      tsNLC_ = t.vertices().size();
      tsRawE_ = t.raw_energy();
      tsRawEmE_ = t.raw_em_energy();
      tsRegE_ = t.regressed_energy();
      tsEta_ = t.barycenter().eta();
      for (int k = 0; k < 8; ++k)
        tsProb_[k] = t.id_probabilities(k);
      tsTruthDeposit_ = tot;
      std::fill(std::begin(tsFrac_), std::end(tsFrac_), 0.f);
      for (auto const& [obj, e] : c) {
        const int a = std::abs(graph.particles()[obj].pdgId);
        const int sp = speciesOfObj(obj);
        const int k = a == 22 ? 0 : a == 11 ? 1 : a == 111 ? 2 : sp == 1 ? 3 : sp == 2 ? 4 : a == 13 ? 5 : 6;
        tsFrac_[k] += tot > 0. ? e / tot : 0.;
      }
      {
        namespace share = ticl::modelfeatures::neutralShare;
        std::array<float, share::kN> x{};
        ticl::modelfeatures::fillNeutralShareFeatures(t, lcs, {0.f, 0, 0.f}, std::span<float, share::kN>(x));
        std::copy(x.begin() + share::kMaxClusterShare, x.begin() + share::kAbsZExtent + 1, tsLcProf_);
      }
      tsPurity_ = tot > 0. ? domE / tot : -1.f;
      tsCompleteness_ = dom >= 0 && objDeposit[dom] > 0. ? domE / objDeposit[dom] : 0.f;
      tsDomSpecies_ = dom >= 0 ? speciesOfObj(dom) : -1;
      tsDomSignal_ = dom >= 0 ? isSignalObj(dom) : -1;
      tsDomPdg_ = dom >= 0 ? graph.particles()[dom].pdgId : 0;
      tsDomE_ = dom >= 0 ? objE[dom] : 0.f;
      tsNObj10_ = n10;
      tsT_->Fill();
      for (auto const& [obj, e] : c) {
        const double dep = objDeposit[obj];
        if (dep <= 0.)
          continue;
        auto& b = best[obj];
        const float cpl = e / dep;
        if (cpl >= 0.1f)
          ++b.n10;
        b.attrib += tot > 0. ? t.regressed_energy() * e / tot : 0.;
        if (cpl > b.compl_) {
          b.compl_ = cpl;
          b.purity = tot > 0. ? e / tot : 0.;
          b.regE = t.regressed_energy();
          b.rawE = t.raw_energy();
        }
      }
    }
    for (auto const& [obj, dep] : objDeposit) {
      const bool sig = isSignalObj(obj);
      if (!sig && dep < minObjDeposit_)
        continue;
      osStage_ = s;
      osSpecies_ = speciesOfObj(obj);
      osSignal_ = sig;
      osPdg_ = graph.particles()[obj].pdgId;
      osObjE_ = objE[obj];
      osEta_ = objE[obj] > 0. ? objEtaW[obj] / objE[obj] : 0.f;
      auto it = best.find(obj);
      const Best b = it != best.end() ? it->second : Best{};
      osBestCompl_ = b.compl_;
      osBestPurity_ = b.purity;
      osBestRegE_ = b.regE;
      osBestRawE_ = b.rawE;
      osNTs10_ = b.n10;
      osAttribRegE_ = b.attrib;
      objsT_->Fill();
    }
  }

  // Tree cands.
  for (auto const& c : cands) {
    cPdg_ = c.pdgId();
    cCharge_ = c.charge();
    cNTs_ = c.tracksters().size();
    cE_ = c.energy();
    cEta_ = c.eta();
    std::unordered_map<int, double> comp;
    double tot = 0.;
    for (auto const& t : c.tracksters())
      for (auto v : t->vertices())
        if (v < nLC)
          for (auto const& [obj, e] : lcComp[v]) {
            comp[obj] += e;
            tot += e;
          }
    int dom = -1;
    double domE = 0.;
    for (auto const& [obj, e] : comp)
      if (e > domE) {
        domE = e;
        dom = obj;
      }
    cPurity_ = tot > 0. ? domE / tot : -1.f;
    cDomSpecies_ = dom >= 0 ? speciesOfObj(dom) : -1;
    cDomSignal_ = dom >= 0 ? isSignalObj(dom) : -1;
    cDomE_ = dom >= 0 ? objE[dom] : 0.f;
    cTrackSpecies_ = -1;
    if (!c.trackPtrs().empty() && c.trackPtrs()[0].isNonnull()) {
      const auto key = c.trackPtrs()[0].key();
      if (key < tkR2t.size() && !tkR2t[key].empty()) {
        auto b = std::min_element(
            tkR2t[key].begin(), tkR2t[key].end(), [](auto const& a, auto const& b) { return a.score() < b.score(); });
        if (b->index() < nP)
          cTrackSpecies_ = speciesOfObj(objectFor(b->index()));
      }
    }
    candT_->Fill();
  }
}

DEFINE_FWK_MODULE(StageTruthProbe);
