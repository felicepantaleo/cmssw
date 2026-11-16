// Study probe, not for integration. One TTree row per (trackster collection, truth target of one
// level) in the endcap: the validator outcome inputs and the geometry of the two largest fragments.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"

#include "TTree.h"

class CornettoLinkingProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit CornettoLinkingProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
  struct Coll {
    std::string name;
    edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksters;
    edm::EDGetTokenT<MapType> truthToReco;
    edm::EDGetTokenT<MapType> recoToTruth;
  };
  edm::EDGetTokenT<truth::Graph> graphToken_;
  edm::EDGetTokenT<std::vector<unsigned int>> targetsToken_;
  std::vector<Coll> colls_;
  const double minAbsEta_, maxAbsEta_, minPt_;

  TTree* tree_;
  int event_, coll_, pdgId_, signal_;
  float pt_, eta_, energy_;
  float leadFrac_, leadRecoScore_, sumFrac_, leadE_;
  int nFrag_;
  float secFrac_, secE_, secRecoScore_, sProj_, dTProj_, sPca_, dTPca_, dEta_, dPhi_, dTime_;
};

CornettoLinkingProbe::CornettoLinkingProbe(const edm::ParameterSet& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      targetsToken_(consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("targets"))),
      minAbsEta_(cfg.getParameter<double>("minAbsEta")),
      maxAbsEta_(cfg.getParameter<double>("maxAbsEta")),
      minPt_(cfg.getParameter<double>("minPt")) {
  usesResource(TFileService::kSharedResource);
  const auto associator = cfg.getParameter<std::string>("associator");
  for (auto const& name : cfg.getParameter<std::vector<std::string>>("collections")) {
    colls_.push_back({name,
                      consumes<std::vector<ticl::Trackster>>(edm::InputTag(name)),
                      consumes<MapType>(edm::InputTag(associator, name + "TruthToReco")),
                      consumes<MapType>(edm::InputTag(associator, name + "RecoToTruthFixed"))});
  }
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("probe", "probe");
  tree_->Branch("event", &event_);
  tree_->Branch("coll", &coll_);
  tree_->Branch("pdgId", &pdgId_);
  tree_->Branch("signal", &signal_);
  tree_->Branch("pt", &pt_);
  tree_->Branch("eta", &eta_);
  tree_->Branch("energy", &energy_);
  tree_->Branch("leadFrac", &leadFrac_);
  tree_->Branch("leadRecoScore", &leadRecoScore_);
  tree_->Branch("sumFrac", &sumFrac_);
  tree_->Branch("leadE", &leadE_);
  tree_->Branch("nFrag", &nFrag_);
  tree_->Branch("secFrac", &secFrac_);
  tree_->Branch("secE", &secE_);
  tree_->Branch("secRecoScore", &secRecoScore_);
  tree_->Branch("sProj", &sProj_);
  tree_->Branch("dTProj", &dTProj_);
  tree_->Branch("sPca", &sPca_);
  tree_->Branch("dTPca", &dTPca_);
  tree_->Branch("dEta", &dEta_);
  tree_->Branch("dPhi", &dPhi_);
  tree_->Branch("dTime", &dTime_);
}

void CornettoLinkingProbe::analyze(const edm::Event& event, const edm::EventSetup&) {
  auto const& graph = event.get(graphToken_);
  auto const& targets = event.get(targetsToken_);
  event_ = event.id().event();
  for (std::size_t c = 0; c < colls_.size(); ++c) {
    coll_ = c;
    auto const& ts = event.get(colls_[c].tracksters);
    auto const& t2r = event.get(colls_[c].truthToReco).getMap();
    auto const& r2t = event.get(colls_[c].recoToTruth).getMap();
    auto recoScoreOf = [&r2t](unsigned int r, unsigned int b) {
      if (r >= r2t.size())
        return 1.f;
      for (auto const& m : r2t[r])
        if (m.index() == b)
          return m.score();
      return 1.f;
    };
    for (unsigned int b : targets) {
      if (b >= graph.particles().size())
        continue;
      auto const& p4 = graph.particles()[b].momentum;
      if (p4.pt() < minPt_ || std::abs(p4.eta()) < minAbsEta_ || std::abs(p4.eta()) > maxAbsEta_)
        continue;
      pt_ = p4.pt();
      pdgId_ = graph.particles()[b].pdgId;
      signal_ = graph.particles()[b].isSignal() ? 1 : 0;
      eta_ = p4.eta();
      energy_ = p4.E();
      std::vector<std::pair<float, unsigned int>> frags;  // (shared fraction, reco index)
      sumFrac_ = 0.f;
      if (b < t2r.size())
        for (auto const& m : t2r[b]) {
          frags.emplace_back(m.sharedEnergy(), m.index());
          sumFrac_ += m.sharedEnergy();
        }
      std::sort(frags.begin(), frags.end(), [](auto const& x, auto const& y) { return x.first > y.first; });
      nFrag_ = std::count_if(frags.begin(), frags.end(), [](auto const& f) { return f.first > 0.05f; });
      leadFrac_ = leadRecoScore_ = leadE_ = -1.f;
      secFrac_ = secE_ = secRecoScore_ = sProj_ = dTProj_ = sPca_ = dTPca_ = dEta_ = dPhi_ = dTime_ = -999.f;
      if (!frags.empty() && frags[0].second < ts.size()) {
        auto const& lead = ts[frags[0].second];
        leadFrac_ = frags[0].first;
        leadRecoScore_ = recoScoreOf(frags[0].second, b);
        leadE_ = lead.raw_energy();
        if (frags.size() > 1 && frags[1].second < ts.size()) {
          auto const& sec = ts[frags[1].second];
          secFrac_ = frags[1].first;
          secE_ = sec.raw_energy();
          secRecoScore_ = recoScoreOf(frags[1].second, b);
          const auto D = sec.barycenter() - lead.barycenter();
          const auto proj = lead.barycenter().Unit();
          sProj_ = D.Dot(proj);
          dTProj_ = std::sqrt(std::max(0.f, static_cast<float>(D.Mag2()) - sProj_ * sProj_));
          const auto pca = lead.eigenvectors(0);
          sPca_ = D.Dot(pca);
          dTPca_ = std::sqrt(std::max(0.f, static_cast<float>(D.Mag2()) - sPca_ * sPca_));
          dEta_ = sec.barycenter().eta() - lead.barycenter().eta();
          dPhi_ = reco::deltaPhi(sec.barycenter().phi(), lead.barycenter().phi());
          if (lead.timeError() > 0.f && sec.timeError() > 0.f)
            dTime_ = (sec.time() - lead.time()) /
                     std::sqrt(lead.timeError() * lead.timeError() + sec.timeError() * sec.timeError());
        }
      }
      tree_->Fill();
    }
  }
}

DEFINE_FWK_MODULE(CornettoLinkingProbe);
