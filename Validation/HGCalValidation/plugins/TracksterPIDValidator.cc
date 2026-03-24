/**
DQM plots for trackster PID, based on simulation truth
Takes as input a trackster collection and a mask to build the baseline (efficiency denominator)
 Computes the efficiency as EFF=#(baseline & passing PID cut) / #(baseline)
Another mask is used to define a "fake" baseline region (populated by unmatched tracksters),
 the fake rate is then defined as FR=#(fake baseline & passing signal PID cut)/#(fake baseline)

Author: Théo Cuisset (LLR)
*/
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"

using namespace ticl;

struct Histograms_TracksterPIDValidation {
  dqm::reco::MonitorElement* pt_eta_pid_;  // 3D histo pt-abs(eta)-PID value

  dqm::reco::MonitorElement* pt_eta_reco2SimSelected_;  // 2D pt-eta after selection on reco2Sim score but before PID cut
  dqm::reco::MonitorElement* pt_eta_noReco2SimSelection_;  // 2D pt-eta before selection on reco2Sim score (->denominator)
  dqm::reco::MonitorElement* pt_eta_pidNum_;               // 2D pt-eta after PID cut

  dqm::reco::MonitorElement* pt_eta_pid_fakes_;     // 3D histo pt-abs(eta)-PID value in fakes region
  dqm::reco::MonitorElement* pt_eta_fakes_;         // 2D pt-eta fakes selection before PID cut
  dqm::reco::MonitorElement* pt_eta_fakes_pid_Num;  // 2D pt-eta fakes selection after PID cut
};

class TICLTracksterPIDValidation : public DQMGlobalEDAnalyzer<Histograms_TracksterPIDValidation> {
public:
  explicit TICLTracksterPIDValidation(const edm::ParameterSet&);
  ~TICLTracksterPIDValidation() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&,
                      edm::Run const&,
                      edm::EventSetup const&,
                      Histograms_TracksterPIDValidation&) const override;

  void dqmAnalyze(edm::Event const&, edm::EventSetup const&, Histograms_TracksterPIDValidation const&) const override;

  std::string folder_;
  edm::EDGetTokenT<ticl::TracksterCollection> tracksters_token_;
  edm::EDGetTokenT<std::vector<int>> tracksters_mask_token_;
  edm::EDGetTokenT<std::vector<int>> tracksters_mask_fakes_token_;

  bool doFakes_;

  double pidCut_;

  std::vector<ticl::Trackster::ParticleType> pidsToConsider_;
  std::string label_;
};

TICLTracksterPIDValidation::TICLTracksterPIDValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      tracksters_token_(consumes<ticl::TracksterCollection>(iConfig.getParameter<edm::InputTag>("tracksters"))),
      doFakes_(iConfig.exists("tracksterMaskFakes")),
      pidCut_(iConfig.getParameter<double>("pidCut")),
      pidsToConsider_({ticl::Trackster::ParticleType::electron, ticl::Trackster::ParticleType::photon}) {
  if (iConfig.exists("tracksterMask"))
    tracksters_mask_token_ = consumes<std::vector<int>>(iConfig.getParameter<edm::InputTag>("tracksterMask"));
  if (iConfig.exists("tracksterMaskFakes"))
    tracksters_mask_fakes_token_ =
        consumes<std::vector<int>>(iConfig.getParameter<edm::InputTag>("tracksterMaskFakes"));
  else
    edm::LogInfo("PIDValidation") << "Not using any mask for fakes, will not run validation for fakes";
}

void TICLTracksterPIDValidation::dqmAnalyze(edm::Event const& iEvent,
                                            edm::EventSetup const& iSetup,
                                            Histograms_TracksterPIDValidation const& histos) const {
  ticl::TracksterCollection const& tracksters = iEvent.get(tracksters_token_);
  std::vector<int> const& tracksterMask = tracksters_mask_token_.isUninitialized()
                                              ? std::vector<int>(tracksters.size(), 1)
                                              : iEvent.get(tracksters_mask_token_);
  if (tracksters_mask_token_.isUninitialized())
    edm::LogWarning("PIDValidation") << "Not using any mask";

  assert(tracksterMask.size() == tracksters.size());

  auto doFill = [](dqm::reco::MonitorElement* h, Trackster const& ts) {
    h->Fill(ts.raw_pt(), std::abs(ts.barycenter().eta()));
  };

  // Signal
  for (std::size_t i = 0; i < tracksters.size(); i++) {
    if (tracksterMask[i] == 1)
      continue;  // Trackster is not the best-matched one to CaloParticle
    ticl::Trackster const& ts = tracksters[i];

    doFill(histos.pt_eta_noReco2SimSelection_, ts);

    if (tracksterMask[i] != 0)
      continue;  // Trackster is not the best-matched one to CaloParticle

    doFill(histos.pt_eta_reco2SimSelected_, ts);

    const double pidValue = std::transform_reduce(
        pidsToConsider_.begin(), pidsToConsider_.end(), 0., std::plus<>{}, [&ts](Trackster::ParticleType partType) {
          return ts.id_probability(partType);
        });

    histos.pt_eta_pid_->Fill(ts.raw_pt(), std::abs(ts.barycenter().eta()), pidValue);

    if (pidValue > pidCut_)
      doFill(histos.pt_eta_pidNum_, ts);
  }

  // Fakes
  if (!doFakes_) {
    return;
  }

  std::vector<int> const& tracksterMaskFakes = iEvent.get(tracksters_mask_fakes_token_);
  assert(tracksterMaskFakes.size() == tracksters.size());

  for (std::size_t i = 0; i < tracksters.size(); i++) {
    if (tracksterMaskFakes[i] != 0)
      continue;  // Trackster is not in the fakes mask (too signal like for example)
    ticl::Trackster const& ts = tracksters[i];

    doFill(histos.pt_eta_fakes_, ts);

    const double pidValue = std::transform_reduce(
        pidsToConsider_.begin(), pidsToConsider_.end(), 0., std::plus<>{}, [&ts](Trackster::ParticleType partType) {
          return ts.id_probability(partType);
        });
    histos.pt_eta_pid_fakes_->Fill(ts.raw_pt(), std::abs(ts.barycenter().eta()), pidValue);

    doFill(histos.pt_eta_fakes_, ts);

    if (pidValue > pidCut_)
      doFill(histos.pt_eta_fakes_pid_Num, ts);
  }
}

void TICLTracksterPIDValidation::bookHistograms(DQMStore::IBooker& ibook,
                                                edm::Run const& run,
                                                edm::EventSetup const& iSetup,
                                                Histograms_TracksterPIDValidation& histos) const {
  ibook.setCurrentFolder(folder_);
  constexpr std::array<float, 20> ptBins = {0., 0.5, 1.,  1.5, 2.,  2.5, 3.,  4.,  5.,  6.,
                                            7., 8.,  10., 12., 15., 20., 30., 40., 50., 100.};
  //    constexpr std::array etaBins = {1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1};
  constexpr std::array<float, 9> etaBins = {1.6, 1.8, 2.1, 2.3, 2.5, 2.7, 2.8, 2.9, 3.};
  constexpr std::array<float, 50> pidBins = {
      0.,         0.02040816, 0.04081633, 0.06122449, 0.08163265, 0.10204082, 0.12244898, 0.14285714, 0.16326531,
      0.18367347, 0.20408163, 0.2244898,  0.24489796, 0.26530612, 0.28571429, 0.30612245, 0.32653061, 0.34693878,
      0.36734694, 0.3877551,  0.40816327, 0.42857143, 0.44897959, 0.46938776, 0.48979592, 0.51020408, 0.53061224,
      0.55102041, 0.57142857, 0.59183673, 0.6122449,  0.63265306, 0.65306122, 0.67346939, 0.69387755, 0.71428571,
      0.73469388, 0.75510204, 0.7755102,  0.79591837, 0.81632653, 0.83673469, 0.85714286, 0.87755102, 0.89795918,
      0.91836735, 0.93877551, 0.95918367, 0.97959184, 1.};
  auto make3D = [&](const char* name, const char* title) -> TH3F* {
    return new TH3F(name,
                    title,
                    ptBins.size() - 1,
                    ptBins.data(),  // pt
                    etaBins.size() - 1,
                    etaBins.data(),  //abs(eta)
                    pidBins.size() - 1,
                    pidBins.data()  //PID
    );
  };
  histos.pt_eta_pid_ = ibook.book3D("pt_eta_pid", make3D("pt_eta_pid", "Pt-abs(eta)-PID value for trackster (signal)"));
  histos.pt_eta_pid_fakes_ =
      ibook.book3D("pt_eta_pid_fakes", make3D("pt_eta_pid_fakes", "Pt-abs(eta)-PID value for trackster (fakes)"));

  histos.pt_eta_noReco2SimSelection_ =
      ibook.book2D("pt_eta_noReco2SimSelection",
                   "Pt-abs(eta) for the trackster best-associated (shared energy) to sim",
                   ptBins.size() - 1,
                   ptBins.data(),  // pt
                   etaBins.size() - 1,
                   etaBins.data()  //abs(eta)
      );
  histos.pt_eta_reco2SimSelected_ = ibook.book2D(
      "pt_eta_reco2SimSelected",
      "Pt-abs(eta) for the trackster best-associated (shared energy) to sim (additionally passing reco2sim cut)",
      ptBins.size() - 1,
      ptBins.data(),  // pt
      etaBins.size() - 1,
      etaBins.data()  //abs(eta)
  );

  histos.pt_eta_pidNum_ = ibook.book2D("pt_eta_pidNum",
                                       "Pt-abs(eta) for trackster after PID cut",
                                       ptBins.size() - 1,
                                       ptBins.data(),  // pt
                                       etaBins.size() - 1,
                                       etaBins.data()  //abs(eta)
  );

  // fakes
  if (doFakes_) {
    histos.pt_eta_fakes_ = ibook.book2D("pt_eta_fakes",
                                        "Pt-abs(eta) for the 'fake' tracksters before PID selection",
                                        ptBins.size() - 1,
                                        ptBins.data(),  // pt
                                        etaBins.size() - 1,
                                        etaBins.data()  //abs(eta)
    );
    histos.pt_eta_fakes_pid_Num =
        ibook.book2D("pt_eta_fakes_pid_Num",
                     "Pt-abs(eta) for the 'fake' tracksters after (signal) PID selection (numerator for fake rate)",
                     ptBins.size() - 1,
                     ptBins.data(),  // pt
                     etaBins.size() - 1,
                     etaBins.data()  //abs(eta)
        );
  }
}

void TICLTracksterPIDValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "HGCAL/TICLTracksterPIDValidation/");  // Please keep the trailing '/'
  desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersCLUE3DHigh"));
  desc.addOptional<edm::InputTag>("tracksterMask");
  desc.addOptional<edm::InputTag>("tracksterMaskFakes");
  desc.add<double>("pidCut", 0.5)->setComment("Cut on the PID score to apply while making PID efficiency plots");
  descriptions.add("ticlTracksterPIDValidation", desc);
}

DEFINE_FWK_MODULE(TICLTracksterPIDValidation);
