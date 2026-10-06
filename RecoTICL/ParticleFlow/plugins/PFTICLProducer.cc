// This producer converts a list of TICLCandidates to a list of PFCandidates.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/HGCalReco/interface/TICLCandidate.h"

#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "RecoTICL/Interpretation/interface/MuonKinematics.h"

class PFTICLProducer : public edm::stream::EDProducer<> {
public:
  PFTICLProducer(const edm::ParameterSet&);
  ~PFTICLProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // parameters
  const bool energy_from_regression_;
  // The candidate producer has set the muon kinematics and records its muon decisions: the converter only copies them.
  const bool muonsFromCandidates_;
  // inputs
  const edm::EDGetTokenT<edm::View<TICLCandidate>> ticl_candidates_;
  edm::EDGetTokenT<edm::ValueMap<reco::MuonRef>> candidateMuons_;
  edm::EDGetTokenT<std::vector<int>> muonTrackType_;
  // Read by PFMuonAlgo when the candidates do not carry the muon decisions.
  const edm::EDGetTokenT<reco::MuonCollection> muons_;
  // For PFMuonAlgo
  std::unique_ptr<PFMuonAlgo> pfmu_;
};

DEFINE_FWK_MODULE(PFTICLProducer);

PFTICLProducer::PFTICLProducer(const edm::ParameterSet& conf)
    : energy_from_regression_(conf.getParameter<bool>("energyFromRegression")),
      muonsFromCandidates_(conf.getParameter<bool>("muonsFromCandidates")),
      ticl_candidates_(consumes<edm::View<TICLCandidate>>(conf.getParameter<edm::InputTag>("ticlCandidateSrc"))),
      muons_(muonsFromCandidates_ ? edm::EDGetTokenT<reco::MuonCollection>()
                                  : consumes<reco::MuonCollection>(conf.getParameter<edm::InputTag>("muonSrc"))),
      pfmu_(std::make_unique<PFMuonAlgo>(conf.getParameterSet("pfMuonAlgoParameters"),
                                         false)) {  // postMuonCleaning = false
  if (muonsFromCandidates_) {
    const auto& src = conf.getParameter<edm::InputTag>("ticlCandidateSrc");
    candidateMuons_ = consumes<edm::ValueMap<reco::MuonRef>>(edm::InputTag(src.label(), "muons", src.process()));
    muonTrackType_ = consumes<std::vector<int>>(edm::InputTag(src.label(), "muonTrackType", src.process()));
  }
  produces<reco::PFCandidateCollection>();
}

void PFTICLProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("ticlCandidateSrc", edm::InputTag("ticlCandidate"));
  desc.add<bool>("energyFromRegression", true);
  desc.add<bool>("muonsFromCandidates", false)
      ->setComment(
          "Copy the muon decisions of the candidate producer (the muon of each candidate and its track type) instead "
          "of PFMuonAlgo.");
  // For PFMuonAlgo
  desc.add<edm::InputTag>("muonSrc", edm::InputTag("muons1stStep"));
  edm::ParameterSetDescription psd_PFMuonAlgo;
  PFMuonAlgo::fillPSetDescription(psd_PFMuonAlgo);
  desc.add<edm::ParameterSetDescription>("pfMuonAlgoParameters", psd_PFMuonAlgo);
  //
  descriptions.add("pfTICLProducer", desc);
}

void PFTICLProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  //get TICLCandidates
  edm::Handle<edm::View<TICLCandidate>> ticl_cand_h;
  evt.getByToken(ticl_candidates_, ticl_cand_h);
  const auto& ticl_candidates = *ticl_cand_h;
  edm::Handle<reco::MuonCollection> muonH;
  edm::Handle<edm::ValueMap<reco::MuonRef>> candidateMuonsH;
  static const std::vector<int> noMuons;
  if (muonsFromCandidates_)
    candidateMuonsH = evt.getHandle(candidateMuons_);
  else
    muonH = evt.getHandle(muons_);
  const auto& muonTrackType = muonsFromCandidates_ ? evt.get(muonTrackType_) : noMuons;
  if (muonsFromCandidates_ && muonTrackType.size() != ticl_candidates.size())
    throw cms::Exception("LogicError") << "PFTICLProducer: " << ticl_candidates.size() << " candidates but "
                                       << muonTrackType.size() << " muon track types";

  auto candidates = std::make_unique<reco::PFCandidateCollection>();

  for (size_t iCand = 0; iCand < ticl_candidates.size(); ++iCand) {
    const auto& ticl_cand = ticl_candidates[iCand];
    const auto abs_pdg_id = std::abs(ticl_cand.pdgId());
    const auto charge = ticl_cand.charge();
    const auto& four_mom = ticl_cand.p4();
    float total_raw_energy = 0.f;
    float total_em_raw_energy = 0.f;
    for (const auto& t : ticl_cand.tracksters()) {
      total_raw_energy += t->raw_energy();
      total_em_raw_energy += t->raw_em_energy();
    }
    float ecal_energy_fraction = (total_raw_energy > 0.f) ? (total_em_raw_energy / total_raw_energy) : 0.f;
    float ecal_energy = energy_from_regression_ ? ticl_cand.p4().energy() * ecal_energy_fraction
                                                : ticl_cand.rawEnergy() * ecal_energy_fraction;
    float hcal_energy =
        energy_from_regression_ ? ticl_cand.p4().energy() - ecal_energy : ticl_cand.rawEnergy() - ecal_energy;
    // fix for floating point rounding could go slightly below 0
    hcal_energy = std::max(0.f, hcal_energy);
    reco::PFCandidate::ParticleType part_type;
    switch (abs_pdg_id) {
      case 11:
        part_type = reco::PFCandidate::e;
        break;
      case 13:
        part_type = reco::PFCandidate::mu;
        break;
      // A pi0 is an EM object; PFCandidate has no pi0 type.
      case 22:
      case 111:
        part_type = reco::PFCandidate::gamma;
        break;
      case 130:
        part_type = reco::PFCandidate::h0;
        break;
      case 211:
        part_type = reco::PFCandidate::h;
        break;
      default:
        part_type = reco::PFCandidate::X;
    }

    candidates->emplace_back(charge, four_mom, part_type);

    auto& candidate = candidates->back();
    candidate.setEcalEnergy(ecal_energy, ecal_energy);
    candidate.setHcalEnergy(hcal_energy, hcal_energy);
    if (candidate.charge()) {  // otherwise PFCandidate throws
      // Construct edm::Ref from edm::Ptr. As of now, assumes type to be reco::Track. To be extended (either via
      // dynamic type checking or configuration) if additional track types are needed.
      reco::TrackRef trackref(ticl_cand.trackPtr().id(), int(ticl_cand.trackPtr().key()), &evt.productGetter());
      candidate.setTrackRef(trackref);
      if (muonsFromCandidates_) {
        if (const auto& muonRef = candidateMuonsH->get(ticl_cand_h.id(), iCand); muonRef.isNonnull()) {
          candidate.setMuonRef(muonRef);
          candidate.setMuonTrackType(static_cast<reco::Muon::MuonTrackType>(muonTrackType[iCand]));
          candidate.setVertex(ticl_cand.vertex());
        }
      } else if (const int muId = PFMuonAlgo::muAssocToTrack(trackref, *muonH); muId != -1) {
        // Utilize PFMuonAlgo
        const reco::MuonRef muonref = reco::MuonRef(muonH, muId);
        const bool allowLoose = (part_type == reco::PFCandidate::mu);
        if (ticl::takesMuonKinematics(muonref, allowLoose, !ticl_cand.tracksters().empty())) {
          // Redefine pfmuon candidate kinematics and add muonref
          pfmu_->reconstructMuon(candidate, muonref, allowLoose);
        }
      }
    }

    candidate.setTime(ticl_cand.time(), ticl_cand.timeError());
  }

  evt.put(std::move(candidates));
}
