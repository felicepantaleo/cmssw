// Author: Felice Pantaleo (CERN) - felice.pantaleo@cern.ch
// Date: 10/2026
//
// Candidate assembly: the TICLCandidates from the final tracksters and the per-track assignment maps of
// TICLInterpretationProducer. The GSF tracks are downstream of the final tracksters: an electron candidate takes the
// direction and the charge of its GSF track.

#include <cmath>
#include <memory>
#include <algorithm>
#include <map>
#include <limits>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/HGCalReco/interface/MtdHostCollection.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "RecoTICL/Interpretation/interface/AssignmentMaps.h"
#include "RecoTICL/Interpretation/interface/CandidateTime.h"
#include "RecoTICL/Interpretation/interface/MuonKinematics.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "Geometry/CommonTopologies/interface/GlobalTrackingGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

using namespace ticl;

class TICLCandidateArbitrationProducer
    : public edm::stream::EDProducer<edm::GlobalCache<cms::Ort::ONNXRuntime>, edm::stream::WatchRuns> {
public:
  TICLCandidateArbitrationProducer(const edm::ParameterSet &ps, const cms::Ort::ONNXRuntime *neutralShareModel);
  static std::unique_ptr<cms::Ort::ONNXRuntime> initializeGlobalCache(const edm::ParameterSet &ps);
  static void globalEndJob(const cms::Ort::ONNXRuntime *) {}
  void produce(edm::Event &, const edm::EventSetup &) override;
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  const edm::EDGetTokenT<std::vector<Trackster>> tracksters_token_;
  const edm::EDGetTokenT<std::vector<int>> trackToTrackster_token_;
  const edm::EDGetTokenT<std::vector<int>> trackMode_token_;
  const edm::EDGetTokenT<std::vector<int>> neutralIdx_token_;
  const edm::EDGetTokenT<std::vector<int>> neutralPdg_token_;
  const edm::EDGetTokenT<std::vector<reco::Track>> tracks_token_;
  // Without the GSF tracks an electron keeps the track kinematics.
  const bool useGsfTracks_;
  edm::EDGetTokenT<std::vector<reco::GsfTrack>> gsf_tracks_token_;
  edm::EDGetTokenT<MtdHostCollection> inputTimingToken_;
  const bool useMTDTiming_;
  const bool useTimingAverage_;
  const float timingQualityThreshold_;
  // (eta, phi) window between an electron track and its GSF track.
  const float delta_tk_gsf_;
  // A trackster shared by tracks that take their energy from the track gives a neutral residual E - sum(p) when it is
  // at least max(floor, fraction x E).
  const float residualEnergyFloor_;
  const float residualEnergyFraction_;
  // A charged candidate on an EM trackster takes the inverse-variance mean of the track energy and the trackster energy
  // when they agree within energyCompatibilityNSigma. Otherwise a charged hadron takes the track energy and an electron
  // keeps the trackster energy. Momentum error: trackMomentumErrorScale x sigma(p) of the track, or
  // gsfMomentumErrorScale x sigma(p) of the GSF mode for an electron with a GSF track. EM trackster error:
  // sigma/E = sqrt(S^2/E + C^2) at the track energy.
  const float trackMomentumErrorScale_;
  const float gsfMomentumErrorScale_;
  const float emTracksterStochastic_;
  const float emTracksterConstant_;
  const float energyCompatibilityNSigma_;

  // A charged candidate whose track belongs to a muon takes the kinematics of the best muon track (PFMuonAlgo).
  const edm::EDGetTokenT<reco::MuonCollection> muons_token_;
  std::unique_ptr<PFMuonAlgo> pfmu_;

  // Share of a neutral trackster energy that comes from the particle it represents, predicted from 31 features of the
  // trackster, its shape and its surroundings. A neutral candidate below neutralMinEnergy_ after the correction is not
  // produced.
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClustersToken_;
  const cms::Ort::ONNXRuntime &neutralShareModel_;
  const float neutralMinEnergy_;
  static constexpr unsigned int kNNeutralShareFeatures = 31;
  std::vector<float> neutralEnergyShares(const std::vector<Trackster> &tracksters,
                                         const std::vector<reco::CaloCluster> &layerClusters,
                                         const std::vector<int> &neutralIdx) const;

  // Energy of a momentum p with error sigmaP and mass hypothesis mass on an EM trackster: the inverse-variance mean of
  // the two when they are compatible, else the track energy.
  struct CombinedEnergy {
    float energy;
    bool compatible;
  };
  CombinedEnergy combinedEnergy(float p, float sigmaP, float mass, const Trackster &ts) const;

  const std::string propName_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bfield_token_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagator_token_;
  const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> trackingGeometry_token_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hdc_token_;

  const HGCalDDDConstants *hgcons_;
  edm::ESHandle<MagneticField> bfield_;
  edm::ESHandle<Propagator> propagator_;
  edm::ESHandle<GlobalTrackingGeometry> trackingGeometry_;
};

namespace {
  // InputTag of a product instance of the interpretation stage.
  edm::InputTag interpretationTag(const edm::ParameterSet &ps, const char *instance) {
    const auto &tag = ps.getParameter<edm::InputTag>("interpretations");
    return edm::InputTag(tag.label(), instance, tag.process());
  }
}  // namespace

std::unique_ptr<cms::Ort::ONNXRuntime> TICLCandidateArbitrationProducer::initializeGlobalCache(
    const edm::ParameterSet &ps) {
  return std::make_unique<cms::Ort::ONNXRuntime>(ps.getParameter<edm::FileInPath>("neutralShareModel").fullPath());
}

TICLCandidateArbitrationProducer::TICLCandidateArbitrationProducer(const edm::ParameterSet &ps,
                                                                   const cms::Ort::ONNXRuntime *neutralShareModel)
    : tracksters_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("interpretations"))),
      trackToTrackster_token_(consumes<std::vector<int>>(interpretationTag(ps, "trackToTrackster"))),
      trackMode_token_(consumes<std::vector<int>>(interpretationTag(ps, "trackMode"))),
      neutralIdx_token_(consumes<std::vector<int>>(interpretationTag(ps, "neutralIdx"))),
      neutralPdg_token_(consumes<std::vector<int>>(interpretationTag(ps, "neutralPdg"))),
      tracks_token_(consumes<std::vector<reco::Track>>(ps.getParameter<edm::InputTag>("tracks"))),
      useGsfTracks_(ps.getParameter<bool>("useGsfTracks")),
      useMTDTiming_(ps.getParameter<bool>("useMTDTiming")),
      useTimingAverage_(ps.getParameter<bool>("useTimingAverage")),
      timingQualityThreshold_(ps.getParameter<float>("timingQualityThreshold")),
      delta_tk_gsf_(ps.getParameter<float>("delta_tk_gsf")),
      residualEnergyFloor_(ps.getParameter<float>("residualEnergyFloor")),
      residualEnergyFraction_(ps.getParameter<float>("residualEnergyFraction")),
      trackMomentumErrorScale_(ps.getParameter<float>("trackMomentumErrorScale")),
      gsfMomentumErrorScale_(ps.getParameter<float>("gsfMomentumErrorScale")),
      emTracksterStochastic_(ps.getParameter<float>("emTracksterStochastic")),
      emTracksterConstant_(ps.getParameter<float>("emTracksterConstant")),
      energyCompatibilityNSigma_(ps.getParameter<float>("energyCompatibilityNSigma")),
      muons_token_(consumes<reco::MuonCollection>(ps.getParameter<edm::InputTag>("muonSrc"))),
      pfmu_(std::make_unique<PFMuonAlgo>(ps.getParameterSet("pfMuonAlgoParameters"), false)),
      layerClustersToken_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layerClusters"))),
      neutralShareModel_(*neutralShareModel),
      neutralMinEnergy_(ps.getParameter<float>("neutralMinEnergy")),
      propName_(ps.getParameter<std::string>("propagator")),
      bfield_token_(esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()),
      propagator_token_(
          esConsumes<Propagator, TrackingComponentsRecord, edm::Transition::BeginRun>(edm::ESInputTag("", propName_))),
      trackingGeometry_token_(
          esConsumes<GlobalTrackingGeometry, GlobalTrackingGeometryRecord, edm::Transition::BeginRun>()),
      hdc_token_(esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag("", "HGCalEESensitive"))),
      hgcons_(nullptr) {
  if (useMTDTiming_) {
    inputTimingToken_ = consumes<MtdHostCollection>(ps.getParameter<edm::InputTag>("timingSoA"));
  }
  if (useGsfTracks_) {
    gsf_tracks_token_ = consumes<std::vector<reco::GsfTrack>>(ps.getParameter<edm::InputTag>("gsf_tracks"));
  }
  produces<std::vector<TICLCandidate>>();
  // Per candidate: its muon (null when none) and the type of the muon track it takes (-1 when none).
  produces<edm::ValueMap<reco::MuonRef>>("muons");
  produces<std::vector<int>>("muonTrackType");
}

void TICLCandidateArbitrationProducer::beginRun(edm::Run const &, edm::EventSetup const &es) {
  edm::ESHandle<HGCalDDDConstants> hdc = es.getHandle(hdc_token_);
  hgcons_ = hdc.product();
  bfield_ = es.getHandle(bfield_token_);
  propagator_ = es.getHandle(propagator_token_);
  trackingGeometry_ = es.getHandle(trackingGeometry_token_);
}

TICLCandidateArbitrationProducer::CombinedEnergy TICLCandidateArbitrationProducer::combinedEnergy(
    float p, float sigmaP, float mass, const Trackster &ts) const {
  const float eTrack = std::sqrt(p * p + mass * mass);
  const float sigmaTrack = sigmaP;
  const float eTrackster = ts.regressed_energy();
  const float sigmaTrackster =
      eTrack * std::sqrt(emTracksterStochastic_ * emTracksterStochastic_ / std::max(eTrack, 0.1f) +
                         emTracksterConstant_ * emTracksterConstant_);
  if (!(std::abs(eTrackster - eTrack) < energyCompatibilityNSigma_ * std::hypot(sigmaTrack, sigmaTrackster)))
    return {eTrack, false};
  const float wTrack = 1.f / std::max(sigmaTrack * sigmaTrack, 1e-12f);
  const float wTrackster = 1.f / (sigmaTrackster * sigmaTrackster);
  return {(wTrack * eTrack + wTrackster * eTrackster) / (wTrack + wTrackster), true};
}

std::vector<float> TICLCandidateArbitrationProducer::neutralEnergyShares(
    const std::vector<Trackster> &tracksters,
    const std::vector<reco::CaloCluster> &layerClusters,
    const std::vector<int> &neutralIdx) const {
  std::vector<float> share(neutralIdx.size(), 1.f);
  if (neutralIdx.empty())
    return share;
  float sideEnergy[2] = {0.f, 0.f};
  for (auto const &ts : tracksters)
    sideEnergy[ts.barycenter().z() > 0.f ? 1 : 0] += ts.raw_energy();
  // Eta-phi tiles of the trackster barycenters, one per side, for the search of the tracksters within kNearDeltaR.
  auto tiles = std::make_unique<std::array<ticl::TICLLayerTile, 2>>();
  std::vector<float> barycenterEta(tracksters.size()), barycenterPhi(tracksters.size());
  for (size_t j = 0; j < tracksters.size(); ++j) {
    const auto &c = tracksters[j].barycenter();
    barycenterEta[j] = c.eta();
    barycenterPhi[j] = c.phi();
    (*tiles)[c.z() > 0.f ? 1 : 0].fill(barycenterEta[j], barycenterPhi[j], j);
  }
  constexpr float kNearDeltaR = 0.2f;

  std::vector<float> x;
  x.reserve(neutralIdx.size() * kNNeutralShareFeatures);
  for (const int idx : neutralIdx) {
    const auto &ts = tracksters[idx];
    const auto &b = ts.barycenter();
    const float bEta = barycenterEta[idx], bPhi = barycenterPhi[idx];
    const int side = b.z() > 0.f ? 1 : 0;
    float nearEnergy = 0.f;
    int nNear = 0;
    const auto &tile = (*tiles)[side];
    const auto box =
        tile.searchBoxEtaPhi(bEta - kNearDeltaR, bEta + kNearDeltaR, bPhi - kNearDeltaR, bPhi + kNearDeltaR);
    for (int iEta = box[0]; iEta <= box[1]; ++iEta)
      for (int iPhi = box[2]; iPhi <= box[3]; ++iPhi)
        for (const unsigned int j : tile[tile.globalBin(iEta, iPhi % ticl::TileConstants::nPhiBins)]) {
          if (static_cast<int>(j) == idx ||
              reco::deltaR2(barycenterEta[j], barycenterPhi[j], bEta, bPhi) >= kNearDeltaR * kNearDeltaR)
            continue;
          nearEnergy += tracksters[j].raw_energy();
          ++nNear;
        }
    x.push_back(std::log(std::max(ts.regressed_energy(), 1e-3f)));
    x.push_back(std::log(std::max(ts.raw_energy(), 1e-3f)));
    x.push_back(ts.raw_em_energy() / std::max(ts.raw_energy(), 1e-6f));
    x.push_back(std::abs(bEta));
    x.push_back(std::log1p(static_cast<float>(ts.vertices().size())));
    x.insert(x.end(), ts.id_probabilities().begin(), ts.id_probabilities().end());
    x.push_back(std::log1p(nearEnergy));
    x.push_back(std::log1p(sideEnergy[side]));
    x.push_back(static_cast<float>(nNear));
    for (int k = 0; k < 3; ++k)
      x.push_back(std::log(std::max(ts.eigenvalues()[k], 1e-3f)));
    for (int k = 0; k < 3; ++k)
      x.push_back(ts.sigmasPCA()[k]);
    for (int k = 0; k < 3; ++k)
      x.push_back(ts.sigmas()[k]);
    // Layer clusters: the largest energy share, the energy share within 2 and 5 cm of the main axis, the
    // energy-weighted transverse rms (cm), the smallest |z| and the |z| extent.
    auto axis = ts.eigenvectors(0);
    if (!(axis.mag2() > 0.f) || !std::isfinite(axis.mag2()))
      axis = b;
    axis = axis.unit();
    float eSum = 0.f, eMax = 0.f, eCore2 = 0.f, eCore5 = 0.f, r2Sum = 0.f;
    float zMin = std::numeric_limits<float>::max(), zMax = 0.f;
    for (size_t i = 0; i < ts.vertices().size(); ++i) {
      const auto &lc = layerClusters[ts.vertices(i)];
      const float e = lc.energy() / std::max<float>(1.f, ts.vertex_multiplicity(i));
      const Trackster::Vector d(lc.x() - b.x(), lc.y() - b.y(), lc.z() - b.z());
      const float along = d.Dot(axis);
      const float r = std::sqrt(std::max(0.f, d.mag2() - along * along));
      eSum += e;
      eMax = std::max(eMax, e);
      eCore2 += r < 2.f ? e : 0.f;
      eCore5 += r < 5.f ? e : 0.f;
      r2Sum += e * r * r;
      zMin = std::min(zMin, std::abs(static_cast<float>(lc.z())));
      zMax = std::max(zMax, std::abs(static_cast<float>(lc.z())));
    }
    if (eSum > 0.f)
      x.insert(x.end(), {eMax / eSum, eCore2 / eSum, eCore5 / eSum, std::sqrt(r2Sum / eSum), zMin, zMax - zMin});
    else
      x.insert(x.end(), {-1.f, -1.f, -1.f, -1.f, -1.f, 0.f});
  }
  for (auto &v : x)
    if (!std::isfinite(v))
      v = 0.f;

  const int64_t rows = neutralIdx.size();
  cms::Ort::FloatArrays input{std::move(x)};
  auto result = neutralShareModel_.run({"features"}, input, {{rows, kNNeutralShareFeatures}}, {}, rows);
  if (result.empty() || result[0].size() != neutralIdx.size())
    throw cms::Exception("LogicError") << "TICLCandidateArbitrationProducer: expected " << rows
                                       << " outputs from the neutral share model";
  // A trackster without layer clusters keeps its energy.
  for (size_t k = 0; k < neutralIdx.size(); ++k)
    share[k] = tracksters[neutralIdx[k]].vertices().empty() ? 1.f : result[0][k];
  return share;
}

void TICLCandidateArbitrationProducer::produce(edm::Event &evt, const edm::EventSetup &es) {
  edm::Handle<std::vector<Trackster>> tracksters_h;
  evt.getByToken(tracksters_token_, tracksters_h);
  const auto &trackToTrackster = evt.get(trackToTrackster_token_);
  const auto &trackMode = evt.get(trackMode_token_);
  const auto &neutralIdx = evt.get(neutralIdx_token_);
  const auto &neutralPdg = evt.get(neutralPdg_token_);

  edm::Handle<std::vector<reco::Track>> tracks_h;
  evt.getByToken(tracks_token_, tracks_h);
  const auto &tracks = *tracks_h;
  if (trackMode.size() != tracks.size()) {
    throw cms::Exception("LogicError") << "TICLCandidateArbitrationProducer: the assignment maps cover "
                                       << trackMode.size() << " tracks but the configured tracks collection has "
                                       << tracks.size() << "; the two producers must consume the same tracks.";
  }
  edm::Handle<std::vector<reco::GsfTrack>> gsfTracks_h;
  static const std::vector<reco::GsfTrack> emptyGsf;
  if (useGsfTracks_) {
    evt.getByToken(gsf_tracks_token_, gsfTracks_h);
  }
  const auto &gsfTracks = useGsfTracks_ ? *gsfTracks_h : emptyGsf;

  edm::Handle<MtdHostCollection> inputTiming_h;
  MtdHostCollection::ConstView inputTimingView;
  if (useMTDTiming_) {
    evt.getByToken(inputTimingToken_, inputTiming_h);
    inputTimingView = (*inputTiming_h).const_view();
  }

  auto resultCandidates = std::make_unique<std::vector<TICLCandidate>>();

  // Summed energy given to the tracks of each trackster, for the neutral residuals.
  std::map<int, float> trackSumP;
  // Four-momentum of the given energy and mass along a direction.
  auto p4Along = [](const math::XYZVector &dir, float energy, float mass) {
    const float pMag = std::sqrt(std::max(energy * energy - mass * mass, 0.f));
    const auto u = dir.unit();
    return math::XYZTLorentzVector(pMag * u.x(), pMag * u.y(), pMag * u.z(), energy);
  };
  // A GSF track goes to one electron only.
  std::vector<bool> gsfUsed(gsfTracks.size(), false);

  // Charged candidates from the per-track assignment.
  for (size_t iTrack = 0; iTrack < trackMode.size(); ++iTrack) {
    const auto mode = static_cast<TrackMode>(trackMode[iTrack]);
    if (mode == TrackMode::kNotSelected)
      continue;
    auto trackPtr = edm::Ptr<reco::Track>(tracks_h, iTrack);
    auto const &tk = *trackPtr;
    const int tsIdx = trackToTrackster[iTrack];
    edm::Ptr<Trackster> tracksterPtr;
    if (tsIdx >= 0)
      tracksterPtr = edm::Ptr<Trackster>(tracksters_h, tsIdx);

    if (mode == TrackMode::kMuon) {
      // Muon: energy from the track momentum.
      TICLCandidate cand(trackPtr, tracksterPtr);
      cand.setPdgId(-13 * tk.charge());
      math::PtEtaPhiMLorentzVector p4Polar(tk.pt(), tk.eta(), tk.phi(), ticl::mmuon);
      cand.setP4(p4Polar);
      resultCandidates->push_back(cand);
    } else if (mode == TrackMode::kElectron) {
      // Electron: the trackster energy along the direction of the GSF track.
      TICLCandidate cand(trackPtr, tracksterPtr);
      int bestGsf = -1;
      float bestDR = delta_tk_gsf_;
      for (size_t iGsf = 0; iGsf < gsfTracks.size(); ++iGsf) {
        if (gsfUsed[iGsf])
          continue;
        const float dR = reco::deltaR(gsfTracks[iGsf].eta(), gsfTracks[iGsf].phi(), tk.eta(), tk.phi());
        if (dR < bestDR) {
          bestDR = dR;
          bestGsf = static_cast<int>(iGsf);
        }
      }
      if (bestGsf >= 0 && tracksterPtr.isNonnull()) {
        gsfUsed[bestGsf] = true;
        const auto &gsf = gsfTracks[bestGsf];
        cand.addGsfTrackPtr(edm::Ptr<reco::GsfTrack>(gsfTracks_h, bestGsf));
        cand.setPdgId(11 * gsf.charge());
        cand.setCharge(gsf.charge());
        // The GSF mode momentum combined with the trackster energy, or the trackster energy when they disagree.
        const float p = gsf.pMode();
        const auto combined =
            combinedEnergy(p, gsfMomentumErrorScale_ * gsf.qoverpModeError() * p * p, 0.f, *tracksterPtr);
        cand.setP4(
            p4Along(gsf.momentum(), combined.compatible ? combined.energy : tracksterPtr->regressed_energy(), 0.f));
      } else {
        // No GSF track: the constructor sets the kinematics from the track and the trackster.
        cand.setPdgId(11 * tk.charge());
      }
      resultCandidates->push_back(cand);
    } else if (mode == TrackMode::kJetMember) {
      // Jet member: a charged candidate from the track alone. The shared trackster goes to the neutral residual.
      edm::Ptr<Trackster> noTrackster;
      TICLCandidate cand(trackPtr, noTrackster);
      resultCandidates->push_back(cand);
      if (tsIdx >= 0)
        trackSumP[tsIdx] += tk.p();
    } else if (tracksterPtr.isNonnull() && !tracksterPtr->isHadronic()) {
      // Charged hadron or recovery on an EM trackster: an electron (the trackster PID), with the combined energy along
      // the track. A compatible trackster is all in the combined energy; else its excess becomes a neutral residual below.
      TICLCandidate cand(trackPtr, tracksterPtr);
      const float p = tk.p();
      const auto combined = combinedEnergy(p, trackMomentumErrorScale_ * tk.qoverpError() * p * p, 0.f, *tracksterPtr);
      cand.setP4(p4Along(tk.momentum(), combined.energy, 0.f));
      resultCandidates->push_back(cand);
      trackSumP[tsIdx] += combined.compatible ? tracksterPtr->regressed_energy() : combined.energy;
    } else {
      // Charged hadron or recovery on a hadronic trackster or with no trackster: kinematics from the track. The
      // calorimetric excess of the trackster becomes a neutral residual below.
      TICLCandidate cand(trackPtr, tracksterPtr);
      math::PtEtaPhiMLorentzVector p4Polar(tk.pt(), tk.eta(), tk.phi(), ticl::mpion);
      cand.setP4(p4Polar);
      resultCandidates->push_back(cand);
      if (tsIdx >= 0)
        trackSumP[tsIdx] += tk.p();
    }
  }

  // Neutral residuals, typed by the trackster PID.
  for (auto const &[tsIdx, sumP] : trackSumP) {
    edm::Ptr<Trackster> tracksterPtr(tracksters_h, tsIdx);
    const float residual = tracksterPtr->regressed_energy() - sumP;
    if (residual < std::max(residualEnergyFloor_, residualEnergyFraction_ * tracksterPtr->regressed_energy()))
      continue;
    edm::Ptr<reco::Track> noTrack;
    TICLCandidate cand(noTrack, tracksterPtr);
    const auto dir = tracksterPtr->barycenter().unit();
    math::XYZTLorentzVector p4(residual * dir.x(), residual * dir.y(), residual * dir.z(), residual);
    cand.setP4(p4);
    resultCandidates->push_back(cand);
  }

  // Neutral candidates. Each takes the share of its trackster energy that comes from the particle it represents.
  const auto neutralShare = neutralEnergyShares(*tracksters_h, evt.get(layerClustersToken_), neutralIdx);
  for (size_t k = 0; k < neutralIdx.size(); ++k) {
    edm::Ptr<Trackster> tracksterPtr(tracksters_h, neutralIdx[k]);
    edm::Ptr<reco::Track> trackPtr;
    TICLCandidate cand(trackPtr, tracksterPtr);
    if (neutralPdg[k] != 0)
      cand.setPdgId(neutralPdg[k]);
    cand.setP4(cand.p4() * neutralShare[k]);
    if (cand.energy() < neutralMinEnergy_)
      continue;
    resultCandidates->push_back(cand);
  }

  ticl::assignTimeToCandidates(*resultCandidates,
                               inputTimingView,
                               {useMTDTiming_, useTimingAverage_, timingQualityThreshold_},
                               bfield_.product(),
                               *propagator_,
                               *trackingGeometry_,
                               *hgcons_);

  // Muons: a charged candidate takes the kinematics of the best muon track when takesMuonKinematics holds. A loose muon
  // is accepted for a muon candidate only.
  std::vector<reco::MuonRef> candidateMuons(resultCandidates->size());
  auto muonTrackType = std::make_unique<std::vector<int>>(resultCandidates->size(), -1);
  const auto muonH = evt.getHandle(muons_token_);
  for (size_t i = 0; i < resultCandidates->size(); ++i) {
    auto &cand = (*resultCandidates)[i];
    if (cand.charge() == 0 || cand.trackPtr().isNull())
      continue;
    const reco::TrackRef trackRef(tracks_h, cand.trackPtr().key());
    const int muId = PFMuonAlgo::muAssocToTrack(trackRef, *muonH);
    if (muId < 0)
      continue;
    const reco::MuonRef muonRef(muonH, muId);
    const bool muonCandidate = std::abs(cand.pdgId()) == 13;
    if (!takesMuonKinematics(muonRef, muonCandidate, !cand.tracksters().empty()))
      continue;
    reco::PFCandidate pf(cand.charge(), cand.p4(), muonCandidate ? reco::PFCandidate::mu : reco::PFCandidate::h);
    pf.setTrackRef(trackRef);
    if (!pfmu_->reconstructMuon(pf, muonRef, muonCandidate))
      continue;
    cand.setP4(pf.p4());
    cand.setCharge(pf.charge());
    cand.setPdgId(-13 * pf.charge());
    cand.setVertex(pf.vertex());
    candidateMuons[i] = muonRef;
    (*muonTrackType)[i] = pf.bestMuonTrackType();
  }

  const auto candidates_h = evt.put(std::move(resultCandidates));
  auto muonMap = std::make_unique<edm::ValueMap<reco::MuonRef>>();
  edm::ValueMap<reco::MuonRef>::Filler filler(*muonMap);
  filler.insert(candidates_h, candidateMuons.begin(), candidateMuons.end());
  filler.fill();
  evt.put(std::move(muonMap), "muons");
  evt.put(std::move(muonTrackType), "muonTrackType");
}

void TICLCandidateArbitrationProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("interpretations", edm::InputTag("ticlTracksterInterpretations"))
      ->setComment("Final tracksters and assignment maps of TICLInterpretationProducer.");
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<bool>("useGsfTracks", true)->setComment("Give the electrons the direction and the charge of the GSF track.");
  desc.add<edm::InputTag>("gsf_tracks", edm::InputTag("electronGsfTracks"));
  desc.add<edm::InputTag>("timingSoA", edm::InputTag("mtdSoA"));
  desc.add<bool>("useMTDTiming", true);
  desc.add<bool>("useTimingAverage", true);
  desc.add<float>("timingQualityThreshold", 0.5f);
  desc.add<float>("delta_tk_gsf", 0.05f)->setComment("(eta,phi) window between an electron track and its GSF track.");
  desc.add<float>("residualEnergyFloor", 2.0f)->setComment("Min energy [GeV] of a neutral residual.");
  desc.add<float>("residualEnergyFraction", 0.1f)
      ->setComment("Min energy of a neutral residual as a fraction of the trackster energy.");
  desc.add<float>("trackMomentumErrorScale", 2.0f)->setComment("Scale of the track momentum error.");
  desc.add<float>("gsfMomentumErrorScale", 3.8f)->setComment("Scale of the GSF mode momentum error.");
  desc.add<float>("emTracksterStochastic", 0.30f)->setComment("Stochastic term of the EM trackster resolution.");
  desc.add<float>("emTracksterConstant", 0.22f)->setComment("Constant term of the EM trackster resolution.");
  desc.add<float>("energyCompatibilityNSigma", 3.0f)
      ->setComment("Track and trackster energies agree within this number of combined errors.");
  desc.add<edm::FileInPath>("neutralShareModel",
                            edm::FileInPath("RecoTICL/Interpretation/data/neutralEnergy/neutralShare_mlp_v1.onnx"))
      ->setComment("ONNX model of the share of a neutral trackster energy that comes from the particle it represents.");
  desc.add<float>("neutralMinEnergy", 1.f)
      ->setComment("Neutral candidates below this corrected energy are not produced.");
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"))
      ->setComment("Layer clusters of the tracksters, read for the neutral energy share.");
  desc.add<edm::InputTag>("muonSrc", edm::InputTag("muons1stStep"));
  edm::ParameterSetDescription pfMuonAlgoDesc;
  PFMuonAlgo::fillPSetDescription(pfMuonAlgoDesc);
  desc.add<edm::ParameterSetDescription>("pfMuonAlgoParameters", pfMuonAlgoDesc);
  desc.add<std::string>("propagator", "PropagatorWithMaterial");
  descriptions.add("ticlCandidateArbitrationProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TICLCandidateArbitrationProducer);
