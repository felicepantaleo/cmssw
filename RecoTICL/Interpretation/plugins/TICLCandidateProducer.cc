// Author: Felice Pantaleo, Wahid Redjeb, Aurora Perego (CERN) - felice.pantaleo@cern.ch, wahid.redjeb@cern.ch, aurora.perego@cern.ch Date: 12/2023
//
// Candidate assembly: consumes the final tracksters and the per-track assignment
// maps produced by the interpretation instance of TracksterLinksProducer
// (ticlTracksterInterpretations) and builds the TICLCandidates. The interpretations
// themselves (masking passes or opinion arbitration) run upstream in that trackster
// module; since PF clustering and GSF seeding depend only on it, this producer can
// legally consume the GSF tracks: electron candidates (trackMode 3) take the GSF
// kinematics, recovering brem electrons whose KF momentum is underestimated.

#include <memory>
#include <map>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/HGCalReco/interface/MtdHostCollection.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "Geometry/CommonTopologies/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

using namespace ticl;

class TICLCandidateProducer : public edm::stream::EDProducer<edm::stream::WatchRuns> {
public:
  explicit TICLCandidateProducer(const edm::ParameterSet &ps);
  ~TICLCandidateProducer() override {}
  void produce(edm::Event &, const edm::EventSetup &) override;
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  template <typename F>
  void assignTimeToCandidates(std::vector<TICLCandidate> &resultCandidates,
                              edm::Handle<std::vector<reco::Track>> track_h,
                              MtdHostCollection::ConstView &inputTimingView,
                              F func) const;

  const edm::EDGetTokenT<std::vector<Trackster>> tracksters_token_;
  const edm::EDGetTokenT<std::vector<int>> trackToTrackster_token_;
  const edm::EDGetTokenT<std::vector<int>> trackMode_token_;
  const edm::EDGetTokenT<std::vector<int>> neutralIdx_token_;
  const edm::EDGetTokenT<std::vector<int>> neutralPdg_token_;
  const edm::EDGetTokenT<std::vector<reco::Track>> tracks_token_;
  const edm::EDGetTokenT<std::vector<reco::GsfTrack>> gsf_tracks_token_;
  edm::EDGetTokenT<MtdHostCollection> inputTimingToken_;
  const bool useMTDTiming_;
  const bool useTimingAverage_;
  const float timingQualityThreshold_;
  // (eta,phi) window to match the electron's KF track to its GSF twin.
  const double delta_tk_gsf_;
  // Track-only charged candidates (particle-flow convention): a selected track that
  // no interpretation assigned and whose trajectory points at (almost) no
  // calorimetric energy becomes a charged-hadron candidate from the track alone.
  // Its momentum otherwise vanishes from the event description, since pfTICL is the
  // only particle flow for HGCAL. The nearby-energy veto is the double-counting
  // protection: tracks whose deposits sit in emitted or claimed tracksters are left
  // to the linking/jet interpretations.
  const bool buildTrackOnlyCandidates_;
  const double trackOnlyDeltaR_;
  const double trackOnlyNearbyEnergyFloor_;
  const double trackOnlyNearbyEnergyFraction_;

  const std::string propName_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bfield_token_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagator_token_;
  const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> trackingGeometry_token_;
  edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hdc_token_;

  const HGCalDDDConstants *hgcons_;
  edm::ESHandle<MagneticField> bfield_;
  edm::ESHandle<Propagator> propagator_;
  edm::ESHandle<GlobalTrackingGeometry> trackingGeometry_;
  static constexpr float c_light_ = CLHEP::c_light * CLHEP::ns / CLHEP::cm;
  static constexpr float timeRes = 0.02f;
};

TICLCandidateProducer::TICLCandidateProducer(const edm::ParameterSet &ps)
    : tracksters_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("interpretations"))),
      trackToTrackster_token_(
          consumes<std::vector<int>>(edm::InputTag(ps.getParameter<edm::InputTag>("interpretations").label(),
                                                   "trackToTrackster",
                                                   ps.getParameter<edm::InputTag>("interpretations").process()))),
      trackMode_token_(
          consumes<std::vector<int>>(edm::InputTag(ps.getParameter<edm::InputTag>("interpretations").label(),
                                                   "trackMode",
                                                   ps.getParameter<edm::InputTag>("interpretations").process()))),
      neutralIdx_token_(
          consumes<std::vector<int>>(edm::InputTag(ps.getParameter<edm::InputTag>("interpretations").label(),
                                                   "neutralIdx",
                                                   ps.getParameter<edm::InputTag>("interpretations").process()))),
      neutralPdg_token_(
          consumes<std::vector<int>>(edm::InputTag(ps.getParameter<edm::InputTag>("interpretations").label(),
                                                   "neutralPdg",
                                                   ps.getParameter<edm::InputTag>("interpretations").process()))),
      tracks_token_(consumes<std::vector<reco::Track>>(ps.getParameter<edm::InputTag>("tracks"))),
      gsf_tracks_token_(consumes<std::vector<reco::GsfTrack>>(ps.getParameter<edm::InputTag>("gsf_tracks"))),
      useMTDTiming_(ps.getParameter<bool>("useMTDTiming")),
      useTimingAverage_(ps.getParameter<bool>("useTimingAverage")),
      timingQualityThreshold_(ps.getParameter<double>("timingQualityThreshold")),
      delta_tk_gsf_(ps.getParameter<double>("delta_tk_gsf")),
      buildTrackOnlyCandidates_(ps.getParameter<bool>("buildTrackOnlyCandidates")),
      trackOnlyDeltaR_(ps.getParameter<double>("trackOnlyDeltaR")),
      trackOnlyNearbyEnergyFloor_(ps.getParameter<double>("trackOnlyNearbyEnergyFloor")),
      trackOnlyNearbyEnergyFraction_(ps.getParameter<double>("trackOnlyNearbyEnergyFraction")),
      propName_(ps.getParameter<std::string>("propagator")),
      bfield_token_(esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()),
      propagator_token_(
          esConsumes<Propagator, TrackingComponentsRecord, edm::Transition::BeginRun>(edm::ESInputTag("", propName_))),
      trackingGeometry_token_(
          esConsumes<GlobalTrackingGeometry, GlobalTrackingGeometryRecord, edm::Transition::BeginRun>()),
      hgcons_(nullptr) {
  std::string detectorName_ =
      (ps.getParameter<std::string>("detector") == "HFNose") ? "HGCalHFNoseSensitive" : "HGCalEESensitive";
  hdc_token_ =
      esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag("", detectorName_));
  if (useMTDTiming_) {
    inputTimingToken_ = consumes<MtdHostCollection>(ps.getParameter<edm::InputTag>("timingSoA"));
  }
  produces<std::vector<TICLCandidate>>();
}

void TICLCandidateProducer::beginRun(edm::Run const &, edm::EventSetup const &es) {
  edm::ESHandle<HGCalDDDConstants> hdc = es.getHandle(hdc_token_);
  hgcons_ = hdc.product();
  bfield_ = es.getHandle(bfield_token_);
  propagator_ = es.getHandle(propagator_token_);
  trackingGeometry_ = es.getHandle(trackingGeometry_token_);
}

void TICLCandidateProducer::produce(edm::Event &evt, const edm::EventSetup &es) {
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
    throw cms::Exception("LogicError") << "TICLCandidateProducer: the assignment maps cover " << trackMode.size()
                                       << " tracks but the configured tracks collection has " << tracks.size()
                                       << "; the two producers must consume the same tracks.";
  }
  edm::Handle<std::vector<reco::GsfTrack>> gsfTracks_h;
  evt.getByToken(gsf_tracks_token_, gsfTracks_h);
  const auto &gsfTracks = *gsfTracks_h;

  edm::Handle<MtdHostCollection> inputTiming_h;
  MtdHostCollection::ConstView inputTimingView;
  if (useMTDTiming_) {
    evt.getByToken(inputTimingToken_, inputTiming_h);
    inputTimingView = (*inputTiming_h).const_view();
  }

  auto const bFieldProd = bfield_.product();
  const Propagator *propagator = propagator_.product();

  auto resultCandidates = std::make_unique<std::vector<TICLCandidate>>();

  // Jet bookkeeping: summed track momentum per shared trackster (trackMode 4), for
  // the neutral residual emitted after the track loop.
  std::map<int, double> jetSumP;

  // Charged candidates from the per-track assignment.
  for (size_t iTrack = 0; iTrack < trackMode.size() && iTrack < tracks.size(); ++iTrack) {
    const int mode = trackMode[iTrack];
    if (mode <= 0)
      continue;  // -1: not selected; 0: selected but unassigned (see buildTrackOnlyCandidates)
    auto trackPtr = edm::Ptr<reco::Track>(tracks_h, iTrack);
    auto const &tk = *trackPtr;
    const int tsIdx = trackToTrackster[iTrack];
    edm::Ptr<Trackster> tracksterPtr;
    if (tsIdx >= 0)
      tracksterPtr = edm::Ptr<Trackster>(tracksters_h, tsIdx);

    if (mode == 1) {
      // Muon: energy from the track momentum, MIP trackster attached if any.
      TICLCandidate cand(trackPtr, tracksterPtr);
      cand.setPdgId(13 * tk.charge());
      math::PtEtaPhiMLorentzVector p4Polar(tk.pt(), tk.eta(), tk.phi(), ticl::mmuon);
      cand.setP4(p4Polar);
      resultCandidates->push_back(cand);
    } else if (mode == 3) {
      // Electron: upgrade to the GSF kinematics. The GSF chain is downstream of the
      // interpretation module, so consuming it here is legal (no dependency cycle).
      TICLCandidate cand(trackPtr, tracksterPtr);
      int bestGsf = -1;
      double bestDR = delta_tk_gsf_;
      for (size_t iGsf = 0; iGsf < gsfTracks.size(); ++iGsf) {
        const double dR = reco::deltaR(gsfTracks[iGsf].eta(), gsfTracks[iGsf].phi(), tk.eta(), tk.phi());
        if (dR < bestDR) {
          bestDR = dR;
          bestGsf = static_cast<int>(iGsf);
        }
      }
      if (bestGsf >= 0 && tracksterPtr.isNonnull()) {
        const auto &gsf = gsfTracks[bestGsf];
        cand.addGsfTrackPtr(edm::Ptr<reco::GsfTrack>(gsfTracks_h, bestGsf));
        cand.setPdgId(11 * gsf.charge());
        cand.setCharge(gsf.charge());
        const auto dir = gsf.momentum().unit();
        const float energy = tracksterPtr->regressed_energy();
        math::XYZTLorentzVector p4(energy * dir.x(), energy * dir.y(), energy * dir.z(), energy);
        cand.setP4(p4);
      } else {
        // No GSF twin found: keep the KF hypothesis (the ctor already set p4/charge).
        cand.setPdgId(11 * tk.charge());
      }
      resultCandidates->push_back(cand);
    } else if (mode == 4) {
      // Jet member: charged candidate from the track alone (particle-flow style); the
      // shared trackster is not attached, its energy enters through the neutral
      // residual below (attaching it to every member would double count).
      edm::Ptr<Trackster> noTrackster;
      TICLCandidate cand(trackPtr, noTrackster);
      resultCandidates->push_back(cand);
      if (tsIdx >= 0)
        jetSumP[tsIdx] += tk.p();
    } else if (mode == 5) {
      // Single-track recovery: attach the claimed trackster (so the calorimetric
      // footprint is associated with this candidate) but take the kinematics from the
      // TRACK, particle-flow style: the calorimeter under-measures fragmented hadronic
      // showers, which is why the tight gates rejected the link in the first place.
      // Any calorimetric excess still becomes a neutral residual below.
      TICLCandidate cand(trackPtr, tracksterPtr);
      cand.setPdgId(211 * tk.charge());
      math::PtEtaPhiMLorentzVector p4Polar(tk.pt(), tk.eta(), tk.phi(), ticl::mpion);
      cand.setP4(p4Polar);
      resultCandidates->push_back(cand);
      if (tsIdx >= 0)
        jetSumP[tsIdx] += tk.p();
    } else {
      // Charged hadron: the constructor sets kinematics and species from the trackster.
      TICLCandidate cand(trackPtr, tracksterPtr);
      resultCandidates->push_back(cand);
    }
  }

  // Jet neutral residuals: the calorimetric excess of a shared trackster over the
  // summed momenta of its tracks, typed by the trackster PID. Below threshold the
  // energy is considered accounted for by the tracks.
  for (auto const &[tsIdx, sumP] : jetSumP) {
    edm::Ptr<Trackster> tracksterPtr(tracksters_h, tsIdx);
    const double residual = tracksterPtr->regressed_energy() - sumP;
    if (residual < std::max(2., 0.1 * tracksterPtr->regressed_energy()))
      continue;
    edm::Ptr<reco::Track> noTrack;
    TICLCandidate cand(noTrack, tracksterPtr);
    const auto dir = tracksterPtr->barycenter().unit();
    math::XYZTLorentzVector p4(residual * dir.x(), residual * dir.y(), residual * dir.z(), residual);
    cand.setP4(p4);
    resultCandidates->push_back(cand);
  }

  // Neutral candidates.
  for (size_t k = 0; k < neutralIdx.size(); ++k) {
    edm::Ptr<Trackster> tracksterPtr(tracksters_h, neutralIdx[k]);
    edm::Ptr<reco::Track> trackPtr;
    TICLCandidate cand(trackPtr, tracksterPtr);
    if (neutralPdg[k] != 0)
      cand.setPdgId(neutralPdg[k]);
    resultCandidates->push_back(cand);
  }

  if (buildTrackOnlyCandidates_) {
    const auto &tracksters = *tracksters_h;
    for (size_t iTrack = 0; iTrack < trackMode.size() && iTrack < tracks.size(); ++iTrack) {
      if (trackMode[iTrack] != 0)
        continue;  // only selected tracks no interpretation assigned
      auto const &tk = tracks[iTrack];
      const auto dir = tk.outerOk() ? tk.outerMomentum() : tk.momentum();
      double nearby = 0.;
      for (auto const &ts : tracksters) {
        const auto &bary = ts.barycenter();
        if (bary.eta() * dir.eta() < 0.)
          continue;
        if (reco::deltaR(bary.eta(), bary.phi(), dir.eta(), dir.phi()) < trackOnlyDeltaR_)
          nearby += ts.raw_energy();
      }
      if (nearby >= std::max(trackOnlyNearbyEnergyFloor_, trackOnlyNearbyEnergyFraction_ * tk.p()))
        continue;
      auto trackPtr = edm::Ptr<reco::Track>(tracks_h, iTrack);
      edm::Ptr<Trackster> noTrackster;
      TICLCandidate cand(trackPtr, noTrackster);
      resultCandidates->push_back(cand);
    }
  }

  auto getPathLength = [&](const reco::Track &track, float zVal) {
    if (!track.innerOk() || !track.outerOk()) {
      return 0.f;
    }
    const auto &fts_inn = trajectoryStateTransform::innerFreeState(track, bFieldProd);
    const auto &fts_out = trajectoryStateTransform::outerFreeState(track, bFieldProd);
    const auto &surf_inn = trajectoryStateTransform::innerStateOnSurface(track, *trackingGeometry_, bFieldProd);
    const auto &surf_out = trajectoryStateTransform::outerStateOnSurface(track, *trackingGeometry_, bFieldProd);

    Basic3DVector<float> pos(track.referencePoint());
    Basic3DVector<float> mom(track.momentum());
    FreeTrajectoryState stateAtBeamspot{GlobalPoint(pos), GlobalVector(mom), track.charge(), bFieldProd};

    float pathlength = propagator->propagateWithPath(stateAtBeamspot, surf_inn.surface()).second;
    if (pathlength) {
      const auto &t_inn_out = propagator->propagateWithPath(fts_inn, surf_out.surface());
      if (t_inn_out.first.isValid()) {
        pathlength += t_inn_out.second;
        std::pair<float, float> rMinMax = hgcons_->rangeR(zVal, true);
        int iSide = int(track.eta() > 0);
        float zSide = (iSide == 0) ? (-1. * zVal) : zVal;
        const auto &disk = std::make_unique<GeomDet>(
            Disk::build(Disk::PositionType(0, 0, zSide),
                        Disk::RotationType(),
                        SimpleDiskBounds(rMinMax.first, rMinMax.second, zSide - 0.5, zSide + 0.5))
                .get());
        const auto &tsos = propagator->propagateWithPath(fts_out, disk->surface());
        if (tsos.first.isValid()) {
          pathlength += tsos.second;
          return pathlength;
        }
      }
    }
    return 0.f;
  };

  assignTimeToCandidates(*resultCandidates, tracks_h, inputTimingView, getPathLength);

  evt.put(std::move(resultCandidates));
}

template <typename F>
void TICLCandidateProducer::assignTimeToCandidates(std::vector<TICLCandidate> &resultCandidates,
                                                   edm::Handle<std::vector<reco::Track>> track_h,
                                                   MtdHostCollection::ConstView &inputTimingView,
                                                   F func) const {
  for (auto &cand : resultCandidates) {
    float beta = 1;
    float time = 0.f;
    float invTimeErr = 0.f;
    float timeErr = -1.f;

    const int trackIndex =
        cand.trackPtr().isNonnull() ? (cand.trackPtr().get() - (edm::Ptr<reco::Track>(track_h, 0)).get()) : -1;
    for (const auto &tr : cand.tracksters()) {
      if (tr->timeError() > 0) {
        const auto invTimeESq = pow(tr->timeError(), -2);
        const auto x = tr->barycenter().X();
        const auto y = tr->barycenter().Y();
        const auto z = tr->barycenter().Z();
        auto path = std::sqrt(x * x + y * y + z * z);
        if (trackIndex != -1) {
          if (useMTDTiming_ and inputTimingView.timeErr()[trackIndex] > 0) {
            const auto xMtd = inputTimingView.posInMTD_x()[trackIndex];
            const auto yMtd = inputTimingView.posInMTD_y()[trackIndex];
            const auto zMtd = inputTimingView.posInMTD_z()[trackIndex];
            beta = inputTimingView.beta()[trackIndex];
            path = std::sqrt((x - xMtd) * (x - xMtd) + (y - yMtd) * (y - yMtd) + (z - zMtd) * (z - zMtd)) +
                   inputTimingView.pathLength()[trackIndex];
          } else {
            float pathLength = func(*(cand.trackPtr().get()), z);
            if (pathLength) {
              path = pathLength;
            }
          }
        }
        time += (tr->time() - path / (beta * c_light_)) * invTimeESq;
        invTimeErr += invTimeESq;
      }
    }
    if (invTimeErr > 0) {
      time = time / invTimeErr;
      timeErr = sqrt(1.f / invTimeErr);
      if (timeErr < timeRes)
        timeErr = timeRes;
      cand.setTime(time, timeErr);
    }

    if (useMTDTiming_ and cand.charge() and trackIndex != -1) {
      const bool assocQuality = inputTimingView.MVAquality()[trackIndex] > timingQualityThreshold_;
      if (assocQuality) {
        const auto timeHGC = cand.time();
        const auto timeEHGC = cand.timeError();
        const auto timeMTD = inputTimingView.time0()[trackIndex];
        const auto timeEMTD = inputTimingView.time0Err()[trackIndex];

        if (useTimingAverage_ && (timeEMTD > 0 && timeEHGC > 0)) {
          const auto invTimeESqHGC = pow(timeEHGC, -2);
          const auto invTimeESqMTD = pow(timeEMTD, -2);
          timeErr = 1.f / (invTimeESqHGC + invTimeESqMTD);
          time = (timeHGC * invTimeESqHGC + timeMTD * invTimeESqMTD) * timeErr;
          timeErr = sqrt(timeErr);
        } else if (timeEMTD > 0) {
          time = timeMTD;
          timeErr = timeEMTD;
        }
      }
      cand.setTime(time, timeErr);
      cand.setMTDTime(inputTimingView.time()[trackIndex], inputTimingView.timeErr()[trackIndex]);
    }
  }
}

void TICLCandidateProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("interpretations", edm::InputTag("ticlTracksterInterpretations"))
      ->setComment("Final tracksters + assignment maps from TICLTracksterInterpretationsProducer.");
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("gsf_tracks", edm::InputTag("electronGsfTracks"));
  desc.add<edm::InputTag>("timingSoA", edm::InputTag("mtdSoA"));
  desc.add<bool>("useMTDTiming", true);
  desc.add<bool>("useTimingAverage", true);
  desc.add<double>("timingQualityThreshold", 0.5);
  desc.add<double>("delta_tk_gsf", 0.05)
      ->setComment("(eta,phi) window to match an electron's KF track to its GSF twin.");
  desc.add<bool>("buildTrackOnlyCandidates", false)
      ->setComment("Emit charged-hadron candidates from selected tracks no interpretation assigned.");
  desc.add<double>("trackOnlyDeltaR", 0.1)->setComment("(eta,phi) window for the nearby-energy veto.");
  desc.add<double>("trackOnlyNearbyEnergyFloor", 2.0)->setComment("Nearby-energy veto floor [GeV].");
  desc.add<double>("trackOnlyNearbyEnergyFraction", 0.2)
      ->setComment("Nearby-energy veto as a fraction of the track momentum.");
  desc.add<std::string>("detector", "HGCAL");
  desc.add<std::string>("propagator", "PropagatorWithMaterial");
  descriptions.add("ticlCandidateProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TICLCandidateProducer);
