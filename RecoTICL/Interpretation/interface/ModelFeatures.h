#ifndef RecoTICL_Interpretation_ModelFeatures_h
#define RecoTICL_Interpretation_ModelFeatures_h

// The input features of the three interpretation models, in the order of their training. A slot index is part of the
// model: test/ModelFeatures_t.cpp checks each slot against the training order.

#include <span>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoTICL/Interpretation/interface/TICLInterpretationAlgoBase.h"

namespace ticl::modelfeatures {

  // Hypothesis model: one row per hypothesis with a footprint or tracks. data/arbitration/hypothesis_mlp_v2.onnx reads
  // the first kNv2 slots.
  namespace hypothesis {
    enum Slot : unsigned int {
      kType = 0,  // 7 one-hot slots, in the order of Hypothesis::Type
      kScore = 7,
      kLogRawEnergy,
      kEMFraction,
      kAbsEta,
      kPID,  // 8 slots, in the order of Trackster::ParticleType
      kNLayerClusters = kPID + 8,
      kNJetTracks,
      kLogTrackP,
      kTrackPt,
      kTrackAbsEta,
      kEOverP,
      kDeltaR,
      kGsf,  // the GSF flag of the training sample, always 0
      kComponentSize,
      kMuonTracks,
      kElectronTracks,
      kNv2,
      kEMRecoveryTracks = kNv2,  // tracks whose Recovery hypothesis has an EM footprint
      kMissingInnerHitTracks,    // tracks with at least one missing inner hit
      kN
    };
  }  // namespace hypothesis
  static_assert(hypothesis::kNv2 == 30);
  static_assert(hypothesis::kN == 32);
  static_assert(static_cast<int>(Hypothesis::Type::RecoveryChargedHadron) == 6, "one-hot slots 0-6 are the types");

  // Context of a hypothesis in its event: the size of its conflict component (without the track-only hypotheses) and,
  // by track, whether the track also has a muon or an electron hypothesis, and whether its Recovery hypothesis has an
  // EM footprint.
  struct HypothesisContext {
    int componentSize;
    const std::vector<bool> &muonTrack;
    const std::vector<bool> &electronTrack;
    const std::vector<bool> &emRecoveryTrack;
  };

  // Fills x, hypothesis::kN zero-initialized slots, for hypothesis h with footprint (null for none).
  void fillHypothesisFeatures(const Hypothesis &h,
                              const Trackster *footprint,
                              const std::vector<reco::Track> &tracks,
                              const HypothesisContext &context,
                              std::span<float, hypothesis::kN> x);

  // Neutral species model (data/arbitration/neutralSpecies_mlp_v2.onnx): one row per neutral trackster with a positive
  // raw energy.
  namespace neutralSpecies {
    enum Slot : unsigned int { kPID = 0 /* 8 slots */, kEMFraction = 8, kLogRawEnergy, kAbsEta, kNLayerClusters, kN };
    // The outputs of the model, in this order.
    enum Output : unsigned int { kPhoton = 0, kPi0, kNeutralHadron, kNOutputs };
  }  // namespace neutralSpecies
  static_assert(neutralSpecies::kN == 12);

  void fillNeutralSpeciesFeatures(const Trackster &ts, std::span<float, neutralSpecies::kN> x);

  // Neutral share model (data/neutralEnergy/neutralShare_mlp_v1.onnx): one row per neutral candidate trackster. A
  // non-finite feature enters as 0.
  namespace neutralShare {
    enum Slot : unsigned int {
      kLogRegressedEnergy = 0,
      kLogRawEnergy,
      kEMFraction,
      kAbsEta,
      kLogNLayerClusters,
      kPID,  // 8 slots
      kLogNearEnergy = kPID + 8,
      kLogSideEnergy,
      kNNear,
      kLogEigenvalues,                   // 3 slots
      kSigmasPCA = kLogEigenvalues + 3,  // 3 slots
      kSigmas = kSigmasPCA + 3,          // 3 slots
      kMaxClusterShare = kSigmas + 3,
      kCoreShare2,
      kCoreShare5,
      kTransverseRMS,
      kMinAbsZ,
      kAbsZExtent,
      kN
    };
  }  // namespace neutralShare
  static_assert(neutralShare::kN == 31);

  // Surroundings of a neutral trackster: the raw energy and the number of the other final tracksters within dR 0.2,
  // and the raw energy of all the final tracksters of its endcap.
  struct NeutralShareContext {
    float nearEnergy;
    int nNear;
    float sideEnergy;
  };

  void fillNeutralShareFeatures(const Trackster &ts,
                                const std::vector<reco::CaloCluster> &layerClusters,
                                const NeutralShareContext &context,
                                std::span<float, neutralShare::kN> x);

}  // namespace ticl::modelfeatures

#endif
