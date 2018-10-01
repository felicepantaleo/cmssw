#define RECOPIXELVERTEXING_PIXELTRIPLETS_FAKERECOTRACK_H

#include "DataFormats/TrackReco/interface/Track.h"


struct FakeRecoTrack {
  char padHitPattern[116];
  float covariance[15];
  float chi2;
  double vertexPos[3];
  double momentum[3];
  char padAlgoMask[8];
  float ndof;
  char charge;
  uint8_t algorithm = reco::TrackBase::undefAlgorithm;
  uint8_t originalAlgo = reco::TrackBase::undefAlgorithm;
  uint8_t quality = reco::TrackBase::undefQuality;
  signed char nloops = 0;
  uint8_t stopreason = 0;
  char padTrackExtra[22];

  inline constexpr double get_chi2() const { return chi2; }

  inline constexpr double get_ndof() const { return ndof; }

  inline constexpr int get_charge() const { return charge; }

  inline constexpr double get_px() const { return momentum[0]; }

  inline constexpr double get_py() const { return momentum[1]; }
  inline constexpr double get_pz() const { return momentum[2]; }

  inline constexpr double get_vx() const { return vertexPos[0]; }

  inline constexpr double get_vy() const { return vertexPos[1]; }
  inline constexpr double get_vz() const { return vertexPos[2]; }

  inline constexpr int covIndex(int i, int j) const {
    int a = (i <= j ? i : j);
    int b = (i <= j ? j : i);
    return b * (b + 1) / 2 + a;
  }
  inline constexpr double get_covariance(int i, int j) const {
    return covariance[covIndex(i, j)];
  }
};
