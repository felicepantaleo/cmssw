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

  inline double getchi2() const
{
    return chi2;
}

inline double getndof() const
{
    return ndof;
}

inline int getcharge() const
{
    return charge;
}

inline double getpx() const
{
    return momentum[0];
}


inline double getpy() const
{
    return momentum[1];
}
inline double getpz() const
{
    return momentum[2];
}


inline double getvx() const
{
    return vertexPos[0];
}

inline double getvy() const
{
    return vertexPos[1];
}
inline double getvz() const
{
    return vertexPos[2];
}

inline int covIndex(int i, int j) const
{
    int a = (i <= j ? i : j);
    int b = (i <= j ? j : i);
    return b * (b + 1) / 2 + a;
}
inline double getcovariance(int i, int j) const
{
    return covariance[covIndex(i, j)];
}

};
