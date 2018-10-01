#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoPixelVertexing/PixelTriplets/interface/FakeRecoTrack.h"
#include <iostream>
#include <stdio.h>
double fakee[] = { 1.1,
             1.2, 2.2,
             1.3, 2.3, 3.3,
             1.4, 2.4, 3.4, 4.4,
             1.5, 2.5, 3.5, 4.5, 5.5 };


int main() {
  reco::TrackBase::CovarianceMatrix covfake(fakee, fakee + 15);

  double chi2 = 1.2;
  double ndof = 3;
  math::XYZPoint pos(5.6, 7.8, 9.10);
  math::XYZVector mom(10, 11, 12);
  int charge = 1;
  reco::Track* emptyTrack = new reco::Track(chi2, ndof, pos, mom, charge, covfake);

  // reco::TrackBase::CovarianceMatrix cov;
  //  for (int i = 0; i < 5; ++i) {
  //        for (int j = 0; j <= i; ++j) {
  //            std::cout << cov(i, j) << std::endl;
  //        }
  //    }

  std::cout << "size of the reco track " << sizeof(*emptyTrack) << std::endl;
  std::cout << "Printing the content of the reco::track " << std::endl;

  char* trackInChars = ((char*)emptyTrack);
  auto nBytes = int(sizeof(*emptyTrack));
  for (auto i = 0; i < nBytes; ++i) {
    if (i % 8 == 0) {
      printf("\n %d\t", i);
    }
    printf("%hhx \t", trackInChars[i]);
  }
  printf("\n");

  FakeRecoTrack fakeTrack;

  fakeTrack.chi2 = chi2;
  fakeTrack.charge = charge;
  fakeTrack.ndof = ndof;
  fakeTrack.vertexPos[0] = 5.6;
  fakeTrack.vertexPos[1] = 7.8;
  fakeTrack.vertexPos[2] = 9.10;
  fakeTrack.momentum[0] = 10;
  fakeTrack.momentum[1] = 11;
  fakeTrack.momentum[2] = 12;
  fakeTrack.algorithm = reco::TrackBase::undefAlgorithm;
  fakeTrack.originalAlgo = reco::TrackBase::undefAlgorithm;
  fakeTrack.quality = reco::TrackBase::undefQuality;
  fakeTrack.covariance[0] = 1.1;
  fakeTrack.covariance[1] = 1.2;
  fakeTrack.covariance[2] = 2.2;
  fakeTrack.covariance[3] = 1.3;
  fakeTrack.covariance[4] = 2.3;
  fakeTrack.covariance[5] = 3.3;
  fakeTrack.covariance[6] = 1.4;
  fakeTrack.covariance[7] = 2.4;
  fakeTrack.covariance[8] = 3.4;
  fakeTrack.covariance[9] = 4.4;
  fakeTrack.covariance[10] = 1.5;
  fakeTrack.covariance[11] = 2.5;
  fakeTrack.covariance[12] = 3.5;
  fakeTrack.covariance[13] = 4.5;
  fakeTrack.covariance[14] = 5.5;
  std::cout << "Printing the content of the fakerecotrack " << std::endl;

  trackInChars = ((char*)&fakeTrack);
  nBytes = int(sizeof(FakeRecoTrack));
  for (auto i = 0; i < nBytes; ++i) {
    if (i % 8 == 0) {
      printf("\n %d\t", i);
    }
    printf("%hhx \t", trackInChars[i]);
  }

  std::cout << "\nChecking that they have the same content " << std::endl;

  assert(fakeTrack.get_chi2() == emptyTrack->chi2());
  assert(fakeTrack.get_ndof() == emptyTrack->ndof());
  assert(fakeTrack.get_charge() == emptyTrack->charge());
  assert(fakeTrack.get_px() == emptyTrack->px());
  assert(fakeTrack.get_py() == emptyTrack->py());
  assert(fakeTrack.get_pz() == emptyTrack->pz());
  assert(fakeTrack.get_vx() == emptyTrack->vx());
  assert(fakeTrack.get_vy() == emptyTrack->vy());
  assert(fakeTrack.get_vz() == emptyTrack->vz());
  assert(fakeTrack.get_covariance(1, 1) == emptyTrack->covariance(1, 1));
  reco::Track* disguisedTrack = (reco::Track*)&fakeTrack;
  assert(disguisedTrack->chi2() == emptyTrack->chi2());
  assert(disguisedTrack->ndof() == emptyTrack->ndof());
  assert(disguisedTrack->charge() == emptyTrack->charge());
  assert(disguisedTrack->px() == emptyTrack->px());
  assert(disguisedTrack->py() == emptyTrack->py());
  assert(disguisedTrack->pz() == emptyTrack->pz());
  assert(disguisedTrack->vx() == emptyTrack->vx());
  assert(disguisedTrack->vy() == emptyTrack->vy());
  assert(disguisedTrack->vz() == emptyTrack->vz());
  assert(disguisedTrack->covariance(1, 1) == emptyTrack->covariance(1, 1));
  assert(disguisedTrack->covariance(2, 1) == emptyTrack->covariance(2, 1));
  assert(disguisedTrack->covariance(1, 3) == emptyTrack->covariance(1, 3));
  assert(disguisedTrack->covariance(4, 3) == emptyTrack->covariance(4, 3));
  assert(disguisedTrack->covariance(4, 2) == emptyTrack->covariance(4, 2));
  printf("\nTest succeded!\n");


}
