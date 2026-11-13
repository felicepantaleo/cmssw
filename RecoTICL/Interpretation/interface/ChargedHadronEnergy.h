#ifndef RecoTICL_Interpretation_ChargedHadronEnergy_h
#define RecoTICL_Interpretation_ChargedHadronEnergy_h

#include <algorithm>

#include "RecoTICL/Interpretation/interface/ExpectedDeposit.h"

namespace ticl {

  // Margin [GeV] of the energy test of a track of momentum p and momentum error sigmaP on tracksters of raw energy
  // rawEnergy: min(0.2 rawEnergy, 10 GeV), or min(nSigma sigmaP, p) when that is larger.
  inline float chargedHadronEnergyMargin(double rawEnergy, double p, double sigmaP, double nSigma) {
    return std::max(std::min(0.2 * rawEnergy, 10.0), std::min(nSigma * sigmaP, p));
  }

  // A charged hadron of momentum p with a claim trackster and no other trackster takes combined, the combined energy
  // of the track and the claim trackster, when the raw energy of the claim trackster is at least deposit.minFraction
  // of the expected raw deposit of the track and combined is above the mass. Else it keeps the track energy.
  inline bool useCombinedClaimEnergy(
      float combined, float mass, float claimRawEnergy, float p, const ExpectedDeposit &deposit) {
    return claimRawEnergy >= deposit.minFraction * deposit(p) && combined > mass;
  }

}  // namespace ticl

#endif
