#ifndef RecoTICL_Interpretation_ExpectedDeposit_h
#define RecoTICL_Interpretation_ExpectedDeposit_h

#include <algorithm>
#include <cmath>

namespace ticl {

  // Expected raw HGCAL deposit of a charged hadron of momentum p [GeV], (response + logSlope ln p) p, and the band
  // [minFraction, maxFraction] x deposit of the raw energy of a recovery footprint.
  struct ExpectedDeposit {
    float response;
    float logSlope;
    float minFraction;
    float maxFraction;

    float operator()(float p) const {
      if (logSlope == 0.f)
        return response * p;
      return (response + logSlope * std::log(p)) * p;
    }
    bool inRecoveryBand(float energy, float deposit) const {
      return !(energy < minFraction * deposit || energy > maxFraction * deposit);
    }
    // 1 when the energy equals the deposit, towards 0 away from it.
    static float score(float energy, float deposit) {
      return 1.f - std::abs(energy - deposit) / std::max(energy, deposit);
    }
  };

}  // namespace ticl

#endif
