#ifndef RecoTICL_Linking_CornettoFeatures_h
#define RecoTICL_Linking_CornettoFeatures_h

// Per-trackster quantities of the Cornetto pair test.

#include <vector>

#include "DataFormats/HGCalReco/interface/Trackster.h"

namespace ticl::cornetto {

  // Number of ticl::Trackster::ParticleType slots.
  inline constexpr int kNumParticleTypes = static_cast<int>(Trackster::ParticleType::unknown) + 1;

  // The EM probability of a trackster: the sum of its id_probabilities over the emCategories (ParticleType slots)
  // divided by the sum over all slots. -1 without a PID: no probability, or only unknown (a merged trackster).
  float emProbability(const Trackster& ts, const std::vector<int>& emCategories);

}  // namespace ticl::cornetto

#endif
