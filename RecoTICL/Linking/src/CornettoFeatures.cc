#include "RecoTICL/Linking/interface/CornettoFeatures.h"

namespace ticl::cornetto {

  float emProbability(const Trackster& ts, const std::vector<int>& emCategories) {
    float sum = 0.f;
    for (float v : ts.id_probabilities())
      sum += v;
    if (!(sum > ts.id_probability(Trackster::ParticleType::unknown)))
      return -1.f;
    float em = 0.f;
    for (int category : emCategories)
      em += ts.id_probabilities(category);
    return em / sum;
  }

}  // namespace ticl::cornetto
