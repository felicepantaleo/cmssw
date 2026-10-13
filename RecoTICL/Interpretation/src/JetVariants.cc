#include "RecoTICL/Interpretation/interface/JetVariants.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

namespace ticl {

  std::vector<JetVariant> jetVariantsWithoutElectrons(
      const std::vector<JetFootprint> &jets,
      const std::vector<std::vector<std::vector<unsigned int>>> &electronFootprints,
      std::span<const double> lcEnergy,
      std::span<const double> trackP,
      const ExpectedDeposit &deposit) {
    std::vector<JetVariant> variants;
    std::set<std::pair<std::vector<int>, std::vector<unsigned int>>> seen;
    std::vector<bool> removed(lcEnergy.size(), false);
    for (const auto &jet : jets) {
      for (const int iElectronTrack : jet.tracks) {
        for (const auto &electronFootprint : electronFootprints[iElectronTrack]) {
          for (auto v : electronFootprint)
            removed[v] = true;
          JetVariant variant;
          float energy = 0.f;
          for (size_t k = 0; k < jet.vertices.size(); ++k) {
            const auto v = jet.vertices[k];
            if (removed[v])
              continue;
            variant.vertices.push_back(v);
            variant.multiplicities.push_back(jet.multiplicities[k]);
            energy += lcEnergy[v] / jet.multiplicities[k];
          }
          for (auto v : electronFootprint)
            removed[v] = false;

          float sumP = 0.f;
          for (const int iTk : jet.tracks)
            if (iTk != iElectronTrack) {
              variant.tracks.push_back(iTk);
              sumP += trackP[iTk];
            }
          if (!(energy > 0.f) || !(sumP > 0.f))
            continue;
          if (variant.tracks.size() >= 2) {
            variant.score = 1.f - std::abs(energy - sumP) / std::max(energy, sumP);
          } else {
            const float expected = deposit(sumP);
            if (!deposit.inRecoveryBand(energy, expected))
              continue;
            variant.score = ExpectedDeposit::score(energy, expected);
          }
          std::pair<std::vector<int>, std::vector<unsigned int>> key(variant.tracks, variant.vertices);
          std::sort(key.first.begin(), key.first.end());
          std::sort(key.second.begin(), key.second.end());
          if (!seen.insert(std::move(key)).second)
            continue;
          variants.push_back(std::move(variant));
        }
      }
    }
    return variants;
  }

}  // namespace ticl
