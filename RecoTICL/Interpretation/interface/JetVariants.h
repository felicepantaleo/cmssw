#ifndef RecoTICL_Interpretation_JetVariants_h
#define RecoTICL_Interpretation_JetVariants_h

#include <span>
#include <vector>

#include "RecoTICL/Interpretation/interface/ExpectedDeposit.h"

namespace ticl {

  // The tracks of a Jet hypothesis and its footprint (layer clusters and their multiplicities).
  struct JetFootprint {
    std::vector<int> tracks;
    std::vector<unsigned int> vertices;
    std::vector<float> multiplicities;
  };

  // A Jet hypothesis without the track and the footprint of one Electron hypothesis of its tracks. One track left
  // makes it a recovery hypothesis of that track.
  struct JetVariant {
    std::vector<int> tracks;
    std::vector<unsigned int> vertices;
    std::vector<float> multiplicities;
    float score;
  };

  // For each jet and each Electron hypothesis of one of its tracks (electronFootprints[track]: the layer clusters of
  // each Electron hypothesis of the track): the jet without that track, on its footprint without the layer clusters of
  // the Electron footprint. Two tracks or more left give a jet variant, scored by the balance of the energy and the
  // summed momentum. One track left gives a recovery variant when the energy is in the recovery band of the expected
  // deposit of the track, with the score of a recovery hypothesis. A variant with the tracks and the layer clusters of
  // an earlier one is not repeated. lcEnergy and trackP are indexed by layer cluster and by track.
  std::vector<JetVariant> jetVariantsWithoutElectrons(
      const std::vector<JetFootprint> &jets,
      const std::vector<std::vector<std::vector<unsigned int>>> &electronFootprints,
      std::span<const double> lcEnergy,
      std::span<const double> trackP,
      const ExpectedDeposit &deposit);

}  // namespace ticl

#endif
