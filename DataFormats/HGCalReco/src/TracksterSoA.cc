#include <tuple>
#include <type_traits>

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TracksterSoA.h"

static_assert(ticl::kTracksterIdProbabilitiesSize ==
                  std::tuple_size_v<std::remove_cvref_t<decltype(std::declval<ticl::Trackster>().id_probabilities())>>,
              "The trackster SoA must hold one PID probability per ticl::Trackster probability");
