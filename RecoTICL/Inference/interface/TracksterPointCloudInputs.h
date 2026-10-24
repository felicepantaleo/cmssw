#ifndef RecoTICL_Inference_TracksterPointCloudInputs_h
#define RecoTICL_Inference_TracksterPointCloudInputs_h

#include <cmath>
#include <utility>
#include <vector>

#include "DataFormats/HGCalReco/interface/MtdHostCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoTICL/Inference/interface/TracksterPointCloud.h"

namespace ticl {

  // The event inputs of TracksterPointCloud that the dataset dumper and the inference producer share.

  // The primary vertex: the first valid, not fake vertex of the collection (offlinePrimaryVertices4D).
  TracksterPointCloud::PrimaryVertex leadingPrimaryVertex(const std::vector<reco::Vertex>& vertices);

  // Number of valid, not fake vertices (offlinePrimaryVertices).
  int countValidVertices(const std::vector<reco::Vertex>& vertices);

  // MTD time and its error [ns] of a track from the MTD SoA; (0, -1) without the SoA.
  std::pair<float, float> mtdTrackTime(const MtdHostCollection* timing, unsigned int track);

  // Tracks that get an HGCAL impact for the point cloud: pT >= minPt [GeV] and |eta| >= minAbsEta.
  struct PointCloudTrackSelection {
    float minPt = 0.5f;
    float minAbsEta = 1.3f;
    bool operator()(const reco::Track& track) const {
      return track.pt() >= minPt && std::abs(track.eta()) >= minAbsEta;
    }
  };

  // The track of each final trackster: among the tracks that trackToTrackster assigns to it, the one of highest p with
  // a valid HGCAL impact (the first in track order on equal p); -1 when none. Impact is ticl::TrackImpact (members
  // valid, position, direction); impacts has one entry per track.
  template <typename Impact>
  std::vector<int> assignedTrackOfHighestP(const std::vector<int>& trackToTrackster,
                                           const std::vector<reco::Track>& tracks,
                                           const std::vector<Impact>& impacts,
                                           unsigned int nTracksters) {
    std::vector<int> best(nTracksters, -1);
    for (size_t k = 0; k < trackToTrackster.size() && k < tracks.size(); ++k) {
      const int f = trackToTrackster[k];
      if (f < 0 || static_cast<unsigned int>(f) >= nTracksters || !impacts[k].valid)
        continue;
      if (best[f] < 0 || tracks[k].p() > tracks[best[f]].p())
        best[f] = static_cast<int>(k);
    }
    return best;
  }

  // The point-cloud track from the HGCAL impact and the MTD time of a track.
  template <typename Impact>
  TracksterPointCloud::Track pointCloudTrack(const Impact& impact, const std::pair<float, float>& time) {
    return {static_cast<float>(impact.position.x()),
            static_cast<float>(impact.position.y()),
            static_cast<float>(impact.position.z()),
            static_cast<float>(impact.direction.x()),
            static_cast<float>(impact.direction.y()),
            static_cast<float>(impact.direction.z()),
            time.first,
            time.second};
  }

}  // namespace ticl

#endif
