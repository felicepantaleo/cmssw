#ifndef RecoTICL_Inference_TracksterPointCloud_h
#define RecoTICL_Inference_TracksterPointCloud_h

#include <array>
#include <utility>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"

namespace ticl {

  // Point cloud and global features of a trackster for the energy, species and pile-up models. The points are the
  // layer clusters of the trackster, then the other layer clusters of its side within a cylinder around the trackster
  // axis (radius per layer: radiusEE in CE-E, radiusEH in CE-H; the transverse offset is taken in the plane of the
  // layer cluster). The cylinder search per layer is a lookup in the layer-cluster tiles of ticlLayerTileProducer. The
  // features use only the layer clusters, their times and tiles, the tracksters of one stage, the primary vertex and
  // an optional track at the HGCAL front face.
  class TracksterPointCloud {
  public:
    struct Config {
      float radiusEE = 8.f;
      float radiusEH = 20.f;
      // The ring for the density features: from the radius to ringOuterFactor x radius.
      float ringOuterFactor = 1.5f;
      // The axis is the PCA axis when it is within maxAxisAngle [rad] of the barycenter direction, else the barycenter
      // direction from the origin.
      float maxAxisAngle = 0.3f;
      // Points: every layer cluster of the trackster (code 0), at most maxOwnPoints by decreasing energy, then the
      // other layer clusters of the cylinder, at most maxNeighbourPoints by decreasing energy.
      unsigned int maxOwnPoints = 96;
      unsigned int maxNeighbourPoints = 48;
    };

    // Point features, in this order. dx, dy [cm]: layer cluster minus the axis point at the layer-cluster z. dz [cm]:
    // layer-cluster z minus barycenter z. logE: log of the energy [GeV]. eFrac: energy over the trackster raw energy.
    // dt [ns]: layer-cluster time minus trackster time, 0 when timeValid is 0. share: 1 / multiplicity in this
    // trackster, 0 when not in it. dtPV [ns]: layer-cluster time minus the primary-vertex time minus the flight time
    // |x - v| / c, -99 when dtPVValid is 0.
    enum PointFeature : unsigned int {
      kDx,
      kDy,
      kDz,
      kLayer,
      kLogE,
      kEFrac,
      kDt,
      kTimeValid,
      kNHits,
      kScintillator,
      kCode,
      kShare,
      kPointDtPV,
      kPointDtPVValid,
      kNPointFeatures
    };
    // Membership code of a point.
    enum Code : int { kThis = 0, kOther = 1, kMaskedSingleSilicon = 2, kNone = 3, kNCodes = 4 };

    // Global features. nPointsAll: layer clusters in the cylinder (all codes); nOwnAll, nNeighbourAll: the own layer
    // clusters and the other cylinder layer clusters before the caps. pv*: the primary vertex; dtPV [ns]: trackster
    // time minus the vertex time minus the flight time |b - v| / c to the barycenter b; dtPVSig: dtPV over the
    // combined time error; both -99 when dtPVValid is 0.
    enum GlobalFeature : unsigned int {
      kRawE,
      kRawEmE,
      kBx,
      kBy,
      kBz,
      kEta,
      kPhi,
      kAx,
      kAy,
      kAz,
      kAxisPCA,
      kAxisAngle,
      kEig0,
      kEig1,
      kEig2,
      kSigPCA0,
      kSigPCA1,
      kSigPCA2,
      kSig0,
      kSig1,
      kSig2,
      kNLC,
      kNLayers,
      kFirstLayer,
      kLastLayer,
      kTime,
      kTimeErr,
      kLcMaxFrac,
      kLcCore2,
      kLcCore5,
      kLcRms,
      kLcZ0,
      kLcZExtent,
      kNearRawE,
      kNNear,
      kSideRawE,
      kCylEThis,
      kCylEOther,
      kCylEMasked,
      kCylENone,
      kRingEOther,
      kRingEMasked,
      kRingENone,
      kNPointsAll,
      kNVertices,
      kTkValid,
      kTkP,
      kTkPt,
      kTkEta,
      kTkPhi,
      kTkDist,
      kTkDt,
      kTkTimeValid,
      kNOwnAll,
      kNNeighbourAll,
      kPvValid,
      kPvZ,
      kPvT,
      kPvTErr,
      kPvTimeValid,
      kDtPV,
      kDtPVSig,
      kDtPVValid,
      kNGlobalFeatures
    };

    // A track at the HGCAL front face: impact position [cm], momentum there [GeV], MTD time [ns] (timeError <= 0:
    // no time).
    struct Track {
      float x, y, z;
      float px, py, pz;
      float time, timeError;
    };

    // The primary vertex: position [cm], time and its error [ns] (tError <= 0: no time).
    struct PrimaryVertex {
      float x = 0.f, y = 0.f, z = 0.f, t = 0.f, tError = -1.f;
      bool valid = false;
    };

    // The layer-cluster information of one event and stage, shared by the builds of its tracksters.
    class Event {
    public:
      Event(const std::vector<Trackster>& stage,
            const std::vector<reco::CaloCluster>& layerClusters,
            const edm::ValueMap<std::pair<float, float>>* layerClusterTimes,
            const TICLLayerTiles& tiles,
            const ticlgeom::Tools& rhtools,
            int nVertices,
            const PrimaryVertex& primaryVertex);

      const std::vector<Trackster>& stage() const { return *stage_; }
      const std::vector<reco::CaloCluster>& layerClusters() const { return *layerClusters_; }
      // Trackster of the stage that holds the layer cluster (the largest 1 / multiplicity), -1 none.
      int owner(unsigned int lc) const { return owner_[lc]; }
      bool masked(unsigned int lc) const { return masked_[lc]; }
      int layer(unsigned int lc) const { return layer_[lc]; }

    private:
      friend class TracksterPointCloud;
      const std::vector<Trackster>* stage_;
      const std::vector<reco::CaloCluster>* layerClusters_;
      const TICLLayerTiles* tiles_;
      int nVertices_;
      PrimaryVertex pv_;
      int lastLayerEE_, lastLayer_;
      // |z| [cm] of each layer (1 to lastLayer).
      std::vector<float> layerZ_;
      std::vector<int> owner_;
      std::vector<bool> masked_;
      std::vector<short> layer_, nHits_, scintillator_;
      std::vector<float> time_, timeError_;
      std::array<float, 2> sideRawE_{{0.f, 0.f}};
      // The tracksters of the stage by barycenter eta and phi, per side (z < 0, z > 0).
      std::array<TICLLayerTile, 2> tracksterTiles_;
    };

    struct Result {
      std::array<float, kNGlobalFeatures> global;
      // nPoints rows of kNPointFeatures values.
      std::vector<float> points;
      std::vector<int> layerCluster;
      std::vector<int> owner;
      unsigned int nPoints() const { return layerCluster.size(); }
    };

    explicit TracksterPointCloud(const Config& config) : config_(config) {}

    Result build(const Event& event, unsigned int trackster, const Track* track = nullptr) const;

    // Tile layer of a layer cluster, as in TICLLayerTileProducer: the layer with offset of its first hit, minus 1,
    // plus lastLayer for z > 0.
    static int tileLayer(const reco::CaloCluster& layerCluster, const ticlgeom::Tools& rhtools);

    // The predicate of the CLUE3DHigh layer-cluster filter: ClusterFilterByAlgoAndSize
    // (RecoTICL/ClusterFilters/plugins/ClusterFilterByAlgoAndSize.h:28-34) with the parameters of
    // filteredLayerClustersCLUE3DHigh (algo_number 6, 7, 8; min_cluster_size 2; max_cluster_size 9999). With these
    // layer clusters it masks exactly the single-hit clusters whose hit is silicon.
    static bool maskedByCLUE3DHighFilter(const reco::CaloCluster& layerCluster, const ticlgeom::Tools& rhtools);

    static const char* pointFeatureName(unsigned int i);
    static const char* globalFeatureName(unsigned int i);

    float radius(int layer, int lastLayerEE) const {
      return layer <= lastLayerEE ? config_.radiusEE : config_.radiusEH;
    }

  private:
    Config config_;
  };

}  // namespace ticl

#endif
