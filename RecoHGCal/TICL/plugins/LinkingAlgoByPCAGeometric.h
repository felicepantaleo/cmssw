#ifndef RecoHGCal_TICL_LinkingAlgoByPCAGeometric_H__
#define RecoHGCal_TICL_LinkingAlgoByPCAGeometric_H__

#include <memory>
#include <vector>
#include <string>
#include <array>
#include "RecoHGCal/TICL/plugins/LinkingAlgoBase.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

//#include "RecoHGCal/TICL/test/PCA_mod.h"

namespace ticl {
  class LinkingAlgoByPCAGeometric final : public LinkingAlgoBase {
  public:
    LinkingAlgoByPCAGeometric(const edm::ParameterSet &conf);
    ~LinkingAlgoByPCAGeometric() override;

    void initialize(const HGCalDDDConstants *hgcons,
                    const hgcal::RecHitTools rhtools,
                    const edm::ESHandle<MagneticField> bfieldH,
                    const edm::ESHandle<Propagator> propH) override;

    void linkTracksters(const edm::Handle<std::vector<reco::Track>>,
                        const StringCutObjectSelector<reco::Track>,
                        const edm::Handle<std::vector<Trackster>>,
                        std::vector<TICLCandidate> &) override;

    static void fillPSetDescription(edm::ParameterSetDescription &desc);

  private:
    typedef math::XYZVector Vector;

    void buildLayers();

    math::XYZVector propagateTrackster(
        const Trackster &t, const unsigned idx, float zVal, std::array<TICLLayerTile, 2>& tracksterTiles);

    std::once_flag initializeGeometry_;

    const HGCalDDDConstants *hgcons_;

    std::unique_ptr<GeomDet> firstDisk_[2];
    std::unique_ptr<GeomDet> interfaceDisk_[2];

    hgcal::RecHitTools rhtools_;

    edm::ESHandle<MagneticField> bfield_;
    edm::ESHandle<Propagator> propagator_;
  };
}  // namespace ticl
#endif
