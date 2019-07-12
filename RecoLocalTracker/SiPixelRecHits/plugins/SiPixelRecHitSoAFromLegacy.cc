#include <cuda_runtime.h>

// hack waiting for if constexpr
#define VIEW_ON_HOST
#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFast.h"

#include "CUDADataFormats/Common/interface/ArrayShadow.h"


class SiPixelRecHitSoAFromLegacy : public edm::global::EDProducer<> {
public:
  explicit SiPixelRecHitSoAFromLegacy(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitSoAFromLegacy() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  using HitModuleStart = std::array<uint32_t,gpuClustering::MaxNumModules + 1>;
  using HMSstorage = ArrayShadow<HitModuleStart>;


private:
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  // The mess with inputs will be cleaned up when migrating to the new framework
  edm::EDGetTokenT<reco::BeamSpot> bsGetToken_;
  edm::EDGetTokenT<SiPixelClusterCollectionNew> clusterToken_;    // Legacy Clusters
  edm::EDPutTokenT<TrackingRecHit2DHost> tokenHit_;

  std::string cpeName_;

};

SiPixelRecHitSoAFromLegacy::SiPixelRecHitSoAFromLegacy(const edm::ParameterSet& iConfig)
    : bsGetToken_{consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))},
      clusterToken_{consumes<SiPixelClusterCollectionNew>(iConfig.getParameter<edm::InputTag>("src"))},
      tokenHit_{produces<TrackingRecHit2DHost>()},
      cpeName_(iConfig.getParameter<std::string>("CPE")) {}

void SiPixelRecHitSoAFromLegacy::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersPreSplitting"));
  desc.add<std::string>("CPE", "PixelCPEFast");
  descriptions.add("siPixelRecHitHostSoA", desc);
}

void SiPixelRecHitSoAFromLegacy::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& es) const {

  
  const TrackerGeometry *geom_ = nullptr;
  const PixelClusterParameterEstimator* cpe_ = nullptr;

  
  edm::ESHandle<TrackerGeometry> geom;
  es.get<TrackerDigiGeometryRecord>().get( geom );
  geom_ = geom.product();
  

  edm::ESHandle<PixelClusterParameterEstimator> hCPE;
  es.get<TkPixelCPERecord>().get(cpeName_, hCPE);
  cpe_ = dynamic_cast<const PixelCPEBase*>(hCPE.product());

  PixelCPEFast const* fcpe = dynamic_cast<const PixelCPEFast*>(cpe_);
  if (!fcpe) {
    throw cms::Exception("Configuration") << "too bad, not a fast cpe gpu processing not possible....";
  }
  auto cpeView = fcpe->getCPUProduct();

  const reco::BeamSpot& bs = iEvent.get(bsGetToken_);


  BeamSpotCUDA::Data bsHost;
  bsHost.x = bs.x0();
  bsHost.y = bs.y0();
  bsHost.z = bs.z0();

  assert(bsHost.z!=0);


  auto const& input = iEvent.get(clusterToken_);
  int numberOfClusters = input.size();


  auto hms = std::make_unique<HMSstorage>();
  auto * hitsModuleStart = hms->data;
  auto dummyStream = cuda::stream::wrap(0,0,false);
  auto output = std::make_unique<TrackingRecHit2DHost>(numberOfClusters,
                                   &cpeView,
                                   hitsModuleStart,
                                   dummyStream
                                  );



    // storage
    std::vector<uint16_t> xx_;
    std::vector<uint16_t> yy_;
    std::vector<uint16_t> adc_;
    std::vector<uint16_t> moduleInd_;
    std::vector<int32_t>  clus_;


  int numberOfDetUnits = 0;
  // int numberOfHits = 0;
  for (auto DSViter = input.begin(); DSViter != input.end(); DSViter++) {
    numberOfDetUnits++;
    unsigned int detid = DSViter->detId();
    DetId detIdObject(detid);
    const GeomDetUnit* genericDet = geom_->idToDetUnit(detIdObject);
    auto gind = genericDet->index();
    assert(gind<2000);
    const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
    assert(pixDet);
    auto const nclus =  DSViter->size();
    if (0==nclus) continue;
    
    //auto fc = m_hitsModuleStart[gind];
    //auto lc = m_hitsModuleStart[gind + 1];
    //std::cout << "in det " << gind << "conv " << nhits << " hits from " << DSViter->size() << " legacy clusters"
    //          <<' '<< lc <<','<<fc<<std::endl;

    // fill digis
    xx_.clear();yy_.clear();adc_.clear();moduleInd_.clear(); clus_.clear();
    uint32_t ic = 0;
    uint32_t ndigi = 0;
    for (auto const& clust : *DSViter) {
      assert(clust.size()>0);
      for (int i=0, nd=clust.size(); i<nd; ++i) {
        auto px = clust.pixel(i);
        xx_.push_back(px.x);
        yy_.push_back(px.y);
        adc_.push_back(px.adc);
        moduleInd_.push_back(gind);
        clus_.push_back(ic);
        ++ndigi;
      }
      ic++;
    }
    assert(nclus==ic);
    assert(clus_.size()==ndigi);  

    // filled creates view
    SiPixelDigisCUDA::DeviceConstView digiView{xx_.data(),yy_.data(),adc_.data(),moduleInd_.data(), clus_.data()};
    assert(digiView.adc(0)!=0);

  }


}

DEFINE_FWK_MODULE(SiPixelRecHitSoAFromLegacy);
