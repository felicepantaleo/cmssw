#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"
#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFast.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

#include "PixelRecHits.h"  // TODO : spit product from kernel

#include <cuda_runtime.h>

class SiPixelRecHitHeterogeneous: public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices <
                                                                   heterogeneous::GPUCuda,
                                                                   heterogeneous::CPU
                                                                   > > {

public:
  using CPUProduct = siPixelRecHitsHeterogeneousProduct::CPUProduct;
  using GPUProduct = siPixelRecHitsHeterogeneousProduct::GPUProduct;
  using Output = siPixelRecHitsHeterogeneousProduct::HeterogeneousPixelRecHit;


  explicit SiPixelRecHitHeterogeneous(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitHeterogeneous() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // CPU implementation
  void produceCPU(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) override;

  // GPU implementation
  void beginStreamGPUCuda(edm::StreamID streamId, cuda::stream_t<>& cudaStream) override;
  void acquireGPUCuda(const edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, cuda::stream_t<>& cudaStream) override;
  void produceGPUCuda(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, cuda::stream_t<>& cudaStream) override;
  void convertGPUtoCPU(edm::Event& iEvent, const edm::Handle<SiPixelClusterCollectionNew>& inputhandle, const siPixelRecHitsHeterogeneousProduct::GPUProduct & gpu) const;

    // Commonalities
  void initialize(const edm::EventSetup& es);

  // CPU
  void run(const edm::Handle<SiPixelClusterCollectionNew>& inputhandle, SiPixelRecHitCollectionNew &output) const;
  // GPU
  void run(const edm::Handle<SiPixelClusterCollectionNew>& inputhandle, SiPixelRecHitCollectionNew &output, const pixelgpudetails::HitsOnCPU& hoc) const;


  // The mess with inputs will be cleaned up when migrating to the new framework
  edm::EDGetTokenT<CUDAProduct<BeamSpotCUDA>> tBeamSpot;
  edm::EDGetTokenT<CUDAProduct<SiPixelClustersCUDA>> token_;
  edm::EDGetTokenT<CUDAProduct<SiPixelDigisCUDA>> tokenDigi_;
  edm::EDGetTokenT<SiPixelClusterCollectionNew> clusterToken_;
  std::string cpeName_;

  const TrackerGeometry *geom_ = nullptr;
  const PixelClusterParameterEstimator *cpe_ = nullptr;

  std::unique_ptr<pixelgpudetails::PixelRecHitGPUKernel> gpuAlgo_;

  bool enableTransfer_;
  bool enableConversion_;
};

SiPixelRecHitHeterogeneous::SiPixelRecHitHeterogeneous(const edm::ParameterSet& iConfig):
  HeterogeneousEDProducer(iConfig),
  tBeamSpot(consumes<CUDAProduct<BeamSpotCUDA>>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
  token_(consumes<CUDAProduct<SiPixelClustersCUDA>>(iConfig.getParameter<edm::InputTag>("heterogeneousSrc"))),
  tokenDigi_(consumes<CUDAProduct<SiPixelDigisCUDA>>(iConfig.getParameter<edm::InputTag>("heterogeneousSrc"))),
  cpeName_(iConfig.getParameter<std::string>("CPE"))
{
  enableConversion_ = iConfig.getParameter<bool>("gpuEnableConversion");
  enableTransfer_ = enableConversion_ || iConfig.getParameter<bool>("gpuEnableTransfer");

  produces<HeterogeneousProduct>();
  if(enableConversion_) {
    clusterToken_ = consumes<SiPixelClusterCollectionNew>(iConfig.getParameter<edm::InputTag>("src"));
    produces<SiPixelRecHitCollectionNew>();
  }
}

void SiPixelRecHitHeterogeneous::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpotCUDA"));
  desc.add<edm::InputTag>("heterogeneousSrc", edm::InputTag("siPixelClustersCUDAPreSplitting"));
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersPreSplitting"));
  desc.add<std::string>("CPE", "PixelCPEFast");

  desc.add<bool>("gpuEnableTransfer", true);
  desc.add<bool>("gpuEnableConversion", true);

  HeterogeneousEDProducer::fillPSetDescription(desc);

  descriptions.add("siPixelRecHitHeterogeneous",desc);
}

void SiPixelRecHitHeterogeneous::initialize(const edm::EventSetup& es) {
  edm::ESHandle<TrackerGeometry> geom;
  es.get<TrackerDigiGeometryRecord>().get( geom );
  geom_ = geom.product();

  edm::ESHandle<PixelClusterParameterEstimator> hCPE;
  es.get<TkPixelCPERecord>().get(cpeName_, hCPE);
  cpe_ = dynamic_cast< const PixelCPEBase* >(hCPE.product());
}

void SiPixelRecHitHeterogeneous::produceCPU(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) {
  throw cms::Exception("NotImplemented") << "CPU version is no longer implemented";
}

void SiPixelRecHitHeterogeneous::run(const edm::Handle<SiPixelClusterCollectionNew>& inputhandle, SiPixelRecHitCollectionNew &output) const {
  const auto& input = *inputhandle;

  edmNew::DetSetVector<SiPixelCluster>::const_iterator DSViter=input.begin();
  for ( ; DSViter != input.end() ; DSViter++) {
    unsigned int detid = DSViter->detId();
    DetId detIdObject( detid );
    const GeomDetUnit * genericDet = geom_->idToDetUnit( detIdObject );
    const PixelGeomDetUnit * pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
    assert(pixDet);
    SiPixelRecHitCollectionNew::FastFiller recHitsOnDetUnit(output,detid);

    edmNew::DetSet<SiPixelCluster>::const_iterator clustIt = DSViter->begin(), clustEnd = DSViter->end();

    for ( ; clustIt != clustEnd; clustIt++) {
      std::tuple<LocalPoint, LocalError,SiPixelRecHitQuality::QualWordType> tuple = cpe_->getParameters( *clustIt, *genericDet );
      LocalPoint lp( std::get<0>(tuple) );
      LocalError le( std::get<1>(tuple) );
      SiPixelRecHitQuality::QualWordType rqw( std::get<2>(tuple) );
      // Create a persistent edm::Ref to the cluster
      edm::Ref< edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster > cluster = edmNew::makeRefTo( inputhandle, clustIt);
      // Make a RecHit and add it to the DetSet
      // old : recHitsOnDetUnit.push_back( new SiPixelRecHit( lp, le, detIdObject, &*clustIt) );
      SiPixelRecHit hit( lp, le, rqw, *genericDet, cluster);
      //
      // Now save it =================
      recHitsOnDetUnit.push_back(hit);
    } //  <-- End loop on Clusters
  } //    <-- End loop on DetUnits
}


void SiPixelRecHitHeterogeneous::beginStreamGPUCuda(edm::StreamID streamId, cuda::stream_t<>& cudaStream) {
  gpuAlgo_ = std::make_unique<pixelgpudetails::PixelRecHitGPUKernel>(cudaStream);
}

void SiPixelRecHitHeterogeneous::acquireGPUCuda(const edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, cuda::stream_t<>& cudaStream) {
  initialize(iSetup);

  PixelCPEFast const * fcpe = dynamic_cast<const PixelCPEFast *>(cpe_);
  if (!fcpe) {
    throw cms::Exception("Configuration") << "too bad, not a fast cpe gpu processing not possible....";
  }

  edm::Handle<CUDAProduct<SiPixelClustersCUDA>> hclusters;
  iEvent.getByToken(token_, hclusters);
  // temporary check (until the migration)
  edm::Service<CUDAService> cs;
  assert(hclusters->device() == cs->getCurrentDevice());
  CUDAScopedContext ctx{*hclusters};
  auto const& clusters = ctx.get(*hclusters);

  edm::Handle<CUDAProduct<SiPixelDigisCUDA>> hdigis;
  iEvent.getByToken(tokenDigi_, hdigis);
  auto const& digis = ctx.get(*hdigis);

  edm::Handle<CUDAProduct<BeamSpotCUDA>> hbs;
  iEvent.getByToken(tBeamSpot, hbs);
  auto const& bs = ctx.get(*hbs);

  // We're processing in a stream given by base class, so need to
  // synchronize explicitly (implementation is from
  // CUDAScopedContext). In practice these should not be needed
  // (because of synchronizations upstream), but let's play generic.
  if (not hclusters->isAvailable()) {
    cudaCheck(cudaStreamWaitEvent(cudaStream.id(), hclusters->event()->id(), 0));
  }
  if (not hdigis->isAvailable()) {
    cudaCheck(cudaStreamWaitEvent(cudaStream.id(), hdigis->event()->id(), 0));
  }
  if (not hbs->isAvailable()) {
    cudaCheck(cudaStreamWaitEvent(cudaStream.id(), hbs->event()->id(), 0));
  }

  gpuAlgo_->makeHitsAsync(digis, clusters, bs, fcpe->getGPUProductAsync(cudaStream), enableTransfer_, cudaStream);

}

void SiPixelRecHitHeterogeneous::produceGPUCuda(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, cuda::stream_t<>& cudaStream) {
  auto output = std::make_unique<GPUProduct>(gpuAlgo_->getOutput());

  if(enableConversion_) {
    // Need the CPU clusters to
    // - properly fill the output DetSetVector of hits
    // - to set up edm::Refs to the clusters
    edm::Handle<SiPixelClusterCollectionNew> hclusters;
    iEvent.getByToken(clusterToken_, hclusters);

    convertGPUtoCPU(iEvent.event(), hclusters, *output);
  }

  iEvent.put<Output>(std::move(output), heterogeneous::DisableTransfer{});
}

void SiPixelRecHitHeterogeneous::convertGPUtoCPU(edm::Event& iEvent, const edm::Handle<SiPixelClusterCollectionNew>& inputhandle, const pixelgpudetails::HitsOnCPU & hoc) const{
  assert(hoc.gpu_d);
  auto output = std::make_unique<SiPixelRecHitCollectionNew>();
  run(inputhandle, *output, hoc);
  iEvent.put(std::move(output));
}

void SiPixelRecHitHeterogeneous::run(const edm::Handle<SiPixelClusterCollectionNew>& inputhandle, SiPixelRecHitCollectionNew &output, const pixelgpudetails::HitsOnCPU& hoc) const {
  auto const & input = *inputhandle;

  int numberOfDetUnits = 0;
  int numberOfClusters = 0;
  for (auto DSViter=input.begin(); DSViter != input.end() ; DSViter++) {
    numberOfDetUnits++;
    unsigned int detid = DSViter->detId();
    DetId detIdObject( detid );
    const GeomDetUnit * genericDet = geom_->idToDetUnit( detIdObject );
    auto gind = genericDet->index();
    const PixelGeomDetUnit * pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
    assert(pixDet);
    SiPixelRecHitCollectionNew::FastFiller recHitsOnDetUnit(output, detid);
    auto fc = hoc.hitsModuleStart[gind];
    auto lc = hoc.hitsModuleStart[gind+1];
    auto nhits = lc-fc;
    uint32_t ic=0;
    auto jnd = [&](int k) { return fc+k; };
    assert(nhits<=DSViter->size());
    if (nhits!=DSViter->size()) {
       edm::LogWarning("GPUHits2CPU") <<"nhits!= ndigi " << nhits << ' ' << DSViter->size() << std::endl;
    }
    for (auto const & clust : *DSViter) {
      if (ic>=nhits) {
        // FIXME add a way to handle this case, or at least notify via edm::LogError
        break;
      }
      auto ij = jnd(clust.originalId());
      assert(clust.originalId()>=0); assert(clust.originalId()<nhits);
      if(clust.charge()!=hoc.charge[ij])
        edm::LogWarning("GPUHits2CPU") << "not a perfect Match "
                                       << gind <<' '<<fc<<' '
                                       << ic<<'/'<<clust.originalId()<<'/'<< (ij-fc) << ' ' << clust.size()
                                       << ' ' << clust.charge()<<"!="<<hoc.charge[ij]
                                       << ' ' << clust.minPixelCol()<<"?"<< hoc.mc[ij]
                                       << ' ' << clust.minPixelRow()<<'/'<< hoc.mr[ij] << std::endl;

      LocalPoint lp(hoc.xl[ij], hoc.yl[ij]);
      LocalError le(hoc.xe[ij], 0, hoc.ye[ij]);
      SiPixelRecHitQuality::QualWordType rqw=0;

      ++ic;
      numberOfClusters++;

      /*   cpu version....  (for reference)
           std::tuple<LocalPoint, LocalError, SiPixelRecHitQuality::QualWordType> tuple = cpe_->getParameters( clust, *genericDet );
           LocalPoint lp( std::get<0>(tuple) );
           LocalError le( std::get<1>(tuple) );
           SiPixelRecHitQuality::QualWordType rqw( std::get<2>(tuple) );
      */

      // Create a persistent edm::Ref to the cluster
      edm::Ref< edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster > cluster = edmNew::makeRefTo( inputhandle, &clust);
      // Make a RecHit and add it to the DetSet
      SiPixelRecHit hit( lp, le, rqw, *genericDet, cluster);
      //
      // Now save it =================
      recHitsOnDetUnit.push_back(hit);
      // =============================

      // std::cout << "SiPixelRecHitGPUVI " << numberOfClusters << ' '<< lp << " " << le << std::endl;

    } //  <-- End loop on Clusters


      //  LogDebug("SiPixelRecHitGPU")
      //std::cout << "SiPixelRecHitGPUVI "
      //	<< " Found " << recHitsOnDetUnit.size() << " RecHits on " << detid //;
      // << std::endl;


  } //    <-- End loop on DetUnits

  /*
  std::cout << "SiPixelRecHitGPUVI $ det, clus, lost "
    <<  numberOfDetUnits << ' '
    << numberOfClusters  << ' '
    << std::endl;
  */
}

DEFINE_FWK_MODULE(SiPixelRecHitHeterogeneous);
