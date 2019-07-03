#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/pixelTuplesHeterogeneousProduct.h"
#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/FitUtils.h"

#include "storeTracks.h"

/**
 * This class will eventually be the one creating the reco::Track
 * objects from the output of GPU CA. Now it is just to produce
 * something persistable.
 */
class PixelTrackProducerFromCUDA
    : public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices<heterogeneous::GPUCuda, heterogeneous::CPU>> {
public:
  using Input = pixelTuplesHeterogeneousProduct::HeterogeneousPixelTuples;
  using TuplesOnCPU = pixelTuplesHeterogeneousProduct::TuplesOnCPU;

  using Output = HeterogeneousProductImpl<heterogeneous::CPUProduct<int>, heterogeneous::GPUCudaProduct<int>>;

  explicit PixelTrackProducerFromCUDA(const edm::ParameterSet &iConfig);
  ~PixelTrackProducerFromCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginStreamGPUCuda(edm::StreamID streamId, cuda::stream_t<> &cudaStream) override {}
  void acquireGPUCuda(const edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) override;
  void produceGPUCuda(edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) override;
  void produceCPU(edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup) override;

private:
  TuplesOnCPU const *tuples_ = nullptr;

  edm::EDGetTokenT<reco::BeamSpot> tBeamSpot_;
  edm::EDGetTokenT<HeterogeneousProduct> gpuToken_;
  edm::EDGetTokenT<RegionsSeedingHitSets> srcToken_;
  uint32_t minNumberOfHits_;
  bool enableConversion_;
};

PixelTrackProducerFromCUDA::PixelTrackProducerFromCUDA(const edm::ParameterSet &iConfig)
    : HeterogeneousEDProducer(iConfig),
      tBeamSpot_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      gpuToken_(consumes<HeterogeneousProduct>(iConfig.getParameter<edm::InputTag>("src"))),
      minNumberOfHits_(iConfig.getParameter<unsigned int>("minNumberOfHits")),
      enableConversion_(iConfig.getParameter<bool>("gpuEnableConversion")) {
  if (enableConversion_) {
    srcToken_ = consumes<RegionsSeedingHitSets>(iConfig.getParameter<edm::InputTag>("src"));
    produces<reco::TrackCollection>();
    produces<TrackingRecHitCollection>();
    produces<reco::TrackExtraCollection>();
  } else {
    produces<int>();  // dummy
  }
  //  produces<HeterogeneousProduct>();
}

void PixelTrackProducerFromCUDA::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("src", edm::InputTag("pixelTracksHitQuadruplets"));
  desc.add<unsigned int>("minNumberOfHits", 4);
  desc.add<bool>("gpuEnableConversion", true);

  HeterogeneousEDProducer::fillPSetDescription(desc);
  descriptions.addWithDefaultLabel(desc);
}

void PixelTrackProducerFromCUDA::acquireGPUCuda(const edm::HeterogeneousEvent &iEvent,
                                                const edm::EventSetup &iSetup,
                                                cuda::stream_t<> &cudaStream) {
  edm::Handle<TuplesOnCPU> gh;
  iEvent.getByToken<Input>(gpuToken_, gh);
  //auto const & gTuples = *gh;
  // std::cout << "tuples from gpu " << gTuples.nTuples << std::endl;

  tuples_ = gh.product();
}

void PixelTrackProducerFromCUDA::produceGPUCuda(edm::HeterogeneousEvent &iEvent,
                                                const edm::EventSetup &iSetup,
                                                cuda::stream_t<> &cudaStream) {
  // iEvent.put(std::make_unique<int>(0));
  if (!enableConversion_)
    return;

  // std::cout << "Converting gpu helix in reco tracks" << std::endl;

  edm::ESHandle<MagneticField> fieldESH;
  iSetup.get<IdealMagneticFieldRecord>().get(fieldESH);

  pixeltrackfitting::TracksWithTTRHs tracks;
  edm::ESHandle<TrackerTopology> httopo;
  iSetup.get<TrackerTopologyRcd>().get(httopo);

  edm::Handle<RegionsSeedingHitSets> hitSets;
  iEvent.getByToken(srcToken_, hitSets);
  const auto &hitSet = *hitSets->begin();
  auto b = hitSet.begin();
  auto e = hitSet.end();
  // std::cout << "reading hitset " << e-b << std::endl;

  // const auto & region = hitSet.region();
  // std::cout << "origin " << region.origin() << std::endl;

  edm::Handle<reco::BeamSpot> bsHandle;
  iEvent.getByToken(tBeamSpot_, bsHandle);
  const auto &bsh = *bsHandle;
  // std::cout << "beamspot " << bsh.x0() << ' ' << bsh.y0() << ' ' << bsh.z0() << std::endl;
  GlobalPoint bs(bsh.x0(), bsh.y0(), bsh.z0());

  std::vector<const TrackingRecHit *> hits;
  hits.reserve(5);

  uint32_t nh = 0;  // current hitset
  assert(tuples_->indToEdm.size() == tuples_->nTuples);
  for (uint32_t it = 0; it < tuples_->nTuples; ++it) {
    auto q = tuples_->quality[it];
    // this should be in phase with selection of hitset....
    if (q != pixelTuplesHeterogeneousProduct::loose)
      continue;                           // FIXME
    assert(tuples_->indToEdm[it] == nh);  // filled on CPU!
    auto const &shits = *(b + nh);
    auto nHits = shits.size();
    ++nh;
    if (nHits< minNumberOfHits_) continue;
    hits.resize(nHits);
    for (unsigned int iHit = 0; iHit < nHits; ++iHit)
      hits[iHit] = shits[iHit];

    // mind: this values are respect the beamspot!
    auto const &fittedTrack = tuples_->helix_fit_results[it];

    // std::cout << "tk " << it << ": " << fittedTrack.q << ' ' << fittedTrack.par[2] << ' ' << std::sqrt(fittedTrack.cov(2, 2)) << std::endl;

    auto iCharge = fittedTrack.q;
    float phi = fittedTrack.par(0);
    float chi2 = fittedTrack.chi2_line + fittedTrack.chi2_circle;

    Rfit::Vector5d opar;
    Rfit::Matrix5d ocov;
    Rfit::transformToPerigeePlane(fittedTrack.par,fittedTrack.cov,opar,ocov,iCharge);

    LocalTrajectoryParameters lpar(opar(0),opar(1),opar(2),opar(3),opar(4),1.);
    AlgebraicSymMatrix55 m;
    for(int i=0; i<5; ++i) for (int j=i; j<5; ++j) m(i,j) = ocov(i,j);

    float sp = std::sin(phi);
    float cp = std::cos(phi);
    Surface::RotationType rot(
                              sp, -cp,    0,
                               0,   0, -1.f,
                              cp,  sp,    0);

    Plane impPointPlane(bs,rot);
    GlobalTrajectoryParameters gp(impPointPlane.toGlobal(lpar.position()), 
                                  impPointPlane.toGlobal(lpar.momentum()),lpar.charge(),fieldESH.product());
    JacobianLocalToCurvilinear jl2c(impPointPlane,lpar,*fieldESH.product());

    AlgebraicSymMatrix55 mo = ROOT::Math::Similarity(jl2c.jacobian(),m);

    int ndof = 2*hits.size()-5;
    GlobalPoint vv = gp.position();
    math::XYZPoint  pos( vv.x(), vv.y(), vv.z() );
    GlobalVector pp = gp.momentum();
    math::XYZVector mom( pp.x(), pp.y(), pp.z() );

    auto track =  std::make_unique<reco::Track> ( chi2, ndof, pos, mom,
                  gp.charge(), CurvilinearTrajectoryError(mo));
    // filter???
    tracks.emplace_back(track.release(), shits);
  }
  assert(nh == e - b);
  // std::cout << "processed " << nh << " good tuples " << tracks.size() << std::endl;

  // store tracks
  storeTracks(iEvent, tracks, *httopo);
}

void PixelTrackProducerFromCUDA::produceCPU(edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup) {
  throw cms::Exception("NotImplemented") << "CPU version is no longer implemented";
}

DEFINE_FWK_MODULE(PixelTrackProducerFromCUDA);
