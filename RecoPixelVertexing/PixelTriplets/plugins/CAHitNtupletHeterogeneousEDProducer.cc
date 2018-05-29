#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"

#include "CAHitQuadrupletGeneratorGPU.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitSeeds.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"

namespace {
void fillNtuplets(RegionsSeedingHitSets::RegionFiller &seedingHitSetsFiller,
                  const OrderedHitSeeds &quadruplets) {
  for (const auto &quad : quadruplets) {
    seedingHitSetsFiller.emplace_back(quad[0], quad[1], quad[2], quad[3]);
  }
}
} // namespace

class CAHitNtupletHeterogeneousEDProducer
    : public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices<
          heterogeneous::GPUCuda, heterogeneous::CPU>> {
public:
  CAHitNtupletHeterogeneousEDProducer(const edm::ParameterSet &iConfig);
  ~CAHitNtupletHeterogeneousEDProducer();

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  void beginStreamGPUCuda(edm::StreamID streamId,
                          cuda::stream_t<> &cudaStream) override;

  void acquireGPUCuda(const edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) override;
  void produceGPUCuda(edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) override;
  void produceCPU(edm::HeterogeneousEvent &iEvent,
                  const edm::EventSetup &iSetup) override;

private:
  edm::EDGetTokenT<IntermediateHitDoublets> doubletToken_;

  edm::RunningAverage localRA_;
  CAHitQuadrupletGeneratorGPU generator_;
  bool emptyRegionDoublets = false;
  std::unique_ptr<RegionsSeedingHitSets> seedingHitSets;
  std::vector<OrderedHitSeeds> ntuplets;
};

CAHitNtupletHeterogeneousEDProducer::CAHitNtupletHeterogeneousEDProducer(
    const edm::ParameterSet &iConfig)
    : HeterogeneousEDProducer(iConfig), doubletToken_(consumes<IntermediateHitDoublets>(iConfig.getParameter<edm::InputTag>("doublets"))),
      generator_(iConfig, consumesCollector()) {
  produces<RegionsSeedingHitSets>();
}

CAHitNtupletHeterogeneousEDProducer::~CAHitNtupletHeterogeneousEDProducer() {

}

void CAHitNtupletHeterogeneousEDProducer::fillDescriptions(
    edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("doublets", edm::InputTag("hitPairEDProducer"));
  CAHitQuadrupletGeneratorGPU::fillDescriptions(desc);

  auto label = CAHitQuadrupletGeneratorGPU::fillDescriptionsLabel() +
               std::string("EDProducer");
  descriptions.add(label, desc);
}

void CAHitNtupletHeterogeneousEDProducer::beginStreamGPUCuda(
    edm::StreamID streamId, cuda::stream_t<> &cudaStream) {
  generator_.allocateOnGPU();
}

void CAHitNtupletHeterogeneousEDProducer::produceGPUCuda(
    edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup,
    cuda::stream_t<> &cudaStream) {

  if (!emptyRegionDoublets) {
    edm::Handle<IntermediateHitDoublets> hdoublets;
    iEvent.getByToken(doubletToken_, hdoublets);
    const auto &regionDoublets = *hdoublets;
    const SeedingLayerSetsHits &seedingLayerHits =
        regionDoublets.seedingLayerHits();
    int index = 0;
    for (const auto &regionLayerPairs : regionDoublets) {
      const TrackingRegion &region = regionLayerPairs.region();
      auto seedingHitSetsFiller = seedingHitSets->beginRegion(&region);
      generator_.fillResults(regionDoublets, ntuplets, iSetup, seedingLayerHits,
                             cudaStream.id());
      fillNtuplets(seedingHitSetsFiller, ntuplets[index]);
      ntuplets[index].clear();
      index++;
    }
    localRA_.update(seedingHitSets->size());
  }
  iEvent.put(std::move(seedingHitSets));

}
void CAHitNtupletHeterogeneousEDProducer::acquireGPUCuda(
    const edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup,
    cuda::stream_t<> &cudaStream) {
  edm::Handle<IntermediateHitDoublets> hdoublets;
  iEvent.event().getByToken(doubletToken_, hdoublets);
  const auto &regionDoublets = *hdoublets;

  const SeedingLayerSetsHits &seedingLayerHits =
      regionDoublets.seedingLayerHits();
  if (seedingLayerHits.numberOfLayersInSet() <
      CAHitQuadrupletGeneratorGPU::minLayers) {
    throw cms::Exception("LogicError")
        << "CAHitNtupletHeterogeneousEDProducer expects "
           "SeedingLayerSetsHits::numberOfLayersInSet() to be >= "
        << CAHitQuadrupletGeneratorGPU::minLayers << ", got "
        << seedingLayerHits.numberOfLayersInSet()
        << ". This is likely caused by a configuration error of this module, "
           "HitPairEDProducer, or SeedingLayersEDProducer.";
  }

  seedingHitSets = std::make_unique<RegionsSeedingHitSets>();

  if (regionDoublets.empty()) {
    emptyRegionDoublets = true;
  } else {
    seedingHitSets->reserve(regionDoublets.regionSize(), localRA_.upper());
    generator_.initEvent(iEvent.event(), iSetup);

    LogDebug("CAHitNtupletHeterogeneousEDProducer")
        << "Creating ntuplets for " << regionDoublets.regionSize()
        << " regions, and " << regionDoublets.layerPairsSize()
        << " layer pairs";
    ntuplets.clear();
    ntuplets.resize(regionDoublets.regionSize());
    for (auto &ntuplet : ntuplets)
      ntuplet.reserve(localRA_.upper());

    generator_.hitNtuplets(regionDoublets, ntuplets, iSetup, seedingLayerHits,
                           cudaStream.id());
  }
}

void CAHitNtupletHeterogeneousEDProducer::produceCPU(
    edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup) {}

DEFINE_FWK_MODULE(CAHitNtupletHeterogeneousEDProducer);
