// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

#include "CondFormats/HGCalObjects/interface/TICLGeomHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLayersHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLookupHost.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/LayerClusterMaskDeviceCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"

#include "HGCalRecoveryTrackstersSoAAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The tracksters of PatternRecognitionbyRecovery, built on the device: one trackster for each layer cluster with a
  // non-zero filtered mask and an energy of at least minEnergy, in the order of the layer clusters.
  class HGCalRecoveryTrackstersSoAProducer : public stream::SynchronizingEDProducer<> {
  public:
    HGCalRecoveryTrackstersSoAProducer(const edm::ParameterSet& config)
        : SynchronizingEDProducer(config),
          layerClustersToken_{consumes(config.getParameter<edm::InputTag>("layer_clusters"))},
          filteredMaskToken_{consumes(config.getParameter<edm::InputTag>("filtered_mask"))},
          trackstersToken_{produces()},
          ticlGeomToken_{consumesCollector().esConsumes<TICLGeomHost, CaloGeometryRecord>()},
          ticlGeomLookupToken_{consumesCollector().esConsumes<TICLGeomLookupHost, CaloGeometryRecord>()},
          ticlGeomLayersToken_{consumesCollector().esConsumes<TICLGeomLayersHost, CaloGeometryRecord>()},
          minEnergy_{config.getParameter<double>("minEnergy")} {}

    // The number of tracksters is on the device, and the host needs it to allocate them
    void acquire(const device::Event& iEvent, const device::EventSetup&) override {
      const auto layerClusters = iEvent.get(layerClustersToken_).const_view();
      const auto mask = iEvent.get(filteredMaskToken_).const_view();
      nLayerClusters_ = layerClusters.position().metadata().size();
      if (mask.metadata().size() != static_cast<int>(nLayerClusters_)) {
        throw edm::Exception(edm::errors::LogicError)
            << "HGCalRecoveryTrackstersSoAProducer: the filtered mask has " << mask.metadata().size() << " values for "
            << nLayerClusters_ << " layer clusters.";
      }
      nTracksters_ = cms::alpakatools::make_host_buffer<uint32_t>(iEvent.queue());
      *nTracksters_->data() = 0;
      if (nLayerClusters_ == 0) {
        return;
      }
      selected_ = cms::alpakatools::make_device_buffer<uint32_t[]>(iEvent.queue(), nLayerClusters_);
      positions_ = cms::alpakatools::make_device_buffer<uint32_t[]>(iEvent.queue(), nLayerClusters_);
      algo_.selectLayerClusters(iEvent.queue(), minEnergy_, layerClusters, mask, selected_->data(), positions_->data());
      alpaka::memcpy(iEvent.queue(),
                     *nTracksters_,
                     cms::alpakatools::make_device_view<const uint32_t>(alpaka::getDev(iEvent.queue()),
                                                                        positions_->data()[nLayerClusters_ - 1]));
    }

    void produce(device::Event& iEvent, const device::EventSetup& iSetup) override {
      // As PatternRecognitionbyRecovery: the layer clusters up to the last CE-E layer count to the EM energy
      ticlgeom::Tools geometry;
      geometry.setGeometry(
          iSetup.getData(ticlGeomToken_), iSetup.getData(ticlGeomLookupToken_), iSetup.getData(ticlGeomLayersToken_));
      const float zLimitEm = std::abs(geometry.getPositionLayer(geometry.lastLayerEE(false), false).z());

      const uint32_t nTracksters = *nTracksters_->data();
      ticl::TracksterDeviceCollection tracksters(
          iEvent.queue(), nTracksters, nTracksters, nTracksters, nTracksters, nTracksters, nTracksters, nTracksters);
      algo_.fillTracksters(iEvent.queue(),
                           zLimitEm,
                           iEvent.get(layerClustersToken_).const_view(),
                           selected_ ? selected_->data() : nullptr,
                           positions_ ? positions_->data() : nullptr,
                           tracksters.view());
      iEvent.emplace(trackstersToken_, std::move(tracksters));
      selected_.reset();
      positions_.reset();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalSoAMergeLayerClusters"));
      desc.add<edm::InputTag>("filtered_mask", edm::InputTag("filteredLayerClustersSoARecovery"))
          ->setComment("The mask of the layer clusters that the iteration can use.");
      desc.add<double>("minEnergy", 0.)->setComment("Minimum layer cluster energy [GeV] to make a trackster.");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDGetToken<reco::CaloClusterDeviceCollection> layerClustersToken_;
    const device::EDGetToken<ticl::LayerClusterMaskDeviceCollection> filteredMaskToken_;
    const device::EDPutToken<ticl::TracksterDeviceCollection> trackstersToken_;
    const edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
    const edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
    const edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;
    const double minEnergy_;
    const HGCalRecoveryTrackstersSoAAlgoWrapper algo_{};
    // Set in acquire and used in produce. The number of tracksters is pinned, so that its copy does not block acquire.
    uint32_t nLayerClusters_ = 0;
    std::optional<cms::alpakatools::host_buffer<uint32_t>> nTracksters_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> selected_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> positions_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalRecoveryTrackstersSoAProducer);
