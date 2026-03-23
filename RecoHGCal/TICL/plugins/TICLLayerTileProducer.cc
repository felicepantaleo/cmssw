// Author: Marco Rovere, marco.rovere@cern.ch
// Date: 05/2019
//
#include <memory>  // unique_ptr

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace {
template <typename ClusterCollection>
void dumpLayerCluster(const ClusterCollection& layerClusters, unsigned int i) {
  const auto& lc = layerClusters[i];

  std::cerr << "LayerCluster dump\n"
            << "  index: " << i << "\n"
            << "  energy: " << lc.energy() << "\n"
            << "  eta: " << lc.eta() << "\n"
            << "  phi: " << lc.phi() << "\n"
            << "  x: " << lc.x() << "\n"
            << "  y: " << lc.y() << "\n"
            << "  z: " << lc.z() << "\n"
            << "  size: " << lc.size() << "\n"
            << "  algo: " << lc.algo() << "\n"
            << "  flags: " << lc.flags() << "\n"
            << "  seed rawId: " << lc.seed().rawId() << "\n";
  for (size_t ih = 0; ih < lc.hitsAndFractions().size(); ++ih) {
     std::cerr << "  hit[" << ih << "] rawId=" << lc.hitsAndFractions()[ih].first.rawId()
               << " fraction=" << lc.hitsAndFractions()[ih].second << "\n";
    HGCalDetId hitId(lc.hitsAndFractions()[ih].first.rawId());
    std::cerr << "    hit decoded:"
              << " det=" << hitId.det()
              << " subdet=" << hitId.subdetId()
              << " zside=" << hitId.zside()
              << " layer=" << hitId.layer()
              << " waferType=" << hitId.waferType()
              << " wafer=" << hitId.wafer()
              << " cell=" << hitId.cell() << "\n";
  }
  


}
}
class TICLLayerTileProducer : public edm::stream::EDProducer<> {
public:
  explicit TICLLayerTileProducer(const edm::ParameterSet &ps);
  ~TICLLayerTileProducer() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_HFNose_token_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;
  hgcal::RecHitTools rhtools_;
  std::string detector_;
  bool doNose_;
};

TICLLayerTileProducer::TICLLayerTileProducer(const edm::ParameterSet &ps)
    : detector_(ps.getParameter<std::string>("detector")) {
  geometry_token_ = esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>();

  doNose_ = (detector_ == "HFNose");

  if (doNose_) {
    clusters_HFNose_token_ =
        consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_HFNose_clusters"));
    produces<TICLLayerTilesHFNose>();
  } else {
    clusters_token_ = consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"));
    produces<TICLLayerTiles>();
    produces<TICLLayerTilesBarrel>("ticlLayerTilesBarrel");
  }
}

void TICLLayerTileProducer::beginRun(edm::Run const &, edm::EventSetup const &es) {
  edm::ESHandle<CaloGeometry> geom = es.getHandle(geometry_token_);
  rhtools_.setGeometry(*geom);
}

void TICLLayerTileProducer::produce(edm::Event &evt, const edm::EventSetup &) {
  std::unique_ptr<TICLLayerTilesHFNose> resultHFNose;
  std::unique_ptr<TICLLayerTiles> result;
  std::unique_ptr<TICLLayerTilesBarrel> resultBarrel;
  if (doNose_) {
    resultHFNose = std::make_unique<TICLLayerTilesHFNose>();
  } else {
    resultBarrel = std::make_unique<TICLLayerTilesBarrel>();
    result = std::make_unique<TICLLayerTiles>();
  }

  edm::Handle<std::vector<reco::CaloCluster>> cluster_h;
  if (doNose_)
    evt.getByToken(clusters_HFNose_token_, cluster_h);
  else
    evt.getByToken(clusters_token_, cluster_h);

  const auto &layerClusters = *cluster_h;
  int lcId = 0;
  for (auto const &lc : layerClusters) {
    const auto firstHitDetId = lc.hitsAndFractions()[0].first;
    int layer = rhtools_.getLayerWithOffset(firstHitDetId);
    bool isBarrelLC = rhtools_.isBarrel(firstHitDetId);
    if (!isBarrelLC) {
      layer += rhtools_.lastLayer(doNose_) * ((rhtools_.zside(firstHitDetId) + 1) >> 1) - 1;
    }
    assert(layer >= 0);

    if (doNose_) {
      resultHFNose->fill(layer, lc.eta(), lc.phi(), lcId);
    } else if (isBarrelLC) {
      resultBarrel->fill(layer, lc.eta(), lc.phi(), lcId);
    } else {
      if(std::abs(lc.eta()) < 1.35 || std::abs(lc.eta()) > 3.2) {
         LogDebug("TICLLayerTileProducer") << "LayerCluster with index: " << lcId << " has eta: " << lc.eta()
                                           << " which is out of range for the endcap tiles. Dumping the layer cluster details:\n";
        dumpLayerCluster(layerClusters, lcId);
      }
      result->fill(layer, lc.eta(), lc.phi(), lcId);
    }
    LogDebug("TICLLayerTileProducer") << "Adding layerClusterId: " << lcId << " into bin [eta,phi]: [ "
                                      << (*result)[layer].etaBin(lc.eta()) << ", " << (*result)[layer].phiBin(lc.phi())
                                      << "] for layer: " << layer << std::endl;
    lcId++;
  }
  if (doNose_)
    evt.put(std::move(resultHFNose));
  else {
    evt.put(std::move(resultBarrel), "ticlLayerTilesBarrel");
    evt.put(std::move(result));
  }
}

void TICLLayerTileProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCAL");
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("layer_HFNose_clusters", edm::InputTag("hgcalLayerClustersHFNose"));
  descriptions.add("ticlLayerTileProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TICLLayerTileProducer);
