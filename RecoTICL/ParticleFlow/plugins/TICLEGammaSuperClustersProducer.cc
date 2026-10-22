// The HGCAL e/gamma superclusters of TICLv6, one reco::SuperCluster per e/gamma trackster
// (TICLEGammaTrackstersProducer: a final trackster that the interpretation types as an electron or as an EM neutral).
// The final trackster of an electron or photon hypothesis is a supercluster of input tracksters: the layer clusters of
// each input trackster in it make one cluster, and the cluster of the largest energy is the seed. The position is the
// energy-weighted mean of the layer clusters, and the widths are their energy-weighted spreads in eta and phi. The
// energy is the raw energy times the output of the energy regression on trackster features
// (EGammaSuperClusterFeatures.h) when regressionModel is set, else the raw energy. The seed hit of a supercluster is
// the seed hit of its seed cluster. The clusters are the instance "clusters", and tracksterIndex gives the e/gamma
// trackster of each supercluster.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoTICL/ParticleFlow/interface/EGammaSuperClusterFeatures.h"

class TICLEGammaSuperClustersProducer : public edm::stream::EDProducer<edm::GlobalCache<cms::Ort::ONNXRuntime>> {
public:
  TICLEGammaSuperClustersProducer(const edm::ParameterSet &ps, const cms::Ort::ONNXRuntime *);

  static std::unique_ptr<cms::Ort::ONNXRuntime> initializeGlobalCache(const edm::ParameterSet &ps);
  static void globalEndJob(const cms::Ort::ONNXRuntime *) {}
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void produce(edm::Event &evt, const edm::EventSetup &) override;

private:
  // The layer clusters of one input trackster in an e/gamma trackster.
  struct Part {
    int input = -1;
    double energy = 0.;
    double x = 0., y = 0., z = 0.;
    std::vector<std::pair<DetId, float>> hits;
    unsigned int seedLayerCluster = 0;
    double seedLayerClusterEnergy = -1.;
  };

  const edm::EDGetTokenT<std::vector<ticl::Trackster>> eGammaTracksters_token_;
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> inputTracksters_token_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClusters_token_;
  const double superClusterEtThreshold_;
  const edm::EDPutTokenT<reco::CaloClusterCollection> clusters_put_token_;
  const edm::EDPutTokenT<reco::SuperClusterCollection> superClusters_put_token_;
  const edm::EDPutTokenT<std::vector<int>> index_put_token_;
};

TICLEGammaSuperClustersProducer::TICLEGammaSuperClustersProducer(const edm::ParameterSet &ps,
                                                                 const cms::Ort::ONNXRuntime *)
    : eGammaTracksters_token_(consumes(ps.getParameter<edm::InputTag>("eGammaTracksters"))),
      inputTracksters_token_(consumes(ps.getParameter<edm::InputTag>("inputTracksters"))),
      layerClusters_token_(consumes(ps.getParameter<edm::InputTag>("layerClusters"))),
      superClusterEtThreshold_(ps.getParameter<double>("superClusterEtThreshold")),
      clusters_put_token_(produces("clusters")),
      superClusters_put_token_(produces()),
      index_put_token_(produces<std::vector<int>>("tracksterIndex")) {}

std::unique_ptr<cms::Ort::ONNXRuntime> TICLEGammaSuperClustersProducer::initializeGlobalCache(
    const edm::ParameterSet &ps) {
  const auto &model = ps.getParameter<std::string>("regressionModel");
  if (model.empty())
    return nullptr;
  return std::make_unique<cms::Ort::ONNXRuntime>(edm::FileInPath(model).fullPath());
}

void TICLEGammaSuperClustersProducer::produce(edm::Event &evt, const edm::EventSetup &) {
  const auto &eGammaTracksters = evt.get(eGammaTracksters_token_);
  const auto &inputTracksters = evt.get(inputTracksters_token_);
  const auto &layerClusters = evt.get(layerClusters_token_);

  // Input trackster of each layer cluster (-1: none).
  std::vector<int> inputOf(layerClusters.size(), -1);
  for (std::size_t j = 0; j < inputTracksters.size(); ++j)
    for (auto v : inputTracksters[j].vertices()) {
      if (v >= layerClusters.size())
        throw cms::Exception("LogicError") << "TICLEGammaSuperClustersProducer: layer cluster " << v << " of input "
                                           << "trackster " << j << " out of range";
      inputOf[v] = static_cast<int>(j);
    }

  // The clusters of each e/gamma trackster, seed first, then in decreasing energy.
  reco::CaloClusterCollection clusters;
  std::vector<std::size_t> firstCluster(eGammaTracksters.size() + 1, 0);
  std::vector<double> etaWidth(eGammaTracksters.size(), 0.), phiWidth(eGammaTracksters.size(), 0.);
  std::vector<math::XYZPoint> position(eGammaTracksters.size());
  std::vector<Part> parts;
  for (std::size_t k = 0; k < eGammaTracksters.size(); ++k) {
    const auto &ts = eGammaTracksters[k];
    const auto &vertices = ts.vertices();
    // A layer cluster shared by several final tracksters gives each the same fraction of its energy.
    auto fraction = [&ts](std::size_t i) {
      const float multiplicity = i < ts.vertex_multiplicity().size() ? ts.vertex_multiplicity(i) : 1.f;
      return multiplicity > 0.f ? 1.f / multiplicity : 1.f;
    };
    parts.clear();
    double energy = 0., x = 0., y = 0., z = 0.;
    for (std::size_t i = 0; i < vertices.size(); ++i) {
      const unsigned int v = vertices[i];
      if (v >= layerClusters.size())
        throw cms::Exception("LogicError") << "TICLEGammaSuperClustersProducer: layer cluster " << v
                                           << " of e/gamma trackster " << k << " out of range";
      const auto &lc = layerClusters[v];
      const float f = fraction(i);
      const double e = lc.energy() * f;
      auto part = std::find_if(parts.begin(), parts.end(), [&](const Part &p) { return p.input == inputOf[v]; });
      if (part == parts.end()) {
        parts.emplace_back();
        parts.back().input = inputOf[v];
        part = parts.end() - 1;
      }
      part->energy += e;
      part->x += e * lc.x();
      part->y += e * lc.y();
      part->z += e * lc.z();
      for (auto const &[id, hitFraction] : lc.hitsAndFractions())
        part->hits.emplace_back(id, hitFraction * f);
      if (e > part->seedLayerClusterEnergy) {
        part->seedLayerClusterEnergy = e;
        part->seedLayerCluster = v;
      }
      energy += e;
      x += e * lc.x();
      y += e * lc.y();
      z += e * lc.z();
    }
    std::stable_sort(parts.begin(), parts.end(), [](const Part &a, const Part &b) { return a.energy > b.energy; });
    for (auto &p : parts) {
      if (p.energy <= 0.)
        continue;
      clusters.emplace_back(p.energy,
                            math::XYZPoint(p.x / p.energy, p.y / p.energy, p.z / p.energy),
                            reco::CaloID(reco::CaloID::DET_HGCAL_ENDCAP),
                            std::move(p.hits),
                            reco::CaloCluster::particleFlow,
                            layerClusters[p.seedLayerCluster].seed());
    }
    firstCluster[k + 1] = clusters.size();
    if (energy <= 0.)
      continue;
    position[k] = math::XYZPoint(x / energy, y / energy, z / energy);
    double eta2 = 0., phi2 = 0.;
    for (std::size_t i = 0; i < vertices.size(); ++i) {
      const auto &lc = layerClusters[vertices[i]];
      const double e = lc.energy() * fraction(i);
      eta2 += e * std::pow(lc.eta() - position[k].eta(), 2);
      phi2 += e * std::pow(reco::deltaPhi(lc.phi(), position[k].phi()), 2);
    }
    etaWidth[k] = std::sqrt(eta2 / energy);
    phiWidth[k] = std::sqrt(phi2 / energy);
  }
  const auto clustersH = evt.emplace(clusters_put_token_, std::move(clusters));

  reco::SuperClusterCollection superClusters;
  std::vector<int> tracksterIndex;
  for (std::size_t k = 0; k < eGammaTracksters.size(); ++k) {
    if (firstCluster[k + 1] == firstCluster[k])
      continue;
    reco::CaloClusterPtrVector members;
    for (std::size_t c = firstCluster[k]; c < firstCluster[k + 1]; ++c)
      members.push_back(reco::CaloClusterPtr(clustersH, c));
    double raw = 0.;
    for (auto const &c : members)
      raw += c->energy();
    superClusters.emplace_back(raw, position[k], members[0], members, 0., phiWidth[k], etaWidth[k]);
    superClusters.back().reco::CaloCluster::setSeed(members[0]->seed());
    superClusters.back().setCorrectedEnergy(raw);
    tracksterIndex.push_back(static_cast<int>(k));
  }

  if (globalCache() != nullptr && !superClusters.empty()) {
    using ticl::egammaSuperCluster::kNFeatures;
    const auto n = static_cast<int64_t>(superClusters.size());
    cms::Ort::FloatArrays input(1);
    input[0].reserve(superClusters.size() * kNFeatures);
    for (std::size_t i = 0; i < superClusters.size(); ++i) {
      const auto features =
          ticl::egammaSuperCluster::regressionFeatures(superClusters[i], eGammaTracksters[tracksterIndex[i]]);
      input[0].insert(input[0].end(), features.begin(), features.end());
    }
    const auto output = globalCache()->run({"input"}, input, {{n, static_cast<int64_t>(kNFeatures)}}, {}, n);
    if (output.empty() || output[0].size() != superClusters.size())
      throw cms::Exception("LogicError") << "TICLEGammaSuperClustersProducer: the regression gives "
                                         << (output.empty() ? 0 : output[0].size()) << " values for "
                                         << superClusters.size() << " superclusters";
    for (std::size_t i = 0; i < superClusters.size(); ++i) {
      const double energy = superClusters[i].rawEnergy() * output[0][i];
      superClusters[i].setEnergy(energy);
      superClusters[i].setCorrectedEnergy(energy);
    }
  }

  // Transverse energy threshold, after the regression.
  reco::SuperClusterCollection selected;
  std::vector<int> selectedIndex;
  for (std::size_t i = 0; i < superClusters.size(); ++i) {
    const auto &sc = superClusters[i];
    if (sc.energy() * std::sin(sc.position().theta()) <= superClusterEtThreshold_)
      continue;
    selected.push_back(sc);
    selectedIndex.push_back(tracksterIndex[i]);
  }
  evt.emplace(superClusters_put_token_, std::move(selected));
  evt.emplace(index_put_token_, std::move(selectedIndex));
}

void TICLEGammaSuperClustersProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("eGammaTracksters", edm::InputTag("ticlEGammaTracksters"))
      ->setComment("E/gamma tracksters: the final tracksters typed as electrons or EM neutrals.");
  desc.add<edm::InputTag>("inputTracksters", edm::InputTag("ticlTrackstersCLUE3DHigh"))
      ->setComment(
          "Tracksters that the e/gamma superclustering groups; each one in an e/gamma trackster is a cluster.");
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<double>("superClusterEtThreshold", 4.)
      ->setComment("Min transverse energy [GeV] of a supercluster, after the regression.");
  desc.add<std::string>("regressionModel", "RecoTICL/ParticleFlow/data/egammaSuperClusters/regression_v2.onnx")
      ->setComment("FileInPath of the ONNX energy regression; empty: the supercluster energy is the raw energy.");
  descriptions.add("ticlEGammaSuperClustersProducer", desc);
}

DEFINE_FWK_MODULE(TICLEGammaSuperClustersProducer);
