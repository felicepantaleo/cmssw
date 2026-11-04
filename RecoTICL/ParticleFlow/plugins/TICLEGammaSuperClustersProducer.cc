// The HGCAL e/gamma superclusters of TICLv6, one reco::SuperCluster per e/gamma trackster
// (TICLEGammaTrackstersProducer: a final trackster that the interpretation types as an electron or as an EM neutral).
// The final trackster of an electron or photon hypothesis is a supercluster of input tracksters: the layer clusters of
// each input trackster in it make one cluster, and the cluster of the largest energy is the seed. The position is the
// energy-weighted mean of the layer clusters, and the widths are their energy-weighted spreads in eta and phi. The
// energy is the raw energy times the output of the energy regression on the first regressionInputs features of
// EGammaSuperClusterFeatures.h when regressionModel is set, else the raw energy. With regressionMaxRatio > 0, the
// output of the regression is at most regressionMaxRatio, and a supercluster whose raw transverse energy times
// regressionMaxRatio is at or below superClusterEtThreshold keeps the raw energy. The seed hit of a supercluster is
// the seed hit of its seed cluster. The clusters are the instance "clusters", and tracksterIndex gives the e/gamma
// trackster of each supercluster.
// The seed input trackster is the one of largest energy. With seedOnlyMinAbsEta > 0, a supercluster whose seed input
// trackster has |eta| >= seedOnlyMinAbsEta keeps only the layer clusters of the seed input trackster. Else, with
// maxTimeDifference > 0 and a seed input trackster with a time, a supercluster drops the layer clusters of each input
// trackster whose time at the origin differs from the one of the seed by maxTimeDifference or more. An input
// trackster without time is kept.
// With scoreModel set, a supercluster that passes the threshold gets the score of the ONNX model scoreModel on
// scoreFeatures of EGammaSuperClusterFeatures.h, and a supercluster with a score below minScore (transverse energy
// below scoreEtSplit) or below minScoreHighEt (at or above scoreEtSplit) is dropped. The instance tracksterScore gives,
// for each e/gamma trackster, the output of typeScoreModel on the same features when it is set, else the score, of its
// supercluster before this cut (-1: no supercluster passes the threshold); it is empty without scoreModel and
// typeScoreModel.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/Common/interface/MultiSpan.h"
#include "DataFormats/Common/interface/RefProdVector.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"
#include "RecoTICL/Interpretation/interface/AssignmentMaps.h"
#include "RecoTICL/ParticleFlow/interface/EGammaSuperClusterFeatures.h"

namespace ticl::egammaSuperCluster {
  // The ONNX models of the producer; a model that is not configured is null.
  struct Models {
    std::unique_ptr<cms::Ort::ONNXRuntime> regression;
    std::unique_ptr<cms::Ort::ONNXRuntime> score;
    std::unique_ptr<cms::Ort::ONNXRuntime> typeScore;
  };
}  // namespace ticl::egammaSuperCluster

class TICLEGammaSuperClustersProducer
    : public edm::stream::EDProducer<edm::GlobalCache<ticl::egammaSuperCluster::Models>, edm::stream::WatchRuns> {
public:
  TICLEGammaSuperClustersProducer(const edm::ParameterSet &ps, const ticl::egammaSuperCluster::Models *);

  static std::unique_ptr<ticl::egammaSuperCluster::Models> initializeGlobalCache(const edm::ParameterSet &ps);
  static void globalEndJob(const ticl::egammaSuperCluster::Models *) {}
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginRun(const edm::Run &, const edm::EventSetup &es) override;
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
  const edm::EDGetTokenT<std::unordered_map<DetId, const unsigned int>> hitMap_token_;
  const edm::EDGetTokenT<edm::RefProdVector<HGCRecHitCollection>> hits_token_;
  const edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
  const edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
  const edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;
  const double maxTimeDifference_;
  const double seedOnlyMinAbsEta_;
  const unsigned int regressionInputs_;
  const double regressionMaxRatio_;
  const double superClusterEtThreshold_;
  const double minScore_;
  const double minScoreHighEt_;
  const double scoreEtSplit_;
  edm::EDGetTokenT<TICLLayerTiles> layerTiles_token_;
  edm::EDGetTokenT<std::vector<reco::Track>> tracks_token_;
  edm::EDGetTokenT<std::vector<int>> eGammaIndex_token_, trackToTrackster_token_, trackMode_token_, neutralIdx_token_,
      neutralPdg_token_;
  ticlgeom::Tools tools_;
  const edm::EDPutTokenT<reco::CaloClusterCollection> clusters_put_token_;
  const edm::EDPutTokenT<reco::SuperClusterCollection> superClusters_put_token_;
  const edm::EDPutTokenT<std::vector<int>> index_put_token_;
  const edm::EDPutTokenT<std::vector<float>> score_put_token_;
};

TICLEGammaSuperClustersProducer::TICLEGammaSuperClustersProducer(const edm::ParameterSet &ps,
                                                                 const ticl::egammaSuperCluster::Models *)
    : eGammaTracksters_token_(consumes(ps.getParameter<edm::InputTag>("eGammaTracksters"))),
      inputTracksters_token_(consumes(ps.getParameter<edm::InputTag>("inputTracksters"))),
      layerClusters_token_(consumes(ps.getParameter<edm::InputTag>("layerClusters"))),
      hitMap_token_(consumes(ps.getParameter<edm::InputTag>("hitMap"))),
      hits_token_(consumes(ps.getParameter<edm::InputTag>("hits"))),
      ticlGeomToken_(esConsumes<TICLGeomHost, CaloGeometryRecord, edm::Transition::BeginRun>()),
      ticlGeomLookupToken_(esConsumes<TICLGeomLookupHost, CaloGeometryRecord, edm::Transition::BeginRun>()),
      ticlGeomLayersToken_(esConsumes<TICLGeomLayersHost, CaloGeometryRecord, edm::Transition::BeginRun>()),
      maxTimeDifference_(ps.getParameter<double>("maxTimeDifference")),
      seedOnlyMinAbsEta_(ps.getParameter<double>("seedOnlyMinAbsEta")),
      regressionInputs_(ps.getParameter<unsigned int>("regressionInputs")),
      regressionMaxRatio_(ps.getParameter<double>("regressionMaxRatio")),
      superClusterEtThreshold_(ps.getParameter<double>("superClusterEtThreshold")),
      minScore_(ps.getParameter<double>("minScore")),
      minScoreHighEt_(ps.getParameter<double>("minScoreHighEt")),
      scoreEtSplit_(ps.getParameter<double>("scoreEtSplit")),
      clusters_put_token_(produces("clusters")),
      superClusters_put_token_(produces()),
      index_put_token_(produces<std::vector<int>>("tracksterIndex")),
      score_put_token_(produces<std::vector<float>>("tracksterScore")) {
  if (!ps.getParameter<std::string>("scoreModel").empty() || !ps.getParameter<std::string>("typeScoreModel").empty()) {
    layerTiles_token_ = consumes(ps.getParameter<edm::InputTag>("layerTiles"));
    tracks_token_ = consumes(ps.getParameter<edm::InputTag>("tracks"));
    const auto &eg = ps.getParameter<edm::InputTag>("eGammaTracksters");
    eGammaIndex_token_ = consumes(edm::InputTag(eg.label(), "interpretationIndex", eg.process()));
    const auto &it = ps.getParameter<edm::InputTag>("interpretations");
    trackToTrackster_token_ = consumes(edm::InputTag(it.label(), "trackToTrackster", it.process()));
    trackMode_token_ = consumes(edm::InputTag(it.label(), "trackMode", it.process()));
    neutralIdx_token_ = consumes(edm::InputTag(it.label(), "neutralIdx", it.process()));
    neutralPdg_token_ = consumes(edm::InputTag(it.label(), "neutralPdg", it.process()));
  }
  if (regressionInputs_ < 1 || regressionInputs_ > ticl::egammaSuperCluster::kNFeatures)
    throw cms::Exception("Configuration") << "TICLEGammaSuperClustersProducer: regressionInputs must be in [1, "
                                          << ticl::egammaSuperCluster::kNFeatures << "]";
}

void TICLEGammaSuperClustersProducer::beginRun(const edm::Run &, const edm::EventSetup &es) {
  tools_.setGeometry(es.getData(ticlGeomToken_), es.getData(ticlGeomLookupToken_), es.getData(ticlGeomLayersToken_));
}

namespace {
  // Time [ns] of a trackster at the origin: its time at the barycenter minus the flight time from the origin.
  double timeAtOrigin(const ticl::Trackster &trackster) {
    constexpr double kSpeedOfLight = 29.9792458;  // cm/ns
    return trackster.time() - trackster.barycenter().r() / kSpeedOfLight;
  }
}  // namespace

std::unique_ptr<ticl::egammaSuperCluster::Models> TICLEGammaSuperClustersProducer::initializeGlobalCache(
    const edm::ParameterSet &ps) {
  auto models = std::make_unique<ticl::egammaSuperCluster::Models>();
  if (const auto &model = ps.getParameter<std::string>("regressionModel"); !model.empty())
    models->regression = std::make_unique<cms::Ort::ONNXRuntime>(edm::FileInPath(model).fullPath());
  if (const auto &model = ps.getParameter<std::string>("scoreModel"); !model.empty())
    models->score = std::make_unique<cms::Ort::ONNXRuntime>(edm::FileInPath(model).fullPath());
  if (const auto &model = ps.getParameter<std::string>("typeScoreModel"); !model.empty())
    models->typeScore = std::make_unique<cms::Ort::ONNXRuntime>(edm::FileInPath(model).fullPath());
  return models;
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
  std::vector<char> kept;
  std::vector<std::pair<int, double>> inputEnergy;
  for (std::size_t k = 0; k < eGammaTracksters.size(); ++k) {
    const auto &ts = eGammaTracksters[k];
    const auto &vertices = ts.vertices();
    // A layer cluster shared by several final tracksters gives each the same fraction of its energy.
    auto fraction = [&ts](std::size_t i) {
      const float multiplicity = i < ts.vertex_multiplicity().size() ? ts.vertex_multiplicity(i) : 1.f;
      return multiplicity > 0.f ? 1.f / multiplicity : 1.f;
    };
    for (auto v : vertices)
      if (v >= layerClusters.size())
        throw cms::Exception("LogicError") << "TICLEGammaSuperClustersProducer: layer cluster " << v
                                           << " of e/gamma trackster " << k << " out of range";
    kept.assign(vertices.size(), 1);
    if (maxTimeDifference_ > 0. || seedOnlyMinAbsEta_ > 0.) {
      inputEnergy.clear();
      for (std::size_t i = 0; i < vertices.size(); ++i) {
        const int input = inputOf[vertices[i]];
        auto it =
            std::find_if(inputEnergy.begin(), inputEnergy.end(), [input](const auto &p) { return p.first == input; });
        if (it == inputEnergy.end()) {
          inputEnergy.emplace_back(input, 0.);
          it = inputEnergy.end() - 1;
        }
        it->second += layerClusters[vertices[i]].energy() * fraction(i);
      }
      const auto seed = std::max_element(
          inputEnergy.begin(), inputEnergy.end(), [](const auto &a, const auto &b) { return a.second < b.second; });
      if (seed != inputEnergy.end() && seed->first >= 0) {
        const auto &seedTrackster = inputTracksters[seed->first];
        if (seedOnlyMinAbsEta_ > 0. && std::abs(seedTrackster.barycenter().eta()) >= seedOnlyMinAbsEta_) {
          for (std::size_t i = 0; i < vertices.size(); ++i)
            kept[i] = inputOf[vertices[i]] == seed->first;
        } else if (maxTimeDifference_ > 0. && seedTrackster.timeError() > 0.f) {
          const double seedTime = timeAtOrigin(seedTrackster);
          for (std::size_t i = 0; i < vertices.size(); ++i) {
            const int input = inputOf[vertices[i]];
            if (input < 0 || input == seed->first || inputTracksters[input].timeError() <= 0.f)
              continue;
            kept[i] = std::abs(timeAtOrigin(inputTracksters[input]) - seedTime) < maxTimeDifference_;
          }
        }
      }
    }
    parts.clear();
    double energy = 0., x = 0., y = 0., z = 0.;
    for (std::size_t i = 0; i < vertices.size(); ++i) {
      if (!kept[i])
        continue;
      const unsigned int v = vertices[i];
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
      if (!kept[i])
        continue;
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

  namespace egsc = ticl::egammaSuperCluster;
  const auto *regressionModel = globalCache()->regression.get();
  const auto *scoreModel = globalCache()->score.get();
  const auto *typeScoreModel = globalCache()->typeScore.get();
  if (regressionModel != nullptr && !superClusters.empty()) {
    // The cell-level features are computed only for a regression that reads them.
    const bool cellLevel = regressionInputs_ > egsc::kNFeaturesV2;
    const std::unordered_map<DetId, const unsigned int> *hitMap = nullptr;
    edm::MultiSpan<HGCRecHit> hits;
    if (cellLevel) {
      hitMap = &evt.get(hitMap_token_);
      hits = edm::MultiSpan<HGCRecHit>(evt.get(hits_token_));
    }
    std::vector<std::size_t> regressed;
    for (std::size_t i = 0; i < superClusters.size(); ++i) {
      const auto &sc = superClusters[i];
      if (regressionMaxRatio_ <= 0. ||
          sc.rawEnergy() * std::sin(sc.position().theta()) * regressionMaxRatio_ > superClusterEtThreshold_)
        regressed.push_back(i);
    }
    const auto n = static_cast<int64_t>(regressed.size());
    cms::Ort::FloatArrays input(1);
    input[0].reserve(regressed.size() * regressionInputs_);
    for (auto i : regressed) {
      const auto cell = cellLevel ? egsc::cellFeatures(superClusters[i], *hitMap, hits, tools_) : egsc::CellFeatures();
      const auto features =
          egsc::regressionFeatures(superClusters[i], eGammaTracksters[tracksterIndex[i]], cell, hits.size());
      input[0].insert(input[0].end(), features.begin(), features.begin() + regressionInputs_);
    }
    if (n > 0) {
      const auto output = regressionModel->run({"input"}, input, {{n, static_cast<int64_t>(regressionInputs_)}}, {}, n);
      if (output.empty() || output[0].size() != regressed.size())
        throw cms::Exception("LogicError")
            << "TICLEGammaSuperClustersProducer: the regression gives " << (output.empty() ? 0 : output[0].size())
            << " values for " << regressed.size() << " superclusters";
      for (std::size_t k = 0; k < regressed.size(); ++k) {
        double ratio = output[0][k];
        if (regressionMaxRatio_ > 0.)
          ratio = std::min(ratio, regressionMaxRatio_);
        auto &sc = superClusters[regressed[k]];
        const double energy = sc.rawEnergy() * ratio;
        sc.setEnergy(energy);
        sc.setCorrectedEnergy(energy);
      }
    }
  }

  // Transverse energy threshold, after the regression.
  std::vector<std::size_t> passing;
  for (std::size_t i = 0; i < superClusters.size(); ++i) {
    const auto &sc = superClusters[i];
    if (sc.energy() * std::sin(sc.position().theta()) > superClusterEtThreshold_)
      passing.push_back(i);
  }
  // Score of the superclusters that pass the threshold.
  std::vector<char> keep(passing.size(), 1);
  std::vector<float> tracksterScore;
  if (scoreModel != nullptr || typeScoreModel != nullptr)
    tracksterScore.assign(eGammaTracksters.size(), -1.f);
  if ((scoreModel != nullptr || typeScoreModel != nullptr) && !passing.empty()) {
    const auto &hitMap = evt.get(hitMap_token_);
    const edm::MultiSpan<HGCRecHit> hits(evt.get(hits_token_));
    const auto &tiles = evt.get(layerTiles_token_);
    const auto &tracks = evt.get(tracks_token_);
    auto trackTiles = std::make_unique<egsc::TrackTiles>();
    egsc::fillTrackTiles(tracks, *trackTiles);
    // Interpretation type of each e/gamma trackster: 11 for the trackster of a track in electron mode, else the pdg
    // id of the neutral.
    const auto &eGammaIndex = evt.get(eGammaIndex_token_);
    const auto &trackToTrackster = evt.get(trackToTrackster_token_);
    const auto &trackMode = evt.get(trackMode_token_);
    const auto &neutralIdx = evt.get(neutralIdx_token_);
    const auto &neutralPdg = evt.get(neutralPdg_token_);
    std::unordered_map<int, int> typeOfFinal;
    for (std::size_t k = 0; k < neutralIdx.size() && k < neutralPdg.size(); ++k)
      typeOfFinal[neutralIdx[k]] = neutralPdg[k];
    for (std::size_t t = 0; t < trackMode.size() && t < trackToTrackster.size(); ++t)
      if (trackMode[t] == static_cast<int>(ticl::TrackMode::kElectron) && trackToTrackster[t] >= 0)
        typeOfFinal[trackToTrackster[t]] = 11;
    auto typeOf = [&](int k) {
      if (k < 0 || k >= static_cast<int>(eGammaIndex.size()))
        return 0;
      const auto it = typeOfFinal.find(eGammaIndex[k]);
      return it == typeOfFinal.end() ? 0 : it->second;
    };
    const auto n = static_cast<int64_t>(passing.size());
    cms::Ort::FloatArrays input(1);
    input[0].reserve(passing.size() * egsc::kNScoreFeatures);
    for (auto i : passing) {
      const auto &sc = superClusters[i];
      const auto &ts = eGammaTracksters[tracksterIndex[i]];
      const auto regression =
          egsc::regressionFeatures(sc, ts, egsc::cellFeatures(sc, hitMap, hits, tools_), hits.size());
      const auto isolation =
          egsc::isolationEt(sc, tiles, layerClusters, egsc::kIsolationInnerDR, egsc::kIsolationOuterDR);
      const int type = typeOf(tracksterIndex[i]);
      const auto features = egsc::scoreFeatures(
          regression, sc, ts, isolation, type == 11, type == 111, egsc::trackFeatures(sc, tracks, *trackTiles));
      input[0].insert(input[0].end(), features.begin(), features.end());
    }
    const std::vector<std::vector<int64_t>> shape{{n, static_cast<int64_t>(egsc::kNScoreFeatures)}};
    auto runModel = [&](const cms::Ort::ONNXRuntime &model, const char *name) {
      auto output = model.run({"input"}, input, shape, {}, n);
      if (output.empty() || output[0].size() != passing.size())
        throw cms::Exception("LogicError")
            << "TICLEGammaSuperClustersProducer: the " << name << " model gives "
            << (output.empty() ? 0 : output[0].size()) << " values for " << passing.size() << " superclusters";
      return output[0];
    };
    if (scoreModel != nullptr) {
      const auto score = runModel(*scoreModel, "score");
      for (std::size_t k = 0; k < passing.size(); ++k) {
        const auto &sc = superClusters[passing[k]];
        const double et = sc.energy() * std::sin(sc.position().theta());
        keep[k] = score[k] >= (et < scoreEtSplit_ ? minScore_ : minScoreHighEt_);
        if (typeScoreModel == nullptr)
          tracksterScore[tracksterIndex[passing[k]]] = score[k];
      }
    }
    if (typeScoreModel != nullptr) {
      const auto typeScore = runModel(*typeScoreModel, "type score");
      for (std::size_t k = 0; k < passing.size(); ++k)
        tracksterScore[tracksterIndex[passing[k]]] = typeScore[k];
    }
  }
  reco::SuperClusterCollection selected;
  std::vector<int> selectedIndex;
  for (std::size_t k = 0; k < passing.size(); ++k) {
    if (!keep[k])
      continue;
    selected.push_back(superClusters[passing[k]]);
    selectedIndex.push_back(tracksterIndex[passing[k]]);
  }
  evt.emplace(superClusters_put_token_, std::move(selected));
  evt.emplace(index_put_token_, std::move(selectedIndex));
  evt.emplace(score_put_token_, std::move(tracksterScore));
}

void TICLEGammaSuperClustersProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("eGammaTracksters", edm::InputTag("ticlEGammaTracksters"))
      ->setComment("E/gamma tracksters: the final tracksters typed as electrons or EM neutrals.");
  desc.add<edm::InputTag>("inputTracksters", edm::InputTag("ticlTrackstersCLUE3DHigh"))
      ->setComment(
          "Tracksters that the e/gamma superclustering groups; each one in an e/gamma trackster is a cluster.");
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("hitMap", edm::InputTag("recHitMapProducer", "hgcalRecHitMap"));
  desc.add<edm::InputTag>("hits", edm::InputTag("recHitMapProducer", "RefProdVectorHGCRecHitCollection"));
  desc.add<double>("maxTimeDifference", 0.05)
      ->setComment(
          "Max difference [ns] of the times at the origin of a cluster and of the seed cluster; <= 0: no cut.");
  desc.add<double>("seedOnlyMinAbsEta", 2.4)
      ->setComment(
          "Min |eta| of the seed cluster above which a supercluster keeps the seed cluster only; <= 0: never.");
  desc.add<unsigned int>("regressionInputs", static_cast<unsigned int>(ticl::egammaSuperCluster::kNFeatures))
      ->setComment("Number of features (EGammaSuperClusterFeatures.h) that the regression reads.");
  desc.add<double>("regressionMaxRatio", 1.5)->setComment("Max output of the regression; <= 0: no limit.");
  desc.add<double>("superClusterEtThreshold", 4.)
      ->setComment("Min transverse energy [GeV] of a supercluster, after the regression.");
  desc.add<std::string>("regressionModel", "RecoTICL/ParticleFlow/data/egammaSuperClusters/regression_v3.onnx")
      ->setComment("FileInPath of the ONNX energy regression; empty: the supercluster energy is the raw energy.");
  desc.add<std::string>("scoreModel", "")
      ->setComment("FileInPath of the ONNX supercluster score on scoreFeatures; empty: no score.");
  desc.add<double>("minScore", 0.5)->setComment("Min score of a supercluster below scoreEtSplit, with scoreModel set.");
  desc.add<double>("minScoreHighEt", 0.5)
      ->setComment("Min score of a supercluster at or above scoreEtSplit, with scoreModel set.");
  desc.add<double>("scoreEtSplit", 10.)
      ->setComment("Transverse energy [GeV] after the regression that separates minScore and minScoreHighEt.");
  desc.add<std::string>("typeScoreModel", "")
      ->setComment("FileInPath of the ONNX score on scoreFeatures for the instance tracksterScore; empty: the score.");
  desc.add<edm::InputTag>("layerTiles", edm::InputTag("ticlLayerTileProducer"))
      ->setComment(
          "Tiles of the layer clusters for the isolation of the score features, with scoreModel or typeScoreModel "
          "set.");
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"))
      ->setComment("Tracks of the track features of the score features, with scoreModel or typeScoreModel set.");
  desc.add<edm::InputTag>("interpretations", edm::InputTag("ticlTracksterInterpretations"))
      ->setComment(
          "Interpretation stage whose maps type the e/gamma tracksters, with scoreModel or typeScoreModel set.");
  descriptions.add("ticlEGammaSuperClustersProducer", desc);
}

DEFINE_FWK_MODULE(TICLEGammaSuperClustersProducer);
