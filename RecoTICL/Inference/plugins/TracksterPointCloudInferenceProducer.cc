// Energy, energy uncertainty, pile-up fraction and species of every final trackster from a point-cloud model (ONNX).
// The inputs are the features of ticl::TracksterPointCloud, the function that also makes the training dataset: the
// global features (glob [B, nG]), the points (points [B, N, nP], padded to the largest point count of the batch) and
// the point mask (valid [B, N], 1 for a point). The rows are sorted by point count before the batches are made. The
// outputs are energy [B], sigma [B], pu [B] and species [B, 5]. A trackster below minRawEnergy gets its raw energy,
// sigma 0, pile-up fraction -1 and species 0.

#include <algorithm>
#include <chrono>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/HGCalReco/interface/MtdHostCollection.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"
#include "RecoTICL/Common/interface/TICLUtils.h"
#include "RecoTICL/Inference/interface/TracksterPointCloud.h"
#include "RecoTICL/Inference/interface/TracksterPointCloudInputs.h"
#include "RecoTICL/Interpretation/interface/TrackImpact.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

namespace {
  constexpr unsigned int kNSpecies = 5;

  ticl::TracksterPointCloud::Config pointCloudConfig(const edm::ParameterSet& ps) {
    ticl::TracksterPointCloud::Config c;
    c.radiusEE = ps.getParameter<double>("radiusEE");
    c.radiusEH = ps.getParameter<double>("radiusEH");
    c.ringOuterFactor = ps.getParameter<double>("ringOuterFactor");
    c.maxAxisAngle = ps.getParameter<double>("maxAxisAngle");
    c.maxOwnPoints = ps.getParameter<unsigned int>("maxOwnPoints");
    c.maxNeighbourPoints = ps.getParameter<unsigned int>("maxNeighbourPoints");
    return c;
  }
}  // namespace

class TracksterPointCloudInferenceProducer
    : public edm::stream::EDProducer<edm::GlobalCache<cms::Ort::ONNXRuntime>, edm::stream::WatchRuns> {
public:
  TracksterPointCloudInferenceProducer(const edm::ParameterSet& ps, const cms::Ort::ONNXRuntime* cache);
  static std::unique_ptr<cms::Ort::ONNXRuntime> initializeGlobalCache(const edm::ParameterSet& ps);
  static void globalEndJob(const cms::Ort::ONNXRuntime*) {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void beginRun(edm::Run const&, edm::EventSetup const& es) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void produce(edm::Event& evt, const edm::EventSetup& es) override;
  void endStream() override;

private:
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksterToken_;
  const edm::EDGetTokenT<std::vector<int>> trackToTracksterToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> lcTimeToken_;
  const edm::EDGetTokenT<TICLLayerTiles> tilesToken_;
  const edm::EDGetTokenT<std::vector<reco::Vertex>> vertexToken_, primaryVertexToken_;
  const edm::EDGetTokenT<std::vector<reco::Track>> trackToken_;
  const edm::EDGetTokenT<MtdHostCollection> timingToken_;
  const edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
  const edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
  const edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propToken_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hgconsToken_;
  const double minRawEnergy_;
  const unsigned int batchSize_;
  const ticl::TracksterPointCloud pointCloud_;
  const ticl::PointCloudTrackSelection trackSelection_;
  ticlgeom::Tools rhtools_;
  std::array<std::unique_ptr<GeomDet>, 2> frontDisks_;
  // Sums over the events of the stream of the time [ms] of the features (vertex, tracks, builds) and of the inference.
  double sumBuildMs_ = 0., sumInferMs_ = 0.;
  unsigned long nEvents_ = 0, nTracksters_ = 0;
};

TracksterPointCloudInferenceProducer::TracksterPointCloudInferenceProducer(const edm::ParameterSet& ps,
                                                                           const cms::Ort::ONNXRuntime*)
    : tracksterToken_(consumes<std::vector<ticl::Trackster>>(ps.getParameter<edm::InputTag>("tracksters"))),
      trackToTracksterToken_(consumes<std::vector<int>>(ps.getParameter<edm::InputTag>("trackToTrackster"))),
      lcToken_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layerClusters"))),
      lcTimeToken_(
          consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("layerClustersTime"))),
      tilesToken_(consumes<TICLLayerTiles>(ps.getParameter<edm::InputTag>("layerClusterTiles"))),
      vertexToken_(consumes<std::vector<reco::Vertex>>(ps.getParameter<edm::InputTag>("vertices"))),
      primaryVertexToken_(consumes<std::vector<reco::Vertex>>(ps.getParameter<edm::InputTag>("primaryVertices"))),
      trackToken_(consumes<std::vector<reco::Track>>(ps.getParameter<edm::InputTag>("tracks"))),
      timingToken_(consumes<MtdHostCollection>(ps.getParameter<edm::InputTag>("timingSoA"))),
      ticlGeomToken_(esConsumes<TICLGeomHost, CaloGeometryRecord, edm::Transition::BeginRun>()),
      ticlGeomLookupToken_(esConsumes<TICLGeomLookupHost, CaloGeometryRecord, edm::Transition::BeginRun>()),
      ticlGeomLayersToken_(esConsumes<TICLGeomLayersHost, CaloGeometryRecord, edm::Transition::BeginRun>()),
      fieldToken_(esConsumes()),
      propToken_(esConsumes(edm::ESInputTag("", ps.getParameter<std::string>("propagator")))),
      hgconsToken_(esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag("", "HGCalEESensitive"))),
      minRawEnergy_(ps.getParameter<double>("minRawEnergy")),
      batchSize_(std::max(1u, ps.getParameter<unsigned int>("batchSize"))),
      pointCloud_(pointCloudConfig(ps)),
      trackSelection_{static_cast<float>(ps.getParameter<double>("minTrackPt")),
                      static_cast<float>(ps.getParameter<double>("minTrackAbsEta"))} {
  produces<std::vector<float>>("energy");
  produces<std::vector<float>>("sigma");
  produces<std::vector<float>>("puFraction");
  produces<std::vector<float>>("species");
}

std::unique_ptr<cms::Ort::ONNXRuntime> TracksterPointCloudInferenceProducer::initializeGlobalCache(
    const edm::ParameterSet& ps) {
  const auto model = ps.getParameter<std::string>("modelFile");
  auto session =
      std::make_unique<cms::Ort::ONNXRuntime>(model.starts_with("/") ? model : edm::FileInPath(model).fullPath());
  // The model must take the features of TracksterPointCloud: one inference on one row of one point.
  using PC = ticl::TracksterPointCloud;
  cms::Ort::FloatArrays inputs{std::vector<float>(PC::kNGlobalFeatures, 0.f),
                               std::vector<float>(PC::kNPointFeatures, 0.f),
                               std::vector<float>(1, 1.f)};
  cms::Ort::FloatArrays out;
  try {
    out = session->run({"glob", "points", "valid"},
                       inputs,
                       {{1, PC::kNGlobalFeatures}, {1, 1, PC::kNPointFeatures}, {1, 1}},
                       {"energy", "sigma", "pu", "species"},
                       1);
  } catch (std::exception const& e) {
    throw cms::Exception("Configuration")
        << "TracksterPointCloudInferenceProducer: the model " << model << " does not take glob [B, "
        << PC::kNGlobalFeatures << "], points [B, N, " << PC::kNPointFeatures << "] and valid [B, N]: " << e.what();
  }
  if (out.size() != 4 || out[0].size() != 1 || out[1].size() != 1 || out[2].size() != 1 || out[3].size() != kNSpecies)
    throw cms::Exception("Configuration")
        << "TracksterPointCloudInferenceProducer: the model " << model
        << " does not give energy [B], sigma [B], pu [B] and species [B, " << kNSpecies << "]";
  return session;
}

void TracksterPointCloudInferenceProducer::beginRun(edm::Run const&, edm::EventSetup const& es) {
  rhtools_.setGeometry(es.getData(ticlGeomToken_), es.getData(ticlGeomLookupToken_), es.getData(ticlGeomLayersToken_));
  frontDisks_ = ticl::utils::buildHGCalFirstDisks(es.getData(hgconsToken_));
}

void TracksterPointCloudInferenceProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  using Clock = std::chrono::steady_clock;
  using PC = ticl::TracksterPointCloud;
  const auto t0 = Clock::now();
  auto const& tracksters = evt.get(tracksterToken_);
  auto const& trackToTrackster = evt.get(trackToTracksterToken_);
  auto const& lcs = evt.get(lcToken_);
  auto const& lcTime = evt.get(lcTimeToken_);
  auto const& tiles = evt.get(tilesToken_);
  auto const& tracks = evt.get(trackToken_);
  const auto timingH = evt.getHandle(timingToken_);
  const MtdHostCollection* timing = timingH.isValid() ? timingH.product() : nullptr;
  auto const* field = &es.getData(fieldToken_);
  auto const& prop = es.getData(propToken_);
  const unsigned int nTs = tracksters.size();

  const PC::Event pcEvent(tracksters,
                          lcs,
                          &lcTime,
                          tiles,
                          rhtools_,
                          ticl::countValidVertices(evt.get(vertexToken_)),
                          ticl::leadingPrimaryVertex(evt.get(primaryVertexToken_)));
  // HGCAL impacts of the selected tracks that are assigned to a final trackster.
  std::vector<ticl::TrackImpact> impacts(tracks.size());
  for (size_t k = 0; k < tracks.size() && k < trackToTrackster.size(); ++k)
    if (trackToTrackster[k] >= 0 && trackSelection_(tracks[k]))
      impacts[k] = ticl::propagateToHGCalFront(tracks[k], field, prop, frontDisks_);
  const auto trackOf = ticl::assignedTrackOfHighestP(trackToTrackster, tracks, impacts, nTs);

  auto energy = std::make_unique<std::vector<float>>(nTs);
  auto sigma = std::make_unique<std::vector<float>>(nTs, 0.f);
  auto pu = std::make_unique<std::vector<float>>(nTs, -1.f);
  auto species = std::make_unique<std::vector<float>>(kNSpecies * nTs, 0.f);
  std::vector<unsigned int> rows;
  std::vector<PC::Result> results(nTs);
  for (unsigned int i = 0; i < nTs; ++i) {
    (*energy)[i] = tracksters[i].raw_energy();
    if (tracksters[i].raw_energy() < minRawEnergy_)
      continue;
    PC::Track track;
    const bool hasTrack = trackOf[i] >= 0;
    if (hasTrack)
      track = ticl::pointCloudTrack(impacts[trackOf[i]], tracks[trackOf[i]], timing, trackOf[i]);
    results[i] = pointCloud_.build(pcEvent, i, hasTrack ? &track : nullptr);
    rows.push_back(i);
  }
  std::stable_sort(rows.begin(), rows.end(), [&](unsigned int a, unsigned int b) {
    return results[a].nPoints() < results[b].nPoints();
  });
  const auto t1 = Clock::now();

  const std::vector<std::string> inputNames{"glob", "points", "valid"};
  const std::vector<std::string> outputNames{"energy", "sigma", "pu", "species"};
  constexpr unsigned int nG = PC::kNGlobalFeatures, nP = PC::kNPointFeatures;
  cms::Ort::FloatArrays inputs(3);
  for (size_t first = 0; first < rows.size(); first += batchSize_) {
    const size_t last = std::min(rows.size(), first + batchSize_);
    const unsigned int nB = last - first;
    // Rows are sorted by point count: the last row of the batch has the most points (at least 1 for the shapes).
    const unsigned int nMax = std::max(1u, results[rows[last - 1]].nPoints());
    inputs[0].assign(nB * nG, 0.f);
    inputs[1].assign(static_cast<size_t>(nB) * nMax * nP, 0.f);
    inputs[2].assign(static_cast<size_t>(nB) * nMax, 0.f);
    for (unsigned int b = 0; b < nB; ++b) {
      const auto& r = results[rows[first + b]];
      std::copy(r.global.begin(), r.global.end(), inputs[0].begin() + b * nG);
      std::copy(r.points.begin(), r.points.end(), inputs[1].begin() + static_cast<size_t>(b) * nMax * nP);
      std::fill_n(inputs[2].begin() + static_cast<size_t>(b) * nMax, r.nPoints(), 1.f);
    }
    const std::vector<std::vector<int64_t>> shapes{{nB, nG}, {nB, nMax, nP}, {nB, nMax}};
    const auto out = globalCache()->run(inputNames, inputs, shapes, outputNames, nB);
    for (unsigned int b = 0; b < nB; ++b) {
      const unsigned int i = rows[first + b];
      (*energy)[i] = out[0][b];
      (*sigma)[i] = out[1][b];
      (*pu)[i] = out[2][b];
      for (unsigned int s = 0; s < kNSpecies; ++s)
        (*species)[kNSpecies * i + s] = out[3][kNSpecies * b + s];
    }
  }
  const auto t2 = Clock::now();
  sumBuildMs_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
  sumInferMs_ += std::chrono::duration<double, std::milli>(t2 - t1).count();
  ++nEvents_;
  nTracksters_ += rows.size();

  evt.put(std::move(energy), "energy");
  evt.put(std::move(sigma), "sigma");
  evt.put(std::move(pu), "puFraction");
  evt.put(std::move(species), "species");
}

void TracksterPointCloudInferenceProducer::endStream() {
  if (nEvents_ > 0)
    edm::LogInfo("TracksterPointCloudInference")
        << "TracksterPointCloudInferenceProducer: " << nEvents_ << " events, " << nTracksters_ / nEvents_
        << " tracksters per event, features " << sumBuildMs_ / nEvents_ << " ms and inference "
        << sumInferMs_ / nEvents_ << " ms per event";
}

void TracksterPointCloudInferenceProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlTracksterInterpretations"));
  desc.add<edm::InputTag>("trackToTrackster", edm::InputTag("ticlTracksterInterpretations", "trackToTrackster"));
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("layerClustersTime", edm::InputTag("hgcalMergeLayerClusters", "timeLayerCluster"));
  desc.add<edm::InputTag>("layerClusterTiles", edm::InputTag("ticlLayerTileProducer"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"))
      ->setComment("The number of valid vertices is a feature.");
  desc.add<edm::InputTag>("primaryVertices", edm::InputTag("offlinePrimaryVertices4D"))
      ->setComment("The leading valid vertex gives the primary-vertex features.");
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("timingSoA", edm::InputTag("mtdSoA"));
  desc.add<std::string>("propagator", "PropagatorWithMaterial");
  desc.add<std::string>("modelFile", "")
      ->setComment(
          "ONNX model with the inputs glob, points, valid: an absolute path or a path of the CMSSW search path.");
  desc.add<double>("minRawEnergy", 0.)->setComment("Tracksters below [GeV] keep their raw energy.");
  desc.add<unsigned int>("batchSize", 512);
  desc.add<double>("radiusEE", 8.);
  desc.add<double>("radiusEH", 20.);
  desc.add<double>("ringOuterFactor", 1.5);
  desc.add<double>("maxAxisAngle", 0.3);
  desc.add<unsigned int>("maxOwnPoints", 96);
  desc.add<unsigned int>("maxNeighbourPoints", 48);
  desc.add<double>("minTrackPt", 0.5);
  desc.add<double>("minTrackAbsEta", 1.3);
  descriptions.add("tracksterPointCloudInferenceProducer", desc);
}

DEFINE_FWK_MODULE(TracksterPointCloudInferenceProducer);
