// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

#include "CondFormats/HGCalObjects/interface/TICLGeomHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLayersHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLookupHost.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterLinksDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterPidDeviceCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"
#include "RecoTICL/Common/interface/alpaka/TracksterCollections.h"

#include "CornettoAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Cornetto linking on the device: the groups and the emission of ticl::TracksterLinkingbyCornetto (without its second
  // pass) for the trackster collections of the ordered list, and for each linked trackster the global indices of its
  // input tracksters. The linked tracksters have the energy, barycenter, time and principal components of all their
  // layer clusters, and no PID (unknown = 1). A collection with a PID SoA takes its PID probabilities from it; the
  // Recovery tracksters have none.
  class TracksterLinksSoAProducer : public stream::SynchronizingEDProducer<> {
  public:
    TracksterLinksSoAProducer(const edm::ParameterSet& config)
        : SynchronizingEDProducer(config),
          layerClustersToken_{consumes(config.getParameter<edm::InputTag>("layer_clusters"))},
          trackstersRefsToken_{consumes(config.getParameter<edm::InputTag>("trackstersRefs"))},
          trackstersToken_{produces()},
          linksToken_{produces("links")},
          ticlGeomToken_{consumesCollector().esConsumes<TICLGeomHost, CaloGeometryRecord>()},
          ticlGeomLookupToken_{consumesCollector().esConsumes<TICLGeomLookupHost, CaloGeometryRecord>()},
          ticlGeomLayersToken_{consumesCollector().esConsumes<TICLGeomLayersHost, CaloGeometryRecord>()},
          parameters_{readParameters(config)} {
      const auto& tracksters = config.getParameter<std::vector<edm::InputTag>>("tracksters");
      const auto& pid = config.getParameter<std::vector<edm::InputTag>>("pid");
      if (tracksters.empty() || tracksters.size() > ticl::cornetto::kMaxCollections ||
          pid.size() != tracksters.size()) {
        throw cms::Exception("Configuration")
            << "TracksterLinksSoAProducer: 'tracksters' needs 1 to " << ticl::cornetto::kMaxCollections
            << " collections, and 'pid' one entry per collection; got " << tracksters.size() << " and " << pid.size()
            << ".";
      }
      for (std::size_t i = 0; i < tracksters.size(); ++i) {
        trackstersTokens_.emplace_back(consumes(tracksters[i]));
        pidTokens_.emplace_back();
        if (!pid[i].label().empty()) {
          const device::EDGetToken<ticl::TracksterPidDeviceCollection> token = consumes(pid[i]);
          pidTokens_.back() = token;
        }
      }
      if (config.getParameter<bool>("mergeFragments") || config.getParameter<bool>("mergeDropped")) {
        throw cms::Exception("Configuration")
            << "TracksterLinksSoAProducer: the second pass (mergeFragments, mergeDropped) is not implemented.";
      }
    }

    // The number of linked tracksters and of their layer clusters and input tracksters are on the device, and the
    // host needs them to allocate the output
    void acquire(const device::Event& iEvent, const device::EventSetup&) override {
      inputs_ = readInputs(iEvent);
      scratch_ = ticl::cornetto::Scratch{};
      algo_.group(iEvent.queue(), parameters_, inputs_, scratch_);
    }

    void produce(device::Event& iEvent, const device::EventSetup& iSetup) override {
      // As the host linking: the layer clusters up to the last CE-E layer count to the EM energy
      ticlgeom::Tools geometry;
      geometry.setGeometry(
          iSetup.getData(ticlGeomToken_), iSetup.getData(ticlGeomLookupToken_), iSetup.getData(ticlGeomLayersToken_));
      const float zLimitEm = geometry.getPositionLayer(geometry.lastLayerEE()).z();

      const uint32_t nLinked = scratch_.nLinked();
      ticl::TracksterDeviceCollection tracksters(
          iEvent.queue(), nLinked, nLinked, nLinked, nLinked, nLinked, nLinked, scratch_.nVertices());
      ticl::TracksterLinksDeviceCollection links(iEvent.queue(), nLinked, scratch_.nMembers());
      algo_.fill(iEvent.queue(),
                 inputs_,
                 scratch_,
                 zLimitEm,
                 iEvent.get(layerClustersToken_).const_view(),
                 tracksters.view(),
                 links.view());
      iEvent.emplace(trackstersToken_, std::move(tracksters));
      iEvent.emplace(linksToken_, std::move(links));
      scratch_ = ticl::cornetto::Scratch{};
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::vector<edm::InputTag>>("tracksters", {edm::InputTag("ticlSoATracksters")})
          ->setComment("The trackster collections to link, at most 4.");
      desc.add<edm::InputTag>("trackstersRefs", edm::InputTag("ticlTrackstersRefs"))
          ->setComment("The ordered list of the collections of 'tracksters', which gives the global indices.");
      desc.add<std::vector<edm::InputTag>>("pid", {edm::InputTag("ticlTrackstersCLUEsteringPidSoA")})
          ->setComment("One entry per collection: the PID SoA of its tracksters, or empty for the PID of the SoA.");
      desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalSoAMergeLayerClusters"))
          ->setComment("The layer clusters that the vertices of the tracksters index.");
      // the parameters of ticl::TracksterLinkingbyCornetto, with the same meaning
      desc.add<double>("etaWindow", 0.3)->setComment("Max |deta| and |dphi| between the barycenters, in (0, pi).");
      desc.add<double>("maxLongitudinalDistance", 60.)
          ->setComment("Max separation [cm] downstream of the anchor along its axis. Positive.");
      desc.add<double>("maxBackwardDistance", 20.)
          ->setComment("Max separation [cm] upstream of the anchor along its axis. Not negative.");
      desc.add<double>("transverseRadius0", 4.)->setComment("Max transverse distance at zero separation [cm].");
      desc.add<double>("transverseSlope", 0.05)
          ->setComment("Growth of the max transverse distance per cm of separation.");
      desc.add<double>("emTransverseRadius0", 1.5)
          ->setComment("Max transverse distance at zero separation [cm] for an EM anchor, at most transverseRadius0.");
      desc.add<double>("emTransverseSlope", 0.05)
          ->setComment("Growth of the max transverse distance per cm for an EM anchor, at most transverseSlope.");
      desc.add<double>("timeCompatibilityNSigma", 3.)
          ->setComment("Max time difference at the anchor depth in combined sigmas. Values <= 0 disable the gate.");
      desc.add<double>("timeResolutionFloor", 0.05)
          ->setComment("Time resolution [ns] added in quadrature to the combined time error.");
      desc.add<double>("typeVetoProbability", 0.6)
          ->setComment("EM/hadronic veto threshold in [0.5, 1]; 0 disables the veto.");
      desc.add<std::vector<int>>("emCategories",
                                 {static_cast<int>(::ticl::Trackster::ParticleType::photon),
                                  static_cast<int>(::ticl::Trackster::ParticleType::electron),
                                  static_cast<int>(::ticl::Trackster::ParticleType::neutral_pion)})
          ->setComment("Trackster::ParticleType slots whose PID probabilities sum to the EM probability.");
      desc.add<double>("minEmittedEnergy", 1.)->setComment("Min raw energy [GeV] of an emitted group.");
      desc.add<double>("minEmittedPt", 0.5)->setComment("Min raw pt [GeV] of an emitted group.");
      desc.add<double>("minEmittedEnergyEM", 1.)->setComment("Min raw energy [GeV] of an emitted EM group.");
      desc.add<double>("minEmittedPtEM", 0.5)->setComment("Min raw pt [GeV] of an emitted EM group.");
      desc.add<double>("minEmittedEnergyAnyPt", 0.)
          ->setComment("A group at or above this raw energy [GeV] is emitted at any pt. Values <= 0 disable it.");
      desc.add<bool>("mergeFragments", false)->setComment("Second pass of the host plugin. Must be false.");
      desc.add<bool>("mergeDropped", false)->setComment("Second pass of the host plugin. Must be false.");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    static ticl::cornetto::Parameters readParameters(const edm::ParameterSet& config) {
      auto get = [&](const char* name) { return static_cast<float>(config.getParameter<double>(name)); };
      ticl::cornetto::Parameters p;
      p.etaWindow = get("etaWindow");
      p.cone = {
          get("maxBackwardDistance"), get("maxLongitudinalDistance"), get("transverseRadius0"), get("transverseSlope")};
      p.emCone = {p.cone.maxBackward, p.cone.maxLongitudinal, get("emTransverseRadius0"), get("emTransverseSlope")};
      const float nSigma = get("timeCompatibilityNSigma");
      p.useTimeGate = nSigma > 0.f;
      p.timeNSigma2 = nSigma * nSigma;
      const float timeFloor = get("timeResolutionFloor");
      p.timeFloor2 = timeFloor * timeFloor;
      p.typeVetoProbability = get("typeVetoProbability");
      p.minEmittedEnergy = get("minEmittedEnergy");
      p.minEmittedPt = get("minEmittedPt");
      p.minEmittedEnergyEM = get("minEmittedEnergyEM");
      p.minEmittedPtEM = get("minEmittedPtEM");
      p.minEmittedEnergyAnyPt = get("minEmittedEnergyAnyPt");
      constexpr int kNumParticleTypes = static_cast<int>(::ticl::Trackster::ParticleType::unknown) + 1;
      const auto categories = config.getParameter<std::vector<int>>("emCategories");
      auto sortedCategories = categories;
      std::sort(sortedCategories.begin(), sortedCategories.end());
      const bool categoriesValid =
          !sortedCategories.empty() && sortedCategories.front() >= 0 && sortedCategories.back() < kNumParticleTypes &&
          std::adjacent_find(sortedCategories.begin(), sortedCategories.end()) == sortedCategories.end();
      p.nEmCategories = 0;
      for (int c : categories) {
        if (c >= 0 && c < kNumParticleTypes && p.nEmCategories < ticl::cornetto::kMaxEmCategories) {
          p.emCategories[p.nEmCategories++] = c;
        }
      }
      const bool vetoValid =
          p.typeVetoProbability == 0.f || (p.typeVetoProbability >= 0.5f && p.typeVetoProbability <= 1.f);
      const auto coneValid = [](const ticl::cornetto::Cone& c, const ticl::cornetto::Cone& em) {
        return c.maxLongitudinal > 0.f && c.maxBackward >= 0.f && c.radius0 >= 0.f && c.slope >= 0.f &&
               em.radius0 >= 0.f && em.radius0 <= c.radius0 && em.slope >= 0.f && em.slope <= c.slope;
      };
      if (!(p.etaWindow > 0.f && p.etaWindow < std::numbers::pi_v<float>) || !coneValid(p.cone, p.emCone) ||
          !(timeFloor >= 0.f) || !vetoValid || !categoriesValid) {
        throw cms::Exception("Configuration")
            << "TracksterLinksSoAProducer: invalid parameters, see the parameter descriptions.";
      }
      return p;
    }

    ticl::cornetto::Inputs readInputs(const device::Event& iEvent) const {
      const auto tracksters = ticl::orderedTracksterCollections(
          iEvent, iEvent.get(trackstersRefsToken_), trackstersTokens_, "TracksterLinksSoAProducer");
      // orderedTracksterCollections matches the tokens to the ordered list; the PID SoA of a token goes with it
      if (tracksters.size() > ticl::cornetto::kMaxCollections) {
        throw edm::Exception(edm::errors::LogicError)
            << "TracksterLinksSoAProducer: " << tracksters.size() << " trackster collections, at most "
            << ticl::cornetto::kMaxCollections << ".";
      }
      ticl::cornetto::Inputs inputs;
      inputs.nCollections = tracksters.size();
      inputs.offsets[0] = 0;
      for (std::size_t i = 0; i < tracksters.size(); ++i) {
        inputs.tracksters[i] = tracksters[i]->const_view();
        const uint32_t n = tracksters[i]->const_view().energy().metadata().size();
        inputs.offsets[i + 1] = inputs.offsets[i] + n;
        // the PID SoA of the token whose collection is tracksters[i]
        for (std::size_t k = 0; k < trackstersTokens_.size(); ++k) {
          if (iEvent.getHandle(trackstersTokens_[k]).product() == tracksters[i] && pidTokens_[k]) {
            const auto& pid = iEvent.get(*pidTokens_[k]);
            if (static_cast<uint32_t>(pid.const_view().metadata().size()) != n) {
              throw edm::Exception(edm::errors::LogicError)
                  << "TracksterLinksSoAProducer: the PID SoA of collection " << i << " has "
                  << pid.const_view().metadata().size() << " entries for " << n << " tracksters.";
            }
            inputs.pid[i] = pid.const_view();
            inputs.hasPid[i] = true;
          }
        }
      }
      return inputs;
    }

    const device::EDGetToken<reco::CaloClusterDeviceCollection> layerClustersToken_;
    std::vector<device::EDGetToken<ticl::TracksterDeviceCollection>> trackstersTokens_;
    std::vector<std::optional<device::EDGetToken<ticl::TracksterPidDeviceCollection>>> pidTokens_;
    const edm::EDGetTokenT<ticl::TracksterCollectionRefs> trackstersRefsToken_;
    const device::EDPutToken<ticl::TracksterDeviceCollection> trackstersToken_;
    const device::EDPutToken<ticl::TracksterLinksDeviceCollection> linksToken_;
    const edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
    const edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
    const edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;
    const ticl::cornetto::Parameters parameters_;
    const ticl::cornetto::Algo algo_{};
    // Set in acquire and used in produce
    ticl::cornetto::Inputs inputs_;
    ticl::cornetto::Scratch scratch_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterLinksSoAProducer);
