// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

// Builds the mixed (signal + pileup) raw TruthGraph as a DigiAccumulatorMixMod. The
// framework gives one sub-event at a time with its own SimTrack, SimVertex and HepMC
// collections, so trackId, vertIndex and parentIndex keep their local meaning.
//
// GEN handling, per realm:
//   collapsePileupGen, collapseSignalGen: if true, the GEN record of that realm is
//        collapsed to the stable (status 1) GEN particles and the species in
//        collapsedGenKeptPdgIds, with one gen vertex per interaction and one decay
//        vertex per kept decaying species. If false, the full HepMC decay chain is
//        kept, with the intermediate particles and GenStatusFlags. A preset seeded on a
//        resonance pdgId needs the full chain.
//   collapseGenShower, collapseGenShowerSignal: for a realm with the full chain,
//        contract the parton shower and the intermediate resonance copies, keeping the
//        ancestry (see truth::collapseGenShower). The main event keeps its shower by
//        default, because the BeamSideInput vertex points at those partons.
//   pileupBunchCrossings: the bunch crossings of the pileup to include.
//
// Each node carries an EncodedEventId: (0,0) for the signal, (bunchCrossing,
// pileupIndex) for pileup.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/Units.h"

#include "PhysicsTools/TruthInfo/interface/GenGraphBuild.h"
#include "SimDataFormats/TruthInfo/interface/TruthGraph.h"

namespace {
  uint64_t packEventId(EncodedEventId const& id) {
    // EncodedEventId is a single uint32 rawId; use the typed accessor rather than a
    // byte copy so the key stays portable and cannot pick up a future member/padding.
    static_assert(sizeof(EncodedEventId) == sizeof(uint32_t));
    return static_cast<uint64_t>(id.rawId());
  }

  // The compact GEN record of a signal Event or a PileUpEventPrincipal (both expose
  // getByLabel), preferring HepMC3, and the position of its interaction. See
  // truth::compactGen.
  template <class EvT>
  std::vector<truth::CompactGenParticle> readCompactGen(EvT const& ev,
                                                        edm::InputTag const& hepmc3Tag,
                                                        edm::InputTag const& hepmc2Tag,
                                                        std::vector<int32_t> const& keptPdgIds,
                                                        std::optional<math::XYZTLorentzVectorD>& interactionPosition) {
    edm::Handle<edm::HepMC3Product> h3;
    if (ev.getByLabel(hepmc3Tag, h3) && h3.isValid() && h3->GetEvent() != nullptr) {
      HepMC3::GenEvent ev3;
      ev3.read_data(*h3->GetEvent());
      ev3.set_units(HepMC3::Units::GEV, HepMC3::Units::MM);
      const auto gb = truth::buildFromHepMC3(ev3);
      interactionPosition = gb.interactionPosition;
      return truth::compactGen(gb, keptPdgIds);
    }
    edm::Handle<edm::HepMCProduct> h2;
    if (ev.getByLabel(hepmc2Tag, h2) && h2.isValid() && h2->GetEvent() != nullptr) {
      const auto gb = truth::buildFromHepMC2(*h2->GetEvent(), false);
      interactionPosition = gb.interactionPosition;
      return truth::compactGen(gb, keptPdgIds);
    }
    return {};
  }

  // The full HepMC record for one sub-event, in the same flattened form
  // TruthGraphProducer builds from an unmixed event, optionally with the parton
  // shower and the intermediate resonance copies contracted away.
  template <class EvT>
  truth::GenBuild readFullGen(EvT const& ev,
                              edm::InputTag const& hepmc3Tag,
                              edm::InputTag const& hepmc2Tag,
                              bool collapseShower,
                              edm::SimTrackContainer const& tracks,
                              bool& degradedCollapseWarned) {
    truth::GenBuild gb;
    edm::Handle<edm::HepMC3Product> h3;
    if (ev.getByLabel(hepmc3Tag, h3) && h3.isValid() && h3->GetEvent() != nullptr) {
      HepMC3::GenEvent ev3;
      ev3.read_data(*h3->GetEvent());
      ev3.set_units(HepMC3::Units::GEV, HepMC3::Units::MM);
      gb = truth::buildFromHepMC3(ev3);
    } else {
      edm::Handle<edm::HepMCProduct> h2;
      if (ev.getByLabel(hepmc2Tag, h2) && h2.isValid() && h2->GetEvent() != nullptr)
        gb = truth::buildFromHepMC2(*h2->GetEvent());
    }
    if (collapseShower && !gb.empty()) {
      // The degraded path is a property of the sample, so it is reported once per stream.
      if (!truth::collapseGenShower(gb, truth::simContinuedGenBarcodes(tracks)) && !degradedCollapseWarned) {
        degradedCollapseWarned = true;
        edm::LogWarning("TruthGraphAccumulator")
            << "collapseGenShower ran on a GEN record with no packed status flags, which "
               "buildFromHepMC3 does not fill. The isHardProcess and isLastCopy keep rules "
               "are then dead and every intermediate resonance is dropped, so a selection "
               "preset seeded on a resonance pdgId will match nothing. Set "
               "collapseGenShower=False on a HepMC3 sample.";
      }
    }
    return gb;
  }
}  // namespace

class TruthGraphAccumulator : public DigiAccumulatorMixMod {
public:
  TruthGraphAccumulator(edm::ParameterSet const&, edm::ProducesCollector, edm::ConsumesCollector&);

  void initializeEvent(edm::Event const&, edm::EventSetup const&) override;
  void accumulate(edm::Event const&, edm::EventSetup const&) override;
  void accumulate(PileUpEventPrincipal const&, edm::EventSetup const&, edm::StreamID const&) override;
  void finalizeEvent(edm::Event&, edm::EventSetup const&) override;

private:
  // Append one sub-event. SimTrack/SimVertex ids are local to this sub-event.
  // The GEN half is `fullGen` when that is non-null and non-empty, otherwise the
  // collapsed `compactGen` when that is non-empty, otherwise absent. Either GEN form
  // is linked to the primary SimTracks by GenToSim edges. `genEvent` identifies the
  // sub-event the GEN nodes belong to.
  void addSubEvent(std::vector<truth::CompactGenParticle> const& compactGen,
                   std::optional<math::XYZTLorentzVectorD> const& compactInteractionPosition,
                   truth::GenBuild const* fullGen,
                   edm::SimTrackContainer const& tracks,
                   edm::SimVertexContainer const& vertices,
                   EncodedEventId const& eid,
                   int32_t genEvent);

  // Append the sim-hits of this sub-event to the merged collections, re-tagged with `eid`.
  template <class EvT>
  void addSubEventHits(EvT const& ev, EncodedEventId const& eid);

  // Merge one sim-hit collection family from the sub-event, re-tagging the eventId of
  // each hit. One output per family, so a consumer can relabel DetIds per collection.
  template <class HitT, class EvT>
  void mergeHits(EvT const& ev,
                 std::vector<edm::InputTag> const& tags,
                 EncodedEventId const& eid,
                 std::vector<HitT>& out);

  const edm::InputTag simTrackTag_;
  const edm::InputTag simVertexTag_;
  const edm::InputTag hepmc3Tag_;
  const edm::InputTag hepmc2Tag_;
  const std::vector<edm::InputTag> caloHitTags_;
  const std::vector<edm::InputTag> ecalHitTags_;
  const std::vector<edm::InputTag> hcalHitTags_;
  const std::vector<edm::InputTag> trackerHitTags_;
  const std::vector<edm::InputTag> muonHitTags_;
  const std::vector<edm::InputTag> mtdHitTags_;
  const std::vector<int> pileupBunchCrossings_;
  // The bunch spacing of the mixing in ns, which it adds to the time of every SimVertex of
  // an out-of-time interaction.
  const int bunchSpace_;
  const bool collapsePileupGen_;
  const bool collapseSignalGen_;
  const bool collapseGenShower_;
  const bool collapseGenShowerSignal_;
  const std::vector<int32_t> collapsedGenKeptPdgIds_;

  // One counter per bunch crossing, keyed by bx, as the MixingModule numbers its
  // sub-events. Reset per event.
  std::map<int, int> pileupCount_;
  // Warn once per missing collection, so one missing collection does not hide another.
  std::set<std::string> missingHitsWarned_;
  bool degradedCollapseWarned_ = false;

  // Merged calorimeter sim-hits of the signal and the kept pileup, re-tagged with the
  // sub-event EncodedEventId, so the (eventId, trackId) key resolves pileup nodes at RECO.
  std::vector<PCaloHit> mergedCaloHits_;
  std::vector<PCaloHit> mergedEcalHits_;
  std::vector<PCaloHit> mergedHcalHits_;
  // Tracking sim-hits (tracker, muon chambers, MTD), with the same re-tagging.
  std::vector<PSimHit> mergedTrackerHits_;
  std::vector<PSimHit> mergedMuonHits_;
  std::vector<PSimHit> mergedMtdHits_;

  // Rejected GenToSim links in the event: a SimTrack whose genpartIndex resolves to a
  // GEN particle of a different pdgId gets no link.
  unsigned int rejectedGenToSimLinks_ = 0;

  std::vector<TruthGraph::NodeRef> nodes_;
  std::vector<int32_t> pdgId_;
  std::vector<int16_t> status_;
  std::vector<uint16_t> statusFlags_;
  std::vector<int32_t> genEventOfNode_;
  std::vector<uint64_t> eventId_;
  std::vector<int32_t> simTrackToVtx_;
  std::vector<int32_t> simTrackToGen_;
  std::vector<std::pair<uint32_t, uint32_t>> edges_;
  std::vector<uint8_t> edgeKinds_;
  std::vector<uint16_t> simVertexProcessType_;  // node-parallel; G4 process subtype (SimVertex only)
  std::vector<uint8_t> simTrackBackscattered_;  // node-parallel; albedo flag (SimTrack only)
  // GEN payload from the record of each sub-event: the GEN node ids in ascending order
  // and, in step, the four-momentum of a GenParticle or the (cm, ns) position of a
  // GenVertex. After mixing, this is the only copy of the pileup GEN record. The time
  // has no bunch-crossing offset.
  std::vector<uint32_t> genPayloadNodes_;
  std::vector<math::XYZTLorentzVectorD> genPayload_;
  // The SimTracks and SimVertices of every sub-event, tagged with the sub-event id, in
  // the order the sub-events are added.
  edm::SimTrackContainer mergedSimTracks_;
  edm::SimVertexContainer mergedSimVertices_;

  [[nodiscard]] bool keepBx(int bx) const {
    return std::find(pileupBunchCrossings_.begin(), pileupBunchCrossings_.end(), bx) != pileupBunchCrossings_.end();
  }
};

TruthGraphAccumulator::TruthGraphAccumulator(edm::ParameterSet const& cfg,
                                             edm::ProducesCollector producesCollector,
                                             edm::ConsumesCollector& iC)
    : simTrackTag_(cfg.getParameter<edm::InputTag>("simTracks")),
      simVertexTag_(cfg.getParameter<edm::InputTag>("simVertices")),
      hepmc3Tag_(cfg.getParameter<edm::InputTag>("genEventHepMC3")),
      hepmc2Tag_(cfg.getParameter<edm::InputTag>("genEventHepMC")),
      caloHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("caloHits")),
      ecalHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("ecalHits")),
      hcalHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("hcalHits")),
      trackerHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("trackerHits")),
      muonHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("muonHits")),
      mtdHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("mtdHits")),
      pileupBunchCrossings_(cfg.getParameter<std::vector<int>>("pileupBunchCrossings")),
      bunchSpace_(cfg.getParameter<int>("bunchSpace")),
      collapsePileupGen_(cfg.getParameter<bool>("collapsePileupGen")),
      collapseSignalGen_(cfg.getParameter<bool>("collapseSignalGen")),
      collapseGenShower_(cfg.getParameter<bool>("collapseGenShower")),
      collapseGenShowerSignal_(cfg.getParameter<bool>("collapseGenShowerSignal")),
      collapsedGenKeptPdgIds_(cfg.getParameter<std::vector<int32_t>>("collapsedGenKeptPdgIds")) {
  producesCollector.produces<TruthGraph>();
  producesCollector.produces<std::vector<PCaloHit>>("mergedHGCHits");
  producesCollector.produces<std::vector<PCaloHit>>("mergedEcalHits");
  producesCollector.produces<std::vector<PCaloHit>>("mergedHcalHits");
  producesCollector.produces<std::vector<PSimHit>>("mergedTrackerHits");
  producesCollector.produces<std::vector<PSimHit>>("mergedMuonHits");
  producesCollector.produces<edm::SimTrackContainer>("mergedSimTracks");
  producesCollector.produces<edm::SimVertexContainer>("mergedSimVertices");
  producesCollector.produces<std::vector<PSimHit>>("mergedMtdHits");
  producesCollector.produces<std::vector<uint32_t>>("genPayloadNodes");
  producesCollector.produces<std::vector<math::XYZTLorentzVectorD>>("genPayload");
  iC.consumes<edm::SimTrackContainer>(simTrackTag_);
  iC.consumes<edm::SimVertexContainer>(simVertexTag_);
  iC.mayConsume<edm::HepMC3Product>(hepmc3Tag_);
  iC.mayConsume<edm::HepMCProduct>(hepmc2Tag_);
  for (auto const* tags : {&caloHitTags_, &ecalHitTags_, &hcalHitTags_})
    for (auto const& tag : *tags)
      iC.mayConsume<std::vector<PCaloHit>>(tag);
  for (auto const* tags : {&trackerHitTags_, &muonHitTags_, &mtdHitTags_})
    for (auto const& tag : *tags)
      iC.mayConsume<std::vector<PSimHit>>(tag);
}

void TruthGraphAccumulator::initializeEvent(edm::Event const&, edm::EventSetup const&) {
  pileupCount_.clear();
  rejectedGenToSimLinks_ = 0;
  mergedCaloHits_.clear();
  mergedEcalHits_.clear();
  mergedHcalHits_.clear();
  mergedTrackerHits_.clear();
  mergedMuonHits_.clear();
  mergedMtdHits_.clear();
  nodes_.clear();
  pdgId_.clear();
  status_.clear();
  statusFlags_.clear();
  genEventOfNode_.clear();
  eventId_.clear();
  simTrackToVtx_.clear();
  simTrackToGen_.clear();
  edges_.clear();
  edgeKinds_.clear();
  simVertexProcessType_.clear();
  simTrackBackscattered_.clear();
  genPayloadNodes_.clear();
  genPayload_.clear();
  mergedSimTracks_.clear();
  mergedSimVertices_.clear();
}

void TruthGraphAccumulator::addSubEvent(std::vector<truth::CompactGenParticle> const& compactGen,
                                        std::optional<math::XYZTLorentzVectorD> const& compactInteractionPosition,
                                        truth::GenBuild const* fullGen,
                                        edm::SimTrackContainer const& tracks,
                                        edm::SimVertexContainer const& vertices,
                                        EncodedEventId const& eid,
                                        int32_t genEvent) {
  mergedSimTracks_.reserve(mergedSimTracks_.size() + tracks.size());
  for (SimTrack t : tracks) {
    t.setEventId(eid);
    mergedSimTracks_.push_back(std::move(t));
  }
  mergedSimVertices_.reserve(mergedSimVertices_.size() + vertices.size());
  // The mixing adds the crossing offset in ns to a SimVertex time that is in seconds.
  // Remove it and add it back in seconds, so every merged vertex time is in seconds.
  constexpr double kNsToS = 1e-9;
  const double offsetNs = static_cast<double>(eid.bunchCrossing()) * bunchSpace_;
  for (SimVertex v : vertices) {
    v.setEventId(eid);
    if (offsetNs != 0.)
      v.setTof(v.position().t() - offsetNs + offsetNs * kNsToS);
    mergedSimVertices_.push_back(std::move(v));
  }

  const uint64_t packed = packEventId(eid);
  auto pushNode = [&](TruthGraph::NodeKind kind, int64_t key, int32_t pdg, int16_t st) {
    const uint32_t node = static_cast<uint32_t>(nodes_.size());
    nodes_.push_back(TruthGraph::NodeRef{kind, key});
    pdgId_.push_back(pdg);
    status_.push_back(st);
    statusFlags_.push_back(0);
    genEventOfNode_.push_back(-1);
    eventId_.push_back(packed);
    simTrackToVtx_.push_back(-1);
    simTrackToGen_.push_back(-1);
    simVertexProcessType_.push_back(0);
    simTrackBackscattered_.push_back(0);
    return node;
  };
  // Called right after the node is pushed, so the node ids stay in ascending order.
  auto setGenPayload = [&](uint32_t node, math::XYZTLorentzVectorD const& value) {
    genPayloadNodes_.push_back(node);
    genPayload_.push_back(value);
  };
  auto pushEdge = [&](uint32_t src, uint32_t dst, TruthGraph::EdgeKind k) {
    edges_.emplace_back(src, dst);
    edgeKinds_.push_back(static_cast<uint8_t>(k));
  };

  // GEN realm, one of two forms. Either way genBarcodeToNode maps a HepMC barcode to
  // its GenParticle node, which is what GenToSim linking below needs.
  std::unordered_map<int, uint32_t> genBarcodeToNode;
  const bool useFullGen = (fullGen != nullptr && !fullGen->empty());

  if (useFullGen) {
    // Full HepMC decay chain: every particle at its own status and both Gen edge
    // directions, with a GenEvent node as the source.
    const uint32_t genEventNode = pushNode(TruthGraph::NodeKind::GenEvent, static_cast<int64_t>(genEvent), 0, 0);
    genEventOfNode_[genEventNode] = genEvent;

    std::unordered_map<int, uint32_t> genVtxBarcodeToNode;
    genVtxBarcodeToNode.reserve(fullGen->vtxBarcodes.size() * 2);
    for (int vbc : fullGen->vtxBarcodes) {
      const uint32_t vn = pushNode(TruthGraph::NodeKind::GenVertex, static_cast<int64_t>(vbc), 0, 0);
      genEventOfNode_[vn] = genEvent;
      if (const auto it = fullGen->vertexPositionByBarcode.find(vbc); it != fullGen->vertexPositionByBarcode.end())
        setGenPayload(vn, it->second);
      genVtxBarcodeToNode.emplace(vbc, vn);
    }

    genBarcodeToNode.reserve(fullGen->partBarcodes.size() * 2);
    for (int pbc : fullGen->partBarcodes) {
      const auto itPdg = fullGen->particlePdgIdByBarcode.find(pbc);
      const auto itStatus = fullGen->particleStatusByBarcode.find(pbc);
      const int32_t pdg = (itPdg != fullGen->particlePdgIdByBarcode.end()) ? itPdg->second : 0;
      const int16_t st = (itStatus != fullGen->particleStatusByBarcode.end()) ? itStatus->second : 0;
      const uint32_t pn = pushNode(TruthGraph::NodeKind::GenParticle, static_cast<int64_t>(pbc), pdg, st);
      if (const auto it = fullGen->particleMomentumByBarcode.find(pbc); it != fullGen->particleMomentumByBarcode.end())
        setGenPayload(pn, it->second);
      const auto itFlags = fullGen->particleStatusFlagsByBarcode.find(pbc);
      if (itFlags != fullGen->particleStatusFlagsByBarcode.end())
        statusFlags_[pn] = itFlags->second;
      genEventOfNode_[pn] = genEvent;
      genBarcodeToNode.emplace(pbc, pn);
    }

    std::unordered_map<int, unsigned int> vtxIncoming;
    for (auto const& [pbc, vbc] : fullGen->partToVtx)
      ++vtxIncoming[vbc];

    for (auto const& [vbc, pbc] : fullGen->vtxToPart) {
      auto itV = genVtxBarcodeToNode.find(vbc);
      auto itP = genBarcodeToNode.find(pbc);
      if (itV != genVtxBarcodeToNode.end() && itP != genBarcodeToNode.end())
        pushEdge(itV->second, itP->second, TruthGraph::EdgeKind::Gen);
    }
    for (auto const& [pbc, vbc] : fullGen->partToVtx) {
      auto itP = genBarcodeToNode.find(pbc);
      auto itV = genVtxBarcodeToNode.find(vbc);
      if (itP != genBarcodeToNode.end() && itV != genVtxBarcodeToNode.end())
        pushEdge(itP->second, itV->second, TruthGraph::EdgeKind::Gen);
    }

    // Attach the GenEvent node per connected component, as TruthGraphProducer does: to
    // each source vertex, or to every vertex of a component with no source. In a collider
    // record the beam particles feed the first vertex, so no vertex is a source.
    std::unordered_map<int, int> componentOfVtx;
    {
      // Two vertices are in the same component when a particle touches both.
      std::unordered_map<int, std::vector<int>> partAdjacency;
      std::unordered_map<int, std::vector<int>> vtxNeighbours;
      for (auto const& [vbc, pbc] : fullGen->vtxToPart)
        partAdjacency[pbc].push_back(vbc);
      for (auto const& [pbc, vbc] : fullGen->partToVtx)
        partAdjacency[pbc].push_back(vbc);
      for (auto const& [pbc, vtxs] : partAdjacency) {
        for (std::size_t i = 1; i < vtxs.size(); ++i) {
          vtxNeighbours[vtxs[0]].push_back(vtxs[i]);
          vtxNeighbours[vtxs[i]].push_back(vtxs[0]);
        }
      }

      int nextComponent = 0;
      std::vector<int> stack;
      for (int vbc : fullGen->vtxBarcodes) {
        if (componentOfVtx.count(vbc) != 0)
          continue;
        const int component = nextComponent++;
        stack.push_back(vbc);
        componentOfVtx.emplace(vbc, component);
        while (!stack.empty()) {
          const int current = stack.back();
          stack.pop_back();
          const auto it = vtxNeighbours.find(current);
          if (it == vtxNeighbours.end())
            continue;
          for (const int next : it->second) {
            if (componentOfVtx.emplace(next, component).second)
              stack.push_back(next);
          }
        }
      }
    }

    // Known limit, shared with TruthGraphProducer: a component with a source and a
    // beam-fed branch attaches only the source, and the branch is unreachable.
    std::unordered_map<int, unsigned int> rootsInComponent;
    for (int vbc : fullGen->vtxBarcodes) {
      if (vtxIncoming[vbc] == 0)
        ++rootsInComponent[componentOfVtx.at(vbc)];
    }
    for (int vbc : fullGen->vtxBarcodes) {
      const bool isSource = vtxIncoming[vbc] == 0;
      const bool componentHasNoSource = rootsInComponent[componentOfVtx.at(vbc)] == 0;
      if (isSource || componentHasNoSource)
        pushEdge(genEventNode, genVtxBarcodeToNode.at(vbc), TruthGraph::EdgeKind::Gen);
    }
  } else if (!compactGen.empty()) {
    // Collapsed GEN: one gen vertex for the interaction, and one decay vertex for each
    // kept particle that decays into another kept particle.
    const uint32_t genVtxNode = pushNode(TruthGraph::NodeKind::GenVertex, 0, 0, 0);
    genEventOfNode_[genVtxNode] = genEvent;
    if (compactInteractionPosition)
      setGenPayload(genVtxNode, *compactInteractionPosition);
    genBarcodeToNode.reserve(compactGen.size() * 2);
    std::unordered_map<int, int> decayVertexOf;
    std::unordered_map<int, math::XYZTLorentzVectorD> decayPositionOf;
    for (auto const& particle : compactGen) {
      const uint32_t pn =
          pushNode(TruthGraph::NodeKind::GenParticle, particle.barcode, particle.pdgId, particle.status);
      genEventOfNode_[pn] = genEvent;
      setGenPayload(pn, particle.momentum);
      genBarcodeToNode.emplace(particle.barcode, pn);
      if (particle.decayVertex != 0) {
        decayVertexOf.emplace(particle.barcode, particle.decayVertex);
        decayPositionOf.emplace(particle.barcode, particle.decayPosition);
      }
    }
    std::unordered_map<int, uint32_t> decayVertexNode;
    for (auto const& particle : compactGen) {
      uint32_t source = genVtxNode;
      const auto itDecay = decayVertexOf.find(particle.parent);
      if (particle.parent != 0 && itDecay != decayVertexOf.end()) {
        auto [itNode, inserted] = decayVertexNode.try_emplace(particle.parent, 0);
        if (inserted) {
          itNode->second = pushNode(TruthGraph::NodeKind::GenVertex, itDecay->second, 0, 0);
          genEventOfNode_[itNode->second] = genEvent;
          setGenPayload(itNode->second, decayPositionOf.at(particle.parent));
          pushEdge(genBarcodeToNode.at(particle.parent), itNode->second, TruthGraph::EdgeKind::Gen);
        }
        source = itNode->second;
      }
      pushEdge(source, genBarcodeToNode.at(particle.barcode), TruthGraph::EdgeKind::Gen);
    }
  }

  // SIM realm (native local ids).
  std::unordered_map<uint32_t, uint32_t> vertexIdToNode;
  vertexIdToNode.reserve(vertices.size() * 2);
  const uint32_t baseVtx = static_cast<uint32_t>(nodes_.size());
  for (auto const& v : vertices) {
    const uint32_t node = pushNode(TruthGraph::NodeKind::SimVertex, static_cast<int64_t>(v.vertexId()), 0, 0);
    simVertexProcessType_[node] = static_cast<uint16_t>(v.processType());
    vertexIdToNode.emplace(static_cast<uint32_t>(v.vertexId()), node);
  }
  const uint32_t baseTrk = static_cast<uint32_t>(nodes_.size());
  std::unordered_map<uint32_t, uint32_t> trackIdToNode;
  trackIdToNode.reserve(tracks.size() * 2);
  for (auto const& t : tracks) {
    const uint32_t node = pushNode(TruthGraph::NodeKind::SimTrack, static_cast<int64_t>(t.trackId()), t.type(), 0);
    simTrackBackscattered_[node] = t.isFromBackScattering() ? 1 : 0;
    trackIdToNode.emplace(t.trackId(), node);
  }

  // Production edge: track.vertIndex() is the local vector index into `vertices`.
  for (std::size_t i = 0; i < tracks.size(); ++i) {
    const int vi = tracks[i].vertIndex();
    if (vi < 0 || static_cast<std::size_t>(vi) >= vertices.size())
      continue;
    const uint32_t trkNode = baseTrk + static_cast<uint32_t>(i);
    const uint32_t prodVtxNode = baseVtx + static_cast<uint32_t>(vi);
    pushEdge(prodVtxNode, trkNode, TruthGraph::EdgeKind::Sim);
    simTrackToVtx_[trkNode] = static_cast<int32_t>(prodVtxNode);
  }

  // Decay edge: vertex.parentIndex() is the trackId of the parent track.
  for (auto const& v : vertices) {
    if (v.parentIndex() < 0)
      continue;
    auto pIt = trackIdToNode.find(static_cast<uint32_t>(v.parentIndex()));
    auto vIt = vertexIdToNode.find(static_cast<uint32_t>(v.vertexId()));
    if (pIt != trackIdToNode.end() && vIt != vertexIdToNode.end())
      pushEdge(pIt->second, vIt->second, TruthGraph::EdgeKind::Sim);
  }

  // GenToSim: a primary SimTrack's genpartIndex is its GEN particle's barcode. The
  // two must agree on pdgId, otherwise the barcode does not identify this track's
  // generator particle and no link is written.
  if (!genBarcodeToNode.empty()) {
    for (auto const& t : tracks) {
      auto gIt = genBarcodeToNode.find(t.genpartIndex());
      if (gIt == genBarcodeToNode.end())
        continue;
      auto sIt = trackIdToNode.find(t.trackId());
      if (sIt == trackIdToNode.end())
        continue;
      if (pdgId_[gIt->second] != t.type()) {
        ++rejectedGenToSimLinks_;
        continue;
      }
      pushEdge(gIt->second, sIt->second, TruthGraph::EdgeKind::GenToSim);
      simTrackToGen_[sIt->second] = static_cast<int32_t>(gIt->second);
    }
  }
}

template <class HitT, class EvT>
void TruthGraphAccumulator::mergeHits(EvT const& ev,
                                      std::vector<edm::InputTag> const& tags,
                                      EncodedEventId const& eid,
                                      std::vector<HitT>& out) {
  for (auto const& tag : tags) {
    edm::Handle<std::vector<HitT>> hits;
    ev.getByLabel(tag, hits);
    if (!hits.isValid()) {
      // Two causes: the collection is not in the running geometry (the strip tracker in
      // Run4), or the pileup is premixed.
      if (missingHitsWarned_.insert(tag.encode()).second) {
        edm::LogWarning("TruthGraphAccumulator")
            << "sim-hit collection " << tag.encode() << " not found in a sub-event, so it contributes no truth hits."
            << " Either this collection does not exist in the running geometry, or the pileup is premixed and its"
            << " sim-hits were digitized away; pileup-aware truth needs classic (non-premixed) pileup.";
      }
      continue;
    }
    out.reserve(out.size() + hits->size());
    for (HitT hit : *hits) {  // copy, to re-tag the eventId
      hit.setEventId(eid);
      out.push_back(hit);
    }
  }
}

template <class EvT>
void TruthGraphAccumulator::addSubEventHits(EvT const& ev, EncodedEventId const& eid) {
  mergeHits(ev, caloHitTags_, eid, mergedCaloHits_);
  mergeHits(ev, ecalHitTags_, eid, mergedEcalHits_);
  mergeHits(ev, hcalHitTags_, eid, mergedHcalHits_);
  mergeHits(ev, trackerHitTags_, eid, mergedTrackerHits_);
  mergeHits(ev, muonHitTags_, eid, mergedMuonHits_);
  mergeHits(ev, mtdHitTags_, eid, mergedMtdHits_);
}

void TruthGraphAccumulator::accumulate(edm::Event const& event, edm::EventSetup const&) {
  edm::Handle<edm::SimTrackContainer> tracks;
  edm::Handle<edm::SimVertexContainer> vertices;
  event.getByLabel(simTrackTag_, tracks);
  event.getByLabel(simVertexTag_, vertices);
  if (!tracks.isValid() || !vertices.isValid())
    return;
  std::vector<truth::CompactGenParticle> compactGen;
  std::optional<math::XYZTLorentzVectorD> interactionPosition;
  truth::GenBuild fullGen;
  if (collapseSignalGen_)
    compactGen = readCompactGen(event, hepmc3Tag_, hepmc2Tag_, collapsedGenKeptPdgIds_, interactionPosition);
  else
    fullGen = readFullGen(event, hepmc3Tag_, hepmc2Tag_, collapseGenShowerSignal_, *tracks, degradedCollapseWarned_);
  const EncodedEventId sigEid(0, 0);
  addSubEvent(compactGen, interactionPosition, &fullGen, *tracks, *vertices, sigEid, 0);
  addSubEventHits(event, sigEid);
}

void TruthGraphAccumulator::accumulate(PileUpEventPrincipal const& pep, edm::EventSetup const&, edm::StreamID const&) {
  const int bx = pep.bunchCrossing();
  if (!keepBx(bx))
    return;

  // One counter per bunch crossing, starting at 1, as the MixingModule numbers its
  // sub-events. The tracker digi links carry these numbers, so the two must agree. The
  // counter advances also for a sub-event that this accumulator cannot read.
  const int puIndex = ++pileupCount_[bx];
  // EncodedEventId packs the event number into 16 bits.
  if (puIndex > 0xFFFF)
    throw cms::Exception("TruthGraphAccumulator")
        << "pileup sub-event count " << puIndex << " exceeds the 16-bit EncodedEventId event field";

  edm::Handle<edm::SimTrackContainer> tracks;
  edm::Handle<edm::SimVertexContainer> vertices;
  pep.getByLabel(simTrackTag_, tracks);
  pep.getByLabel(simVertexTag_, vertices);
  if (!tracks.isValid() || !vertices.isValid())
    return;

  std::vector<truth::CompactGenParticle> compactGen;
  std::optional<math::XYZTLorentzVectorD> interactionPosition;
  truth::GenBuild fullGen;
  if (collapsePileupGen_)
    compactGen = readCompactGen(pep, hepmc3Tag_, hepmc2Tag_, collapsedGenKeptPdgIds_, interactionPosition);
  else
    fullGen = readFullGen(pep, hepmc3Tag_, hepmc2Tag_, collapseGenShower_, *tracks, degradedCollapseWarned_);

  const EncodedEventId puEid(bx, puIndex);
  addSubEvent(compactGen, interactionPosition, &fullGen, *tracks, *vertices, puEid, puIndex);
  addSubEventHits(pep, puEid);
}

void TruthGraphAccumulator::finalizeEvent(edm::Event& event, edm::EventSetup const&) {
  auto out = std::make_unique<TruthGraph>();
  const uint32_t nNodes = static_cast<uint32_t>(nodes_.size());

  out->nodes() = std::move(nodes_);
  out->pdgId() = std::move(pdgId_);
  out->status() = std::move(status_);
  out->eventId() = std::move(eventId_);
  out->simTrackToVtx() = std::move(simTrackToVtx_);
  out->simTrackToGen() = std::move(simTrackToGen_);
  out->simVertexProcessType() = std::move(simVertexProcessType_);
  out->simTrackBackscattered() = std::move(simTrackBackscattered_);
  out->statusFlags() = std::move(statusFlags_);
  out->genEventOfNode() = std::move(genEventOfNode_);
  out->simVtxToGen().assign(nNodes, -1);

  if (rejectedGenToSimLinks_ != 0) {
    edm::LogWarning("TruthGraphAccumulator")
        << rejectedGenToSimLinks_ << " GenToSim links dropped in this event because the SimTrack pdgId disagreed with"
        << " the GEN particle its genpartIndex points at.";
  }

  // CSR out-edges by a counting-sort scatter.
  out->offsets().assign(nNodes + 1, 0);
  for (auto const& e : edges_)
    ++out->offsets()[e.first + 1];
  for (uint32_t i = 1; i <= nNodes; ++i)
    out->offsets()[i] += out->offsets()[i - 1];

  out->edges().resize(edges_.size());
  out->edgeKind().resize(edges_.size());
  std::vector<uint32_t> cursor = out->offsets();
  for (std::size_t e = 0; e < edges_.size(); ++e) {
    const uint32_t pos = cursor[edges_[e].first]++;
    out->edges()[pos] = edges_[e].second;
    out->edgeKind()[pos] = edgeKinds_[e];
  }

  if (!out->isConsistent())
    throw cms::Exception("TruthGraphAccumulator") << "Produced TruthGraph is not consistent";

  event.put(std::move(out));
  event.put(std::make_unique<std::vector<uint32_t>>(std::move(genPayloadNodes_)), "genPayloadNodes");
  event.put(std::make_unique<std::vector<math::XYZTLorentzVectorD>>(std::move(genPayload_)), "genPayload");

  event.put(std::make_unique<std::vector<PCaloHit>>(std::move(mergedCaloHits_)), "mergedHGCHits");
  event.put(std::make_unique<std::vector<PCaloHit>>(std::move(mergedEcalHits_)), "mergedEcalHits");
  event.put(std::make_unique<std::vector<PCaloHit>>(std::move(mergedHcalHits_)), "mergedHcalHits");
  event.put(std::make_unique<std::vector<PSimHit>>(std::move(mergedTrackerHits_)), "mergedTrackerHits");
  event.put(std::make_unique<std::vector<PSimHit>>(std::move(mergedMuonHits_)), "mergedMuonHits");
  event.put(std::make_unique<std::vector<PSimHit>>(std::move(mergedMtdHits_)), "mergedMtdHits");
  event.put(std::make_unique<edm::SimTrackContainer>(std::move(mergedSimTracks_)), "mergedSimTracks");
  event.put(std::make_unique<edm::SimVertexContainer>(std::move(mergedSimVertices_)), "mergedSimVertices");
}

DEFINE_DIGI_ACCUMULATOR(TruthGraphAccumulator);
