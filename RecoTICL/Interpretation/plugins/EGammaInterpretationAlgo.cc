#include <algorithm>
#include "RecoTICL/Interpretation/plugins/EGammaInterpretationAlgo.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace ticl;

EGammaInterpretationAlgo::EGammaInterpretationAlgo(const edm::ParameterSet &conf, edm::ConsumesCollector iC)
    : TICLInterpretationAlgoBase(conf, iC),
      delta_tk_sc_(conf.getParameter<float>("delta_tk_sc")),
      eop_min_(conf.getParameter<float>("eop_min")),
      eop_max_(conf.getParameter<float>("eop_max")),
      min_em_fraction_(conf.getParameter<float>("min_em_fraction")),
      max_matches_(conf.getParameter<unsigned int>("max_matches")),
      min_supercluster_energy_(conf.getParameter<float>("min_supercluster_energy")) {}

EGammaInterpretationAlgo::~EGammaInterpretationAlgo() {}

void EGammaInterpretationAlgo::initialize(const HGCalDDDConstants * /*hgcons*/,
                                          const ticlgeom::Tools /*rhtools*/,
                                          const edm::ESHandle<MagneticField> /*bfieldH*/,
                                          const edm::ESHandle<Propagator> /*propH*/) {}

void EGammaInterpretationAlgo::makeCandidates(const Inputs & /*input*/,
                                              edm::Handle<MtdHostCollection> /*inputTiming_h*/,
                                              std::vector<Trackster> & /*resultTracksters*/,
                                              std::vector<int> & /*resultCandidate*/,
                                              std::vector<bool> & /*maskedTracksters*/,
                                              std::vector<std::vector<unsigned int>> & /*linkedResultTracksters*/) {}

void EGammaInterpretationAlgo::makeOpinions(const Inputs &input,
                                            edm::Handle<MtdHostCollection> /*inputTiming_h*/,
                                            std::vector<Trackster> &hypothesisTracksters,
                                            std::vector<Hypothesis> &hypotheses) {
  const auto &tracks = *input.tracksHandle;
  const auto &maskTracks = input.maskedTracks;
  const auto &superclusters = input.tracksters;
  if (input.impacts == nullptr)
    throw cms::Exception("Configuration") << "EGammaInterpretationAlgo: the track impacts are required";

  // A supercluster barycenter moved to the plane z = zVal along its direction from the origin.
  auto projectToZ = [](const Vector &baryc, float zVal) {
    const Vector dirn = baryc.unit();
    const float par = (zVal - baryc.Z()) / dirn.Z();
    return Vector(par * dirn.X() + baryc.X(), par * dirn.Y() + baryc.Y(), zVal);
  };

  // One hypothesis trackster per supercluster, shared by the electron and the photon hypotheses.
  std::vector<int> hypoTracksterOf(superclusters.size(), -1);
  auto hypoTracksterFor = [&](unsigned scIdx) {
    if (hypoTracksterOf[scIdx] < 0) {
      hypoTracksterOf[scIdx] = static_cast<int>(hypothesisTracksters.size());
      hypothesisTracksters.push_back(superclusters[scIdx]);
    }
    return hypoTracksterOf[scIdx];
  };

  for (size_t iTrack = 0; iTrack < tracks.size(); ++iTrack) {
    if (!maskTracks[iTrack])
      continue;
    const auto &impact = (*input.impacts)[iTrack];
    if (!impact.valid)
      continue;
    const auto &tk = tracks[iTrack];
    const auto dir = tk.outerOk() ? tk.outerMomentum() : tk.momentum();
    const float trackP = tk.p();
    const Vector trackAtFace(impact.position.x(), impact.position.y(), impact.position.z());

    std::vector<std::pair<float, unsigned>> inWindow;
    for (unsigned iSc = 0; iSc < superclusters.size(); ++iSc) {
      const auto &sc = superclusters[iSc];
      if (sc.raw_energy() < min_supercluster_energy_)
        continue;
      const auto &bary = sc.barycenter();
      if (bary.eta() * dir.eta() < 0.f)  // same endcap
        continue;
      const Vector scAtFace = projectToZ(bary, trackAtFace.Z());
      const float dR = reco::deltaR(scAtFace.Eta(), scAtFace.Phi(), trackAtFace.Eta(), trackAtFace.Phi());
      if (dR < delta_tk_sc_)
        inWindow.emplace_back(dR, iSc);
    }
    std::sort(inWindow.begin(), inWindow.end());
    if (inWindow.size() > max_matches_)
      inWindow.resize(max_matches_);
    for (auto const &[bestDR, best] : inWindow) {
      const auto &sc = superclusters[best];
      const float eop = trackP > 0.f ? sc.raw_energy() / trackP : 0.f;
      const float emFraction = sc.raw_energy() > 0.f ? sc.raw_em_energy() / sc.raw_energy() : 0.f;
      if (eop < eop_min_ || eop > eop_max_ || emFraction < min_em_fraction_)
        continue;
      Hypothesis h;
      h.type = Hypothesis::Type::Electron;
      // Product of the window distance, of the E/p distance from 1 relative to the window, and of the EM fraction.
      const float geom = 1.f - bestDR / delta_tk_sc_;
      const float eopSpan = std::max(eop_max_ - 1.f, 1.f - eop_min_);
      const float eopTerm = eopSpan > 0.f ? std::max(0.f, 1.f - std::abs(eop - 1.f) / eopSpan) : 1.f;
      h.score = static_cast<float>(geom * eopTerm * std::min(emFraction, 1.f));
      h.trackIdx = static_cast<int>(iTrack);
      h.tracksterIdx = hypoTracksterFor(best);
      hypotheses.push_back(h);
    }
  }

  // A photon hypothesis for every EM-like supercluster, the superclusters matched to a track included.
  for (unsigned iSc = 0; iSc < superclusters.size(); ++iSc) {
    const auto &sc = superclusters[iSc];
    if (sc.raw_energy() < min_supercluster_energy_)
      continue;
    const float emFraction = sc.raw_energy() > 0.f ? sc.raw_em_energy() / sc.raw_energy() : 0.f;
    if (emFraction < min_em_fraction_)
      continue;
    Hypothesis h;
    h.type = Hypothesis::Type::Photon;
    h.score = static_cast<float>(std::min(emFraction, 1.f));
    h.tracksterIdx = hypoTracksterFor(iSc);
    hypotheses.push_back(h);
  }
}

void EGammaInterpretationAlgo::fillPSetDescription(edm::ParameterSetDescription &desc) {
  desc.add<float>("delta_tk_sc", 0.05f)
      ->setComment("(eta,phi) window at the HGCAL front between the track impact and the supercluster.");
  desc.add<unsigned int>("max_matches", 3)
      ->setComment("Electron hypotheses per track: the nearest superclusters in the window, at most this number.");
  desc.add<float>("eop_min", 0.2f)->setComment("Min supercluster E / track p for an electron hypothesis.");
  desc.add<float>("eop_max", 10.f)->setComment("Max supercluster E / track p for an electron hypothesis.");
  desc.add<float>("min_em_fraction", 0.5f)
      ->setComment("Min EM energy fraction of a supercluster for an electron or a photon hypothesis.");
  desc.add<float>("min_supercluster_energy", 1.0f)->setComment("Min supercluster raw energy [GeV].");
  TICLInterpretationAlgoBase::fillPSetDescription(desc);
}
