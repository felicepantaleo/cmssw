#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <tuple>
#include <vector>

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "TracksterLinkingbySuperClusteringDNNHeterogeneous.h"

using namespace ticl;

TracksterLinkingbySuperClusteringDNNHeterogeneous::TracksterLinkingbySuperClusteringDNNHeterogeneous(
    const edm::ParameterSet& ps, edm::ConsumesCollector iC, cms::Ort::ONNXRuntime const* onnxRuntime)
    : TracksterLinkingAlgoBase(ps, iC, onnxRuntime),
      pairsToken_(iC.consumes<SuperclusteringDNNPairsHost>(ps.getParameter<edm::InputTag>("pairs"))),
      scoresToken_(iC.consumes<SuperclusteringDNNPairScoresHost>(ps.getParameter<edm::InputTag>("scores"))),
      nnWorkingPoint_(ps.getParameter<float>("nnWorkingPoint")),
      seedPtThreshold_(ps.getParameter<float>("seedPtThreshold")),
      filterByTracksterPID_(ps.getParameter<bool>("filterByTracksterPID")),
      tracksterPIDCategoriesToFilter_(ps.getParameter<std::vector<int>>("tracksterPIDCategoriesToFilter")),
      emissionPIDThreshold_(ps.getParameter<float>("emissionPIDThreshold")) {
  if (!filterByTracksterPID_ && emissionPIDThreshold_ != 0.f)
    throw cms::Exception("Configuration")
        << "TracksterLinkingbySuperClusteringDNNHeterogeneous: emissionPIDThreshold applies only with "
           "filterByTracksterPID";
}

bool TracksterLinkingbySuperClusteringDNNHeterogeneous::passesPID(const Trackster& ts, float threshold) const {
  if (!filterByTracksterPID_)
    return true;
  float probTotal = 0.0f;
  for (int cat : tracksterPIDCategoriesToFilter_) {
    probTotal += ts.id_probabilities(cat);
  }
  return probTotal >= threshold;
}

/**
 * The same outputs as TracksterLinkingbySuperClusteringDNN::linkTracksters, from the pairs and scores of the device:
 * resultTracksters[i] has seed input.tracksters[linkedTracksterIdToInputTracksterId[i][0]], merged with the tracksters
 * linkedTracksterIdToInputTracksterId[i][1..N]; outputSuperclusters is the same list of indices.
*/
void TracksterLinkingbySuperClusteringDNNHeterogeneous::linkTracksters(
    const Inputs& input,
    std::vector<Trackster>& resultTracksters,
    std::vector<std::vector<unsigned int>>& outputSuperclusters,
    std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) {
  auto const& inputTracksters = input.tracksters;
  const auto tracksterCount = static_cast<unsigned int>(inputTracksters.size());
  if (tracksterCount == 0) {
    return;
  }

  const auto pairs = input.ev.get(pairsToken_).const_view();
  const auto scores = input.ev.get(scoresToken_).const_view();
  const auto pairCount = static_cast<unsigned int>(pairs.metadata().size());
  if (static_cast<unsigned int>(scores.metadata().size()) != pairCount) {
    throw cms::Exception("LogicError") << "TracksterLinkingbySuperClusteringDNNHeterogeneous: " << pairCount
                                       << " pairs but " << scores.metadata().size() << " scores";
  }

  // The legacy order: by decreasing raw pT, ties in input order.
  std::vector<unsigned int> trackstersIndicesPt(tracksterCount);
  std::iota(trackstersIndicesPt.begin(), trackstersIndicesPt.end(), 0u);
  std::stable_sort(
      trackstersIndicesPt.begin(), trackstersIndicesPt.end(), [&inputTracksters](unsigned int a, unsigned int b) {
        return inputTracksters[a].raw_pt() > inputTracksters[b].raw_pt();
      });
  std::vector<unsigned int> ptRank(tracksterCount);
  for (unsigned int rank = 0; rank < tracksterCount; ++rank) {
    ptRank[trackstersIndicesPt[rank]] = rank;
  }

  // The device writes the pairs by candidate index. The association needs them by candidate pT rank, as the legacy
  // loop makes them; within a candidate, by seed pT rank (the legacy code goes through its tile instead, which only
  // matters for two seeds with exactly the same score).
  std::vector<unsigned int> pairOrder(pairCount);
  std::iota(pairOrder.begin(), pairOrder.end(), 0u);
  for (unsigned int p = 0; p < pairCount; ++p) {
    if (pairs[p].seedIndex() >= tracksterCount || pairs[p].candidateIndex() >= tracksterCount) {
      throw cms::Exception("LogicError")
          << "TracksterLinkingbySuperClusteringDNNHeterogeneous: a pair indexes trackster "
          << std::max(pairs[p].seedIndex(), pairs[p].candidateIndex()) << " of " << tracksterCount
          << ". The linking must read the tracksters the pairs were built from, and those alone.";
    }
  }
  std::sort(pairOrder.begin(), pairOrder.end(), [&](unsigned int a, unsigned int b) {
    return std::make_tuple(ptRank[pairs[a].candidateIndex()], ptRank[pairs[a].seedIndex()]) <
           std::make_tuple(ptRank[pairs[b].candidateIndex()], ptRank[pairs[b].seedIndex()]);
  });

  std::vector<bool> tracksterMask(tracksterCount, false);
  std::vector<bool> usedAsCandidate(tracksterCount, false);
  std::vector<int> seedToOutputIndex(tracksterCount, -1);

  constexpr auto kInvalid = std::numeric_limits<unsigned int>::max();
  unsigned int previousCand = kInvalid;
  unsigned int bestSeed = kInvalid;
  float bestScore = nnWorkingPoint_;

  // From here on, as in TracksterLinkingbySuperClusteringDNN::linkTracksters.
  auto onCandidateTransition = [&](unsigned int candIdx) {
    if (bestSeed == kInvalid) {
      return;
    }

    tracksterMask[candIdx] = true;
    usedAsCandidate[candIdx] = true;

    int& outIdxRef = seedToOutputIndex[bestSeed];
    if (outIdxRef < 0) {
      outputSuperclusters.emplace_back(std::initializer_list<unsigned int>{bestSeed});
      resultTracksters.emplace_back(inputTracksters[bestSeed]);
      linkedTracksterIdToInputTracksterId.emplace_back(std::initializer_list<unsigned int>{bestSeed});
      outIdxRef = static_cast<int>(outputSuperclusters.size()) - 1;
      tracksterMask[bestSeed] = true;
    }

    const auto outIdx = static_cast<unsigned int>(outIdxRef);
    outputSuperclusters[outIdx].push_back(candIdx);
    resultTracksters[outIdx].mergeTracksters(inputTracksters[candIdx]);
    linkedTracksterIdToInputTracksterId[outIdx].push_back(candIdx);

    bestSeed = kInvalid;
    bestScore = nnWorkingPoint_;
  };

  for (unsigned int p : pairOrder) {
    const unsigned int seedIdx = pairs[p].seedIndex();
    const unsigned int candIdx = pairs[p].candidateIndex();

    if (previousCand != kInvalid && candIdx != previousCand) {
      onCandidateTransition(previousCand);
    }

    const float score = scores[p].score();
    if (score > bestScore && !usedAsCandidate[seedIdx]) {
      bestSeed = seedIdx;
      bestScore = score;
    }

    previousCand = candIdx;
  }
  if (previousCand != kInvalid) {
    onCandidateTransition(previousCand);
  }

  // A single trackster is a supercluster when it passes the emission PID cut.
  for (unsigned int ts_id = 0; ts_id < tracksterCount; ++ts_id) {
    if (!tracksterMask[ts_id] && inputTracksters[ts_id].raw_pt() >= seedPtThreshold_ &&
        passesPID(inputTracksters[ts_id], emissionPIDThreshold_)) {
      outputSuperclusters.emplace_back(std::initializer_list<unsigned int>{ts_id});
      resultTracksters.emplace_back(inputTracksters[ts_id]);
      linkedTracksterIdToInputTracksterId.emplace_back(std::initializer_list<unsigned int>{ts_id});
    }
  }

#ifdef EDM_ML_DEBUG
  for (std::vector<unsigned int> const& sc : outputSuperclusters) {
    std::ostringstream s;
    for (unsigned int trackster_id : sc)
      s << trackster_id << " ";
    LogDebug("HGCalTICLSuperclustering") << "Created supercluster of size " << sc.size()
                                         << " holding tracksters (first one is seed) " << s.str();
  }
#endif
}

void TracksterLinkingbySuperClusteringDNNHeterogeneous::fillPSetDescription(edm::ParameterSetDescription& desc) {
  TracksterLinkingAlgoBase::fillPSetDescription(desc);  // adds algo_verbosity
  desc.add<edm::InputTag>("pairs", edm::InputTag("ticlSuperclusteringDNNPairFeatures"))
      ->setComment("Seed and candidate trackster indices of each pair (SuperclusteringDNNPairFeaturesProducer)");
  desc.add<edm::InputTag>("scores", edm::InputTag("ticlSuperclusteringDNNInference"))
      ->setComment("DNN score of each pair (SuperclusteringDNNInferenceProducer)");
  desc.add<float>("nnWorkingPoint")
      ->setComment("Working point of DNN (in [0, 1]). DNN score above WP will attempt to supercluster.");
  // The pair preselection is applied on the device (SuperclusteringDNNPairFeaturesProducer); these are the
  // parameters of the single-trackster superclusters.
  desc.add<float>("seedPtThreshold", 4.)
      ->setComment("Minimum transverse energy of trackster to be considered as seed of a supercluster");
  desc.add<bool>("filterByTracksterPID", true)->setComment("Filter tracksters before superclustering by PID score");
  desc.add<std::vector<int>>(
          "tracksterPIDCategoriesToFilter",
          {static_cast<int>(Trackster::ParticleType::photon), static_cast<int>(Trackster::ParticleType::electron)})
      ->setComment("List of PID particle types (ticl::Trackster::ParticleType enum) to consider for PID filtering");
  desc.add<float>("emissionPIDThreshold", 0.f)
      ->setComment(
          "Minimum summed PID score for a trackster to be emitted as a standalone single-trackster supercluster. "
          "0 applies no PID on that path.");
}
