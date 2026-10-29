#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNNPairBuilder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {

    constexpr uint32_t kThreadsPerBlock = 128;

    // TracksterLinkingbySuperClusteringDNN::checkExplainedVarianceRatioCut
    ALPAKA_FN_ACC inline bool passesExplainedVarianceRatioCut(::ticl::superclustering::TracksterKinematics const& k,
                                                              SuperclusteringDNNPairSelection const& selection) {
      if (!k.hasEigenvalues) {
        return false;
      }
      return k.explVarRatio >= (k.rawEnergy > selection.explVarRatioCut_energyBoundary
                                    ? selection.explVarRatioMinimum_highEnergy
                                    : selection.explVarRatioMinimum_lowEnergy);
    }

    // seed (the higher pT trackster) - candidate pair of linkTracksters
    ALPAKA_FN_ACC inline bool isPair(SuperclusteringDNNPairingTrackster const* tracksters,
                                     uint32_t seedIndex,
                                     uint32_t candidateIndex,
                                     SuperclusteringDNNPairSelection const& selection) {
      auto const& seed = tracksters[seedIndex];
      auto const& candidate = tracksters[candidateIndex];
      if (!seed.seed) {
        return false;
      }
      // The seed comes first in the legacy std::stable_sort by decreasing raw pT.
      const float seedPt = seed.kinematics.rawPt;
      const float candidatePt = candidate.kinematics.rawPt;
      if (!(seedPt > candidatePt || (!(candidatePt > seedPt) && seedIndex < candidateIndex))) {
        return false;
      }
      // same endcap: the legacy tiles hold one endcap each
      if ((seed.kinematics.eta > 0.f) != (candidate.kinematics.eta > 0.f)) {
        return false;
      }
      if (std::abs(seed.kinematics.eta - candidate.kinematics.eta) >= selection.deltaEtaWindow) {
        return false;
      }
      return std::abs(::ticl::superclustering::deltaPhi(seed.kinematics.phi, candidate.kinematics.phi)) <
             selection.deltaPhiWindow;
    }

    // One thread per trackster: kinematics and preselection, read from the device trackster columns.
    struct KinematicsKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::ticl::TracksterSoA::ConstView tracksters,
                                    SuperclusteringDNNPairSelection selection,
                                    SuperclusteringDNNPairingTrackster* out,
                                    uint32_t nTracksters) const {
        for (auto t : cms::alpakatools::uniform_elements(acc, nTracksters)) {
          const auto energy = tracksters.energy()[t];
          const auto barycenter = tracksters.barycenter()[t];
          const auto pca = tracksters.pca()[t];

          auto& result = out[t];
          result.kinematics =
              ::ticl::superclustering::makeKinematics(Eigen::Vector3f(barycenter.x(), barycenter.y(), barycenter.z()),
                                                      energy.rawEnergy(),
                                                      tracksters.timing()[t].time(),
                                                      pca.eigenvalues(),
                                                      pca.eigenvectors().col(0));
          auto const& k = result.kinematics;
          const bool explainedVariance = passesExplainedVarianceRatioCut(k, selection);
          result.candidate = !(k.rawEnergy < selection.candidateEnergyThreshold) && explainedVariance;

          bool pid = true;
          if (selection.filterByTracksterPID) {
            const auto probabilities = tracksters.id()[t].idProbabilities();
            float em = 0.f;
            for (unsigned int i = 0; i < selection.nPIDCategories; ++i) {
              em += probabilities(selection.pidCategories[i]);
            }
            pid = em >= selection.PIDThreshold;
          }
          result.seed = !(k.rawPt < selection.seedPtThreshold) && explainedVariance && pid;
        }
      }
    };

    // One thread: the indices of the tracksters that can seed, in increasing order. The pairs are looked for among
    // them only: tens of seeds against thousands of tracksters at high pileup, so each candidate checks tens of
    // tracksters instead of all of them, which matters most on the serial backend.
    struct SeedListKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    SuperclusteringDNNPairingTrackster const* tracksters,
                                    uint32_t* seeds,
                                    uint32_t* nSeeds,
                                    uint32_t nTracksters) const {
        if (cms::alpakatools::once_per_grid(acc)) {
          uint32_t n = 0;
          for (uint32_t t = 0; t < nTracksters; ++t) {
            if (tracksters[t].seed) {
              seeds[n++] = t;
            }
          }
          *nSeeds = n;
        }
      }
    };

    // One thread per candidate: its number of pairs.
    struct CountPairsKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    SuperclusteringDNNPairingTrackster const* tracksters,
                                    SuperclusteringDNNPairSelection selection,
                                    uint32_t const* seeds,
                                    uint32_t const* nSeeds,
                                    uint32_t* pairCounts,
                                    uint32_t nTracksters) const {
        const uint32_t seedCount = *nSeeds;
        for (auto candidate : cms::alpakatools::uniform_elements(acc, nTracksters)) {
          uint32_t count = 0;
          if (tracksters[candidate].candidate) {
            for (uint32_t s = 0; s < seedCount; ++s) {
              count += isPair(tracksters, seeds[s], candidate, selection) ? 1 : 0;
            }
          }
          pairCounts[candidate] = count;
        }
      }
    };

    // One thread: the counts become the offsets of each candidate's pairs (an exclusive prefix sum), and the total.
    // There are at most a few thousand tracksters: a serial scan costs microseconds and keeps the order fixed.
    struct PairOffsetsKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    uint32_t* countsToOffsets,
                                    uint32_t* totalPairs,
                                    uint32_t nTracksters) const {
        if (cms::alpakatools::once_per_grid(acc)) {
          uint32_t offset = 0;
          for (uint32_t i = 0; i < nTracksters; ++i) {
            const uint32_t count = countsToOffsets[i];
            countsToOffsets[i] = offset;
            offset += count;
          }
          *totalPairs = offset;
        }
      }
    };

    // One thread per candidate: its pairs and their features, from its offset on.
    struct FillPairsKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    SuperclusteringDNNPairingTrackster const* tracksters,
                                    SuperclusteringDNNPairSelection selection,
                                    uint32_t const* seeds,
                                    uint32_t const* nSeeds,
                                    uint32_t const* pairOffsets,
                                    ::ticl::SuperclusteringDNNPairsSoA::View pairs,
                                    ::ticl::SuperclusteringDNNPairFeaturesSoA::View features,
                                    uint32_t nTracksters) const {
        const uint32_t seedCount = *nSeeds;
        for (auto candidate : cms::alpakatools::uniform_elements(acc, nTracksters)) {
          if (!tracksters[candidate].candidate) {
            continue;
          }
          uint32_t row = pairOffsets[candidate];
          for (uint32_t s = 0; s < seedCount; ++s) {
            const uint32_t seed = seeds[s];
            if (!isPair(tracksters, seed, candidate, selection)) {
              continue;
            }
            pairs[row].seedIndex() = seed;
            pairs[row].candidateIndex() = candidate;

            // the columns past featureCount (v1, v2) stay at zero: the model does not read them
            float f[::ticl::superclustering::kMaxFeatures] = {};
            ::ticl::superclustering::computeFeatures(
                tracksters[seed].kinematics, tracksters[candidate].kinematics, selection.featureCount, f);
            auto out = features[row];
            out.deltaEtaBaryc() = f[0];
            out.deltaPhiBaryc() = f[1];
            out.multiEn() = f[2];
            out.multiEta() = f[3];
            out.multiPt() = f[4];
            out.seedEta() = f[5];
            out.seedPhi() = f[6];
            out.seedEn() = f[7];
            out.seedPt() = f[8];
            out.theta() = f[9];
            out.thetaXZSeedFrame() = f[10];
            out.thetaYZSeedFrame() = f[11];
            out.thetaXYCmsFrame() = f[12];
            out.thetaYZCmsFrame() = f[13];
            out.thetaXZCmsFrame() = f[14];
            out.explVar() = f[15];
            out.explVarRatio() = f[16];
            out.modDeltaTime() = f[17];
            ++row;
          }
        }
      }
    };

  }  // namespace

  SuperclusteringDNNPairBuilder::SuperclusteringDNNPairBuilder(SuperclusteringDNNPairSelection const& selection)
      : selection_{selection} {}

  void SuperclusteringDNNPairBuilder::prepare(Queue& queue, ::ticl::TracksterSoA::ConstView tracksters) {
    nTracksters_ = static_cast<uint32_t>(tracksters.energy().metadata().size());
    pairCountHost_ = cms::alpakatools::make_host_buffer<uint32_t>(queue);
    if (nTracksters_ == 0) {
      // nothing is enqueued, so the host buffer can be written directly
      *(*pairCountHost_) = 0;
      tracksters_.reset();
      seeds_.reset();
      nSeedsDevice_.reset();
      pairOffsets_.reset();
      pairCountDevice_.reset();
      return;
    }

    tracksters_ = cms::alpakatools::make_device_buffer<SuperclusteringDNNPairingTrackster[]>(queue, nTracksters_);
    seeds_ = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nTracksters_);
    nSeedsDevice_ = cms::alpakatools::make_device_buffer<uint32_t>(queue);
    pairOffsets_ = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nTracksters_);
    pairCountDevice_ = cms::alpakatools::make_device_buffer<uint32_t>(queue);

    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(
        cms::alpakatools::divide_up_by(nTracksters_, kThreadsPerBlock), kThreadsPerBlock);
    const auto oneThread = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
    alpaka::exec<Acc1D>(queue, workDiv, KinematicsKernel{}, tracksters, selection_, tracksters_->data(), nTracksters_);
    alpaka::exec<Acc1D>(
        queue, oneThread, SeedListKernel{}, tracksters_->data(), seeds_->data(), nSeedsDevice_->data(), nTracksters_);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        CountPairsKernel{},
                        tracksters_->data(),
                        selection_,
                        seeds_->data(),
                        nSeedsDevice_->data(),
                        pairOffsets_->data(),
                        nTracksters_);
    alpaka::exec<Acc1D>(
        queue, oneThread, PairOffsetsKernel{}, pairOffsets_->data(), pairCountDevice_->data(), nTracksters_);
    alpaka::memcpy(queue, *pairCountHost_, *pairCountDevice_);
  }

  uint32_t SuperclusteringDNNPairBuilder::pairCount() const { return *(*pairCountHost_); }

  void SuperclusteringDNNPairBuilder::fill(Queue& queue,
                                           ticl::SuperclusteringDNNPairsDevice& pairs,
                                           ticl::SuperclusteringDNNPairFeaturesDevice& features) const {
    if (pairCount() == 0) {
      return;
    }
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(
        cms::alpakatools::divide_up_by(nTracksters_, kThreadsPerBlock), kThreadsPerBlock);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        FillPairsKernel{},
                        tracksters_->data(),
                        selection_,
                        seeds_->data(),
                        nSeedsDevice_->data(),
                        pairOffsets_->data(),
                        pairs.view(),
                        features.view(),
                        nTracksters_);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
