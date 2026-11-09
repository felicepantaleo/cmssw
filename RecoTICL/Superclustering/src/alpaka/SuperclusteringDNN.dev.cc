#include <cstdint>
#include <numbers>

#include <Eigen/Geometry>
#include <xtd/math.h>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTICL/Superclustering/interface/SuperclusteringDNNInputs.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNN.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using namespace cms::alpakatools;

  namespace {

    constexpr uint32_t kThreadsPerBlock = 128;

    // ROOT::Math::etaMax<float>()
    constexpr float kEtaMax = 22756.f;
    constexpr float kBigZScaled = 53.8173714f;

    // Following ROOT implementation ROOT::Math::Impl::Eta_FromRhoZ<float>
    // https://root.cern.ch/doc/v636/eta_8h_source.html#l00048
    ALPAKA_FN_ACC inline float eta(const Eigen::Vector3f& v) {
      const float rho = xtd::sqrt(v.x() * v.x() + v.y() * v.y());
      const float z = v.z();
      if (rho > 0.f) {
        const float zScaled = z / rho;
        // the Taylor branch is never reached in HGCAL (|eta| < 3.1), kept for completeness
        if (xtd::fabs(zScaled) < kBigZScaled) {
          return static_cast<float>(xtd::log(zScaled + xtd::sqrt(zScaled * zScaled + 1.0)));
        }
        return z > 0.f ? static_cast<float>(xtd::log(2.0 * zScaled + 0.5 / zScaled))
                       : static_cast<float>(-xtd::log(-2.0 * zScaled));
      }
      if (z == 0.f) {
        return 0.f;
      }
      return z > 0.f ? z + kEtaMax : z - kEtaMax;
    }

    //Follow reco::deltaPhi implementation
    ALPAKA_FN_ACC inline float deltaPhi(float phi1, float phi2) {
      constexpr float kPi = static_cast<float>(std::numbers::pi);
      constexpr float kTwoPi = static_cast<float>(2. * std::numbers::pi);
      constexpr float kOneOverTwoPi = static_cast<float>(1. / (2. * std::numbers::pi));
      const float x = phi1 - phi2;
      if (xtd::fabs(x) <= kPi) {
        return x;
      }
      return x - xtd::round(x * kOneOverTwoPi) * kTwoPi;
    }

    ALPAKA_FN_ACC inline TracksterKinematics makeKinematics(const Eigen::Vector3f& barycenter,
                                                            float rawEnergy,
                                                            float time,
                                                            const Eigen::Vector3f& eigenvalues,
                                                            const Eigen::Vector3f& principalAxis) {
      TracksterKinematics k;
      k.barycenter = barycenter;
      k.eta = eta(barycenter);
      k.phi = xtd::atan2(barycenter.y(), barycenter.x());
      k.rawEnergy = rawEnergy;
      k.rawPt = rawEnergy / xtd::cosh(k.eta);
      k.time = time;
      k.eigenvalue0 = eigenvalues(0);
      // summed in the order of the legacy code, not with eigenvalues.sum(): explVarRatio feeds a cut
      const float sum = 0.f + eigenvalues(0) + eigenvalues(1) + eigenvalues(2);
      k.hasEigenvalues = sum != 0.f;
      k.explVarRatio = k.hasEigenvalues ? eigenvalues(0) / sum : 0.f;
      // fillPCAVariables: the principal axis points away from the interaction point
      k.pca = principalAxis.z() * barycenter.z() < 0.f ? Eigen::Vector3f(-principalAxis) : principalAxis;
      return k;
    }

    // ROOT::Math::VectorUtil::Angle, and angle2D of SuperclusteringDNNInputs.cc
    template <typename Vector>
    ALPAKA_FN_ACC inline float angle(const Vector& a, const Vector& b) {
      const float product = a.squaredNorm() * b.squaredNorm();
      float cosine = 0.f;
      if (product > 0.f) {
        cosine = a.dot(b) / xtd::sqrt(product);
        cosine = cosine > 1.f ? 1.f : (cosine < -1.f ? -1.f : cosine);  //clamp [-1.f, 1.f]
      }
      return xtd::acos(cosine);
    }

    // Fills the features of a pair into its row of the SuperclusteringDNNPairFeaturesSoA:
    // SuperclusteringDNNInputV3::computeInto (src/SuperclusteringDNNInputs.cc)
    template <typename Row>
    ALPAKA_FN_ACC inline void fillFeatures(const TracksterKinematics& seed,
                                           const TracksterKinematics& candidate,
                                           Row out) {
      out.deltaEtaBaryc() = xtd::fabs(candidate.eta) - xtd::fabs(seed.eta);
      out.deltaPhiBaryc() = candidate.phi - seed.phi;  // not wrapped, as in the legacy code --> to be checked.
      out.multiEn() = candidate.rawEnergy;
      out.multiEta() = candidate.eta;
      out.multiPt() = candidate.rawPt;
      out.seedEta() = seed.eta;
      out.seedPhi() = seed.phi;
      out.seedEn() = seed.rawEnergy;
      out.seedPt() = seed.rawPt;

      const Eigen::Vector3f& seedAxis = seed.pca;
      const Eigen::Vector3f& candidateAxis = candidate.pca;
      const Eigen::Vector3f xs = seedAxis.cross(Eigen::Vector3f::UnitZ()).normalized();
      const Eigen::Vector3f ys = xs.cross(seedAxis).normalized();
      Eigen::Matrix3f seedFrame;
      seedFrame.col(0) = xs;
      seedFrame.col(1) = ys;
      seedFrame.col(2) = seedAxis;
      const Eigen::Vector3f inSeedFrame = (seedFrame * candidateAxis);

      const Eigen::Vector2f beam2D(0.f, 1.f);
      out.theta() = angle(candidateAxis, seedAxis);
      out.thetaXZSeedFrame() = angle(Eigen::Vector2f(inSeedFrame.x(), inSeedFrame.z()), beam2D);
      out.thetaYZSeedFrame() = angle(Eigen::Vector2f(inSeedFrame.y(), inSeedFrame.z()), beam2D);
      out.thetaXYCmsFrame() =
          angle(Eigen::Vector2f(candidateAxis.x(), candidateAxis.y()), Eigen::Vector2f(seedAxis.x(), seedAxis.y()));
      out.thetaYZCmsFrame() =
          angle(Eigen::Vector2f(candidateAxis.y(), candidateAxis.z()), Eigen::Vector2f(seedAxis.y(), seedAxis.z()));
      out.thetaXZCmsFrame() =
          angle(Eigen::Vector2f(candidateAxis.x(), candidateAxis.z()), Eigen::Vector2f(seedAxis.x(), seedAxis.z()));
      out.explVar() = candidate.eigenvalue0;
      out.explVarRatio() = candidate.explVarRatio;

      const float deltaTime = candidate.time - seed.time;
      out.modDeltaTime() = (deltaTime < -::ticl::kDeltaTimeDefault || deltaTime > ::ticl::kDeltaTimeDefault)
                               ? ::ticl::kBadDeltaTime
                               : deltaTime;
    }

    // TracksterLinkingbySuperClusteringDNN::checkExplainedVarianceRatioCut
    ALPAKA_FN_ACC inline bool passesExplainedVarianceRatioCut(const TracksterKinematics& k,
                                                              const SuperclusteringDNNPairSelection& selection) {
      if (!k.hasEigenvalues) {
        return false;
      }
      return k.explVarRatio >= (k.rawEnergy > selection.explVarRatioCut_energyBoundary
                                    ? selection.explVarRatioMinimum_highEnergy
                                    : selection.explVarRatioMinimum_lowEnergy);
    }

    // seed (the higher pT trackster) - candidate pair of linkTracksters
    ALPAKA_FN_ACC inline bool isPair(const SuperclusteringDNNPairingTrackster* tracksters,
                                     uint32_t seedIndex,
                                     uint32_t candidateIndex,
                                     const SuperclusteringDNNPairSelection& selection) {
      const auto& seed = tracksters[seedIndex];
      const auto& candidate = tracksters[candidateIndex];
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
      if (xtd::fabs(seed.kinematics.eta - candidate.kinematics.eta) >= selection.deltaEtaWindow) {
        return false;
      }
      return xtd::fabs(deltaPhi(seed.kinematics.phi, candidate.kinematics.phi)) < selection.deltaPhiWindow;
    }

    // One thread per trackster: kinematics and preselection, read from the device trackster columns.
    struct KinematicsKernel {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    ::ticl::TracksterSoA::ConstView tracksters,
                                    SuperclusteringDNNPairSelection selection,
                                    SuperclusteringDNNPairingTrackster* out,
                                    uint32_t nTracksters) const {
        for (auto t : uniform_elements(acc, nTracksters)) {
          const auto energy = tracksters.energy()[t];
          const auto barycenter = tracksters.barycenter()[t];
          const auto pca = tracksters.pca()[t];

          auto& result = out[t];
          result.kinematics = makeKinematics(Eigen::Vector3f(barycenter.x(), barycenter.y(), barycenter.z()),
                                             energy.rawEnergy(),
                                             tracksters.timing()[t].time(),
                                             pca.eigenvalues(),
                                             pca.eigenvectors().col(0));
          const auto& k = result.kinematics;
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
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const SuperclusteringDNNPairingTrackster* tracksters,
                                    uint32_t* seeds,
                                    uint32_t* nSeeds,
                                    uint32_t nTracksters) const {
        if (once_per_grid(acc)) {
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
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const SuperclusteringDNNPairingTrackster* tracksters,
                                    SuperclusteringDNNPairSelection selection,
                                    const uint32_t* seeds,
                                    const uint32_t* nSeeds,
                                    uint32_t* pairCounts,
                                    uint32_t nTracksters) const {
        const uint32_t seedCount = *nSeeds;
        for (auto candidate : uniform_elements(acc, nTracksters)) {
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
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    uint32_t* countsToOffsets,
                                    uint32_t* totalPairs,
                                    uint32_t nTracksters) const {
        if (once_per_grid(acc)) {
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
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const SuperclusteringDNNPairingTrackster* tracksters,
                                    SuperclusteringDNNPairSelection selection,
                                    const uint32_t* seeds,
                                    const uint32_t* nSeeds,
                                    const uint32_t* pairOffsets,
                                    ::ticl::SuperclusteringDNNPairsSoA::View pairs,
                                    ::ticl::SuperclusteringDNNPairFeaturesSoA::View features,
                                    uint32_t nTracksters) const {
        const uint32_t seedCount = *nSeeds;
        for (auto candidate : uniform_elements(acc, nTracksters)) {
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

            fillFeatures(tracksters[seed].kinematics, tracksters[candidate].kinematics, features[row]);
            ++row;
          }
        }
      }
    };

  }  // namespace

  void SuperclusteringDNNAlgo::findPairs(Queue& queue,
                                         ::ticl::TracksterSoA::ConstView tracksters,
                                         const SuperclusteringDNNPairSelection& selection,
                                         SuperclusteringDNNPairSearch& search) {
    const auto n = search.nTracksters;
    const auto workDiv = make_workdiv<Acc1D>(divide_up_by(n, kThreadsPerBlock), kThreadsPerBlock);
    const auto oneThread = make_workdiv<Acc1D>(1, 1);
    alpaka::exec<Acc1D>(queue, workDiv, KinematicsKernel{}, tracksters, selection, search.tracksters.data(), n);
    alpaka::exec<Acc1D>(
        queue, oneThread, SeedListKernel{}, search.tracksters.data(), search.seeds.data(), search.nSeeds.data(), n);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        CountPairsKernel{},
                        search.tracksters.data(),
                        selection,
                        search.seeds.data(),
                        search.nSeeds.data(),
                        search.pairOffsets.data(),
                        n);
    alpaka::exec<Acc1D>(queue, oneThread, PairOffsetsKernel{}, search.pairOffsets.data(), search.nPairs.data(), n);
  }

  void SuperclusteringDNNAlgo::fillPairs(Queue& queue,
                                         const SuperclusteringDNNPairSelection& selection,
                                         const SuperclusteringDNNPairSearch& search,
                                         ::ticl::SuperclusteringDNNPairsSoA::View pairs,
                                         ::ticl::SuperclusteringDNNPairFeaturesSoA::View features) {
    const auto n = search.nTracksters;
    const auto workDiv = make_workdiv<Acc1D>(divide_up_by(n, kThreadsPerBlock), kThreadsPerBlock);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        FillPairsKernel{},
                        search.tracksters.data(),
                        selection,
                        search.seeds.data(),
                        search.nSeeds.data(),
                        search.pairOffsets.data(),
                        pairs,
                        features,
                        n);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl
