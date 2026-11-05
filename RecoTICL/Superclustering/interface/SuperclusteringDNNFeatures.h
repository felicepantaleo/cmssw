#ifndef RecoTICL_Superclustering_SuperclusteringDNNFeatures_h
#define RecoTICL_Superclustering_SuperclusteringDNNFeatures_h

// The superclustering DNN features of a (seed, candidate) pair, for the host and the device.
//
// This is SuperclusteringDNNInputV3::computeInto (src/SuperclusteringDNNInputs.cc)

#include <cmath>
#include <cstdint>
#include <xtd/xtd.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <alpaka/alpaka.hpp>

namespace ticl::superclustering {

  // ROOT::Math::etaMax<float>() 
  inline constexpr float kEtaMax = 22756.f;
//  inline constexpr float kBigZScaled = std::powf(std::numeric_limits<float>::epsilon(), -0.25f); //std::pow constexpr from C++26
  inline constexpr float kBigZScaled = 53.8173714f;
  // SuperclusteringDNNInputs.h: any delta time outside +/- kDeltaTimeDefault is replaced by kBadDeltaTime
  inline constexpr float kDeltaTimeDefault = 50.f;
  inline constexpr float kBadDeltaTime = -5.f;

  // What the features and the preselection read from a trackster.
  struct TracksterKinematics {
    Eigen::Vector3f barycenter;
    float eta, phi;
    float rawEnergy;
    float rawPt;
    float time;
    float eigenvalue0;
    float explVarRatio;  // 0 when the eigenvalues sum to zero
    bool hasEigenvalues;  // the eigenvalues do not sum to zero
    Eigen::Vector3f pca;  // principal axis, pointing away from the interaction point
  };

  // Following ROOT implementation ROOT::Math::Impl::Eta_FromRhoZ<float>
  // https://root.cern.ch/doc/v636/eta_8h_source.html#l00048
  ALPAKA_FN_HOST_ACC inline float eta(Eigen::Vector3f const& v) {
    const float rho = xtd::sqrt(v.x() * v.x() + v.y() * v.y());
    const float z = v.z();
    if (rho > 0.f) {
      const float zScaled = z / rho;
      // the Taylor branch is never reached in HGCAL (|eta| < 3.1), kept for completeness
      if (xtd::fabs(zScaled) < kBigZScaled) {
        return static_cast<float>(xtd::log(zScaled + std::sqrt(zScaled * zScaled + 1.0)));
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
  ALPAKA_FN_HOST_ACC inline float deltaPhi(float phi1, float phi2) {
    constexpr float kPi = static_cast<float>(M_PI);
    constexpr float kTwoPi = static_cast<float>(2. * M_PI);
    constexpr float kOneOverTwoPi = static_cast<float>(1. / (2. * M_PI));
    const float x = phi1 - phi2;
    if (xtd::fabs(x) <= kPi) {
      return x;
    }
    return x - xtd::round(x * kOneOverTwoPi) * kTwoPi;
  }

  ALPAKA_FN_HOST_ACC inline TracksterKinematics makeKinematics(Eigen::Vector3f const& barycenter,
                                                               float rawEnergy,
                                                               float time,
                                                               Eigen::Vector3f const& eigenvalues,
                                                               Eigen::Vector3f const& principalAxis) {
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

  namespace detail {

    // ROOT::Math::VectorUtil::Angle: 
    ALPAKA_FN_HOST_ACC inline float angle3D(Eigen::Vector3f const& a, Eigen::Vector3f const& b) {
      const float product = a.squaredNorm() * b.squaredNorm();
      float cosine = 0.f;
      if (product > 0.f) {
        cosine = a.dot(b) / xtd::sqrt(product);
        cosine = cosine > 1.f ? 1.f : (cosine < -1.f ? -1.f : cosine); //clamp [-1.f, 1.f]
      }
      return xtd::acosf(cosine);
    }

    // angle2D of SuperclusteringDNNInputs.cc, 
    ALPAKA_FN_HOST_ACC inline float angle2D(Eigen::Vector2f const& a, Eigen::Vector2f const& b) {
      const float denominator = a.squaredNorm() * b.squaredNorm();
      float cosine = 0.f;
      if (denominator > 0.f) {
        cosine = a.dot(b) / xtd::sqrt(denominator);
        cosine = cosine > 1.f ? 1.f : (cosine < -1.f ? -1.f : cosine); //clamp [-1.f, 1.f]
      }
      return xtd::acosf(cosine);
    }

  }  // namespace detail

  // Fills the features of a pair into its row of the SuperclusteringDNNPairFeaturesSoA.
  template <typename Row>
  ALPAKA_FN_HOST_ACC inline void fillFeatures(TracksterKinematics const& seed,
                                              TracksterKinematics const& candidate,
                                              Row out) {
    out.deltaEtaBaryc() = std::abs(candidate.eta) - std::abs(seed.eta);
    out.deltaPhiBaryc() = candidate.phi - seed.phi;  // not wrapped, as in the legacy code --> to be checked.
    out.multiEn() = candidate.rawEnergy;
    out.multiEta() = candidate.eta;
    out.multiPt() = candidate.rawPt;
    out.seedEta() = seed.eta;
    out.seedPhi() = seed.phi;
    out.seedEn() = seed.rawEnergy;
    out.seedPt() = seed.rawPt;

    Eigen::Vector3f const& seedAxis = seed.pca;
    Eigen::Vector3f const& candidateAxis = candidate.pca;
    const Eigen::Vector3f xs = seedAxis.cross(Eigen::Vector3f::UnitZ()).normalized();
    const Eigen::Vector3f ys = xs.cross(seedAxis).normalized();
    Eigen::Matrix3f seedFrame;
    seedFrame.col(0) = xs;
    seedFrame.col(1) = ys;
    seedFrame.col(2) = seedAxis;
    const Eigen::Vector3f inSeedFrame = (seedFrame * candidateAxis);

    const Eigen::Vector2f beam2D(0.f, 1.f);
    out.theta() = detail::angle3D(candidateAxis, seedAxis);
    out.thetaXZSeedFrame() = detail::angle2D(Eigen::Vector2f(inSeedFrame.x(), inSeedFrame.z()), beam2D);
    out.thetaYZSeedFrame() = detail::angle2D(Eigen::Vector2f(inSeedFrame.y(), inSeedFrame.z()), beam2D);
    out.thetaXYCmsFrame() = detail::angle2D(Eigen::Vector2f(candidateAxis.x(), candidateAxis.y()),
                                            Eigen::Vector2f(seedAxis.x(), seedAxis.y()));
    out.thetaYZCmsFrame() = detail::angle2D(Eigen::Vector2f(candidateAxis.y(), candidateAxis.z()),
                                            Eigen::Vector2f(seedAxis.y(), seedAxis.z()));
    out.thetaXZCmsFrame() = detail::angle2D(Eigen::Vector2f(candidateAxis.x(), candidateAxis.z()),
                                            Eigen::Vector2f(seedAxis.x(), seedAxis.z()));
    out.explVar() = candidate.eigenvalue0;
    out.explVarRatio() = candidate.explVarRatio;

    const float deltaTime = candidate.time - seed.time;
    out.modDeltaTime() = (deltaTime < -kDeltaTimeDefault || deltaTime > kDeltaTimeDefault) ? kBadDeltaTime : deltaTime;
  }

}  // namespace ticl::superclustering

#endif  // RecoTICL_Superclustering_SuperclusteringDNNFeatures_h
