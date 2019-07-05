#ifndef CUDADataFormatsTrackTrajectoryStateSOA_H
#define CUDADataFormatsTrackTrajectoryStateSOA_H

#include <Eigen/Dense>
#include "HeterogeneousCore/CUDAUtilities/interface/eigenSoA.h"

template <int32_t S>
struct TrajectoryStateSoA {

  using Vector5f = Eigen::Matrix<float, 5, 1>;
  using Vector15f = Eigen::Matrix<float, 15, 1>;

  using Vector5d = Eigen::Matrix<double, 5, 1>;
  using Matrix5d = Eigen::Matrix<double, 5, 5>;


  static constexpr int32_t stride() { return S; }

  eigenSoA::MatrixSoA<Vector5f,S> state;
  eigenSoA::MatrixSoA<Vector15f,S> covariance;

  template<typename V5, typename M5>
  __host__ __device__
  void copyFromDense(V5 const & v, M5 const & cov, int32_t i) {
     state(i) = v.template cast<float>();
     for(int j=0, ind=0; j<5; ++j) for (auto k=j;k<5;++k) covariance(i)(ind++) = cov(j,k); 
  }

  template<typename V5, typename M5>
  __host__ __device__
  void copyToDense(V5 & v, M5 & cov, int32_t i) const {
     v = state(i).template cast<typename V5::Scalar>();
     for(int j=0, ind=0; j<5; ++j) {
        cov(j,j) = covariance(i)(ind++); 
        for (auto k=j+1;k<5;++k) cov(k,j)=cov(j,k) = covariance(i)(ind++);
     }
  }

};

#endif // CUDADataFormatsTrackTrajectoryStateSOA_H


