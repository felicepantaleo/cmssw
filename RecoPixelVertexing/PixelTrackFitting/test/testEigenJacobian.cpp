#include "RecoPixelVertexing/PixelTrackFitting/interface/FitResult.h"
#include<cmath>

using Rfit::Vector5d;
using Rfit::Matrix5d;

// transformation between the "perigee" to cmssw localcoord frame
// the plane of the latter is the perigee plane...
// from   //!<(phi,Tip,pt,cotan(theta)),Zip)
// to 1/p,dx/dz,dy/dz,x,z

Vector5d transf(Vector5d  const & p) {
  Vector5d op;
  auto tipSignNeg = -std::copysign(1.,p(1));
  auto sinTheta = 1/std::sqrt(1+p(3)*p(3));
  op(0) = sinTheta/p(2);
  op(1) = 0.;
  op(2) = tipSignNeg*p(3);
  op(3) = std::abs(p(1));
  op(4) = tipSignNeg*p(4);
  return op;
}

Matrix5d transfFast(Matrix5d cov, Vector5d const &  p) {
  auto sqr = [](auto x) { return x*x;};
  auto sinTheta = 1/std::sqrt(1+p(3)*p(3));
  auto cosTheta = p(3)*sinTheta;
  cov(2,2) = sqr(sinTheta) * (
              cov(2,2)*sqr(1./(p(2)*p(2)))
            + cov(3,3)*sqr(cosTheta*sinTheta/p(2))
            );
  cov(3,2) = cov(2,3) = cov(3,3) * cosTheta * sqr(sinTheta) / p(2); 
  // for (int i=0; i<5; ++i) cov(i,2) *= -sinTheta/(p(2)*p(2));
  // for (int i=0; i<5; ++i) cov(2,i) *= -sinTheta/(p(2)*p(2));
  return cov;


}

Matrix5d Jacobian(Vector5d const &  p) {
/*
  op(0) = sinTheta/p(2);
  op(1) = 0.;
  op(2) = tipSignNeg*p(3);
  op(3) = std::abs(p(1));
  op(4) = tipSignNeg*p(4);
*/

  Matrix5d J = Matrix5d::Zero();

  auto sinTheta2 = 1/(1+p(3)*p(3));
  auto sinTheta = std::sqrt(sinTheta2);
  auto cosTheta = p(3)*sinTheta;
  auto tipSignNeg = -std::copysign(1.,p(1));

  J(0,2) = -sinTheta/(p(2)*p(2));
  J(0,3) = -sinTheta2*cosTheta/p(2);
  J(1,0) = 1.;
  J(2,3) = tipSignNeg;
  J(3,1) = -tipSignNeg;
  J(4,4) = tipSignNeg;
  return J;
}

Matrix5d transf(Matrix5d const & cov, Matrix5d const& J) {

   return J*cov*J.transpose();

}  

Matrix5d loadCov(Vector5d const & e) {

  Matrix5d cov;
  for (int i=0; i<5; ++i) cov(i,i) = e(i)*e(i);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < i; ++j) {
      double v = 0.3*std::sqrt( cov(i,i) * cov(j,j) ); // this makes the matrix pos defined
      cov(i,j) = (i+j)%2 ? -0.4*v  : 0.1*v;
      cov(j,i) = cov(i,j);
    }
   }
  return cov;
}


#include<iostream>
int main() {

  for (auto stip=-1; stip<2; stip+=2)
  for (auto szip=-1; szip<2; szip+=2) {
  Vector5d par0; par0 << 0.2,0.1,3.5,0.8,0.1;
  Vector5d del0; del0 << 0.01,0.01,0.035,-0.03,-0.01;
  //!<(phi,Tip,pt,cotan(theta)),Zip)
    par0(1) *= stip;
    par0(4) *= szip;

  Matrix5d J = Jacobian(par0);


  Vector5d par1 = transf(par0);
  Vector5d par2 = transf(par0+del0);
  // not accurate as the perigee plane move as well...
  Vector5d del1 = par2-par1; 

  Matrix5d cov0 = loadCov(del0);
  Matrix5d cov1 = transf(cov0,J);
  Matrix5d cov2 = transfFast(cov0,par0);

  // don't ask: guess
  std::cout << "par0 " << par0.transpose() << std::endl;
  std::cout << "del0 " << del0.transpose() << std::endl;


  std::cout << "par1 " << par1.transpose() << std::endl;
  std::cout << "del1 " << del1.transpose() << std::endl;
  std::cout << "del2 " << (J*del0).transpose() << std::endl;

  std::cout << "del1^2 " << (del1.array()*del1.array()).transpose() << std::endl;
  std::cout << std::endl;
  std::cout << "J\n" << J << std::endl;
  
  std::cout << "cov0\n" << cov0 << std::endl;
  std::cout << "cov1\n" << cov1 << std::endl;
  std::cout << "cov2\n" << cov2 << std::endl;
  std::cout << std::endl << "----------" << std::endl;

  } // lopp over signs

  return 0;


}
