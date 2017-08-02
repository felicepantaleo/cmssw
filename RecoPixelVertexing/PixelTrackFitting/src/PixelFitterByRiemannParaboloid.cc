#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByRiemannParaboloid.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

//#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

//#include "CommonTools/Statistics/interface/LinearFit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

//#include "RecoPixelVertexing/PixelTrackFitting/interface/RZLine.h"
//#include "CircleFromThreePoints.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackBuilder.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackErrorParam.h"
#include "DataFormats/GeometryVector/interface/Pi.h"

#include "CommonTools/Utils/interface/DynArray.h"

using namespace std;
using namespace Eigen;

namespace
{

constexpr float b = 1.f;
constexpr float d = 1.e4f;
constexpr unsigned int max_nop = 8;
constexpr float halfpi = Geom::fhalfPi();


struct circle_fit
{
        Vector3f par;
        Matrix3f cov;
        int charge;
        float chi2;
};

struct line_fit
{
        Vector2f par;
        Matrix2f cov;
        float chi2;
};

struct helix_fit
{
        Matrix<float, 5, 1> par;
        Matrix<float, 5, 5> cov;
        int charge;
        float chi2;
};

inline float sqr(float a)
{
    return a * a;
}

float chi2_circle(Matrix<float, 2, Dynamic, 0, 2, max_nop> points2D,
        Matrix<float, Dynamic, Dynamic, 0, 2 * max_nop, 2 * max_nop> V, Vector3f par_uvr)
{
    float chi2 = 0;
    int nop = points2D.cols();
    for (int i = 0; i < nop; i++)
    {
        float x_ = points2D(0, i) - par_uvr(0);
        float y_ = points2D(1, i) - par_uvr(1);
        float x_2 = sqr(x_);
        float y_2 = sqr(y_);
        chi2 += sqr(sqrt(x_2 + y_2) - par_uvr(2))
                / ((V(i, i) * x_2 + V(i + nop, i + nop) * y_2 + 2 * V(i, i + nop) * x_ * y_)
                        / (x_2 + y_2));
        //if(debug) cout  << sqrt(x_2+y_2)-par_uvr(2) << "    "
        //                << ((V(i,i)*x_2+V(i+nop,i+nop)*y_2+2*V(i,i+nop)*x_*y_)/(x_2+y_2)) << endl ;
    }
    return chi2;
}

float chi2_line(Matrix<float, 2, Dynamic, 0, 2, max_nop> points2D,
        Matrix<float, Dynamic, 1, 0, max_nop, 1> y_err2, Vector2f par_line)
{
    float chi2 = 0;
    int nop = points2D.cols();
    for (int i = 0; i < nop; i++)
    {
        chi2 +=
                sqr(points2D(1, i) - points2D(0, i) * par_line(0) - par_line(1))
                        / (1 + sqr(par_line(0)))
                        / (y_err2(i) * sqr(cos(atan(par_line(0))))
                                + y_err2(i) * sqr(sin(atan(par_line(0)))));
    }
    return chi2;
}

inline int Charge(Matrix<float, 2, Dynamic, 0, 2, max_nop> points2D, Vector3f par_uvr)
{ //error to be computed TO FIX
    float dir = (points2D(0, 1) - points2D(0, 0)) * (par_uvr(1) - points2D(1, 0))
            - (points2D(1, 1) - points2D(1, 0)) * (par_uvr(0) - points2D(0, 0));
    return (dir > 0) ? -1 : 1;
}

Vector3f par_transformation(Vector3f par_uvr, int charge, float B_field)
{
    Vector3f par_pak;
    float phi = (charge > 0) ? atan2(par_uvr(0), -par_uvr(1)) : atan2(-par_uvr(0), par_uvr(1));
    par_pak << phi, charge * (sqrt(sqr(par_uvr(0)) + sqr(par_uvr(1))) - par_uvr(2)), par_uvr(2)
            * B_field;
    return par_pak;
}

// return the eigenvector associated to the minimum eigenvalue
Vector3f min_eigen3D(Matrix3f A)
{
    EigenSolver < Matrix3f > solver(A); // evaluate eigenvalues and eigenvector
    Vector3cf lambdac = solver.eigenvalues(); //why can't I cast here ??
    Matrix3cf eigenvectc = solver.eigenvectors();
    Vector3f lambda = lambdac.real().cast<float>(); //check if real ! TO FIX
    Matrix3f eigenvect = eigenvectc.real().cast<float>();
    int minindex = 0;
    lambda.minCoeff(&minindex);
    Vector3f n = eigenvect.col(minindex);
    return n;
}

Vector2f min_eigen2D(Matrix2f A)
{
    EigenSolver < Matrix2f > solver(A); // evaluate eigenvalues and eigenvector
    Vector2cf lambdac = solver.eigenvalues(); //why can't I cast here ??
    Matrix2cf eigenvectc = solver.eigenvectors();
    Vector2f lambda = lambdac.real().cast<float>(); //check if real ! TO FIX
    Matrix2f eigenvect = eigenvectc.real().cast<float>();
    int minindex = 0;
    lambda.minCoeff(&minindex);
    Vector2f n = eigenvect.col(minindex);
    return n;
}

//     a       ||   0|   1|   2|   3|   4|   5
// nu(a)=(i,j) || 0,0| 0,1| 0,2| 1,1| 1,2| 2,2
inline int nu(int a, int* i, int* j)
{
    *i = (a == 0 || a == 1 || a == 2) ? 0 : (a == 3 || a == 4) ? 1 : (a == 5) ? 2 : 3;
    *j = (a == 2 || a == 4 || a == 5) ? 2 : (a == 1 || a == 3) ? 1 : (a == 0) ? 0 : 3;
    if (*i == 3)
        return 1;
    return 0;
}

inline float X(Matrix<float, 5, 1> par)
{
    return 2.f
            * asin(
                    sqrt(
                            sqr(par(3) - cos(par(0) - halfpi) * par(1))
                                    + sqr(par(4) - sin(par(0) - halfpi) * par(1))) * par(2) / 2.f)
            / par(2);
}

Matrix<float, Dynamic, 5> J_num(Matrix<float, 3, Dynamic> points3D, circle_fit circle)
{
    int nop = points3D.cols();
    Matrix<float, Dynamic, 5> J(nop, 5);
    for (int i = 0; i < nop; i++)
    {
        Matrix<float, 5, 1> par;
        par << circle.par, points3D(0, i), points3D(1, i);
        for (int j = 0; j < 5; j++)
        {
            Matrix<float, 5, 1> Delta = MatrixXf::Zero(5, 1);
            Delta(j) = abs(par(j) / d);
            J(i, j) = (X(par + Delta) - X(par)) / Delta(j);
        }
    }
    return J;
}

VectorXf X_err(Matrix<float, 3, Dynamic> points3D,
        Matrix<float, Dynamic, Dynamic, 0, 3 * max_nop, 3 * max_nop> V, circle_fit circle)
{
    int nop = points3D.cols();
    VectorXf x_err(nop);
    Matrix<float, Dynamic, 5> J = J_num(points3D, circle);
    for (int i = 0; i < nop; i++)
    {
        Matrix<float, 5, 5> Cov = MatrixXf::Zero(5, 5);
        Cov.block(0, 0, 3, 3) = circle.cov;
        Cov(3, 3) = V(i, i);
        Cov(4, 4) = V(i + nop, i + nop);
        Cov(3, 4) = Cov(4, 3) = V(i, i + nop);
        x_err(i) = J.row(i) * Cov * J.row(i).transpose();
    }
    return x_err;
}

circle_fit Circle_fit(Matrix<float, 2, Dynamic, 0, 2, max_nop> points2D,
        Matrix<float, Dynamic, Dynamic, 0, 2 * max_nop, 2 * max_nop> V, float B_field,
        bool return_err = true)
{

    //INITIALIZATION
    int nop = points2D.cols();
    Matrix<float, 2, Dynamic, 0, 2, max_nop> points2D_ = points2D;
    Matrix<float, 3, Dynamic, 0, 3, max_nop> points3D(3, nop);
    Matrix<float, Dynamic, 1, 0, max_nop, 1> weight = 1.f
            / (V.block(0, 0, nop, nop).diagonal() + V.block(nop, nop, nop, nop).diagonal()).array();
    weight /= weight.sum();

    //CENTER & SCALE 2D POINTS
    float umean = points2D.row(0).mean();
    float vmean = points2D.row(1).mean();
    points2D.row(0) = points2D.row(0).array() - umean;
    points2D.row(1) = points2D.row(1).array() - vmean;
    VectorXf mc(2 * nop);
    mc << points2D.row(0).transpose(), points2D.row(1).transpose(); //useful for error propagation
    float q = points2D.array().square().sum();
    float s = b * sqrt(nop / q); //scaling factor (b is an arbitrary constant)
    points2D *= s;

    //CALCULATE TRASFORMED POINTS IN 3D
    points3D.block(0, 0, 2, nop) = points2D;
    points3D.row(2) = points2D.row(0).array().square() + points2D.row(1).array().square();

    //CALCULATE & MINIMIZE COST FUNCTION
    Matrix3f A = Matrix3f::Zero();
    Vector3f r0 = points3D * weight;

    for (int i = 0; i < nop; i++)
        A += weight(i) * ((points3D.col(i) - r0) * (points3D.col(i) - r0).transpose());

    Vector3f n = min_eigen3D(A);
    n *= (n(2) > 0) ? 1 : -1;
    float c = -n.transpose() * r0;

    //CALCULATE CIRCUMFERENCE PARAMETER
    Vector3f par_uvr_;
    par_uvr_ << -n(0) / (2.f * n(2)), -n(1) / (2.f * n(2)), sqrt(
            (1.f - sqr(n(2)) - 4.f * c * n(2)) / (4.f * sqr(n(2))));
    Vector3f par_uvr;
    par_uvr << par_uvr_(0) / s + umean, par_uvr_(1) / s + vmean, par_uvr_(2) / s;

    circle_fit circle;
    circle.charge = Charge(points2D_, par_uvr);
    circle.par = par_transformation(par_uvr, circle.charge, B_field);
    circle.cov = Matrix3f::Zero();
    circle.chi2 = chi2_circle(points2D_, V, par_uvr);

    //ERROR PROPAGATION
    if (return_err)
    {
        //auxiliary quantities TO FIX
        Matrix<float, Dynamic, 1, 0, 3 * max_nop, 1> r(3 * nop);
        r << points3D.row(0).transpose(), points3D.row(1).transpose(), points3D.row(2).transpose();
        Matrix<float, Dynamic, Dynamic, 0, max_nop, max_nop> H = MatrixXf::Identity(nop, nop)
                - VectorXf::Constant(nop, 1.f) * weight.transpose();
        Matrix<float, Dynamic, 3, 0, max_nop, 3> s_v = H * points3D.transpose();
        Matrix<float, Dynamic, Dynamic, 0, 2 * max_nop, 2 * max_nop> W = weight
                * weight.transpose();
        float h = sqrt(1.f - sqr(n(2)) - 4.f * c * n(2));
        Matrix<float, 1, Dynamic, 1, 1, 2 * max_nop> Jq = mc.transpose() * s / (b * nop);

        //Matrix<float, Dynamic, Dynamic, 0, 2*max_nop, 2*max_nop> J1 = s * (MatrixXf::Identity(2*nop,2*nop) - mc*mc.transpose()/q);
        //MatrixXf Vcs_ = J1 * V * J1.transpose();

        Matrix<float, Dynamic, Dynamic, 0, 2 * max_nop, 2 * max_nop> Vcs = sqr(s) * V
                + pow(s, 4) / sqr(b)
                        * (sqr(1 / (2.f * sqrt(q * nop)))
                                * (2.f * V.array().square().sum() + 4.f * mc.transpose() * V * mc))
                        * mc * mc.transpose();

        Matrix<float, Dynamic, Dynamic, 0, 3 * max_nop, 3 * max_nop> C = MatrixXf::Zero(3 * nop,
                3 * nop);
        C.block(0, 0, 2 * nop, 2 * nop) = Vcs;
        C.block(0, 2 * nop, nop, nop) = 2.f * Vcs.block(0, 0, nop, nop).array()
                * (VectorXf::Constant(nop, 1.f) * points3D.row(0)).array()
                + 2.f * Vcs.block(0, nop, nop, nop).array()
                        * (VectorXf::Constant(nop, 1.f) * points3D.row(1)).array();
        C.block(2 * nop, 0, nop, nop) = C.block(0, 2 * nop, nop, nop).transpose();
        C.block(nop, 2 * nop, nop, nop) = 2.f * Vcs.block(nop, 0, nop, nop).array()
                * (VectorXf::Constant(nop, 1.f) * points3D.row(0)).array()
                + 2.f * Vcs.block(nop, nop, nop, nop).array()
                        * (VectorXf::Constant(nop, 1.f) * points3D.row(1)).array();
        C.block(2 * nop, nop, nop, nop) = C.block(nop, 2 * nop, nop, nop).transpose();
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                C.block(2 * nop, 2 * nop, nop, nop) += (2.f
                        * Vcs.block(i * nop, i * nop, nop, nop).array()
                        * Vcs.block(i * nop, j * nop, nop, nop).array()
                        + 4.f * Vcs.block(i * nop, j * nop, nop, nop).array()
                                * (points3D.row(i).transpose() * points3D.row(j)).array()).matrix();

        Matrix<float, Dynamic, Dynamic, 0, 3 * max_nop, 3 * max_nop> D = MatrixXf::Zero(3 * nop,
                3 * nop);
        for (int i = 0; i < 3; i++)
        {
            for (int j = i; j < 3; j++)
            {
                D.block(i * nop, j * nop, nop, nop) = H * C.block(i * nop, j * nop, nop, nop) * H; // H.transpose();
                if (i != j)
                    D.block(j * nop, i * nop, nop, nop) =
                            D.block(i * nop, j * nop, nop, nop).transpose();
            }
        }

        Matrix<float, 6, 6> E;
        for (int a = 0; a < 6; a++)
        {
            for (int z = a; z < 6; z++)
            {
                int i = 0, j = 0, k = 0, l = 0;
                nu(a, &i, &j);
                nu(z, &k, &l);
                E(a, z) = (D.block(i * nop, k * nop, nop, nop).array() * W.array()
                        * D.block(j * nop, l * nop, nop, nop).array()
                        + D.block(i * nop, l * nop, nop, nop).array() * W.array()
                                * D.block(j * nop, k * nop, nop, nop).array()).sum()
                        + (s_v.col(i).transpose()
                                * (D.block(j * nop, l * nop, nop, nop).array() * W.array()).matrix()
                                * s_v.col(k))
                        + (s_v.col(i).transpose()
                                * (D.block(j * nop, k * nop, nop, nop).array() * W.array()).matrix()
                                * s_v.col(l))
                        + (s_v.col(j).transpose()
                                * (D.block(i * nop, l * nop, nop, nop).array() * W.array()).matrix()
                                * s_v.col(k))
                        + (s_v.col(j).transpose()
                                * (D.block(i * nop, k * nop, nop, nop).array() * W.array()).matrix()
                                * s_v.col(l));
                E(z, a) = E(a, z);
            }
        }

        Matrix<float, 3, 6> J2; //= J2_numeric(A);
        for (int a = 0; a < 6; a++)
        {
            int i = 0, j = 0;
            nu(a, &i, &j);
            Matrix3f Delta = Matrix3f::Zero();
            Delta(i, j) = Delta(j, i) = abs(A(i, j) / d);
            J2.col(a) = min_eigen3D(A + Delta);
            int dn3sign = (J2.col(a)(2) > 0) ? 1 : -1;
            J2.col(a) = (J2.col(a) * dn3sign - n) / Delta(i, j);
        }

        Matrix3f C0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                C0(i, j) = weight.transpose() * C.block(i * nop, j * nop, nop, nop) * weight;

        Matrix4f Cnc;
        Cnc.block(0, 0, 3, 3) = J2 * E * J2.transpose();
        Cnc.block(0, 3, 3, 1) = -Cnc.block(0, 0, 3, 3) * r0;
        Cnc.block(3, 0, 1, 3) = Cnc.block(0, 3, 3, 1).transpose();
        Cnc(3, 3) = (n.transpose() * C0 * n) + (C0.array() * Cnc.block(0, 0, 3, 3).array()).sum()
                + (r0.transpose() * Cnc.block(0, 0, 3, 3) * r0);

        Matrix<float, 3, 4> J3;
        J3 << -1.f / (2.f * n(2)), 0, n(0) / (2.f * sqr(n(2))), 0, 0, -1.f / (2 * n(2)), n(1)
                / (2.f * sqr(n(2))), 0, 0, 0, -h / (2.f * sqr(n(2)))
                - (4.f * c + 2.f * n(2)) / (4.f * h * n(2)), -1.f / h;

        Matrix3f cov_uvr = (J3 * Cnc * J3.transpose()) / sqr(s)
                + (par_uvr_ * par_uvr_.transpose()) * (Jq * V * Jq.transpose()) / sqr(b);

        Matrix3f J4 = Matrix3f::Zero();
        J4 << -par_uvr(1) / sqr(par_uvr(0)) * 1.f / (1 + sqr(par_uvr(1) / par_uvr(0))), 1.f
                / par_uvr(0) * 1.f / (1 + sqr(par_uvr(1) / par_uvr(0))), 0, circle.charge
                * par_uvr(0) / sqrt(sqr(par_uvr(0)) + sqr(par_uvr(1))), circle.charge * par_uvr(1)
                / sqrt(sqr(par_uvr(0)) + sqr(par_uvr(1))), -circle.charge, 0, 0, B_field;
        Matrix3f J4_ = Matrix3f::Zero();
        for (int a = 0; a < 3; a++)
        {
            Vector3f Delta = Vector3f::Zero();
            Delta(a) = abs(par_uvr(a) / d);
            J4_.col(a) = (par_transformation(par_uvr + Delta, circle.charge, B_field) - circle.par)
                    / Delta(a);
        }

        circle.cov = J4 * cov_uvr * J4.transpose();

    }

    return circle;
}

line_fit Line_fit(Matrix<float, 3, Dynamic, 0, 3, max_nop> points3D,
        Matrix<float, Dynamic, Dynamic, 0, 3 * max_nop, 3 * max_nop> V, circle_fit circle,
        float B_field, bool return_err = true)
{

    //INITIALIZATION
    int nop = points3D.cols();
    Matrix<float, 2, Dynamic, 0, 2, max_nop> points2D(2, nop);

    //Matrix<float, Dynamic, 1, 0, max_nop, 1> x_err2 = X_err(points3D, V, circle);
    Matrix<float, Dynamic, 1, 0, max_nop, 1> y_err2 =
            V.block(2 * nop, 2 * nop, nop, nop).diagonal();
    //float k = x_err2.array().sqrt().mean()/y_err2.array().sqrt().mean();
    //if(debug) cout << "k:  " <<  k <<endl;

    Matrix<float, Dynamic, 1, 0, max_nop, 1> weight = 1.f / (y_err2).array();
    weight /= weight.sum();

    //CALCULATE TRASFORMED POINTS IN 2D
    points2D.row(1) = points3D.row(2);
    //points2D.row(0) = points3D.row(1);
    points2D.row(0) =
            2.f
                    * asin(
                            ((points3D.row(0).array() - cos(circle.par(0) - halfpi) * circle.par(1)).square()
                                    + (points3D.row(1).array()
                                            - sin(circle.par(0) - halfpi) * circle.par(1)).square()).sqrt()
                                    / (circle.par(2) / B_field * 2.f)) * (circle.par(2) / B_field);

    //CALCULATE & MINIMIZE COST FUNCTION
    Matrix2f A = Matrix2f::Zero();
    Vector2f r0 = points2D * weight;

    for (int i = 0; i < nop; i++)
        A += weight(i) * ((points2D.col(i) - r0) * (points2D.col(i) - r0).transpose());

    Vector2f n = min_eigen2D(A);
    float c = -n.transpose() * r0;

    //CALCULATE LINE PARAMETER
    line_fit line;
    line.par << -n(0) / n(1), -c * sqrt(sqr(n(0)) + sqr(n(1))) / n(1);
    line.cov = Matrix2f::Zero();
    line.chi2 = chi2_line(points2D, y_err2, line.par);

    //ERROR PROPAGATION
    if (return_err)
    {
        //auxiliary quantities
        float sig2 = y_err2.mean(); //TO FIX
        float S = (A(0, 0) + A(1, 1)) * nop;
        float n0_2 = sqr(n(0));
        float n1_2 = sqr(n(1));
        float sqrt_ = sqrt(n1_2 + n0_2);
        float x_ = points2D.row(0).sum() / nop;
        float y_ = points2D.row(1).sum() / nop;
        float corr = sqr(1.131); //TO FIX
        float C13 = sig2 * n(1) * (n(0) * y_ - n(1) * x_) / S;
        float C23 = -sig2 * n(0) * (n(0) * y_ - n(1) * x_) / S;
        float C33 = corr * sig2 * sqr(1 / nop + n(0) * y_ - n(1) * x_) / S;
        Matrix3f C;
        C << sig2 * n1_2 / S, -sig2 * n(0) * n(1) / S, C13, -sig2 * n(0) * n(1) / S, sig2 * n0_2
                / S, C23, C13, C23, C33;

        Matrix<float, 2, 3> J;
        J << -1.f / n(1), n(0) / n1_2, 0, -c * n(0) / (n(1) * sqrt_), n0_2 * c / (n1_2 * sqrt_), -sqrt_
                / n(1);

        line.cov = J * C * J.transpose();
    }

    return line;
}

helix_fit Helix_fit(Matrix<float, 3, Dynamic, 0, 3, max_nop> hits,
        Matrix<float, Dynamic, Dynamic, 0, 3 * max_nop, 3 * max_nop> hits_cov, float B_field,
        bool return_err = true)
{
    int nop = hits.cols();
    circle_fit circle = Circle_fit(hits.block(0, 0, 2, nop), hits_cov.block(0, 0, 2 * nop, 2 * nop),
            B_field, return_err);
    line_fit line = Line_fit(hits, hits_cov, circle, B_field, return_err);
    helix_fit helix;
    helix.par << circle.par, line.par;
    helix.cov = MatrixXf::Zero(5, 5);
    if (return_err)
    {
        helix.cov.block(0, 0, 3, 3) = circle.cov;
        helix.cov.block(3, 3, 2, 2) = line.cov;
    }
    helix.charge = circle.charge;
    helix.chi2 = circle.chi2 + line.chi2;
    return helix;
}

}

PixelFitterByRiemannParaboloid::PixelFitterByRiemannParaboloid(const edm::EventSetup *es,
        const MagneticField *field) :
        theES(es), theField(field)
{
}

std::unique_ptr<reco::Track> PixelFitterByRiemannParaboloid::run(
        const std::vector<const TrackingRecHit *> & hits, const TrackingRegion & region) const
{
    std::unique_ptr < reco::Track > ret;

    unsigned int nhits = hits.size();

    std::cout << "nhits " << nhits << std::endl;
    if (nhits < 2)
        return ret;

    declareDynArray(GlobalPoint, nhits, points);
    declareDynArray(GlobalError, nhits, errors);
    declareDynArray(bool,nhits, isBarrel);

    for (unsigned int i = 0; i != nhits; ++i)
    {
        auto const & recHit = hits[i];
        points[i] = GlobalPoint(
                recHit->globalPosition().basicVector() - region.origin().basicVector());
        errors[i] = recHit->globalPositionError();
        isBarrel[i] = recHit->detUnit()->type().isBarrel();
    }

    Matrix<float, 3, Dynamic, 0, 3, max_nop> riemannHits(3, nhits);

    Matrix<float, Dynamic, Dynamic, 0, 3 * max_nop, 3 * max_nop> riemannHits_cov = MatrixXf::Zero(3*nhits, 3*nhits);


    for (unsigned int i = 0; i < nhits; ++i)
    {
        riemannHits.col(i) << points[i].x(), points[i].y(), points[i].z();

        const auto& errorMatrix = errors[i].matrix4D();

        for (auto j = 0; j < 3; ++j)
        {
            for (auto l = 0; l < 3; ++l)
            {
                riemannHits_cov(i + j * nhits, i + l * nhits) = errorMatrix(j, l);

                std::cout << "printing error matrix for i j l " << i << " " << j << " " << l << std::endl;

                std::cout << errorMatrix(j, l) << std::endl;

            }
        }
    }


    std::cout << riemannHits_cov << std::endl;
    float bField = 1/PixelRecoUtilities::fieldInInvGev(*theES);
    helix_fit fittedTrack = Helix_fit(riemannHits, riemannHits_cov, bField, true);
    int iCharge = fittedTrack.charge;

    // parameters are:
    // 0: phi
    // 1: tip
    // 2: curvature
    // 3: cottheta
    // 4: zip
    float valPhi = fittedTrack.par(0);

    float valTip = fittedTrack.par(1);

    float valCotTheta = fittedTrack.par(3);

    float valZip = fittedTrack.par(4);
    float valPt = fittedTrack.par(2);
//
//  PixelTrackErrorParam param(valEta, valPt);
    float errValPhi = std::sqrt(fittedTrack.cov(0, 0));
    float errValTip = std::sqrt(fittedTrack.cov(1, 1));

    float errValPt = std::sqrt(fittedTrack.cov(2, 2));

    float errValCotTheta = std::sqrt(fittedTrack.cov(3, 3));
    float errValZip = std::sqrt(fittedTrack.cov(4, 4));


    float chi2 = fittedTrack.chi2;
//  if (nhits > 2) {
//    RZLine rzLine(points,errors,isBarrel);
//    chi2 = rzLine.chi2();
//  }

    PixelTrackBuilder builder;
    Measurement1D phi(valPhi, errValPhi);
    Measurement1D tip(valTip, errValTip);

    Measurement1D pt(valPt, errValPt);
    Measurement1D cotTheta(valCotTheta, errValCotTheta);
    Measurement1D zip(valZip, errValZip);
//    std::cout << "found one track with " << valPhi << " " << valTip << " " << valPt << " "
//            << valCotTheta << " " << valZip << " " << chi2 << std::endl;

    ret.reset(
            builder.build(pt, phi, cotTheta, tip, zip, chi2, iCharge, hits, theField,
                    region.origin()));
    return ret;
}

