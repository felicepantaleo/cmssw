#ifndef SoA_Layout_Test_h
#define SoA_Layout_Test_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"
#include <iostream>

GENERATE_SOA_LAYOUT(PhysicsObjTemplate,
                    SOA_COLUMN(double, x),
                    SOA_COLUMN(double, y),
                    SOA_COLUMN(double, z),
                    SOA_SCALAR(int, detectorType))

using PhysicsObj = PhysicsObjTemplate<>;
using PhysicsObjView = PhysicsObj::View;

GENERATE_SOA_LAYOUT(PhysicsObjExtraTemplate,
                    SOA_COLUMN(double, eigenvalues),
                    SOA_COLUMN(double, eigenvector_1),
                    SOA_COLUMN(double, eigenvector_2),
                    SOA_COLUMN(double, eigenvector_3),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))
                    
using PhysicsObjExtra = PhysicsObjExtraTemplate<>;    
using PhysicsObjExtraView = PhysicsObjExtra::View;       

GENERATE_SOA_LAYOUT(CombinedPhysicsObjectTemplate,
                    SOA_COLUMN(double, x),
                    SOA_COLUMN(double, y),
                    SOA_COLUMN(double, z),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using CombinedPhysicsObject = CombinedPhysicsObjectTemplate<>;
using CombinedPhysicsObjectView = CombinedPhysicsObject::View;

GENERATE_SOA_LAYOUT(NGTSoATemplate,
                    SOA_COLUMN(double, a),
                    SOA_COLUMN(double, b),
                    SOA_COLUMN(double, c),
                    SOA_SCALAR(int, d))

using NGTSoA = NGTSoATemplate<>;
using NGTSoAView = NGTSoA::View;

// Helper per verificare la presenza di un metodo
#define HAS_METHOD(method) \
template <typename T> \
class has_##method { \
private: \
    template <typename U> static auto test(int) -> decltype(std::declval<U>().method(), std::true_type()); \
    template <typename> static std::false_type test(...); \
public: \
    static constexpr bool value = std::is_same<decltype(test<T>(0)), std::true_type>::value; \
}; \

HAS_METHOD(metadata)
HAS_METHOD(x)
HAS_METHOD(y)
HAS_METHOD(z)
HAS_METHOD(candidateDirection)
HAS_METHOD(eigenvalues)
HAS_METHOD(eigenvector_1)
HAS_METHOD(eigenvector_2)
HAS_METHOD(eigenvector_3)
HAS_METHOD(detectorType)


template <typename View>
inline void printSoAView(View view) {
    std::cout << "SoAView:" << std::endl;

    if constexpr (has_detectorType<View>::value) {
        std::cout << "detectorType: " << view.detectorType() << std::endl;
    }

    if constexpr (has_metadata<View>::value) {
        for (auto i = 0; i < view.metadata().size(); ++i) {
            std::cout << "Element " << i << ": ";

            if constexpr (has_x<View>::value) {
                std::cout << "x = " << view.x()[i] << ", ";
            }

            if constexpr (has_y<View>::value) {
                std::cout << "y = " << view.y()[i] << ", ";
            }

            if constexpr (has_z<View>::value) {
                std::cout << "z = " << view.z()[i] << std::endl;
            }

            if constexpr (has_eigenvalues<View>::value) {
                std::cout << "eigenvalues = " << view.eigenvalues()[i] << ", ";
            }

            if constexpr (has_eigenvector_1<View>::value) {
                std::cout << "eigenvector_1 = " << view.eigenvector_1()[i] << ", ";
            }

            if constexpr (has_eigenvector_2<View>::value) {
                std::cout << "eigenvector_2 = " << view.eigenvector_2()[i] << std::endl;
            }

            if constexpr (has_eigenvector_3<View>::value) {
                std::cout << "eigenvector_3 = " << view.eigenvector_3()[i] << std::endl;
            }
            
            if constexpr (has_candidateDirection<View>::value) {
                std::cout << "candidateDirection = " << view[i].candidateDirection().transpose() << std::endl;
            }
        }
    }
}

#endif
