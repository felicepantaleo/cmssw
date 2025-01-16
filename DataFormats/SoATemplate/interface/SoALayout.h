#ifndef DataFormats_SoATemplate_interface_SoALayout_h
#define DataFormats_SoATemplate_interface_SoALayout_h

/*
 * Structure-of-Arrays template with "columns" and "scalars", defined through preprocessor macros,
 * with compile-time size and alignment, and accessors to the "rows" and "columns".
 */

#include <cassert>

#include "FWCore/Reflection/interface/reflex.h"

#include "SoACommon.h"
#include "SoAView.h"

/* dump SoA fields information; these should expand to, for columns:
 * Example:
 * GENERATE_SOA_LAYOUT(SoA,
 *   // predefined static scalars
 *   // size_t size;
 *   // size_t alignment;
 *
 *   // columns: one value per element
 *   SOA_COLUMN(double, x),
 *   SOA_COLUMN(double, y),
 *   SOA_COLUMN(double, z),
 *   SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
 *   SOA_EIGEN_COLUMN(Eigen::Vector3d, b),
 *   SOA_EIGEN_COLUMN(Eigen::Vector3d, r),
 *   SOA_COLUMN(uint16_t, colour),
 *   SOA_COLUMN(int32_t, value),
 *   SOA_COLUMN(double *, py),
 *   SOA_COLUMN(uint32_t, count),
 *   SOA_COLUMN(uint32_t, anotherCount),
 *
 *   // scalars: one value for the whole structure
 *   SOA_SCALAR(const char *, description), 
 *   SOA_SCALAR(uint32_t, someNumber)
 * );
 *
 * dumps as:
 * SoA(32, 64):
 *   sizeof(SoA): 152
 *  Column x_ at offset 0 has size 256 and padding 0
 *  Column y_ at offset 256 has size 256 and padding 0
 *  Column z_ at offset 512 has size 256 and padding 0
 *  Eigen value a_ at offset 768 has dimension (3 x 1) and per column size 256 and padding 0
 *  Eigen value b_ at offset 1536 has dimension (3 x 1) and per column size 256 and padding 0
 *  Eigen value r_ at offset 2304 has dimension (3 x 1) and per column size 256 and padding 0
 *  Column colour_ at offset 3072 has size 64 and padding 0
 *  Column value_ at offset 3136 has size 128 and padding 0
 *  Column py_ at offset 3264 has size 256 and padding 0
 *  Column count_ at offset 3520 has size 128 and padding 0
 *  Column anotherCount_ at offset 3648 has size 128 and padding 0
 *  Scalar description_ at offset 3776 has size 8 and padding 56
 *  Scalar someNumber_ at offset 3840 has size 4 and padding 60
 * Final offset = 3904 computeDataSize(...): 3904
 *
 */

// clang-format off
#define _DECLARE_SOA_STREAM_INFO_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                      \
  _SWITCH_ON_TYPE(                                                                                                     \
      VALUE_TYPE,                                                                                                      \
      /* Dump scalar */                                                                                                \
      _soa_impl_os << " Scalar " BOOST_PP_STRINGIZE(NAME) " at offset " << _soa_impl_offset                            \
         << " has size " << sizeof(CPP_TYPE)                                                                           \
         << " and padding " << ((sizeof(CPP_TYPE) - 1) / alignment + 1) * alignment - sizeof(CPP_TYPE)                 \
         << std::endl;                                                                                                 \
      _soa_impl_offset += ((sizeof(CPP_TYPE) - 1) / alignment + 1) * alignment;                                        \
      ,                                                                                                                \
      /* Dump column */                                                                                                \
      _soa_impl_os << " Column " BOOST_PP_STRINGIZE(NAME) " at offset " << _soa_impl_offset << " has size "            \
         << sizeof(CPP_TYPE) * elements_ << " and padding "                                                            \
         << cms::soa::alignSize(elements_ * sizeof(CPP_TYPE), alignment) - (elements_ * sizeof(CPP_TYPE))              \
         << std::endl;                                                                                                 \
      _soa_impl_offset += cms::soa::alignSize(elements_ * sizeof(CPP_TYPE), alignment);                                \
      ,                                                                                                                \
      /* Dump Eigen column */                                                                                          \
      _soa_impl_os << " Eigen value " BOOST_PP_STRINGIZE(NAME) " at offset " << _soa_impl_offset << " has dimension "  \
         << "(" << CPP_TYPE::RowsAtCompileTime << " x " << CPP_TYPE::ColsAtCompileTime << ")"                          \
         << " and per column size "                                                                                    \
         << sizeof(CPP_TYPE::Scalar) * elements_                                                                       \
         << " and padding "                                                                                            \
         << cms::soa::alignSize(elements_ * sizeof(CPP_TYPE::Scalar), alignment)                                       \
            - (elements_ * sizeof(CPP_TYPE::Scalar))                                                                   \
         << std::endl;                                                                                                 \
      _soa_impl_offset += cms::soa::alignSize(elements_ * sizeof(CPP_TYPE::Scalar), alignment)                         \
                * CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                           \
  )
// clang-format on

#define _DECLARE_SOA_STREAM_INFO(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_SOA_STREAM_INFO_IMPL TYPE_NAME)

/**
 * Metadata member computing column pitch
 */
// clang-format off
#define _DEFINE_METADATA_MEMBERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                      \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      byte_size_type BOOST_PP_CAT(NAME, Pitch()) const {                                                               \
        return cms::soa::alignSize(sizeof(CPP_TYPE), ParentClass::alignment);                                          \
      }                                                                                                                \
      using BOOST_PP_CAT(TypeOf_, NAME) = CPP_TYPE;                                                                    \
      constexpr static cms::soa::SoAColumnType BOOST_PP_CAT(ColumnTypeOf_, NAME) = cms::soa::SoAColumnType::scalar;    \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      CPP_TYPE const* BOOST_PP_CAT(addressOf_, NAME)() const {                                                         \
        return parent_.metadata().BOOST_PP_CAT(parametersOf_, NAME)().addr_;                                           \
      }                                                                                                                \
      using BOOST_PP_CAT(ParametersTypeOf_, NAME) =                                                                    \
        cms::soa::SoAParameters_ColumnType<cms::soa::SoAColumnType::scalar>::DataType<CPP_TYPE>;                       \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      BOOST_PP_CAT(ParametersTypeOf_, NAME) BOOST_PP_CAT(parametersOf_, NAME)() const {                                \
        return  BOOST_PP_CAT(ParametersTypeOf_, NAME) (parent_.BOOST_PP_CAT(NAME, _), parent_.metadata().size());      \
      }                                                                                                                \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      CPP_TYPE* BOOST_PP_CAT(addressOf_, NAME)() {                                                                     \
        return parent_.metadata().BOOST_PP_CAT(parametersOf_, NAME)().addr_;                                           \
      },                                                                                                               \
      /* Column */                                                                                                     \
      using BOOST_PP_CAT(ParametersTypeOf_, NAME) =                                                                    \
         cms::soa::SoAParameters_ColumnType<cms::soa::SoAColumnType::column>::DataType<CPP_TYPE>;                      \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      BOOST_PP_CAT(ParametersTypeOf_, NAME) BOOST_PP_CAT(parametersOf_, NAME)() const {                                \
        return  BOOST_PP_CAT(ParametersTypeOf_, NAME) (parent_.BOOST_PP_CAT(NAME, _), parent_.metadata().size());      \
      }                                                                                                                \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      CPP_TYPE const* BOOST_PP_CAT(addressOf_, NAME)() const {                                                         \
        return parent_.metadata().BOOST_PP_CAT(parametersOf_, NAME)().addr_;                                           \
      }                                                                                                                \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      CPP_TYPE* BOOST_PP_CAT(addressOf_, NAME)() {                                                                     \
        return parent_.metadata().BOOST_PP_CAT(parametersOf_, NAME)().addr_;                                           \
      }                                                                                                                \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      byte_size_type BOOST_PP_CAT(NAME, Pitch()) const {                                                               \
        return cms::soa::alignSize(parent_.elements_ * sizeof(CPP_TYPE), ParentClass::alignment);                      \
      }                                                                                                                \
      using BOOST_PP_CAT(TypeOf_, NAME) = CPP_TYPE;                                                                    \
      constexpr static cms::soa::SoAColumnType BOOST_PP_CAT(ColumnTypeOf_, NAME) = cms::soa::SoAColumnType::column;,   \
      /* Eigen column */                                                                                               \
      using BOOST_PP_CAT(ParametersTypeOf_, NAME) =                                                                    \
          cms::soa::SoAParameters_ColumnType<cms::soa::SoAColumnType::eigen>::DataType<CPP_TYPE>;                      \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      BOOST_PP_CAT(ParametersTypeOf_, NAME) BOOST_PP_CAT(parametersOf_, NAME)() const {                                \
        return BOOST_PP_CAT(ParametersTypeOf_, NAME) (                                                                 \
          parent_.BOOST_PP_CAT(NAME, _),                                                                               \
          parent_.BOOST_PP_CAT(NAME, Stride_),                                                                         \
          parent_.metadata().size());                                                                                  \
      }                                                                                                                \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      byte_size_type BOOST_PP_CAT(NAME, Pitch()) const {                                                               \
        return cms::soa::alignSize(parent_.elements_ * sizeof(CPP_TYPE::Scalar), ParentClass::alignment)               \
               * CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                            \
      }                                                                                                                \
      using BOOST_PP_CAT(TypeOf_, NAME) = CPP_TYPE ;                                                                   \
      constexpr static cms::soa::SoAColumnType BOOST_PP_CAT(ColumnTypeOf_, NAME) = cms::soa::SoAColumnType::eigen;     \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      CPP_TYPE::Scalar const* BOOST_PP_CAT(addressOf_, NAME)() const {                                                 \
        return parent_.metadata().BOOST_PP_CAT(parametersOf_, NAME)().addr_;                                           \
      }                                                                                                                \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      CPP_TYPE::Scalar* BOOST_PP_CAT(addressOf_, NAME)() {                                                             \
        return parent_.metadata().BOOST_PP_CAT(parametersOf_, NAME)().addr_;                                           \
      }                                                                                                                \
)
// clang-format on
#define _DEFINE_METADATA_MEMBERS(R, DATA, TYPE_NAME) _DEFINE_METADATA_MEMBERS_IMPL TYPE_NAME

/**
 * Functions for retreving pointers to layout columns
 */
// clang-format off
#define _DEFINE_SET_COLUMN_FUNCTIONS_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                  \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      void BOOST_PP_CAT(setColumn_, NAME)(CPP_TYPE& newAddr) {                                                         \
          BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE*>(&newAddr);                                               \
      }                                                                                                                \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      void BOOST_PP_CAT(setColumn_, NAME)(CPP_TYPE* newAddr) {                                                         \
          BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE*>(newAddr);                                                \
      }                                                                                                                \
      void BOOST_PP_CAT(setColumn_, NAME)(const CPP_TYPE* newAddr) {                                                   \
          BOOST_PP_CAT(NAME, _) = const_cast<CPP_TYPE*>(newAddr);                                                      \
      }                                                                                                                \
      ,                                                                                                                \
      /* Eigen */                                                                                                      \
      void BOOST_PP_CAT(setColumn_, NAME)(CPP_TYPE::Scalar* newAddr) {                                                 \
          BOOST_PP_CAT(NAME, Stride_) = cms::soa::alignSize(elements_ * sizeof(CPP_TYPE::Scalar), alignment)           \
            / sizeof(CPP_TYPE::Scalar);                                                                                \
          BOOST_PP_CAT(NAME, ElementsWithPadding_) = BOOST_PP_CAT(NAME, Stride_)                                       \
            *  CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                              \
          BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE::Scalar*>(newAddr);                                        \
      }                                                                                                                \
  )   
// clang-format on    

#define _DEFINE_SET_COLUMN_FUNCTIONS(R, DATA, TYPE_NAME) _DEFINE_SET_COLUMN_FUNCTIONS_IMPL TYPE_NAME

// clang-format off
#define _DECLARE_MEMBER_TRIVIAL_CONSTRUCTION_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                          \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      (BOOST_PP_CAT(NAME, _)(nullptr)),                                                                                \
      /* Column */                                                                                                     \
      (BOOST_PP_CAT(NAME, _)(nullptr)),                                                                                \
      /* Eigen column */                                                                                               \
      (BOOST_PP_CAT(NAME, Stride_)(0))                                                                                 \
      (BOOST_PP_CAT(NAME, ElementsWithPadding_)(0))                                                                    \
      (BOOST_PP_CAT(NAME, _)(nullptr))                                                                                 \
  )
// clang-format on

#define _DECLARE_MEMBER_TRIVIAL_CONSTRUCTION(R, DATA, TYPE_NAME) \
  BOOST_PP_EXPAND(_DECLARE_MEMBER_TRIVIAL_CONSTRUCTION_IMPL TYPE_NAME)

// clang-format off
#define _DECLARE_MEMBER_COPY_CONSTRUCTION_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                             \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      (BOOST_PP_CAT(NAME, _){_soa_impl_other.BOOST_PP_CAT(NAME, _)}),                                                  \
      /* Column */                                                                                                     \
      (BOOST_PP_CAT(NAME, _){_soa_impl_other.BOOST_PP_CAT(NAME, _)}),                                                  \
      /* Eigen column */                                                                                               \
      (BOOST_PP_CAT(NAME, Stride_){_soa_impl_other.BOOST_PP_CAT(NAME, Stride_)})                                       \
      (BOOST_PP_CAT(NAME, ElementsWithPadding_){_soa_impl_other.BOOST_PP_CAT(NAME, ElementsWithPadding_)})             \
      (BOOST_PP_CAT(NAME, _){_soa_impl_other.BOOST_PP_CAT(NAME, _)})                                                   \
  )
// clang-format on

#define _DECLARE_MEMBER_COPY_CONSTRUCTION(R, DATA, TYPE_NAME) \
  BOOST_PP_EXPAND(_DECLARE_MEMBER_COPY_CONSTRUCTION_IMPL TYPE_NAME)

// clang-format off
#define _DECLARE_MEMBER_ASSIGNMENT_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                    \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      BOOST_PP_CAT(NAME, _) = _soa_impl_other.BOOST_PP_CAT(NAME, _);,                                                  \
      /* Column */                                                                                                     \
      BOOST_PP_CAT(NAME, _) = _soa_impl_other.BOOST_PP_CAT(NAME, _);,                                                  \
      /* Eigen column */                                                                                               \
      BOOST_PP_CAT(NAME, ElementsWithPadding_) = _soa_impl_other.BOOST_PP_CAT(NAME, ElementsWithPadding_);             \
      BOOST_PP_CAT(NAME, _) = _soa_impl_other.BOOST_PP_CAT(NAME, _);                                                   \
      BOOST_PP_CAT(NAME, Stride_) = _soa_impl_other.BOOST_PP_CAT(NAME, Stride_);                                       \
  )
// clang-format on

#define _DECLARE_MEMBER_ASSIGNMENT(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_MEMBER_ASSIGNMENT_IMPL TYPE_NAME)

/**
 * Declare the value_element data members
 */
// clang-format off
#define _DEFINE_VALUE_ELEMENT_MEMBERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                 \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar (empty) */                                                                                             \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      CPP_TYPE NAME;                                                                                                   \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      CPP_TYPE NAME;                                                                                                   \
  )
// clang-format on

#define _DEFINE_VALUE_ELEMENT_MEMBERS(R, DATA, TYPE_NAME) _DEFINE_VALUE_ELEMENT_MEMBERS_IMPL TYPE_NAME

/**
 * List of data members in the value_element constructor arguments
 */
// clang-format off
#define _VALUE_ELEMENT_CTOR_ARGS_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                      \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar (empty) */                                                                                             \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      (CPP_TYPE NAME),                                                                                                 \
      /* Eigen column */                                                                                               \
      (CPP_TYPE NAME)                                                                                                  \
  )
// clang-format on

#define _VALUE_ELEMENT_CTOR_ARGS(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_VALUE_ELEMENT_CTOR_ARGS_IMPL TYPE_NAME)

/**
 * List-initalise the value_element data members
 */
// clang-format off
#define _VALUE_ELEMENT_INITIALIZERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                   \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar (empty) */                                                                                             \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      (NAME{NAME}),                                                                                                    \
      /* Eigen column */                                                                                               \
      (NAME{NAME})                                                                                                     \
  )
// clang-format on

#define _VALUE_ELEMENT_INITIALIZERS(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_VALUE_ELEMENT_INITIALIZERS_IMPL TYPE_NAME)

// clang-format off
#define _STRUCT_ELEMENT_INITIALIZERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                  \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar (empty) */                                                                                             \
      (BOOST_PP_CAT(NAME, _){parent_.metadata().BOOST_PP_CAT(parametersOf_, NAME)()}),                                 \
      /* Column */                                                                                                     \
      (BOOST_PP_CAT(NAME, _){parent_.metadata().BOOST_PP_CAT(parametersOf_, NAME)()}),                                 \
      /* Eigen column */                                                                                               \
      (BOOST_PP_CAT(NAME, _){parent_.metadata().BOOST_PP_CAT(parametersOf_, NAME)()})                                  \
  )
// clang-format on

#define _STRUCT_ELEMENT_INITIALIZERS(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_STRUCT_ELEMENT_INITIALIZERS_IMPL TYPE_NAME)


/**
 * Freeing of the ROOT-allocated column or scalar buffer
 */
// clang-format off
#define _ROOT_FREE_SOA_COLUMN_OR_SCALAR_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                               \
  delete[] BOOST_PP_CAT(NAME, _);                                                                                      \
  BOOST_PP_CAT(NAME, _) = nullptr;                                                                                     \
  // clang-format on

#define _ROOT_FREE_SOA_COLUMN_OR_SCALAR(R, DATA, TYPE_NAME) _ROOT_FREE_SOA_COLUMN_OR_SCALAR_IMPL TYPE_NAME

/**
 * Computation of the column or scalar pointer location in the memory layout (at SoA construction time)
 */
// clang-format off
#define _ASSIGN_SOA_COLUMN_OR_SCALAR_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                  \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE*>(_soa_impl_curMem);                                           \
      _soa_impl_curMem += cms::soa::alignSize(sizeof(CPP_TYPE), alignment);                                            \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE*>(_soa_impl_curMem);                                           \
      _soa_impl_curMem += cms::soa::alignSize(elements_ * sizeof(CPP_TYPE), alignment);                                \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      BOOST_PP_CAT(NAME, Stride_) = cms::soa::alignSize(elements_ * sizeof(CPP_TYPE::Scalar), alignment)               \
        / sizeof(CPP_TYPE::Scalar);                                                                                    \
      BOOST_PP_CAT(NAME, ElementsWithPadding_) = BOOST_PP_CAT(NAME, Stride_)                                           \
        *  CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                                  \
      BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE::Scalar*>(_soa_impl_curMem);                                   \
      _soa_impl_curMem += cms::soa::alignSize(elements_ * sizeof(CPP_TYPE::Scalar), alignment)                         \
        * CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                                   \
  )                                                                                                                    \
  if constexpr (alignmentEnforcement == AlignmentEnforcement::enforced)                                                \
    if (reinterpret_cast<intptr_t>(BOOST_PP_CAT(NAME, _)) % alignment)                                                 \
      throw std::runtime_error("In layout constructor: misaligned column: " #NAME);
// clang-format on
 
#define _ASSIGN_SOA_COLUMN_OR_SCALAR(R, DATA, TYPE_NAME) _ASSIGN_SOA_COLUMN_OR_SCALAR_IMPL TYPE_NAME

#define _INITIALIZE_MEM_ADDRESS_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                       \
  if(mem_start) {                                                                                                      \
    mem_ = BOOST_PP_CAT(custom_view.metadata().addressOf_, NAME)();                                                    \
    mem_start = false;                                                                                                 \
  }                                                                                                                    \

#define _INITIALIZE_MEM_ADDRESS(R, DATA, TYPE_NAME) _INITIALIZE_MEM_ADDRESS_IMPL TYPE_NAME

#define  _ASSIGN_SOA_COLUMN_OR_SCALAR_FROM_VIEW_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                     \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
    /* Scalar */                                                                                                       \
      BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE*>(_soa_impl_curMem);                                           \
      _soa_impl_curMem += cms::soa::alignSize(sizeof(CPP_TYPE), alignment);                                            \
    memcpy(BOOST_PP_CAT(NAME, _), BOOST_PP_CAT(custom_view.metadata().addressOf_, NAME)(), cms::soa::alignSize(sizeof(CPP_TYPE), alignment));                                                      \
    ,                                                                                                                \
    /* Column */                                                                                                     \
      BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE*>(_soa_impl_curMem);                                           \
      _soa_impl_curMem += cms::soa::alignSize(sizeof(CPP_TYPE), alignment);                                            \
    memcpy(BOOST_PP_CAT(NAME, _), BOOST_PP_CAT(custom_view.metadata().addressOf_, NAME)(), cms::soa::alignSize(elements_ * sizeof(CPP_TYPE), alignment));                                          \
    ,                                                                                                                \
    /* Eigen column */                                                                                               \
      BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE::Scalar*>(_soa_impl_curMem);                                   \
      _soa_impl_curMem += cms::soa::alignSize(elements_ * sizeof(CPP_TYPE::Scalar), alignment)                         \
        * CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                                   \
    BOOST_PP_CAT(NAME, Stride_) = cms::soa::alignSize(elements_ * sizeof(CPP_TYPE::Scalar), alignment)               \
      / sizeof(CPP_TYPE::Scalar);                                                                                    \
    BOOST_PP_CAT(NAME, ElementsWithPadding_) = BOOST_PP_CAT(NAME, Stride_)                                           \
      *  CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                                  \
    memcpy(BOOST_PP_CAT(NAME, _), BOOST_PP_CAT(custom_view.metadata().addressOf_, NAME)(), cms::soa::alignSize(elements_ * sizeof(CPP_TYPE::Scalar), alignment)                                    \
                                                      * CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime);  \
  )                                                                                                                    \

#define _ASSIGN_SOA_COLUMN_OR_SCALAR_FROM_VIEW(R, DATA, TYPE_NAME) _ASSIGN_SOA_COLUMN_OR_SCALAR_FROM_VIEW_IMPL TYPE_NAME

// clang-format off
#define _AGGREGATE_SOA_COLUMNS_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                        \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      memcpy(_soa_impl_curMem, BOOST_PP_CAT(NAME, _), sizeof(CPP_TYPE));                                               \
      BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE*>(_soa_impl_curMem);                                           \
      _soa_impl_curMem += cms::soa::alignSize(sizeof(CPP_TYPE), alignment);                                            \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      memcpy(_soa_impl_curMem, BOOST_PP_CAT(NAME, _), elements_ * sizeof(CPP_TYPE));                                   \
      BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE*>(_soa_impl_curMem);                                           \
      _soa_impl_curMem += cms::soa::alignSize(elements_ * sizeof(CPP_TYPE), alignment);                                \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      memcpy(_soa_impl_curMem, BOOST_PP_CAT(NAME, _), cms::soa::alignSize(elements_ * sizeof(CPP_TYPE::Scalar), alignment)                             \
                                                        * CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime);  \
      BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE::Scalar*>(_soa_impl_curMem);                                   \
      _soa_impl_curMem += cms::soa::alignSize(elements_ * sizeof(CPP_TYPE::Scalar), alignment)                         \
        * CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                                   \
  )                                                                                                                    \
  if constexpr (alignmentEnforcement == AlignmentEnforcement::enforced)                                                \
    if (reinterpret_cast<intptr_t>(BOOST_PP_CAT(NAME, _)) % alignment)                                                 \
      throw std::runtime_error("In layout constructor: misaligned column: " #NAME);
// clang-format on

#define _AGGREGATE_SOA_COLUMNS(R, DATA, TYPE_NAME) _AGGREGATE_SOA_COLUMNS_IMPL TYPE_NAME

/**
 * Computation of the column or scalar size for SoA size computation
 */
// clang-format off
#define _ACCUMULATE_SOA_ELEMENT_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                       \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      _soa_impl_ret += cms::soa::alignSize(sizeof(CPP_TYPE), alignment);                                               \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      _soa_impl_ret += cms::soa::alignSize(elements * sizeof(CPP_TYPE), alignment);                                    \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      _soa_impl_ret += cms::soa::alignSize(elements * sizeof(CPP_TYPE::Scalar), alignment)                             \
             * CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                              \
  )
// clang-format on

#define _ACCUMULATE_SOA_ELEMENT(R, DATA, TYPE_NAME) _ACCUMULATE_SOA_ELEMENT_IMPL TYPE_NAME

/**
 * Direct access to column pointer and indexed access
 */
// clang-format off
#define _DECLARE_SOA_ACCESSOR_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                         \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      SOA_HOST_DEVICE SOA_INLINE CPP_TYPE& NAME() { return *BOOST_PP_CAT(NAME, _); }                                   \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      SOA_HOST_DEVICE SOA_INLINE CPP_TYPE* NAME() { return BOOST_PP_CAT(NAME, _); }                                    \
      SOA_HOST_DEVICE SOA_INLINE CPP_TYPE& NAME(size_type _soa_impl_index) {                                           \
        return BOOST_PP_CAT(NAME, _)[_soa_impl_index];                                                                 \
      }                                                                                                                \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      /* TODO: implement*/                                                                                             \
      BOOST_PP_EMPTY()                                                                                                 \
  )
// clang-format on

#define _DECLARE_SOA_ACCESSOR(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_SOA_ACCESSOR_IMPL TYPE_NAME)

/**
 * Direct access to column pointer (const) and indexed access.
 */
// clang-format off
#define _DECLARE_SOA_CONST_ACCESSOR_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                   \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      SOA_HOST_DEVICE SOA_INLINE CPP_TYPE NAME() const { return *(BOOST_PP_CAT(NAME, _)); }                            \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      SOA_HOST_DEVICE SOA_INLINE CPP_TYPE const* NAME() const { return BOOST_PP_CAT(NAME, _); }                        \
      SOA_HOST_DEVICE SOA_INLINE CPP_TYPE NAME(size_type _soa_impl_index) const {                                      \
        return *(BOOST_PP_CAT(NAME, _) + _soa_impl_index);                                                             \
      }                                                                                                                \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      SOA_HOST_DEVICE SOA_INLINE CPP_TYPE::Scalar const* NAME() const { return BOOST_PP_CAT(NAME, _); }                \
      SOA_HOST_DEVICE SOA_INLINE size_type BOOST_PP_CAT(NAME, Stride)() { return BOOST_PP_CAT(NAME, Stride_); }        \
  )
// clang-format on

#define _DECLARE_SOA_CONST_ACCESSOR(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_SOA_CONST_ACCESSOR_IMPL TYPE_NAME)

/**
 * SoA member ROOT streamer read (column pointers).
 */
// clang-format off
#define _STREAMER_READ_SOA_DATA_MEMBER_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      memcpy(BOOST_PP_CAT(NAME, _), onfile.BOOST_PP_CAT(NAME, _), sizeof(CPP_TYPE));                                   \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      memcpy(BOOST_PP_CAT(NAME, _), onfile.BOOST_PP_CAT(NAME, _), sizeof(CPP_TYPE) * onfile.elements_);                \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      memcpy(BOOST_PP_CAT(NAME, _), onfile.BOOST_PP_CAT(NAME, _),                                                      \
        sizeof(CPP_TYPE::Scalar) * BOOST_PP_CAT(NAME, ElementsWithPadding_));                                          \
  )
// clang-format on

#define _STREAMER_READ_SOA_DATA_MEMBER(R, DATA, TYPE_NAME) \
  BOOST_PP_EXPAND(_STREAMER_READ_SOA_DATA_MEMBER_IMPL TYPE_NAME)

/**
 * SoA class member declaration (column pointers).
 */
// clang-format off
#define _DECLARE_SOA_DATA_MEMBER_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                      \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      CPP_TYPE* BOOST_PP_CAT(NAME, _) EDM_REFLEX_SIZE(scalar_) = nullptr;                                              \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      CPP_TYPE * BOOST_PP_CAT(NAME, _) EDM_REFLEX_SIZE(elements_) = nullptr;                                           \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      byte_size_type BOOST_PP_CAT(NAME, Stride_) = 0;                                                                  \
      size_type BOOST_PP_CAT(NAME, ElementsWithPadding_) = 0; /* For ROOT serialization */                             \
      CPP_TYPE::Scalar * BOOST_PP_CAT(NAME, _) EDM_REFLEX_SIZE(BOOST_PP_CAT(NAME, ElementsWithPadding_)) = nullptr;    \
  )
// clang-format on

#define _DECLARE_SOA_DATA_MEMBER(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_SOA_DATA_MEMBER_IMPL TYPE_NAME)

// clang-format off
#define _ACCESSORS_STRUCT_MEMBERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                     \
    const typename Metadata::BOOST_PP_CAT(ParametersTypeOf_, NAME)& NAME() const { return BOOST_PP_CAT(NAME, _); }
// clang-format on

#define _ACCESSORS_STRUCT_MEMBERS(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_ACCESSORS_STRUCT_MEMBERS_IMPL TYPE_NAME)

// clang-format off
#define _DECLARE_STRUCT_DATA_MEMBER_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                   \
    typename Metadata::BOOST_PP_CAT(ParametersTypeOf_, NAME) BOOST_PP_CAT(NAME, _);
// clang-format on

#define _DECLARE_STRUCT_DATA_MEMBER(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_STRUCT_DATA_MEMBER_IMPL TYPE_NAME)

// clang-format off
#define _DECLARE_STRUCT_MEMBERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                       \
    typename Metadata::BOOST_PP_CAT(ParametersTypeOf_, NAME) NAME;
// clang-format on

#define _DECLARE_STRUCT_MEMBERS(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_STRUCT_MEMBERS_IMPL TYPE_NAME)

#define _COUNT_SOA_COLUMNS(R, DATA, TYPE_NAME)  _soa_column_count++;

/**
 * Array of string column names
 */
// clang-format off
#define _DECLARE_SOA_COLUMNS_NAMES_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                    \
  _columns_names[i++] = BOOST_PP_STRINGIZE(NAME);                                                                      \
// clang-format on 

#define _DECLARE_SOA_COLUMNS_NAMES(R, DATA, NAME) BOOST_PP_EXPAND(_DECLARE_SOA_COLUMNS_NAMES_IMPL  NAME)

/**
 * List of data members in the layout-by-columns constructor arguments
 */
// clang-format off
#define _DECLARE_CONSTRUCTOR_ARGUMENT_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                 \
    _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                        \
        /* Scalar */                                                                                                   \
        (CPP_TYPE& NAME),                                                                                              \
        /* Column */                                                                                                   \
        (CPP_TYPE* NAME),                                                                                              \
        /* Eigen column */                                                                                             \
        (CPP_TYPE::Scalar* NAME)                                                                                       \
    )
// clang-format on

#define _DECLARE_CONSTRUCTOR_ARGUMENT(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_CONSTRUCTOR_ARGUMENT_IMPL TYPE_NAME)

// clang-format off
#define _CALL_SET_COLUMN_FUNCTIONS_FROM_ARGS_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                          \
        setColumn_##NAME(NAME); 
// clang-format on

#define _CALL_SET_COLUMN_FUNCTIONS_FROM_ARGS(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_CALL_SET_COLUMN_FUNCTIONS_FROM_ARGS_IMPL TYPE_NAME)

#define _DECLARE_CONSTRUCTOR_PARAMETERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                               \
    (typename Metadata::BOOST_PP_CAT(ParametersTypeOf_, NAME) NAME)

#define _DECLARE_CONSTRUCTOR_PARAMETERS(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_CONSTRUCTOR_PARAMETERS_IMPL TYPE_NAME)

// clang-format off
#define _INITIALIZE_PARAMETERS_FROM_ARGS_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                              \
    _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                        \
        /* Scalar */                                                                                                   \
        (BOOST_PP_CAT(NAME, _)([&]() -> auto {                                                                         \
          if (elements_ != NAME.size_)                                                                                 \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: number of elements not equal for every column: "                     \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          return NAME.tupleOrPointer();                                                                                \
            }()))                                                                                                      \
            ,                                                                                                          \
        /* Column */                                                                                                   \
        (BOOST_PP_CAT(NAME, _)([&]() -> auto {                                                                         \
          if (elements_ != NAME.size_)                                                                                 \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: number of elements not equal for every column: "                     \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          return NAME.tupleOrPointer();                                                                                \
            }()))                                                                                                      \
        ,                                                                                                              \
        /* Eigen column */                                                                                             \
        (BOOST_PP_CAT(NAME, Stride_)([&]() -> auto {                                                                   \
          if (elements_ != NAME.size_)                                                                                 \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: number of elements not equal for every column: "                     \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          return std::get<1>(NAME.tupleOrPointer());                                                                   \
            }()))                                                                                                      \
        (BOOST_PP_CAT(NAME, ElementsWithPadding_)([&]() -> auto {                                                      \
          if (elements_ != NAME.size_)                                                                                 \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: number of elements not equal for every column: "                     \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          return BOOST_PP_CAT(NAME, Stride_)                                                                           \
                 *  CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                         \
            }()))                                                                                                      \
        (BOOST_PP_CAT(NAME, _)([&]() -> auto {                                                                         \
          if (elements_ != NAME.size_)                                                                                 \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: number of elements not equal for every column: "                     \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          return std::get<0>(NAME.tupleOrPointer());                                                                   \
            }()))                                                                                                      \
        )
// clang-format on                       

#define _INITIALIZE_PARAMETERS_FROM_ARGS(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_INITIALIZE_PARAMETERS_FROM_ARGS_IMPL TYPE_NAME)

// clang-format off
#define _INITIALIZE_PARAMETERS_FROM_STRUCT_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                            \
    _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                        \
        /* Scalar */                                                                                                   \
        (BOOST_PP_CAT(NAME, _)([&]() -> auto {                                                                         \
          if (elements_ != helper.NAME.size_)                                                                          \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: number of elements not equal for every column: "                     \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          return helper.NAME.tupleOrPointer();                                                                         \
            }()))                                                                                                      \
            ,                                                                                                          \
        /* Column */                                                                                                   \
        (BOOST_PP_CAT(NAME, _)([&]() -> auto {                                                                         \
          if (elements_ != helper.NAME.size_)                                                                          \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: number of elements not equal for every column: "                     \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          return helper.NAME.tupleOrPointer();                                                                         \
            }()))                                                                                                      \
        ,                                                                                                              \
        /* Eigen column */                                                                                             \
        (BOOST_PP_CAT(NAME, Stride_)([&]() -> auto {                                                                   \
          if (elements_ != helper.NAME.size_)                                                                          \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: number of elements not equal for every column: "                     \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          return std::get<1>(helper.NAME.tupleOrPointer());                                                            \
            }()))                                                                                                      \
        (BOOST_PP_CAT(NAME, ElementsWithPadding_)([&]() -> auto {                                                      \
          if (elements_ != helper.NAME.size_)                                                                          \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: number of elements not equal for every column: "                     \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          return BOOST_PP_CAT(NAME, Stride_)                                                                           \
                 *  CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                         \
            }()))                                                                                                      \
        (BOOST_PP_CAT(NAME, _)([&]() -> auto {                                                                         \
          if (elements_ != helper.NAME.size_)                                                                          \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: number of elements not equal for every column: "                     \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          return std::get<0>(helper.NAME.tupleOrPointer());                                                            \
            }()))                                                                                                      \
    )
// clang-format on

#define _INITIALIZE_PARAMETERS_FROM_STRUCT(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_INITIALIZE_PARAMETERS_FROM_STRUCT_IMPL TYPE_NAME)

// clang-format off
#define _INITIALIZE_PARAMETERS_AND_SIZE_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                               \
    _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                        \
        /* Scalar */                                                                                                   \
        if (not readyToSet) {                                                                                          \
          elements_ = NAME.size_;                                                                                      \
          readyToSet = true;                                                                                           \
        }                                                                                                              \
        auto BOOST_PP_CAT(NAME, _tmp) = [&]() -> auto {                                                                \
          if (elements_ != NAME.size_)                                                                                 \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: number of elements not equal for every column: "                     \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          return NAME.tupleOrPointer();                                                                                \
            }();                                                                                                       \
        BOOST_PP_CAT(NAME, _) = BOOST_PP_CAT(NAME, _tmp);                                                              \
        ,                                                                                                              \
        /* Column */                                                                                                   \
        if (not readyToSet) {                                                                                          \
          elements_ = NAME.size_;                                                                                      \
          readyToSet = true;                                                                                           \
        }                                                                                                              \
        auto BOOST_PP_CAT(NAME, _tmp) = [&]() -> auto {                                                                \
          if (elements_ != NAME.size_)                                                                                 \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: number of elements not equal for every column: "                     \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          return NAME.tupleOrPointer();                                                                                \
            }();                                                                                                       \
        BOOST_PP_CAT(NAME, _) = BOOST_PP_CAT(NAME, _tmp);                                                              \
        ,                                                                                                              \
        /* Eigen column */                                                                                             \
        if (not readyToSet) {                                                                                          \
          elements_ = NAME.size_;                                                                                      \
          readyToSet = true;                                                                                           \
        }                                                                                                              \
        auto BOOST_PP_CAT(NAME, Stride_tmp)([&]() -> auto {                                                            \
          if (elements_ != NAME.size_)                                                                                 \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: number of elements not equal for every column: "                     \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          return std::get<1>(NAME.tupleOrPointer());                                                                   \
            }());                                                                                                      \
        auto BOOST_PP_CAT(NAME, ElementsWithPadding_tmp)([&]() -> auto {                                               \
          if (elements_ != NAME.size_)                                                                                 \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: number of elements not equal for every column: "                     \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          return BOOST_PP_CAT(NAME, Stride_)                                                                           \
                 *  CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                         \
            }());                                                                                                      \
        auto BOOST_PP_CAT(NAME, _tmp) = [&]() -> auto {                                                                \
          if (elements_ != NAME.size_)                                                                                 \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: number of elements not equal for every column: "                     \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          return std::get<0>(NAME.tupleOrPointer());                                                                   \
            }();                                                                                                       \
        BOOST_PP_CAT(NAME, Stride_) = BOOST_PP_CAT(NAME, Stride_tmp);                                                  \
        BOOST_PP_CAT(NAME, ElementsWithPadding_) = BOOST_PP_CAT(NAME, ElementsWithPadding_tmp);                        \
        BOOST_PP_CAT(NAME, _) = BOOST_PP_CAT(NAME, _tmp);                                                              \
      ) 
// clang-format on

#define _INITIALIZE_PARAMETERS_AND_SIZE(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_INITIALIZE_PARAMETERS_AND_SIZE_IMPL TYPE_NAME)

#define _COPY_COLUMN_BY_COLUMN_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                        \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      memcpy(BOOST_PP_CAT(soa.metadata().addressOf_, NAME)(), BOOST_PP_CAT(NAME, _), cms::soa::alignSize(sizeof(CPP_TYPE), alignment));                                                      \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      memcpy(BOOST_PP_CAT(soa.metadata().addressOf_, NAME)(), BOOST_PP_CAT(NAME, _), cms::soa::alignSize(elements_ * sizeof(CPP_TYPE), alignment));                                          \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      memcpy(BOOST_PP_CAT(soa.metadata().addressOf_, NAME)(), BOOST_PP_CAT(NAME, _), cms::soa::alignSize(elements_ * sizeof(CPP_TYPE::Scalar), alignment)                                    \
                                                        * CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime);  \
  )                                                                                                                    \

#define _COPY_COLUMN_BY_COLUMN(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_COPY_COLUMN_BY_COLUMN_IMPL TYPE_NAME)

#define _COPY_VIEW_COLUMNS_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                        \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      memcpy(BOOST_PP_CAT(soa.metadata().addressOf_, NAME)(), BOOST_PP_CAT(view.metadata().addressOf_, NAME)(), cms::soa::alignSize(sizeof(CPP_TYPE), alignment));                                                      \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      memcpy(BOOST_PP_CAT(soa.metadata().addressOf_, NAME)(), BOOST_PP_CAT(view.metadata().addressOf_, NAME)(), cms::soa::alignSize(soa.elements_ * sizeof(CPP_TYPE), alignment));                                          \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      memcpy(BOOST_PP_CAT(soa.metadata().addressOf_, NAME)(), BOOST_PP_CAT(view.metadata().addressOf_, NAME)(), cms::soa::alignSize(soa.elements_ * sizeof(CPP_TYPE::Scalar), alignment)                                    \
                                                        * CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime);  \
  )                                                                                                                    \

#define _COPY_VIEW_COLUMNS(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_COPY_VIEW_COLUMNS_IMPL TYPE_NAME)

#ifdef DEBUG
#define _DO_RANGECHECK true
#else
#define _DO_RANGECHECK false
#endif

/*
 * A macro defining a SoA layout (collection of scalars and columns of equal lengths)
 */
// clang-format off
#define GENERATE_SOA_LAYOUT(CLASS, ...)                                                                                \
  template <CMS_SOA_BYTE_SIZE_TYPE ALIGNMENT = cms::soa::CacheLineSize::defaultSize,                                   \
            bool ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::relaxed>                                      \
  struct CLASS {                                                                                                       \
    /* these could be moved to an external type trait to free up the symbol names */                                   \
    using self_type = CLASS;                                                                                           \
    using AlignmentEnforcement = cms::soa::AlignmentEnforcement;                                                       \
                                                                                                                       \
    /* For CUDA applications, we align to the 128 bytes of the cache lines.                                            \
     * See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory-3-0 this is still valid      \
     * up to compute capability 8.X.                                                                                   \
     */                                                                                                                \
    using size_type = cms::soa::size_type;                                                                             \
    using byte_size_type = cms::soa::byte_size_type;                                                                   \
    constexpr static byte_size_type defaultAlignment = 128;                                                            \
    constexpr static byte_size_type alignment = ALIGNMENT;                                                             \
    constexpr static bool alignmentEnforcement = ALIGNMENT_ENFORCEMENT;                                                \
    constexpr static byte_size_type conditionalAlignment =                                                             \
        alignmentEnforcement == cms::soa::AlignmentEnforcement::enforced ? alignment : 0;                              \
    /* Those typedefs avoid having commas in macros (which is problematic) */                                          \
    template <cms::soa::SoAColumnType COLUMN_TYPE, class C>                                                            \
    using SoAValueWithConf = cms::soa::SoAValue<COLUMN_TYPE, C, conditionalAlignment>;                                 \
                                                                                                                       \
    template <cms::soa::SoAColumnType COLUMN_TYPE, class C>                                                            \
    using SoAConstValueWithConf = cms::soa::SoAConstValue<COLUMN_TYPE, C, conditionalAlignment>;                       \
                                                                                                                       \
    template <CMS_SOA_BYTE_SIZE_TYPE VIEW_ALIGNMENT = cms::soa::CacheLineSize::defaultSize,                            \
            bool VIEW_ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::relaxed,                                 \
            bool RESTRICT_QUALIFY = cms::soa::RestrictQualify::Default,                                                \
            bool RANGE_CHECKING = cms::soa::RangeChecking::Default>                                                    \
    struct ViewTemplateFreeParams;                                                                                     \
                                                                                                                       \
    /* dump the SoA internal structure */                                                                              \
    SOA_HOST_ONLY                                                                                                      \
    void soaToStreamInternal(std::ostream & _soa_impl_os) const {                                                      \
      _soa_impl_os << #CLASS "(" << elements_ << " elements, byte alignement= " << alignment << ", @"<< mem_ <<"): "   \
         << std::endl;                                                                                                 \
      _soa_impl_os << "  sizeof(" #CLASS "): " << sizeof(CLASS) << std::endl;                                          \
      byte_size_type _soa_impl_offset = 0;                                                                             \
      _ITERATE_ON_ALL(_DECLARE_SOA_STREAM_INFO, ~, __VA_ARGS__)                                                        \
      _soa_impl_os << "Final offset = " << _soa_impl_offset << " computeDataSize(...): " << computeDataSize(elements_) \
              << std::endl;                                                                                            \
      _soa_impl_os << std::endl;                                                                                       \
    }                                                                                                                  \
                                                                                                                       \
    /* Helper function used by caller to externally allocate the storage */                                            \
    static constexpr byte_size_type computeDataSize(size_type elements) {                                              \
      byte_size_type _soa_impl_ret = 0;                                                                                \
      _ITERATE_ON_ALL(_ACCUMULATE_SOA_ELEMENT, ~, __VA_ARGS__)                                                         \
      return _soa_impl_ret;                                                                                            \
    }                                                                                                                  \
                                                                                                                       \
    /* Helper function to compute the total number of columns */                                                       \
    static constexpr size_type computeColumnNumber() {                                                                 \
      size_type _soa_column_count = 0;                                                                                 \
      _ITERATE_ON_ALL(_COUNT_SOA_COLUMNS, ~, __VA_ARGS__)                                                              \
      return _soa_column_count;                                                                                        \
    }                                                                                                                  \
                                                                                                                       \
    /* Helper function used by caller to generate an array of strings containing column names */                       \
    static constexpr std::array<const char*, computeColumnNumber()> generateColumnNames() {                            \
      std::array<const char*, computeColumnNumber()> _columns_names = {};                                              \
      size_type i = 0;                                                                                                 \
      _ITERATE_ON_ALL(_DECLARE_SOA_COLUMNS_NAMES, ~, __VA_ARGS__)                                                      \
      return _columns_names;                                                                                           \
    }                                                                                                                  \
                                                                                                                       \
    SOA_HOST_ONLY                                                                                                      \
    void aggregateInPlace() {                                                                                          \
      std::byte* buffer {                                                                                              \
        reinterpret_cast<std::byte *>(aligned_alloc(alignment, byteSize_))};                                           \
      mem_ = buffer;                                                                                                   \
      if constexpr (alignmentEnforcement == cms::soa::AlignmentEnforcement::enforced)                                  \
        if (reinterpret_cast<intptr_t>(mem_) % alignment)                                                              \
          throw std::runtime_error("In " #CLASS "::" #CLASS ": misaligned buffer");                                    \
      auto _soa_impl_curMem = mem_;                                                                                    \
      _ITERATE_ON_ALL(_AGGREGATE_SOA_COLUMNS, ~, __VA_ARGS__)                                                          \
      /* Sanity check: we should have reached the computed size, only on host code */                                  \
      byteSize_ = computeDataSize(elements_);                                                                          \
      if (mem_ + byteSize_ != _soa_impl_curMem)                                                                        \
        throw std::runtime_error("In " #CLASS "::" #CLASS ": unexpected end pointer.");                                \
    }                                                                                                                  \
                                                                                                                       \
    SOA_HOST_ONLY                                                                                                      \
    CLASS aggregate() {                                                                                                \
      std::byte* buffer {                                                                                              \
        reinterpret_cast<std::byte *>(aligned_alloc(alignment, byteSize_))};                                           \
      CLASS soa(buffer, elements_);                                                                                    \
      _ITERATE_ON_ALL(_COPY_COLUMN_BY_COLUMN, ~, __VA_ARGS__)                                                          \
      return soa;                                                                                                      \
    }                                                                                                                  \
                                                                                                                       \
    /* Helper function to set the starting memory byte */                                                              \
    inline void setData(std::byte* newMem) {mem_ = newMem;}                                                            \
                                                                                                                       \
                                                                                                                       \
    _ITERATE_ON_ALL(_DEFINE_SET_COLUMN_FUNCTIONS, ~, __VA_ARGS__)                                                      \
                                                                                                                       \
    /**                                                                                                                \
     * Helper/friend class allowing SoA introspection.                                                                 \
     */                                                                                                                \
    struct Metadata {                                                                                                  \
      friend CLASS;                                                                                                    \
      SOA_HOST_DEVICE SOA_INLINE size_type size() const { return parent_.elements_; }                                  \
      SOA_HOST_DEVICE SOA_INLINE byte_size_type byteSize() const { return parent_.byteSize_; }                         \
      SOA_HOST_DEVICE SOA_INLINE byte_size_type alignment() const { return CLASS::alignment; }                         \
      SOA_HOST_DEVICE SOA_INLINE std::byte* data() { return parent_.mem_; }                                            \
      SOA_HOST_DEVICE SOA_INLINE const std::byte* data() const { return parent_.mem_; }                                \
      SOA_HOST_DEVICE SOA_INLINE std::byte* nextByte() const { return parent_.mem_ + parent_.byteSize_; }              \
      SOA_HOST_DEVICE SOA_INLINE CLASS cloneToNewAddress(std::byte* _soa_impl_addr) const {                            \
        return CLASS(_soa_impl_addr, parent_.elements_);                                                               \
      }                                                                                                                \
                                                                                                                       \
      _ITERATE_ON_ALL(_DEFINE_METADATA_MEMBERS, ~, __VA_ARGS__)                                                        \
                                                                                                                       \
      struct value_element {                                                                                           \
        SOA_HOST_DEVICE SOA_INLINE value_element                                                                       \
          BOOST_PP_IF(                                                                                                 \
            BOOST_PP_SEQ_SIZE(_ITERATE_ON_ALL(_VALUE_ELEMENT_CTOR_ARGS, ~, __VA_ARGS__) ),                             \
            (_ITERATE_ON_ALL_COMMA(_VALUE_ELEMENT_CTOR_ARGS, ~, __VA_ARGS__)):,                                        \
            ())                                                                                                        \
          BOOST_PP_TUPLE_ENUM(BOOST_PP_IF(                                                                             \
            BOOST_PP_SEQ_SIZE(_ITERATE_ON_ALL(_VALUE_ELEMENT_CTOR_ARGS, ~, __VA_ARGS__)),                              \
            BOOST_PP_SEQ_TO_TUPLE(_ITERATE_ON_ALL(_VALUE_ELEMENT_INITIALIZERS, ~, __VA_ARGS__)),                       \
            ()                                                                                                         \
          )                                                                                                            \
        )                                                                                                              \
        {}                                                                                                             \
                                                                                                                       \
        _ITERATE_ON_ALL(_DEFINE_VALUE_ELEMENT_MEMBERS, ~, __VA_ARGS__)                                                 \
      };                                                                                                               \
                                                                                                                       \
      Metadata& operator=(const Metadata&) = delete;                                                                   \
      Metadata(const Metadata&) = delete;                                                                              \
                                                                                                                       \
                                                                                                                       \
    private:                                                                                                           \
      SOA_HOST_DEVICE SOA_INLINE Metadata(const CLASS& _soa_impl_parent) : parent_(_soa_impl_parent) {}                \
      const CLASS& parent_;                                                                                            \
      using ParentClass = CLASS;                                                                                       \
    };                                                                                                                 \
                                                                                                                       \
    friend Metadata;                                                                                                   \
                                                                                                                       \
      /**                                                                                                              \
     * Helper class allowing for SoA costruction with names.                                                           \
     */                                                                                                                \
    struct Records {                                                                                                   \
      size_type capacity;                                                                                              \
      _ITERATE_ON_ALL(_DECLARE_STRUCT_MEMBERS, ~, __VA_ARGS__)                                                         \
    };                                                                                                                 \
                                                                                                                       \
    struct Metarecords {                                                                                               \
      friend CLASS;                                                                                                    \
      Metarecords(const CLASS& _soa_impl_parent) : parent_(_soa_impl_parent), _ITERATE_ON_ALL_COMMA(_STRUCT_ELEMENT_INITIALIZERS, ~, __VA_ARGS__) {} \
      _ITERATE_ON_ALL(_ACCESSORS_STRUCT_MEMBERS, ~, __VA_ARGS__)                                                       \
      private:                                                                                                         \
        const CLASS& parent_;                                                                                          \
        _ITERATE_ON_ALL(_DECLARE_STRUCT_DATA_MEMBER, ~, __VA_ARGS__)                                                   \
    };                                                                                                                 \
                                                                                                                       \
    SOA_HOST_DEVICE SOA_INLINE const Metadata metadata() const { return Metadata(*this); }                             \
    SOA_HOST_DEVICE SOA_INLINE Metadata metadata() { return Metadata(*this); }                                         \
    SOA_HOST_DEVICE SOA_INLINE const Metarecords records() const { return Metarecords(*this); }                        \
    SOA_HOST_DEVICE SOA_INLINE Metarecords records() { return Metarecords(*this); }                                    \
                                                                                                                       \
    /* Generate the ConstView template */                                                                              \
    _GENERATE_SOA_TRIVIAL_CONST_VIEW(CLASS,                                                                            \
                    SOA_VIEW_LAYOUT_LIST(                                                                              \
                        SOA_VIEW_LAYOUT(BOOST_PP_CAT(CLASS, _parametrized) , BOOST_PP_CAT(instance_, CLASS))),         \
                    SOA_VIEW_VALUE_LIST(_ITERATE_ON_ALL_COMMA(                                                         \
                    _VIEW_FIELD_FROM_LAYOUT, BOOST_PP_CAT(instance_, CLASS), __VA_ARGS__)))                            \
                                                                                                                       \
    template <bool RESTRICT_QUALIFY, bool RANGE_CHECKING>                                                              \
    using ConstViewTemplate = ConstViewTemplateFreeParams<ALIGNMENT, ALIGNMENT_ENFORCEMENT, RESTRICT_QUALIFY,          \
      RANGE_CHECKING>;                                                                                                 \
                                                                                                                       \
    using ConstView = ConstViewTemplate<cms::soa::RestrictQualify::Default, cms::soa::RangeChecking::Default>;         \
                                                                                                                       \
    /* Generate the mutable View template */                                                                           \
    _GENERATE_SOA_TRIVIAL_VIEW(CLASS,                                                                                  \
                    SOA_VIEW_LAYOUT_LIST(                                                                              \
                        SOA_VIEW_LAYOUT(BOOST_PP_CAT(CLASS, _parametrized), BOOST_PP_CAT(instance_, CLASS))),          \
                    SOA_VIEW_VALUE_LIST(_ITERATE_ON_ALL_COMMA(                                                         \
                    _VIEW_FIELD_FROM_LAYOUT, BOOST_PP_CAT(instance_, CLASS), __VA_ARGS__)),                            \
                    __VA_ARGS__)                                                                                       \
                                                                                                                       \
    template <bool RESTRICT_QUALIFY, bool RANGE_CHECKING>                                                              \
    using ViewTemplate = ViewTemplateFreeParams<ALIGNMENT, ALIGNMENT_ENFORCEMENT, RESTRICT_QUALIFY, RANGE_CHECKING>;   \
                                                                                                                       \
    using View = ViewTemplate<cms::soa::RestrictQualify::Default, cms::soa::RangeChecking::Default>;                   \
                                                                                                                       \
    /* Trivial constuctor */                                                                                           \
    CLASS()                                                                                                            \
        : mem_(nullptr),                                                                                               \
          elements_(0),                                                                                                \
          byteSize_(0),                                                                                                \
          _ITERATE_ON_ALL_COMMA(_DECLARE_MEMBER_TRIVIAL_CONSTRUCTION, ~, __VA_ARGS__) {}                               \
                                                                                                                       \
    /* Constructor relying on user provided storage (implementation shared with ROOT streamer) */                      \
    SOA_HOST_ONLY CLASS(std::byte* mem, size_type elements) : mem_(mem), elements_(elements), byteSize_(0) {           \
      organizeColumnsFromBuffer();                                                                                     \
    }                                                                                                                  \
                                                                                                                       \
    /* Explicit copy constructor and assignment operator */                                                            \
    SOA_HOST_ONLY CLASS(CLASS const& _soa_impl_other)                                                                  \
        : mem_(_soa_impl_other.mem_),                                                                                  \
          elements_(_soa_impl_other.elements_),                                                                        \
          byteSize_(_soa_impl_other.byteSize_),                                                                        \
          _ITERATE_ON_ALL_COMMA(_DECLARE_MEMBER_COPY_CONSTRUCTION, ~, __VA_ARGS__) {}                                  \
                                                                                                                       \
    /* Constructor relying on user-provided column pointers */                                                         \
    SOA_HOST_ONLY CLASS(_ITERATE_ON_ALL_COMMA(_DECLARE_CONSTRUCTOR_PARAMETERS, ~, __VA_ARGS__))                        \
      : mem_(nullptr)                                                                                                  \
    {                                                                                                                  \
      bool readyToSet = false;                                                                                         \
      _ITERATE_ON_ALL(_INITIALIZE_PARAMETERS_AND_SIZE, ~, __VA_ARGS__)                                                 \
      byteSize_ = computeDataSize(elements_);                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    /* Constructor relying on user-provided column pointers */                                                         \
    /* SOA_HOST_ONLY CLASS(size_type elements, _ITERATE_ON_ALL_COMMA(_DECLARE_CONSTRUCTOR_ARGUMENT, ~, __VA_ARGS__)) */\
    /*  : elements_(elements), mem_(nullptr)                    */                                                     \
    /* {                                                */                                                             \
    /*  _ITERATE_ON_ALL(_CALL_SET_COLUMN_FUNCTIONS_FROM_ARGS, ~, __VA_ARGS__)       */                                 \
    /* }                                                 */                                                            \
    SOA_HOST_ONLY CLASS(View const& custom_view, std::byte* mem) : mem_(mem), elements_(custom_view.metadata().size()), byteSize_(0) {     \
      auto _soa_impl_curMem = mem_;                                                                                    \
      _ITERATE_ON_ALL(_ASSIGN_SOA_COLUMN_OR_SCALAR_FROM_VIEW, ~, __VA_ARGS__)                                          \
      byteSize_ = computeDataSize(elements_);                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    /* Construnctor having only the number of elements */                                                              \
    SOA_HOST_ONLY CLASS(size_type elements)                                                                            \
      : elements_(elements)                                                                                            \
    {}                                                                                                                 \
                                                                                                                       \
    SOA_HOST_ONLY CLASS(Records helper)                                                                                \
      : mem_(nullptr), elements_(helper.capacity), _ITERATE_ON_ALL_COMMA(_INITIALIZE_PARAMETERS_FROM_STRUCT, ~, __VA_ARGS__) \
    {}                                                                                                                 \
                                                                                                                       \
    SOA_HOST_ONLY CLASS& operator=(CLASS const& _soa_impl_other) {                                                     \
        mem_ = _soa_impl_other.mem_;                                                                                   \
        elements_ = _soa_impl_other.elements_;                                                                         \
        byteSize_ = _soa_impl_other.byteSize_;                                                                         \
        _ITERATE_ON_ALL(_DECLARE_MEMBER_ASSIGNMENT, ~, __VA_ARGS__)                                                    \
        return *this;                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    SOA_HOST_ONLY                                                                                                      \
    static const CLASS aggregate(ConstView const& view) {                                                                    \
    std::byte* buffer {                                                                                                \
        reinterpret_cast<std::byte *>(aligned_alloc(alignment, computeDataSize(view.metadata().size())))};             \
      CLASS soa(buffer, view.metadata().size());                                                                       \
      _ITERATE_ON_ALL(_COPY_VIEW_COLUMNS, ~, __VA_ARGS__)                                                              \
      return soa;                                                                                                      \
    }                                                                                                                  \
                                                                                                                       \
    /* ROOT read streamer */                                                                                           \
    template <typename T>                                                                                              \
    void ROOTReadStreamer(T & onfile) {                                                                                \
      _ITERATE_ON_ALL(_STREAMER_READ_SOA_DATA_MEMBER, ~, __VA_ARGS__)                                                  \
    }                                                                                                                  \
                                                                                                                       \
    /* ROOT allocation cleanup */                                                                                      \
    void ROOTStreamerCleaner() {                                                                                       \
      /* This function should only be called from the PortableCollection ROOT streamer */                              \
      _ITERATE_ON_ALL(_ROOT_FREE_SOA_COLUMN_OR_SCALAR, ~, __VA_ARGS__)                                                 \
    }                                                                                                                  \
                                                                                                                       \
    /* Dump the SoA internal structure */                                                                              \
    template <typename T>                                                                                              \
    SOA_HOST_ONLY friend void dump();                                                                                  \
                                                                                                                       \
  private:                                                                                                             \
    /* Helper method for the user provided storage constructor and ROOT streamer */                                    \
    void organizeColumnsFromBuffer() {                                                                                 \
      if constexpr (alignmentEnforcement == cms::soa::AlignmentEnforcement::enforced)                                  \
        if (reinterpret_cast<intptr_t>(mem_) % alignment)                                                              \
          throw std::runtime_error("In " #CLASS "::" #CLASS ": misaligned buffer");                                    \
      auto _soa_impl_curMem = mem_;                                                                                    \
      _ITERATE_ON_ALL(_ASSIGN_SOA_COLUMN_OR_SCALAR, ~, __VA_ARGS__)                                                    \
      /* Sanity check: we should have reached the computed size, only on host code */                                  \
      byteSize_ = computeDataSize(elements_);                                                                          \
      if (mem_ + byteSize_ != _soa_impl_curMem)                                                                        \
        throw std::runtime_error("In " #CLASS "::" #CLASS ": unexpected end pointer.");                                \
    }                                                                                                                  \
                                                                                                                       \
    /* Data members */                                                                                                 \
    std::byte* mem_ EDM_REFLEX_TRANSIENT;                                                                              \
    size_type elements_;                                                                                               \
    size_type const scalar_ = 1;                                                                                       \
    byte_size_type byteSize_ EDM_REFLEX_TRANSIENT;                                                                     \
    _ITERATE_ON_ALL(_DECLARE_SOA_DATA_MEMBER, ~, __VA_ARGS__)                                                          \
    /* static constexpr std::array<const char*, computeColumnNumber()> columnNames_ = generateColumnNames(); */        \
    /* Making the code conditional is problematic in macros as the commas will interfere with parameter lisings     */ \
    /* So instead we make the code unconditional with paceholder names which are protected by a private protection. */ \
    /* This will be handled later as we handle the integration of the view as a subclass of the layout.             */ \
                                                                                                                       \
  };
// clang-format on

#endif  // DataFormats_SoATemplate_interface_SoALayout_h
