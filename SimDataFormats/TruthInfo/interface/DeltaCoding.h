// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#ifndef SimDataFormats_TruthInfo_interface_DeltaCoding_h
#define SimDataFormats_TruthInfo_interface_DeltaCoding_h

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

// Exact, reversible codings of integer arrays for the stored form of the truth
// products. Sorted or slowly varying indices become small repeated values, which the
// file compression reduces much better than the absolute values.
namespace truth::delta {

  // out[i] = in[i] - in[i - 1], with in[-1] = 0. The differences wrap.
  template <typename T>
  void encode(std::vector<T> const& in, std::vector<T>& out) {
    using U = std::make_unsigned_t<T>;
    out.resize(in.size());
    U previous = 0;
    for (std::size_t i = 0; i < in.size(); ++i) {
      const U value = static_cast<U>(in[i]);
      out[i] = static_cast<T>(static_cast<U>(value - previous));
      previous = value;
    }
  }

  // The inverse of encode.
  template <typename T>
  void decode(std::vector<T> const& in, std::vector<T>& out) {
    using U = std::make_unsigned_t<T>;
    out.resize(in.size());
    U previous = 0;
    for (std::size_t i = 0; i < in.size(); ++i) {
      previous = static_cast<U>(previous + static_cast<U>(in[i]));
      out[i] = static_cast<T>(previous);
    }
  }

  // out[i] = in[i] - i for an index array with the value -1 for "none". -1 is kept as
  // kNone. Every other value must be >= 0.
  inline constexpr int32_t kNone = INT32_MIN;

  inline void encodeRelativeToIndex(std::vector<int32_t> const& in, std::vector<int32_t>& out) {
    out.resize(in.size());
    for (std::size_t i = 0; i < in.size(); ++i)
      out[i] = in[i] == -1 ? kNone : static_cast<int32_t>(static_cast<uint32_t>(in[i]) - static_cast<uint32_t>(i));
  }

  // The inverse of encodeRelativeToIndex.
  inline void decodeRelativeToIndex(std::vector<int32_t> const& in, std::vector<int32_t>& out) {
    out.resize(in.size());
    for (std::size_t i = 0; i < in.size(); ++i)
      out[i] = in[i] == kNone ? -1 : static_cast<int32_t>(static_cast<uint32_t>(in[i]) + static_cast<uint32_t>(i));
  }

}  // namespace truth::delta

#endif
