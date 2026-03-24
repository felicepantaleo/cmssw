/*
 * ONNXRuntime.h
 *
 * A convenience wrapper of the ONNXRuntime C++ API.
 * Based on https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp.
 *
 *  Created on: Jun 28, 2019
 *      Author: hqu
 */

#ifndef PHYSICSTOOLS_ONNXRUNTIME_INTERFACE_ONNXRUNTIME_H_
#define PHYSICSTOOLS_ONNXRUNTIME_INTERFACE_ONNXRUNTIME_H_

#include <numeric>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <cassert>

#include "onnxruntime/onnxruntime_cxx_api.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace cms::Ort {
  typedef std::vector<std::vector<float>> FloatArrays;

  enum class Backend {
    cpu,
    cuda,
  };

  class ONNXRuntime {
  public:
    ONNXRuntime(const std::string& model_path, const ::Ort::SessionOptions* session_options = nullptr);
    ONNXRuntime(const ONNXRuntime&) = delete;
    ONNXRuntime& operator=(const ONNXRuntime&) = delete;
    ~ONNXRuntime();

    static ::Ort::SessionOptions defaultSessionOptions(Backend backend = Backend::cpu);

    // Run inference and get outputs
    // input_names: list of the names of the input nodes.
    // input_values: list of input arrays for each input node. The order of `input_values` must match `input_names`.
    // input_shapes: list of `int64_t` arrays specifying the shape of each input node. Can leave empty if the model does not have dynamic axes.
    // output_names: names of the output nodes to get outputs from. Empty list means all output nodes.
    // batch_size: number of samples in the batch. Each array in `input_values` must have a shape layout of (batch_size, ...).
    // Returns: a std::vector<std::vector<float>>, with the order matched to `output_names`.
    // When `output_names` is empty, will return all outputs ordered as in `getOutputNames()`.
    FloatArrays run(const std::vector<std::string>& input_names,
                    FloatArrays& input_values,
                    const std::vector<std::vector<int64_t>>& input_shapes = {},
                    const std::vector<std::string>& output_names = {},
                    int64_t batch_size = 1) const;

    // Run inference writing outputs into user-provided buffers when possible.
    //
    // - output_values will be resized as needed; callers can reuse capacity across events to reduce allocations.
    // - If output_shapes is provided (non-empty), outputs are written directly into output_values (preallocated path).
    // - If output_shapes is empty:
    //     * for outputs with fully-known shapes (or only dynamic batch dim at index 0), we still use the preallocated path
    //     * for outputs with other dynamic dims (-1 at index > 0), the implementation falls back to ORT-allocated outputs
    //       and copies the results into output_values (capacity still reusable across events).
    void runInto(const std::vector<std::string>& input_names,
                 FloatArrays& input_values,
                 const std::vector<std::vector<int64_t>>& input_shapes,
                 const std::vector<std::string>& output_names,
                 FloatArrays& output_values,
                 const std::vector<std::vector<int64_t>>& output_shapes = {},
                 int64_t batch_size = 1) const;

    // Parameters of one input tensor for runIntoTemplated.
    template <typename T>
    struct InputTensorConfig {
      std::string input_name;            ///< Name of the input node in ONNX model
      std::vector<T> input_value;        ///< actual tensor values
      std::vector<int64_t> input_shape;  ///< shape of the input_value tensor

      typedef T onnx_type;  ///< type of the ONNX tensor (by default same as that of std::vector)
      onnx_type* convertToPointer() { return input_value.data(); }
    };
    // Speicalization in case of bool tensor. We make it out of a std::vector<uint8>
    struct InputTensorConfigBool : public InputTensorConfig<uint8_t> {
      typedef bool onnx_type;
      // The reinterpret_cast is unfortunately needed because we cannot use vector<bool> (it does not have a .data())
      // ideally we could have some equivalent vector<bool> to avoid possible UB in the reinterpret_cast
      // another option would be to cast to void* then manually specify the ONNXTensorElementDataType
      onnx_type* convertToPointer() { return reinterpret_cast<bool*>(input_value.data()); }
    };

    /**
    * Run inference with inputs that may not be the same type. Otherwise identical to ONNXRuntime::runInto
    * @param inputTensorConfigTuple an std::tuple of InputTensorConfig<T>, where the T do not have to be the same (ie std::tuple<InputTensorConfig<float>, InputTensorConfig<uint8_t>)
    */
    template <typename InputTensorConfigTupleT>
    void runIntoTemplated(InputTensorConfigTupleT inputTensorConfigTuple,
                          const std::vector<std::string>& output_names,
                          FloatArrays& output_values,
                          const std::vector<std::vector<int64_t>>& output_shapes,
                          int64_t batch_size = 1) const;

    // Get a list of names of all the output nodes
    const std::vector<std::string>& getOutputNames() const;

    // Get the shape of a output node
    // The 0th dim depends on the batch size, therefore is set to -1
    const std::vector<int64_t>& getOutputShape(const std::string& output_name) const;

  private:
    static const ::Ort::Env env_;
    std::unique_ptr<::Ort::Session> session_;

    std::vector<std::string> input_node_strings_;
    std::vector<const char*> input_node_names_;
    std::map<std::string, std::vector<int64_t>> input_node_dims_;

    std::vector<std::string> output_node_strings_;
    std::vector<const char*> output_node_names_;
    std::map<std::string, std::vector<int64_t>> output_node_dims_;
  };

  inline int64_t numel(const std::vector<int64_t>& dims) {
    return std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>());
  }

  inline bool hasDynamicDimsExceptBatch(const std::vector<int64_t>& dims) {
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i] == -1 && i != 0) {
        return true;
      }
    }
    return false;
  }

  template <typename InputTensorConfigTupleT>
  void ONNXRuntime::runIntoTemplated(InputTensorConfigTupleT inputTensorConfigTuple,
                                     const std::vector<std::string>& output_names,
                                     FloatArrays& output_values,
                                     const std::vector<std::vector<int64_t>>& output_shapes,
                                     int64_t batch_size) const {
    assert(output_shapes.empty() || (!output_names.empty() && output_shapes.size() == output_names.size()));
    static_assert(std::tuple_size<InputTensorConfigTupleT>{} > 0, "inputTensorConfigTuple must be a std::tuple");
    assert(batch_size > 0);

    // create input tensor objects from data values
    std::vector<::Ort::Value> input_tensors;
    input_tensors.reserve(input_node_strings_.size());

    auto memory_info = ::Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // The following lambda is to loop over inputTensorConfigTuple. ArgsT will be of type InputTensorConfig<T>
    auto create_tensor_input = [&]<typename ArgsT>(ArgsT& args) {
      // Get input dimensions: use provided shapes if available, otherwise fall back to ONNX model defaults
      const auto& onnx_dims = input_node_dims_.at(args.input_name);
      std::vector<int64_t> input_dims = args.input_shape.empty() ? onnx_dims : args.input_shape;

      // Dynamic batch handling (-1 in dim0)
      const bool has_dynamic_batch = !onnx_dims.empty() && (onnx_dims[0] == -1);
      if (has_dynamic_batch) {
        if (args.input_shape.empty()) {
          input_dims[0] = batch_size;
        } else if (input_dims[0] != batch_size) {
          batch_size = input_dims[0];
        }
      }

      const int64_t expected_len = numel(input_dims);
      if (expected_len != static_cast<int64_t>(args.input_value.size())) {
        throw cms::Exception("RuntimeError") << "Input array " << args.input_name << " has a wrong size of "
                                             << args.input_value.size() << ", expected " << expected_len;
      }

      auto input_tensor =  // float
          ::Ort::Value::CreateTensor<typename ArgsT::onnx_type>(
              memory_info, args.convertToPointer(), args.input_value.size(), input_dims.data(), input_dims.size());
      assert(input_tensor.IsTensor());
      input_tensors.emplace_back(std::move(input_tensor));
    };
    std::apply(
        [&](auto&&... args) { ((create_tensor_input(args)), ...); },
        inputTensorConfigTuple);  // essentially does : "for (auto args : inputTensorConfigTuple) create_tensor_input(args);"

    // Resolve output node names; will get all outputs if `output_names` is not provided
    std::vector<std::string> resolved_output_names;
    if (output_names.empty()) {
      resolved_output_names = output_node_strings_;
    } else {
      resolved_output_names = output_names;
    }

    std::vector<const char*> run_output_node_names;
    run_output_node_names.reserve(resolved_output_names.size());
    for (const auto& n : resolved_output_names) {
      run_output_node_names.push_back(n.c_str());
    }

    // Decide whether we can use the preallocated output path.
    // - If caller provided output_shapes => always preallocated.
    // - Otherwise => preallocated only if ALL outputs have no dynamic dims except batch.
    bool need_fallback_allocation = false;
    if (output_shapes.empty()) {
      for (const auto& out_name : resolved_output_names) {
        std::vector<int64_t> out_dims = getOutputShape(out_name);
        if (!out_dims.empty() && out_dims[0] == -1) {
          out_dims[0] = batch_size;
        }
        if (hasDynamicDimsExceptBatch(out_dims)) {
          need_fallback_allocation = true;
          break;
        }
      }
    }

    if (need_fallback_allocation) {
      // Fallback: let ORT allocate outputs, then copy into output_values (capacity reused across events).
      auto ort_outputs = session_->Run(::Ort::RunOptions{nullptr},
                                       input_node_names_.data(),
                                       input_tensors.data(),
                                       input_tensors.size(),
                                       run_output_node_names.data(),
                                       run_output_node_names.size());

      output_values.resize(ort_outputs.size());

      for (size_t i = 0; i < ort_outputs.size(); ++i) {
        auto& out_tensor = ort_outputs[i];
        assert(out_tensor.IsTensor());

        auto tensor_info = out_tensor.GetTensorTypeAndShapeInfo();
        const size_t length = static_cast<size_t>(tensor_info.GetElementCount());

        const float* data = out_tensor.GetTensorData<float>();

        auto& out_buf = output_values[i];
        out_buf.resize(length);
        std::copy(data, data + length, out_buf.begin());
      }
      return;
    }

    // Preallocated path (output_shapes provided, or outputs are statically known except batch)
    output_values.resize(resolved_output_names.size());

    std::vector<::Ort::Value> output_tensors;
    output_tensors.reserve(resolved_output_names.size());

    for (size_t i = 0; i < resolved_output_names.size(); ++i) {
      const auto& out_name = resolved_output_names[i];

      std::vector<int64_t> out_dims;
      if (!output_shapes.empty()) {
        out_dims = output_shapes[i];
      } else {
        out_dims = getOutputShape(out_name);
        if (!out_dims.empty() && out_dims[0] == -1) {
          out_dims[0] = batch_size;
        }
        // safe here because need_fallback_allocation == false
      }

      const int64_t out_len = numel(out_dims);
      if (out_len <= 0) {
        throw cms::Exception("RuntimeError") << "Output " << out_name << " has invalid inferred size " << out_len;
      }

      auto& out_buf = output_values[i];
      if (static_cast<int64_t>(out_buf.capacity()) < out_len) {
        out_buf.reserve(static_cast<size_t>(out_len));
      }
      out_buf.resize(static_cast<size_t>(out_len));

      auto out_tensor = ::Ort::Value::CreateTensor<float>(
          memory_info, out_buf.data(), out_buf.size(), out_dims.data(), out_dims.size());
      assert(out_tensor.IsTensor());
      output_tensors.emplace_back(std::move(out_tensor));
    }

    session_->Run(::Ort::RunOptions{nullptr},
                  input_node_names_.data(),
                  input_tensors.data(),
                  input_tensors.size(),
                  run_output_node_names.data(),
                  output_tensors.data(),
                  output_tensors.size());
  }

}  // namespace cms::Ort

#endif /* PHYSICSTOOLS_ONNXRUNTIME_INTERFACE_ONNXRUNTIME_H_ */
