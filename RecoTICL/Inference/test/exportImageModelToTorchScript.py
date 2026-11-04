#!/usr/bin/env python3
"""Export a trackster image model (CNN, DNN or PFN, ticlv5/onnx_models of RecoHGCal-TICL) to TorchScript, for the device
inference of Trackster{CNN,DNN,PFN}InferenceProducer.

The TorchScript module runs the ONNX graph op by op, with the weights and constants read from the ONNX file, after a
permutation from the device layout [B, F, 50, 10] to the [B, 50, 10, F] of the ONNX models. A second input, the [B, 7]
trackster quantities, is passed as it is. The export is saved only if the module reproduces onnxruntime on sparse inputs
that have the memory layout of the device inputs, in which the batch index is the fastest one.

    python3 exportImageModelToTorchScript.py <onnx_models>/DNN/linking/id_v0.onnx \
        -o RecoTICL/Inference/data/trackster_pid_dnn.pt
"""

import argparse
import math
import os
import sys

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn.functional as F
from onnx import numpy_helper


def _tensor(array):
    # the CMSSW PyTorch has no NumPy bridge
    array = np.asarray(array)
    return torch.tensor(array.tolist(), dtype=torch.float32 if array.dtype.kind == "f" else torch.int64)


def _ints(tensor):
    return [int(value) for value in tensor.tolist()]


def _pads(x, attributes, kernel, strides, dilations):
    """The [top, left, bottom, right] padding of a Conv or MaxPool."""
    mode = attributes.get("auto_pad", b"NOTSET").decode()
    if mode not in ("SAME_UPPER", "SAME_LOWER"):
        return attributes.get("pads", [0, 0, 0, 0])
    total = [max((math.ceil(x.shape[2 + i] / strides[i]) - 1) * strides[i] + (kernel[i] - 1) * dilations[i] + 1
                 - x.shape[2 + i], 0) for i in range(2)]
    first = [t // 2 if mode == "SAME_UPPER" else t - t // 2 for t in total]
    return [first[0], first[1], total[0] - first[0], total[1] - first[1]]


class OnnxGraph(torch.nn.Module):
    """The operators of the image models of ticlv5/onnx_models; any other one raises NotImplementedError."""

    def __init__(self, path):
        super().__init__()
        graph = onnx.load(path).graph
        initializers = {t.name for t in graph.initializer}
        self.nodes = [(n.op_type, list(n.input), n.output[0],
                       {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}) for n in graph.node]
        self.inputs = [i.name for i in graph.input if i.name not in initializers]
        self.output = graph.output[0].name
        self.buffer_of = {}
        for k, initializer in enumerate(graph.initializer):
            self.buffer_of[initializer.name] = "constant%d" % k
            self.register_buffer("constant%d" % k, _tensor(numpy_helper.to_array(initializer)))

    def forward(self, *inputs):
        values = {name: getattr(self, buffer) for name, buffer in self.buffer_of.items()}
        values[self.inputs[0]] = inputs[0].permute(0, 2, 3, 1)
        if len(inputs) > 1:
            values[self.inputs[1]] = inputs[1]
        for op, node_inputs, output, attributes in self.nodes:
            values[output] = self.run(op, [values[name] if name else None for name in node_inputs], attributes)
        return values[self.output]

    @staticmethod
    def run(op, x, a):
        if op == "Constant":
            return _tensor(numpy_helper.to_array(a["value"]))
        if op == "Relu":
            return torch.relu(x[0])
        if op == "LeakyRelu":
            return F.leaky_relu(x[0], a.get("alpha", 0.01))
        if op == "Sigmoid":
            return torch.sigmoid(x[0])
        if op == "Softmax":
            return torch.softmax(x[0], a.get("axis", -1))
        if op == "Add":
            return x[0] + x[1]
        if op == "Mul":
            return x[0] * x[1]
        if op == "MatMul":
            return torch.matmul(x[0], x[1])
        if op == "Gemm":
            y = a.get("alpha", 1.0) * ((x[0].t() if a.get("transA", 0) else x[0]) @
                                       (x[1].t() if a.get("transB", 0) else x[1]))
            return y + a.get("beta", 1.0) * x[2] if len(x) > 2 else y
        if op == "Transpose":
            return x[0].permute(a["perm"])
        if op == "Concat":
            return torch.cat([e.to(x[0].dtype) for e in x], a["axis"])
        if op == "Flatten":
            return torch.flatten(x[0], a.get("axis", 1))
        if op == "Shape":
            return torch._shape_as_tensor(x[0])
        if op == "Gather":
            return torch.index_select(x[0], a.get("axis", 0), x[1].reshape(-1)).reshape(x[1].shape)
        if op == "Unsqueeze":
            y = x[0]
            for axis in sorted(_ints(x[1]) if len(x) > 1 else a["axes"]):
                y = y.unsqueeze(axis)
            return y
        if op == "Reshape":
            # one element at a time, so that the trace keeps the batch size of the Shape nodes dynamic
            return x[0].reshape([x[1][k] for k in range(x[1].shape[0])])
        if op == "Slice":
            y = x[0]
            axes = _ints(x[3]) if len(x) > 3 and x[3] is not None else list(range(len(x[1])))
            for start, end, axis in zip(_ints(x[1]), _ints(x[2]), axes):
                size = y.shape[axis]
                start = start if start >= 0 else size + start
                end = min(end, size) if end >= 0 else size + end
                y = y.narrow(axis, start, end - start)
            return y
        if op == "BatchNormalization":
            return F.batch_norm(x[0], x[3], x[4], x[1], x[2], False, 0.0, a.get("epsilon", 1e-5))
        if op == "GlobalAveragePool":
            return x[0].mean(dim=[2, 3], keepdim=True)
        if op == "Conv":
            kernel = a.get("kernel_shape", list(x[1].shape[2:]))
            strides, dilations = a.get("strides", [1, 1]), a.get("dilations", [1, 1])
            top, left, bottom, right = _pads(x[0], a, kernel, strides, dilations)
            return F.conv2d(F.pad(x[0], [left, right, top, bottom]), x[1], x[2] if len(x) > 2 else None, strides, 0,
                            dilations, a.get("group", 1))
        if op == "MaxPool":
            kernel, strides, dilations = a["kernel_shape"], a.get("strides", [1, 1]), a.get("dilations", [1, 1])
            top, left, bottom, right = _pads(x[0], a, kernel, strides, dilations)
            return F.max_pool2d(F.pad(x[0], [left, right, top, bottom], value=-float("inf")), kernel, strides, 0,
                                dilations, bool(a.get("ceil_mode", 0)))
        raise NotImplementedError(op)


def _sample(rng, batch, features, trackster_features):
    """Images with most slots empty, as the producer fills them, and the trackster quantities."""
    shape = (batch, 50, 10)
    columns = [rng.exponential(2.0, shape), rng.uniform(1.5, 3.0, shape), rng.uniform(-3.14, 3.14, shape)]
    if features == 7:
        columns += [rng.normal(0.0, 80.0, shape), rng.normal(0.0, 80.0, shape), rng.uniform(320.0, 520.0, shape),
                    rng.integers(1, 40, shape).astype(float)]
    occupied = rng.random(shape) < 0.08
    image = np.stack([np.where(occupied, column, 0.0) for column in columns], axis=1).astype(np.float32)
    if not trackster_features:
        return [image]
    quantities = [rng.exponential(30.0, batch), rng.exponential(15.0, batch), rng.normal(0.0, 80.0, batch),
                  rng.normal(0.0, 80.0, batch), rng.uniform(320.0, 400.0, batch), rng.uniform(1.5, 3.0, batch),
                  rng.uniform(-3.14, 3.14, batch)]
    return [image, np.stack(quantities, axis=1).astype(np.float32)]


def _device_layout(array):
    """The same values with the batch index the fastest one in memory, as the SoA that the producer wraps."""
    tensor = _tensor(array)
    order = list(range(1, tensor.dim())) + [0]
    return tensor.permute(order).contiguous().permute([tensor.dim() - 1] + list(range(tensor.dim() - 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx", help="image model")
    parser.add_argument("-o", "--output", required=True, help="TorchScript file")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="max |torch - onnxruntime| / max(|onnxruntime|, 1e-3)")
    args = parser.parse_args()

    graph = onnx.load(args.onnx).graph
    initializers = {t.name for t in graph.initializer}
    names = [i.name for i in graph.input if i.name not in initializers]
    features = [i for i in graph.input if i.name == names[0]][0].type.tensor_type.shape.dim[3].dim_value
    two_inputs = len(names) > 1

    rng = np.random.default_rng(12345)
    with torch.no_grad():
        module = torch.jit.trace(OnnxGraph(args.onnx).eval(),
                                 tuple(_tensor(a) for a in _sample(rng, 4, features, two_inputs)), check_trace=False)
    module.save(args.output)
    module = torch.jit.load(args.output)

    session = onnxruntime.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    worst = 0.0
    for batch in (1, 7, 64, 300):
        arrays = _sample(rng, batch, features, two_inputs)
        feed = {names[0]: np.ascontiguousarray(arrays[0].transpose(0, 2, 3, 1))}
        if two_inputs:
            feed[names[1]] = arrays[1]
        reference = session.run(None, feed)[0]
        with torch.no_grad():
            result = np.array(module(*[_device_layout(a) for a in arrays]).tolist(), dtype=np.float32)
        if result.shape != reference.shape:
            print("output shape %s, expected %s" % (result.shape, reference.shape), file=sys.stderr)
            worst = math.inf
            break
        worst = max(worst, float((np.abs(result - reference) / np.maximum(np.abs(reference), 1e-3)).max()))
    print("%s: max relative |torch - onnxruntime| = %.3g" % (args.onnx, worst))
    if not worst <= args.tolerance:
        print("the TorchScript module does not reproduce the ONNX model", file=sys.stderr)
        os.remove(args.output)
        return 1
    print("saved", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
