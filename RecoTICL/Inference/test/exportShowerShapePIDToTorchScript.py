#!/usr/bin/env python3
"""Export a mixture-mode shower-shape PID model (RecoTICL/Inference/data/showerShapePID/mixPID_*.onnx) to TorchScript,
for the device inference of TracksterShowerShapeInferenceProducer.

The TorchScript module is the ONNX graph op for op, with the weights and constants read from the ONNX file, followed by
the clamp to [0, 1] that TracksterInferenceByShowerShape applies on the host. The export is saved only if the module
reproduces onnxruntime (with the clamp) on random inputs around the standardisation of the model.

    python3 exportShowerShapePIDToTorchScript.py RecoTICL/Inference/data/showerShapePID/mixPID_hi_v1.onnx
"""

import argparse
import sys

import numpy as np
import onnx
import onnxruntime
import torch
from onnx import numpy_helper


def _tensor(array):
    # the CMSSW PyTorch has no NumPy bridge
    return torch.tensor(array.tolist(), dtype=torch.float32)


class ShowerShapePID(torch.nn.Module):
    """Input [N, 22]: the 16 trackster quantities and the 6 values of the layer-cluster profile of
    TracksterInferenceByShowerShape. Output [N, 8]: the probabilities in ticl::Trackster::ParticleType order."""

    def __init__(self, init, const):
        super().__init__()
        t = lambda name: _tensor(init[name])
        self.register_buffer("mean", t("em.net.mean"))
        self.register_buffer("std", t("em.net.std"))
        layers = []
        for k in (0, 2, 4, 6):
            w = t("em.net.body.%d.weight" % k)
            layer = torch.nn.Linear(w.shape[1], w.shape[0])
            with torch.no_grad():
                layer.weight.copy_(w)
                layer.bias.copy_(t("em.net.body.%d.bias" % k))
            layers.append(layer)
        self.hidden = torch.nn.ModuleList(layers[:-1])
        self.output = layers[-1]
        for name in ("knots", "dslope", "c0", "c1", "lo", "hi", "first", "last"):
            self.register_buffer(name, t("mapped." + name))
        self.register_buffer("lo_width", t("onnx::Div_203"))
        self.register_buffer("hi_width", t("onnx::Div_206"))
        self.s_min = float(const["/Constant_output_0"])
        self.s_max = float(const["/Constant_1_output_0"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the layer-cluster profile: max fraction, core fractions within 2 and 5 cm, log(1 + rms radius), z / 100
        profile = x[:, 16:]
        profile = torch.stack([profile[:, 0], profile[:, 1], profile[:, 2],
                               torch.log(1.0 + torch.clamp(profile[:, 3], min=0.0)),
                               profile[:, 4] / 100.0, profile[:, 5] / 100.0], dim=1)
        h = (torch.cat([x[:, :16], profile], dim=1) - self.mean) / self.std
        for layer in self.hidden:
            h = torch.nn.functional.silu(layer(h))
        p = torch.softmax(self.output(h), dim=1)
        # EM score: photon + electron + pi0, calibrated in logit space in bins of log(1 + raw energy)
        s = torch.clamp(p[:, 0] + p[:, 1] + p[:, 2], self.s_min, self.s_max)
        logit = torch.log(s) - torch.log(1.0 - s)
        spline = self.c0 + (torch.relu(logit[:, None, None] - self.knots) * self.dslope).sum(dim=2)
        spline = torch.minimum(torch.maximum(spline, self.c0), self.c1)
        energy = x[:, 0:1]
        w_lo = torch.clamp((energy - self.lo) / self.lo_width, 0.0, 1.0)
        w_hi = torch.clamp((self.hi - energy) / self.hi_width, 0.0, 1.0)
        weight = torch.minimum(torch.maximum(w_lo, self.first), torch.maximum(w_hi, self.last))
        mapped = torch.clamp((weight * spline).sum(dim=1), 0.0, 1.0)
        em = mapped / s
        had = (1.0 - mapped) / (1.0 - s)
        out = torch.stack([p[:, 0] * em,             # photon
                           p[:, 1] * em,             # electron
                           p[:, 5] * had,            # muon
                           p[:, 2] * em,             # neutral pi0
                           p[:, 3] * had,            # charged hadron
                           p[:, 4] * had,            # neutral hadron
                           torch.zeros_like(s),      # ambiguous
                           p[:, 6] * had], dim=1)    # unknown
        return torch.clamp(out, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx", help="mixture-mode shower-shape PID model")
    parser.add_argument("-o", "--output", help="TorchScript file (default: the ONNX path with .pt)")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="max |torch - onnxruntime| per probability")
    args = parser.parse_args()
    output = args.output or args.onnx[:-len(".onnx")] + ".pt"

    graph = onnx.load(args.onnx).graph
    init = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    const = {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in graph.node if n.op_type == "Constant"}
    module = torch.jit.script(ShowerShapePID(init, const).eval())

    # inputs around the standardisation of the model, with the tails and the profile defaults of empty tracksters
    rng = np.random.default_rng(12345)
    x = (init["em.net.mean"] + rng.normal(size=(20000, 22)) * 2.0 * init["em.net.std"]).astype(np.float32)
    x[:100, 16:21] = -1.0
    x[100:200, 21] = 0.0
    session = onnxruntime.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    reference = np.clip(session.run(None, {graph.input[0].name: x})[0], 0.0, 1.0)
    with torch.no_grad():
        result = np.array(module(_tensor(x)).tolist(), dtype=np.float32)
    diff = np.abs(result - reference).max()
    print("%s: max |torch - onnxruntime| = %.3g on %d inputs" % (args.onnx, diff, len(x)))
    if not diff <= args.tolerance:
        print("the TorchScript module does not reproduce the ONNX model: not saved", file=sys.stderr)
        return 1
    module.save(output)
    reloaded = torch.jit.load(output)
    with torch.no_grad():
        assert np.array_equal(np.array(reloaded(_tensor(x)).tolist(), dtype=np.float32), result)
    print("saved", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
