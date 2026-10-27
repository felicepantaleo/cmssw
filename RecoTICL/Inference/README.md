# Portable Trackster inference

This implementation extends `heterogeneous-trackster` at
`3ca7bfc14e16169b5df847653be120d62fa9d8cf`. Production execution uses PyTorchAlpaka.
The CPU ONNX implementations are unchanged and remain the physics references.

## Configuration

```python
from RecoTICL.Configuration.tracksterInference_cff import makeTracksterInference

process.ticlTracksterInferenceFeatures, inference = makeTracksterInference(
    model="PFN", doPID=True, doRegression=True, batchSize=64,
    tracksters="ticlTrackstersToSoAProducer",
)
process.ticlTracksterInference = inference.clone(
    features=cms.InputTag("ticlTracksterInferenceFeatures", "features")
)
process.ticlTracksterInferenceTask = cms.Task(
    process.ticlTracksterInferenceFeatures, process.ticlTracksterInference
)
process.reconstruction_step.associate(process.ticlTracksterInferenceTask)
process.myConsumer.tracksters = cms.InputTag("ticlTracksterInference")
```

The factory returns a pair of EDM producer configurations: a feature extractor
and an inference producer. Assign both to the process, give them unique labels,
and set the inference producer's `features` InputTag to the extractor's
`features` instance. Schedule them together in a `cms.Task`. Calling the
factory again creates independent module configurations.

Other supported selections:

```python
cnn_features, cnn_inference = makeTracksterInference(
    model="CNN", doPID=True, doRegression=False
)
dnn_features, dnn_inference = makeTracksterInference(
    model="DNN", doPID=True, doRegression=True
)
pfn_features, pfn_inference = makeTracksterInference(
    model="PFN", doPID=False, doRegression=True,
    regressionModel="RecoTICL/Inference/data/my_energy.pt"
)
```

`pidModel` and `regressionModel` override the defaults independently. Both tasks
may be disabled, in which case the output is an unchanged device copy and the
feature product is not retrieved. CNN regression is rejected in Python and C++.
The three standalone inference plugins share one producer template and the existing
runners and tensor adapters. `CLUEsteringStep_cff` uses this factory to schedule CNN PID feature extraction
and inference before conversion to legacy Tracksters.
The superseded `TracksterInferenceProducer@alpaka` plugin and hand-written
`Configuration.tracksterInferenceFeaturesProducer_cfi` are removed; custom
configurations should use the factory (`pidModel` replaces `model`). The output
remains `ticl::TracksterDevice` with no instance label. CPU ONNX settings are unchanged.

Validation is opt-in: load
`RecoTICL.Configuration.tracksterPIDComparisonAnalyzer_cfi` and schedule
`process.ticlTracksterPIDComparison` on an EndPath. This configured analyzer is
no longer exported by the production CLUE3D fragment; it was never in its task.


`layerClusters` has three entries, in exactly the concatenation order used by
the input Trackster vertex indices (by default EE, HSi, HSci). All families
publish the same `ticl::TracksterDevice` product with unchanged associations and
Trackster order. This iteration retains the current HGCAL geometry convention;
it does not add a barrel layer-clustering pipeline.

CMSSW's experimental `Process.setStrict(True)` is not supported here: at the
base revision, assigning a `cms.Task` to a strict process itself raises
`AttributeError: Task has no attribute setIsFrozen`. Ordinary process
assignment, template cloning, task association and dump/reload are tested.

## TorchScript contracts

Names below are registration labels. PyTorchAlpaka calls `forward` with
**positional** tensor arguments in the listed order; it does not resolve ONNX
input/output names. Each independent model returns a single floating-point
Tensor on the input device, with no dictionary or tuple wrapper.

| Family | First argument, registration `input` | Second argument | PID result | Regression result |
|---|---|---|---|---|
| CNN | `[B, 3, 50, 10]` | none | `[B, 8]` | unsupported |
| DNN | `[B, 3, 50, 10]` | none | `[B, 8]` | `[B, 1]` |
| PFN | `[B, 7, 50, 10]` | `input_tr_features`: `[B, 7]` | `[B, 8]` | `[B, 1]` |

PID and regression output labels are `pid_output` and `enreg_output`.
Input and stored output features use float32 by default. The optional existing
`convertToFP16` setting converts model weights and input tensors to float16;
results are copied back to float32 SoAs. Validate FP32 first. The final
minibatch has its actual size, not `batchSize` padded rows. Models must support
dynamic B and strided inputs, or explicitly call `contiguous()` where needed.
They must not mutate their inputs. Zero selected rows bypass model execution.
Warmup uses zero features and the configured full batch size; set
`warmupIterations=0` if a future model cannot accept those inputs.

**CNN layout discrepancy in the request:** the actual supplied
`trackster_pid_cnn.pt` contains a Conv2d model with three input channels, taking
NCHW directly. Its `[B, 3, 50, 10]` contract is preserved. The semantic image is
still 50 layers, 10 clusters and 3 features. SoA registration exposes
`[B, F, L, C]` for all families. DNN/PFN exports whose internal network expects
`[B, L, C, F]` must perform `image.permute(0, 2, 3, 1)` inside the model.
PFN's second tensor needs no permutation. Existing channels-last-only exports
are incompatible until wrapped/re-exported; no model files are modified here.
`TensorCollection::change_order()` orders model arguments, not tensor axes.
Regression uses the standard PyTorchAlpaka output handling with registered
`[B,1]` storage; TICL does not reshape or unsqueeze the model result.

Default model paths, relative to the CMSSW search path:

- `RecoTICL/Inference/data/trackster_pid_cnn.pt` (existing)
- `RecoTICL/Inference/data/trackster_pid_dnn.pt` (to be supplied)
- `RecoTICL/Inference/data/trackster_energy_dnn.pt` (to be supplied)
- `RecoTICL/Inference/data/trackster_pid_pfn.pt` (to be supplied)
- `RecoTICL/Inference/data/trackster_energy_pfn.pt` (to be supplied)

Missing enabled models produce a `TracksterInferenceModel` exception naming the
parameter, path and remedy. Paths are configuration strings: missing files do
not prevent configuration generation or C++ compilation. Only enabled tasks
resolve/load their files. No model was created, exported or trained here.

## Physics and input provenance

| Behavior | CNN | DNN | PFN |
|---|---|---|---|
| Selection | First prefix of raw cluster energies reaching threshold | Same, excluding barrel clusters | All Tracksters containing no barrel cluster, including empty ones |
| Barrel contributions | Included as in CPU CNN | Skipped in selection and image | Entire Trackster excluded |
| Selected output initialization | PID overwritten; regression preserved | Both PID and regression zeroed before enabled tasks | Same as DNN |
| Regression | none | Assign model energy directly | Assign model energy only when raw Trackster energy is strictly above threshold; otherwise raw energy |
| Non-selected output | Preserve original values | Preserve original values | Preserve original values |

Cluster features are ordered as follows:

- CNN/DNN: `energy / vertex_multiplicity`, `abs(eta)`, `phi`.
- PFN: those three, then `x`, `y`, `abs(z)`, `number of cells`.
- PFN Trackster features: `raw_energy`, `raw_em_energy`, `barycenterX`,
  `barycenterY`, `abs(barycenterZ)`, `abs(barycenterEta)`, `barycenterPhi`.

Clusters are sorted by **uncorrected** energy before the first ten per layer
are retained; multiplicity correction is applied afterwards. Missing slots
are zero. Selection uses an inclusive prefix scan, preserving original
Trackster order and an explicit selected-to-original mapping.

The existing CaloCluster position block provides `x/y/z/layer/cells`; the
indexes block provides `seedID`. `HGCalLayerClustersSoAAlgoKernelEnergy` counts
assigned cells and records the seed; no shared data format change is needed.
The barrel predicate follows `ticlgeom::Tools::isBarrel` (EB, HB, HO).
The layer is the upstream zero-based layer modulo the endcap layer count.
This corresponds to CPU `getLayerWithOffset(firstHit)-1` for the HGCAL
layer-homogeneous clusters produced here. A future source of mixed-layer
clusters must provide the CPU-equivalent layer; this code cannot reconstruct
first-hit identity from a SoA that does not store the hit list. Likewise,
upstream SoA cell counts and coordinates must actually represent the same
clusters supplied to the CPU reference. Full event-data parity remains to be
measured; synthetic fixtures cannot establish that upstream equivalence.

Two inherited preprocessing discrepancies were corrected:

1. Threshold selection now stops when the running sum reaches the threshold,
   matching the CPU even with negative later cluster energies; an empty
   Trackster is not selected by CNN/DNN at a nonpositive threshold.
2. Zero transverse coordinates retain energy and follow ROOT's finite eta
   convention (`abs(z) + 22756`, or zero at the origin). The old GPU CNN
   silently zeroed the whole slot; the CPU code did not.

The original per-layer top-k also did not reproduce CPU `std::sort` ordering
for tied energies. `TracksterClusterOrder` uses a device-compatible introsort
matching libstdc++'s partition/insertion/heap strategy. A host parity test
compares the complete index permutation, including ties. Equivalent-element
ordering is not guaranteed by the C++ standard: rerun that test with the actual
CMSSW compiler/STL and after upgrades. Sorting now needs an event-local device
index buffer (four bytes per vertex association) and one-thread-per-Trackster
scratch. Its performance has not been benchmarked.

ROOT convention sources:
[eta.h](https://github.com/root-project/root/blob/master/math/genvector/inc/Math/GenVector/eta.h),
[etaMax.h](https://github.com/root-project/root/blob/master/math/genvector/inc/Math/GenVector/etaMax.h).

## Architecture and changed areas

- `TracksterFeatureExtractor<Family>` shares selection/ordering/image kernels
  while retaining compile-time family-specific behavior. DNN uses the existing
  three-feature SoA; PFN has its own seven-image/seven-scalar SoA and dictionaries.
- `TracksterFeatureExtractionRunner` retains the acquire/produce lifecycle and
  sole selected-count device-to-host transfer. The cluster seed view is passed
  alongside energy/position views.
- Input adapters describe each tensor; output adapters scatter PID or energy.
  `TracksterInferenceRunner::runInto` updates an already copied Trackster output.
- `TracksterMultiTaskInferenceProducerT` owns independent optional runners for
  PID and regression, extracts no features itself and copies Tracksters once.
  Each framework stream owns separate TorchScript instances on its fixed queue.
- `TracksterInferenceRunner` owns `AlpakaModel` directly and calls its batched
  `forward(queue, inputs, outputs, dtype)` overload for both warmup and events.
  PyTorchAlpaka handles batch consistency, materialization, input copying,
  device placement, queue guarding and output conversion. TICL retains a
  `NoGradGuard` because this overload invokes the underlying script module.
- No TICL batching loop, input transform, output-shape normalization or custom
  output checks remain. Model contracts must match the registered tensors.
  No PhysicsTools changes are needed for this simplification.

PID and regression share extracted device features, but each model still has
its own tensor registration/materialization. This does not promise zero
additional internal PyTorch allocations or eliminate the input staging already
used by PyTorchAlpaka.

## Validation

Executed in the patch-authoring environment:

- Four Python unittest methods using the branch's real CMSSW Python code:
  supported task combinations, invalid configurations, unique module wiring,
  template cloning and full `process.dumpPython()`/reload. Passed.
- Standalone C++20 `testTracksterClusterOrder`: 16,384 full permutation
  comparisons with GCC 13.3/libstdc++ `std::sort`; includes identical energies,
  monotone/organ-pipe sequences and random ties, sizes 0 through 2047. Passed.
- The same test under AddressSanitizer and UndefinedBehaviorSanitizer. Passed
  with leak scanning disabled because this sandbox prevents LeakSanitizer from
  reading process metadata. No claim of leak testing is made.
- Python syntax, dictionary structure checks, whitespace and patch application checks.

**Not executed:** CMSSW/SCRAM compilation, serial/CUDA/ROCm preprocessing
binaries, model runtime, multistream runtime, event-data comparisons and
performance. This environment has no `scram`, `nvcc`, `hipcc`, CMSSW externals
or GPU tooling. The new DNN/PFN production models are also absent.
**Physics equivalence and backend compilation are not established.**

The model-free Alpaka test `testTracksterPreprocessing.dev.cc` builds synthetic
Trackster/CaloCluster SoAs, runs the production extraction runner, transfers
features only for validation, and compares them to an independent host oracle
transcribed from the CPU ONNX loops. It checks all image slots and all PFN
scalar features, mapping/order, multiplicity, layer positions, top ten,
zero-padding, both endcaps, invalid layers, barrel-only/mixed Tracksters,
empty input/selection, zero transverse coordinates and ties. It additionally
materializes minibatches of two through the real tensor adapters and checks
tensor dimensions, strides and partial tails. No mock neural network is used.
The CPU oracle uses full `std::sort`, independently of the device algorithm;
it does not invoke `runInference()` because that requires a model/session.

In an initialized CMSSW 20_1_0_pre4 environment containing this branch:

```bash
scram b -j 8
python3 RecoTICL/Inference/test/testTracksterInferenceConfig.py
scram b runtests TEST=testTracksterClusterOrder
# Discover the backend-suffixed executables produced by SCRAM:
find "$CMSSW_BASE/test/$SCRAM_ARCH" -maxdepth 1 -type f -name 'testTracksterPreprocessing*' -print
```

Run each listed preprocessing executable on a matching serial/CUDA/ROCm node.
They require no model files or input events. Exit 77 explicitly means that no
device was available; it must not be counted as a pass. Build and run each
backend supported by the release/toolchain; building CUDA does not validate
ROCm. To run the compiler/STL parity test without SCRAM:

```bash
g++ -std=c++20 -O2 -I. RecoTICL/Inference/test/testTracksterClusterOrder.cc -o /tmp/testTracksterClusterOrder
/tmp/testTracksterClusterOrder
```

Next validate real event preprocessing against CPU CaloCluster/Trackster
objects and geometry, especially seed-vs-first-hit layer provenance, SoA float
rounding and cell counts. Keep inference disabled during that comparison.
Then supply production models, validate inference separately and measure
multistream memory/throughput.

## Toward fewer producers

The runners have no EDM Event dependency. A future device-native Trackster
producer can invoke extraction and inference directly while retaining the same
policies/adapters. First remove the legacy Trackster-to-SoA bridge upstream.
Then consider device-count-aware batching to remove the selected-count host
round trip, and share materialized input tensors across the two models. Measure
those changes before fusing producers or replacing the now-tested ordering
policy with a faster kernel.
