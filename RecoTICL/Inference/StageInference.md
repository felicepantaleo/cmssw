# Candidate-stage inference design notes

These notes describe the earlier candidate-stage implementation. The development
branch has since removed its C++ bridge and producer hook; the retained Python
helpers and model contracts do not by themselves enable candidate PyTorch inference.
They are retained for future candidate work, outside this superclustering cleanup.

Superclustering uses the existing CPU ONNX implementation. The experimental
PyTorch pair-scoring interface, tensor adaptation and configuration helper have
been removed so that preprocessing and integration can be redesigned separately.

## Candidate reconstruction

`TICLCandidateProducer` selects `inferenceAlgo="PyTorchAlpaka"` through its existing
`TracksterInferenceAlgoBase` hook. Each CPU EDM stream owns its backend plugin,
queue and independent optional PID/regression models. Inference still happens
**after both interpretation algorithms and PCA assignment, before the reconstructed
Tracksters are put into the event and before final candidates are constructed**.
There is no additional producer, published intermediate collection or changed token.

The bridge stages the reconstructed Tracksters' vertices, multiplicities, energies
and PCA-derived barycenter into the existing Trackster/CaloCluster SoAs. It runs the
same `TracksterFeatureExtractionRunner` and `TracksterInferenceRunner` used by the
previous device producers. PID and regression share feature extraction and update
the same private device collection. Only the two prediction blocks (eight PID
columns and regressed energy, including portable column padding) come back to the
CPU. No vertex/PCA payload is copied back. Only these predictions are written into
the existing CPU objects; timing, eigenvalues/eigenvectors, vertices, tracks, GSF
tracks and other associations are untouched. Input interpretation Tracksters remain
distinct from the reconstructed output Tracksters. Candidate construction, muon
momentum treatment and the original/reconstructed-object links are unchanged.

DNN/PFN reset both predictions for selected Tracksters, then run enabled tasks,
as in the CPU implementations. Unselected Tracksters retain their predictions.
PFN regression returns raw energy when raw energy is **not strictly greater than**
`minClusterEnergy`; PID still runs for those selected objects. CNN remains PID-only.
Disabling both tasks is a no-op and does not resolve any model paths. Setting the
producer's `regressionAndPid=False` bypasses plugin construction entirely.

Geometry comes from the first cluster hit, matching CPU preprocessing, whereas
barrel exclusion uses the cluster seed. Host layers are already endcap-local and
are not folded by the detector layer count. Empty-hit clusters are assigned layer
-1 and cannot occupy an image slot. Valid CPU inputs are expected to have hits.
CaloCluster coordinates/energy are staged as floats, as in the existing SoAs:
real-event checks must still assess rounding near selection thresholds and equal
energies. The PCA scalar ordering for PFN is unchanged; eigenvalues are not PFN
inputs in this implementation.

```python
# After loading the ordinary iterative TICL configuration:
from RecoTICL.Configuration.stageInference_cff import withCandidateInference

process.ticlCandidate = withCandidateInference(
    process.ticlCandidate,
    model="PFN", doPID=True, doRegression=True,
    pidModel="RecoTICL/Inference/data/candidate_pid_pfn.pt",
    regressionModel="RecoTICL/Inference/data/candidate_energy_pfn.pt",
    minClusterEnergy=2.5,
    backend="cuda_async", batchSize=64,
    convertToFP16=False,
)
# DNN is selected with model="DNN" and the corresponding two model paths.
# CNN: model="CNN", doPID=True, doRegression=False.
```

Alternatively, `makeCandidateInference(...)` returns just the PSet:

```python
from RecoTICL.Configuration.stageInference_cff import makeCandidateInference
process.ticlCandidate.inferenceAlgo = "PyTorchAlpaka"
process.ticlCandidate.pluginInferenceAlgoPyTorchAlpaka = makeCandidateInference(
    model="DNN", doPID=False, doRegression=True,
    regressionModel="RecoTICL/Inference/data/candidate_energy_dnn.pt",
)
```

The CPU PFN PSet remains available; select `inferenceAlgo="TracksterInferenceByPFN"`
to restore it. The model factory helpers are also imported by `iterativeTICL_cff`.

## Production model contracts

All paths below are **required future data**, resolved with `edm::FileInPath` only
for enabled tasks. They are not supplied by these patches. Other CMSSW data-package
paths may be configured. Missing enabled models fail explicitly; there is no silent
ONNX fallback. Serial/CUDA/ROCm plugin selection is explicit via `serial_sync`,
`cuda_async`, `rocm_async`. An unavailable backend/device is an error.

| Task | Default path in `RecoTICL/Inference/data/` | Forward inputs | Forward output |
|---|---|---|---|
| Candidate CNN PID | `candidate_pid_cnn.pt` | `[B,3,50,10]` | `[B,8]` |
| Candidate DNN PID | `candidate_pid_dnn.pt` | `[B,50,10,3]` | `[B,8]` |
| Candidate DNN regression | `candidate_energy_dnn.pt` | `[B,50,10,3]` | `[B,1]` or `[B]` |
| Candidate PFN PID | `candidate_pid_pfn.pt` | `[B,50,10,7]`, then `[B,7]` | `[B,8]` |
| Candidate PFN regression | `candidate_energy_pfn.pt` | `[B,50,10,7]`, then `[B,7]` | `[B,1]` or `[B]` |

The TorchScript forward result must be one floating-point Tensor on the selected
Alpaka device, not a tuple/dictionary. Models must support dynamic positive B,
including the final partial minibatch, use no hard-coded device, and not mutate
inputs. No model is called for an empty collection. Missing image slots are zero;
minibatch tails contain only real rows, without extra padded samples. Portable SoA
alignment padding is not part of a tensor's logical shape. `convertToFP16=False`
uses FP32; True converts both weights and inputs to FP16 and copies predictions
into the FP32 output storage. FP16 suitability requires validation with real models.

CNN/DNN cluster channels are corrected cluster energy (energy/multiplicity),
absolute eta, phi. PFN appends x, y, absolute z, cell count. The PFN second tensor is
raw energy, raw EM energy, barycenter x, barycenter y, absolute barycenter z, absolute
barycenter eta, barycenter phi. Full descending cluster-energy ordering precedes
per-layer top-ten truncation. PID class order is photon, electron, muon, neutral
pion, charged hadron, neutral hadron, ambiguous, unknown. Predictions are used as
returned: no added activation/normalization. Regression must supply physical energy;
candidate-stage models are distinct from first-stage models.

## Shared execution

The standalone Trackster runner owns `AlpakaModel` directly and delegates batching
and output handling to its existing `forward()` overload. TICL performs no input
permutation or regression-output normalization. See the current standalone tensor
contracts in [README.md](README.md); the earlier candidate contracts above are
historical and would need updating when its bridge is reintroduced.

## Validation

The previous stage/preprocessing test sources have been removed from the upstream
development branch. Historical test results do not validate its current state.
Full CMSSW builds, backend tests and inference-output comparisons remain required.
