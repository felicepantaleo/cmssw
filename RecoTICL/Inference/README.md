# Trackster inference on the device

`Trackster{CNN,DNN,PFN,ShowerShape}InferenceProducer@alpaka` (`plugins/alpaka/TracksterInferenceProducer.cc`) read
the device tracksters (`ticl::TracksterDeviceCollection`) and put a copy with the PID, and with a regression the
regressed energy, of the tracksters that the model selects. Per event:

1. `acquire()`: select the tracksters and count them; the count is copied to the host.
2. `produce()`: fill the inputs of the selected tracksters, run the PID and regression models with PyTorchAlpaka in
   batches of `batchSize`, and write the predictions into the copy.

The device work is in `src/alpaka/TracksterInference.dev.cc`, the model inputs in `interface/alpaka/TracksterInference.h`
and their layouts in `interface/TracksterInferenceSoA.h`.

## Configuration

`RecoTICL/Configuration/python/tracksterInference_cff.py` makes the producer of any model:

```python
from RecoTICL.Configuration.tracksterInference_cff import makeTracksterInference
ticlTrackstersPID = makeTracksterInference(model="PFN", doPID=True, doRegression=True, tracksters="ticlSoATracksters")
```

pyTICL schedules it on the TICLv6 CLUEstering tracksters with `TICLConfig.device_pid()`, by default the shower-shape PID
(`python/tracksterShowerShapePID_cff.py`). Another model is `device_pid(model="DNN", doRegression=True)`.

## Models

| Model | Selection | Inputs | Before the predictions | Regression |
|---|---|---|---|---|
| CNN | summed layer-cluster energy >= `energyThreshold` | image | - | none |
| DNN | as CNN, without the barrel layer clusters | image without the barrel layer clusters | PID and regressed energy zeroed | model energy |
| PFN | no barrel layer cluster (empty tracksters too) | image + trackster quantities | PID and regressed energy zeroed | model energy if raw energy > `energyThreshold`, else raw energy |
| ShowerShape | raw energy >= `energyThreshold` | 22 shower-shape quantities | - | none |

The tracksters that a model does not select keep their values.

- Image `[B, F, 50, 10]`: per layer, the 10 most energetic layer clusters by decreasing uncorrected energy, zero padded.
  CNN/DNN: `F = 3`, energy, `|eta|`, `phi` (the trackster stores no vertex multiplicity). PFN: `F = 7`, those, then `x`,
  `y`, `|z|` and the number of cells. A model trained on `[B, 50, 10, F]` permutes inside its export.
- PFN trackster quantities `[B, 7]`: raw energy, raw EM energy, barycenter `x`, `y`, `|z|`, `|eta|`, `phi`.
- Shower shape `[B, 22]`: the 16 quantities of `TracksterInferenceByShowerShape::features` and the energy profile of the
  trackster (`TracksterSoA` block `profile`, as `ticl::layerClusterProfile`).
- Outputs: `pid_output` `[B, 8]` in `ticl::Trackster::ParticleType` order, `enreg_output` `[B, 1]`.

Model files, relative to the CMSSW search path (only the enabled models are loaded):

- `RecoTICL/Inference/data/showerShapePID/mixPID_hi_v2.pt`: `mixPID_hi_v2.onnx`, exported with `test/exportShowerShapePIDToTorchScript.py`

## Validation

