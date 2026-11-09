"""The shower-shape PID of the device tracksters of CLUEstering, on the device: a copy of the device tracksters with
the 8 probabilities of the model for each trackster of at least 1 GeV. The model is mixPID_hi_v2.onnx, the
identification of the CLUEstering tracksters on the host, exported to TorchScript with
RecoTICL/Inference/test/exportShowerShapePIDToTorchScript.py."""

from RecoTICL.Configuration.tracksterInference_cff import makeTracksterInference

ticlSoATrackstersPID = makeTracksterInference(model="ShowerShape")
