"""Optional TICLv6 shower-shape PID validation: CPU ONNX versus device PyTorchAlpaka."""

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration.iterativeTICL_cff import (
    ticlCLUEsteringStepTask,
    _ticlV6Modules,
)
from RecoTICL.Configuration.presets import shower_shape_pid
from RecoTICL.Configuration.tracksterPIDComparisonAnalyzer_cfi import (
    tracksterPIDComparisonAnalyzer,
)

# Both converters read the same CLUEstering Tracksters. The CPU converter runs the shower-shape PID of
# mixPID_hi_v1.onnx on the host, while the production converter keeps the scores of its TorchScript export on the
# device (ticlSoATrackstersPID).
ticlTrackstersCLUEsteringCPU = _ticlV6Modules["ticlTrackstersCLUEstering"].clone(
    src=cms.InputTag("ticlSoATracksters"),
    inferenceAlgo=cms.string("TracksterInferenceByShowerShape"),
    pluginInferenceAlgoTracksterInferenceByShowerShape=shower_shape_pid("mixPID_hi_v1.onnx"),
    doPidCut=cms.bool(False),
)

ticlTracksterPIDComparison = tracksterPIDComparisonAnalyzer.clone(
    reference=cms.InputTag("ticlTrackstersCLUEsteringCPU"),
    candidate=cms.InputTag("ticlTrackstersCLUEstering"),
)

# Register these modules on process and associate the task with reconstruction.
# Place the analyzer on an EndPath and append it to explicit cms.Schedule.
ticlCLUEsteringPIDComparisonTask = cms.Task(
    ticlCLUEsteringStepTask, ticlTrackstersCLUEsteringCPU
)
ticlTracksterPIDComparisonEndPath = cms.EndPath(ticlTracksterPIDComparison)
