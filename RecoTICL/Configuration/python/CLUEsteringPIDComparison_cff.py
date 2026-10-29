"""Optional TICLv6 CNN PID validation: CPU ONNX versus GPU PyTorchAlpaka."""

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration.iterativeTICL_cff import (
    ticlCLUEsteringStepTask,
    ticlTrackstersCLUEstering,
    _ticlV6Modules,
)
from RecoTICL.Configuration.tracksterPIDComparisonAnalyzer_cfi import (
    tracksterPIDComparisonAnalyzer,
)

# Both converters use the same CLUEstering Tracksters. The CPU converter runs
# the legacy CNN, while the production converter preserves PyTorchAlpaka scores.
ticlTrackstersCLUEsteringCPU = _ticlV6Modules["ticlTrackstersCLUEstering"].clone(
    src=cms.InputTag("ticlSoATracksters"),
    inferenceAlgo=cms.string("TracksterInferenceByCNN"),
    pluginInferenceAlgoTracksterInferenceByCNN=cms.PSet(
        algo_verbosity=cms.int32(0),
        type=cms.string("TracksterInferenceByCNN"),
        onnxModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/CNN/patternrecognition/id_v0.onnx"),
        inputNames=cms.vstring("input"),
        outputNames=cms.vstring("pid_output"),
        eid_min_cluster_energy=cms.float(1.0),
        eid_n_layers=cms.int32(50),
        eid_n_clusters=cms.int32(10),
        doPID=cms.int32(1),
        miniBatchSize=cms.untracked.int32(64),
    ),
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
