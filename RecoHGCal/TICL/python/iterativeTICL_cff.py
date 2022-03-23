import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.FastJetStep_cff import *
from RecoHGCal.TICL.CLUE3DHighStep_cff import *
from RecoHGCal.TICL.CLUE3DLowStep_cff import *
from RecoHGCal.TICL.EMStep_cff import *
from RecoHGCal.TICL.HADStep_cff import *

from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer as _pfTICLProducer
from RecoHGCal.TICL.trackstersMergeProducer_cfi import trackstersMergeProducer as _trackstersMergeProducer
from RecoHGCal.TICL.tracksterSelectionTf_cfi import *

ticlLayerTileTask = cms.Task(ticlLayerTileProducer)

ticlTrackstersMerge = _trackstersMergeProducer.clone()


pfTICL = _pfTICLProducer.clone()
ticlPFTask = cms.Task(pfTICL)

ticlIterationsTask = cms.Task(
    ticlEMStepTask
    ,ticlHADStepTask
)

from Configuration.ProcessModifiers.clue3D_cff import clue3D
clue3D.toModify(ticlIterationsTask, func=lambda x : x.add(ticlCLUE3DHighStepTask,ticlCLUE3DLowStepTask))

from Configuration.ProcessModifiers.fastJetTICL_cff import fastJetTICL
fastJetTICL.toModify(ticlIterationsTask, func=lambda x : x.add(ticlFastJetStepTask))

from Configuration.ProcessModifiers.ticl_v4_cff import ticl_v4
ticl_v4.toModify(ticlIterationsTask, func=lambda x : x.add(ticlCLUE3DHighStepTask))
ticl_v4.toModify(ticlTrackstersMerge, TICLV4 = cms.bool(True))

ticlIterLabels = [_step.itername.value() for _iteration in ticlIterationsTask for _step in _iteration if (_step._TypedParameterizable__type == "TrackstersProducer")]

ticlTracksterMergeTask = cms.Task(ticlTrackstersMerge)

iterTICLTask = cms.Task(ticlLayerTileTask
    ,ticlIterationsTask
    ,ticlTracksterMergeTask
    ,ticlPFTask
)
ticlIterLabelsMerge = ticlIterLabels + ["Merge"]

ticlLayerTileHFNose = ticlLayerTileProducer.clone(
    detector = 'HFNose'
)

ticlLayerTileHFNoseTask = cms.Task(ticlLayerTileHFNose)

iterHFNoseTICLTask = cms.Task(ticlLayerTileHFNoseTask
    ,ticlHFNoseEMStepTask
    ,ticlHFNoseHADStepTask
)
