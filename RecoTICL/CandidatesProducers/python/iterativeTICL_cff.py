import FWCore.ParameterSet.Config as cms

from RecoTICL.TrackstersProducers.FastJetStep_cff import *
from RecoTICL.TrackstersProducers.CLUE3DHighStep_cff import *
from RecoTICL.TrackstersProducers.MIPStep_cff import *
from RecoTICL.TrackstersProducers.TrkEMStep_cff import *
from RecoTICL.TrackstersProducers.TrkStep_cff import *
from RecoTICL.TrackstersProducers.EMStep_cff import *
from RecoTICL.TrackstersProducers.HADStep_cff import *
from RecoTICL.TrackstersProducers.CLUE3DEM_cff import *
from RecoTICL.TrackstersProducers.CLUE3DHAD_cff import *

from RecoTICL.LayerClustersProducers.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoTICL.CandidatesProducers.pfTICLProducer_cfi import pfTICLProducer as _pfTICLProducer
from RecoTICL.LinkingProducers.trackstersMergeProducer_cfi import trackstersMergeProducer as _trackstersMergeProducer
from RecoTICL.TrackstersProducers.tracksterSelectionTf_cfi import *

from RecoTICL.LinkingProducers.tracksterLinksProducer_cfi import tracksterLinksProducer as _tracksterLinksProducer
from RecoTICL.CandidatesProducers.ticlCandidateProducer_cfi import ticlCandidateProducer as _ticlCandidateProducer

from RecoTICL.CandidatesProducers.mtdSoAProducer_cfi import mtdSoAProducer as _mtdSoAProducer

ticlLayerTileTask = cms.Task(ticlLayerTileProducer)

ticlTrackstersMerge = _trackstersMergeProducer.clone()
ticlTracksterLinks = _tracksterLinksProducer.clone()
ticlCandidate = _ticlCandidateProducer.clone()
mtdSoA = _mtdSoAProducer.clone()

pfTICL = _pfTICLProducer.clone()
ticlPFTask = cms.Task(pfTICL)

ticlIterationsTask = cms.Task(
    ticlCLUE3DEMStepTask,
    ticlCLUE3DHADStepTask,
    ticlCLUE3DHighStepTask
)

ticlCandidateTask = cms.Task(mtdSoA, ticlCandidate)


from Configuration.ProcessModifiers.fastJetTICL_cff import fastJetTICL
fastJetTICL.toModify(ticlIterationsTask, func=lambda x : x.add(ticlFastJetStepTask))


ticlIterLabels = ["CLUE3DEM", "CLUE3DHAD", "CLUE3DHigh"]

ticlTracksterMergeTask = cms.Task(ticlTrackstersMerge)
ticlTracksterLinksTask = cms.Task(ticlTracksterLinks)


mergeTICLTask = cms.Task(ticlLayerTileTask
    ,ticlIterationsTask
    ,ticlTracksterMergeTask
    ,ticlTracksterLinksTask
)

ticlIterLabelsMerge = ticlIterLabels + ["Merge"]


iterTICLTask = cms.Task(mergeTICLTask
    ,ticlCandidateTask
    ,ticlPFTask)

ticlLayerTileHFNose = ticlLayerTileProducer.clone(
    detector = 'HFNose'
)

ticlLayerTileHFNoseTask = cms.Task(ticlLayerTileHFNose)

iterHFNoseTICLTask = cms.Task(ticlLayerTileHFNoseTask
    ,ticlHFNoseTrkEMStepTask
    ,ticlHFNoseEMStepTask
    ,ticlHFNoseTrkStepTask
    ,ticlHFNoseHADStepTask
    ,ticlHFNoseMIPStepTask
)
