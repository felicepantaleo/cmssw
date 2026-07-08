import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.FastJetStep_cff import *
from RecoHGCal.TICL.CLUE3DHighStep_cff import *
from RecoHGCal.TICL.MIPStep_cff import *
from RecoHGCal.TICL.TrkEMStep_cff import *
from RecoHGCal.TICL.TrkStep_cff import *
from RecoHGCal.TICL.EMStep_cff import *
from RecoHGCal.TICL.HADStep_cff import *
from RecoHGCal.TICL.CLUE3DEM_cff import *
from RecoHGCal.TICL.CLUE3DHAD_cff import *
from RecoHGCal.TICL.PRbyRecovery_cff import *
from RecoHGCal.TICL.CLUE3DBarrel_cff import *

from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer as _pfTICLProducer

from RecoHGCal.TICL.tracksterLinksProducer_cfi import tracksterLinksProducer as _tracksterLinksProducer
from RecoHGCal.TICL.superclustering_cff import *
from RecoHGCal.TICL.ticlCandidateProducer_cfi import ticlCandidateProducer as _ticlCandidateProducer

from RecoHGCal.TICL.mtdSoAProducer_cfi import mtdSoAProducer as _mtdSoAProducer
from Configuration.ProcessModifiers.ticlv5_TrackLinkingGNN_cff import ticlv5_TrackLinkingGNN

from Configuration.ProcessModifiers.ticl_superclustering_mustache_pf_cff import ticl_superclustering_mustache_pf
from Configuration.ProcessModifiers.ticl_superclustering_mustache_ticl_cff import ticl_superclustering_mustache_ticl
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel

ticlLayerTileTask = cms.Task(ticlLayerTileProducer)


# TICLv5 is now the default configuration
ticlTracksterLinks = _tracksterLinksProducer.clone(
    tracksters_collections = cms.VInputTag(
        'ticlTrackstersCLUE3DHigh',
        'ticlTrackstersRecovery'
    ),
    linkingPSet = cms.PSet(
      cylinder_radius_sqr_split = cms.double(9),
      proj_distance_split = cms.double(5),
      track_time_quality_threshold = cms.double(0.5),
      min_num_lcs = cms.uint32(15),
      min_trackster_energy = cms.double(20),
      pca_quality_th = cms.double(0.85),
      dot_prod_th = cms.double(0.97),
      lower_boundary = cms.vdouble(20, 10),  
      upper_boundary = cms.vdouble(150, 100),  
      upper_distance_projective_sqr = cms.vdouble(4, 60),  
      lower_distance_projective_sqr = cms.vdouble(4, 60),  
      min_distance_z = cms.vdouble(35, 35),  
      upper_distance_projective_sqr_closest_points = cms.vdouble(5, 30),  
      lower_distance_projective_sqr_closest_points = cms.vdouble(10, 50),  
      max_z_distance_closest_points = cms.vdouble(35, 35),
      cylinder_radius_sqr = cms.vdouble(9, 15),  
      deltaRxy = cms.double(4.),
      algo_verbosity = cms.int32(0),
      type = cms.string('Skeletons')
    ),  
    regressionAndPid = cms.bool(False),
    inferenceAlgo = cms.string(''),
    pluginInferenceAlgoTracksterInferenceByDNN = cms.PSet(
        algo_verbosity = cms.int32(0),
        doPID = cms.int32(1),
        doRegression = cms.int32(1),
        inputNames  = cms.vstring('input'),
        output_en   = cms.vstring('enreg_output'),
        output_id   = cms.vstring('pid_output'),
        eid_min_cluster_energy = cms.double(1),
        eid_n_clusters = cms.int32(10),
        eid_n_layers = cms.int32(50),
        onnxEnergyModelPath = cms.string('RecoHGCal/TICL/data/ticlv5/onnx_models/DNN/linking/energy_v0.onnx'),
        onnxPIDModelPath = cms.string('RecoHGCal/TICL/data/ticlv5/onnx_models/DNN/linking/id_v0.onnx'),
        type = cms.string('TracksterInferenceByDNN')
    ),
    pluginInferenceAlgoTracksterInferenceByPFN = cms.PSet(
        algo_verbosity = cms.int32(0),
        doPID = cms.int32(1),
        doRegression = cms.int32(1),
        inputNames  = cms.vstring('input','input_tr_features'),
        output_en   = cms.vstring('enreg_output'),
        output_id   = cms.vstring('pid_output'),
        eid_min_cluster_energy = cms.double(2.5),
        eid_n_clusters = cms.int32(10),
        eid_n_layers = cms.int32(50),
        onnxEnergyModelPath = cms.string('RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/linking/energy_v1.onnx'),
        onnxPIDModelPath = cms.string('RecoHGCal/TICL/data/ticlv5/onnx_models/CNN/linking/id_v0.onnx'),
        type = cms.string('TracksterInferenceByPFN')
    )
)

# Interpretation instance of the trackster-links producer: hosts the track <->
# trackster interpretations (masking passes by default, opinion arbitration behind
# useArbitration) over the Skeletons-linked hadronic view plus the superclustering EM
# view, and produces the FINAL TRACKSTERS (consumed by PF clustering, hence upstream
# of GSF seeding) plus the per-track assignment maps consumed by ticlCandidate.
ticlTracksterInterpretations = _tracksterLinksProducer.clone(
    runInterpretation=cms.bool(True),
    tracksters_collections=cms.VInputTag('ticlTracksterLinks'),
    egamma_tracksters_collections=[cms.InputTag("ticlTracksterLinksSuperclusteringDNN")],
    # The linking plugin is not used in interpretation mode; Recovery is the identity.
    linkingPSet=cms.PSet(
        algo_verbosity=cms.int32(0),
        type=cms.string('Recovery')
    ),
    inferenceAlgo=cms.string('TracksterInferenceByPFN'),
    regressionAndPid=cms.bool(True),
    # dict-merge, not cms.PSet: an explicit PSet freezes the parameter list and
    # silently drops later cfi additions (eid_blend_width was invisible to
    # modifiers until this became a merge).
    pluginInferenceAlgoTracksterInferenceByPFN=dict(
        onnxPIDModelPath='RecoHGCal/TICL/data/ticlv5/onnx_models/CNN/linking/id_v0.onnx',
        onnxEnergyModelPath='RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/linking/energy_v1.onnx',
        inputNames=['input', 'input_tr_features'],
        output_en=['enreg_output'],
        output_id=['pid_output'],
        eid_min_cluster_energy=2.5,
        doPID=1,
        doRegression=1,
    )
)

# Candidate assembly: consumes the final tracksters + assignment maps and the GSF
# tracks (legal now that PF clustering depends on ticlTracksterInterpretations).
ticlCandidate = _ticlCandidateProducer.clone()

# With the Mustache superclustering modifier the DNN module is replaced: keep the
# e/gamma interpretation inputs pointing at the produced collection.
ticl_superclustering_mustache_ticl.toModify(
    ticlTracksterInterpretations,
    egamma_tracksters_collections=[cms.InputTag("ticlTracksterLinksSuperclusteringMustache")],
)


# TICLv6 development (ticl_dev): opinion arbitration across the interpretations and
# GSF electron kinematics in the candidate assembly. Without the modifier the chain
# runs the masking passes, reproducing the TICLv5 candidates.
from Configuration.ProcessModifiers.ticl_dev_cff import ticl_dev
ticl_dev.toModify(
    ticlTracksterInterpretations,
    useArbitration=True,
    egammaInterpretationDescPSet=dict(eop_max=4.0),
    # Smooth raw-to-regressed energy blend: removes the forbidden zone the hard
    # eid_min_cluster_energy switch digs into the trackster energy spectrum (no
    # object could carry 2.5-5 GeV; the candidate momentum notch).
    pluginInferenceAlgoTracksterInferenceByPFN=dict(eid_blend_width=5.0),
)
ticl_dev.toModify(
    ticlCandidate,
    buildTrackOnlyCandidates=True,
)

ticlv5_TrackLinkingGNN.toModify(ticlTracksterInterpretations,
        interpretationDescPSet = cms.PSet(
            onnxTrkLinkingModelFirstDisk = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/TrackLinking_GNN/FirstDiskPropGNN_v0.onnx'),
            onnxTrkLinkingModelInterfaceDisk = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/TrackLinking_GNN/InterfaceDiskPropGNN_v0.onnx'),
            inputNames = cms.vstring('x', 'edge_index', 'edge_attr'),
            output = cms.vstring('output'),
            delta_tk_ts = cms.double(0.1),
            thr_gnn = cms.double(0.5),
            type = cms.string('GNNLink')
        )
    )
mtdSoA = _mtdSoAProducer.clone()

# pfTICL uses ticlCandidate by default in v5
pfTICL = _pfTICLProducer.clone(
    ticlCandidateSrc = cms.InputTag('ticlCandidate'), 
    useTimingAverage=True
)



ticlPFTask = cms.Task(pfTICL)

# v5 iterations: CLUE3DHigh + Recovery
ticlIterationsTask = cms.Task(
    ticlCLUE3DHighStepTask,
    ticlRecoveryStepTask
)

ticlIterLabelsPSet = cms.PSet(
    labels=cms.vstring(
        "ticlTrackstersCLUE3DHigh",
        "ticlTracksterLinks",
        "ticlTracksterInterpretations",
        "ticlTracksterLinksSuperclusteringDNN"
    )
)

ticl_superclustering_mustache_ticl.toModify(
    ticlIterLabelsPSet,
    labels=cms.vstring(
        "ticlTrackstersCLUE3DHigh",
        "ticlTracksterLinks",
        "ticlTracksterInterpretations",
        "ticlTracksterLinksSuperclusteringMustache"
    )
)

associatorsInstances = []
for labelts in ticlIterLabelsPSet.labels:
    for labelsts in ["ticlSimTracksters", "ticlSimTrackstersfromCPs"]:
        associatorsInstances.append(labelts + "To" + labelsts)
        associatorsInstances.append(labelsts + "To" + labelts)

ticlTracksterLinksTask = cms.Task(ticlTracksterLinks, ticlSuperclusteringTask) 

# mergeTICLTask default for v5
mergeTICLTask = cms.Task(
    ticlLayerTileTask,
    ticlIterationsTask,
    ticlTracksterLinksTask
)


mtdSoATask = cms.Task(mtdSoA)
ticlCandidateTask = cms.Task(ticlTracksterInterpretations, ticlCandidate)



# iterTICLTask default for v5
iterTICLTask = cms.Task(
    mergeTICLTask,
    mtdSoATask, 
    ticlCandidateTask,
    ticlPFTask
)


# HFNose remains on legacy iterations
ticlLayerTileHFNose = ticlLayerTileProducer.clone(
    detector = 'HFNose'
)
ticlLayerTileHFNoseTask = cms.Task(ticlLayerTileHFNose)
iterHFNoseTICLTask = cms.Task(
    ticlLayerTileHFNoseTask,
    ticlHFNoseTrkEMStepTask,
    ticlHFNoseEMStepTask,
    ticlHFNoseTrkStepTask,
    ticlHFNoseHADStepTask,
    ticlHFNoseMIPStepTask
)

ticlLayerTileBarrel = ticlLayerTileProducer.clone(
    detector = 'Barrel',
)

ticlLayerTileBarrelTask = cms.Task(filteredLayerClustersCLUE3DBarrel
    ,ticlLayerTileBarrel)

iterBarrelTICLTask = cms.Task(ticlLayerTileBarrel
    ,ticlCLUE3DBarrelTask
)

ticl_barrel.toModify(mergeTICLTask, func=lambda x : x.add(ticlLayerTileBarrelTask, iterBarrelTICLTask))

