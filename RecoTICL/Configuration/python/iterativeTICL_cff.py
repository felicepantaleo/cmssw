import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration.FastJetStep_cff import *
from RecoTICL.Configuration.CLUE3DHighStep_cff import *
from RecoTICL.Configuration.MIPStep_cff import *
from RecoTICL.Configuration.TrkEMStep_cff import *
from RecoTICL.Configuration.TrkStep_cff import *
from RecoTICL.Configuration.EMStep_cff import *
from RecoTICL.Configuration.HADStep_cff import *
from RecoTICL.Configuration.CLUE3DEM_cff import *
from RecoTICL.Configuration.CLUE3DHAD_cff import *
from RecoTICL.Configuration.PRbyRecovery_cff import *
from RecoTICL.Configuration.CLUE3DBarrel_cff import *

from RecoTICL.LayerClustering.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoTICL.ParticleFlow.pfTICLProducer_cfi import pfTICLProducer as _pfTICLProducer

from RecoTICL.Linking.tracksterLinksProducer_cfi import tracksterLinksProducer as _tracksterLinksProducer
from RecoTICL.Superclustering.superclustering_cff import *
from RecoTICL.Interpretation.ticlCandidateProducer_cfi import ticlCandidateProducer as _ticlCandidateProducer

from RecoTICL.Interpretation.mtdSoAProducer_cfi import mtdSoAProducer as _mtdSoAProducer
from Configuration.ProcessModifiers.ticlv5_TrackLinkingGNN_cff import ticlv5_TrackLinkingGNN

from Configuration.ProcessModifiers.ticl_superclustering_mustache_pf_cff import ticl_superclustering_mustache_pf
from Configuration.ProcessModifiers.ticl_superclustering_mustache_ticl_cff import ticl_superclustering_mustache_ticl
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
from Configuration.ProcessModifiers.ticl_v6_cff import ticl_v6
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHGC_cfi import particleFlowClusterHGCal as _particleFlowClusterHGCal

ticlLayerTileTask = cms.Task(ticlLayerTileProducer)


# TICLv5 is now the default configuration
ticlTracksterLinks = _tracksterLinksProducer.clone(
    tracksters_collections = cms.VInputTag(
        'ticlTrackstersCLUE3DHigh',
        'ticlTrackstersRecovery'
    ),
    linkingPSet = cms.PSet(
      cylinder_radius_sqr_split = cms.float(9),
      proj_distance_split = cms.float(5),
      track_time_quality_threshold = cms.float(0.5),
      min_num_lcs = cms.uint32(15),
      min_trackster_energy = cms.float(20),
      pca_quality_th = cms.float(0.85),
      dot_prod_th = cms.float(0.97),
      lower_boundary = cms.vfloat(20, 10),  
      upper_boundary = cms.vfloat(150, 100),  
      upper_distance_projective_sqr = cms.vfloat(4, 60),  
      lower_distance_projective_sqr = cms.vfloat(4, 60),  
      min_distance_z = cms.vfloat(35, 35),  
      upper_distance_projective_sqr_closest_points = cms.vfloat(5, 30),  
      lower_distance_projective_sqr_closest_points = cms.vfloat(10, 50),  
      max_z_distance_closest_points = cms.vfloat(35, 35),
      cylinder_radius_sqr = cms.vfloat(9, 15),  
      deltaRxy = cms.float(4.),
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
        eid_min_cluster_energy = cms.float(1),
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
        eid_min_cluster_energy = cms.float(2.5),
        eid_n_clusters = cms.int32(10),
        eid_n_layers = cms.int32(50),
        onnxEnergyModelPath = cms.string('RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/linking/energy_v1.onnx'),
        onnxPIDModelPath = cms.string('RecoHGCal/TICL/data/ticlv5/onnx_models/CNN/linking/id_v0.onnx'),
        type = cms.string('TracksterInferenceByPFN')
    )
)

ticlCandidate = _ticlCandidateProducer.clone(
    inferenceAlgo=cms.string('TracksterInferenceByPFN'),
    regressionAndPid = cms.bool(True),
    pluginInferenceAlgoTracksterInferenceByPFN=cms.PSet(
        algo_verbosity=cms.int32(0),
        onnxPIDModelPath=cms.string('RecoHGCal/TICL/data/ticlv5/onnx_models/CNN/linking/id_v0.onnx'),
        onnxEnergyModelPath=cms.string('RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/linking/energy_v1.onnx'),
        inputNames=cms.vstring('input', 'input_tr_features'),
        output_en=cms.vstring('enreg_output'),
        output_id=cms.vstring('pid_output'),
        eid_min_cluster_energy=cms.float(2.5),
        eid_n_layers=cms.int32(50),
        eid_n_clusters=cms.int32(10),
        doPID=cms.int32(1),
        doRegression=cms.int32(1),
        type=cms.string('TracksterInferenceByPFN')
    )
)

ticlv5_TrackLinkingGNN.toModify(ticlCandidate,
        interpretationDescPSet = cms.PSet(
            onnxTrkLinkingModelFirstDisk = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/TrackLinking_GNN/FirstDiskPropGNN_v0.onnx'),
            onnxTrkLinkingModelInterfaceDisk = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/TrackLinking_GNN/InterfaceDiskPropGNN_v0.onnx'),
            inputNames = cms.vstring('x', 'edge_index', 'edge_attr'),
            output = cms.vstring('output'),
            delta_tk_ts = cms.float(0.1),
            thr_gnn = cms.float(0.5),
            type = cms.string('GNNLink')
        )
    )
mtdSoA = _mtdSoAProducer.clone()

# pfTICL uses ticlCandidate by default in v5
pfTICL = _pfTICLProducer.clone(
    ticlCandidateSrc = cms.InputTag('ticlCandidate')
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
        "ticlCandidate",
        "ticlTracksterLinksSuperclusteringDNN"
    )
)

ticl_superclustering_mustache_ticl.toModify(
    ticlIterLabelsPSet,
    labels=cms.vstring(
        "ticlTrackstersCLUE3DHigh",
        "ticlTracksterLinks",
        "ticlCandidate",
        "ticlTracksterLinksSuperclusteringMustache"
    )
)

# TICLv6: the final tracksters come from the interpretation stage; ticlCandidate holds the candidates only. The first
# iteration is CLUEstering.
_ticlV6IterLabels = {"ticlCandidate": "ticlTracksterInterpretations",
                     "ticlTrackstersCLUE3DHigh": "ticlTrackstersCLUEstering"}
ticl_v6.toModify(
    ticlIterLabelsPSet,
    labels=[_ticlV6IterLabels.get(label, label) for label in ticlIterLabelsPSet.labels]
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
ticlCandidateTask = cms.Task(ticlCandidate)



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


# TICLv6: the modules of the pyTICL preset v6, with the interpretation stage before the candidates.
from RecoTICL.Configuration.presets import v6 as _v6
_ticlV6Modules = _v6().assemble().modules
ticlTracksterInterpretations = _ticlV6Modules["ticlTracksterInterpretations"]
# TICLv6: the only iteration is CLUEstering on the layer clusters merged on the device; there is no Recovery for now.
ticlCLUEstering = _ticlV6Modules["ticlCLUEstering"]
ticlSoATracksters = _ticlV6Modules["ticlSoATracksters"]
ticlTrackstersCLUEstering = _ticlV6Modules["ticlTrackstersCLUEstering"]
ticlCLUEsteringStepTask = cms.Task(ticlCLUEstering, ticlSoATracksters, ticlTrackstersCLUEstering)
ticl_v6.toReplaceWith(ticlIterationsTask, cms.Task(ticlCLUEsteringStepTask))
# Only the input changes, so that the other modifiers of this module still apply.
ticl_v6.toModify(ticlEGammaSuperClusterProducer,
                 ticlTrackstersEM=_ticlV6Modules["ticlEGammaSuperClusterProducer"].ticlTrackstersEM)
(ticl_v6 & ticl_superclustering_mustache_ticl).toModify(
    ticlTracksterLinksSuperclusteringMustache,
    tracksters_collections=_ticlV6Modules["ticlTracksterLinksSuperclusteringDNN"].tracksters_collections)
ticl_v6.toReplaceWith(ticlTracksterLinks, _ticlV6Modules["ticlTracksterLinks"])
ticl_v6.toReplaceWith(ticlTracksterLinksSuperclusteringDNN, _ticlV6Modules["ticlTracksterLinksSuperclusteringDNN"])
ticl_v6.toReplaceWith(ticlCandidate, _ticlV6Modules["ticlCandidate"])
ticl_v6.toModify(ticlCandidateTask, lambda t: t.add(ticlTracksterInterpretations))
# TICLv6 candidates carry the muon kinematics; pfTICL copies their muon decisions.
ticl_v6.toReplaceWith(pfTICL, _ticlV6Modules["pfTICL"])
# TICLv6: the HGCAL e/gamma superclusters are the final tracksters that the interpretation types as electrons or EM
# neutrals, one supercluster each; no HGCAL PF clusters and no HGCAL PF superclusters are made.
from RecoTICL.ParticleFlow.ticlEGammaTrackstersProducer_cfi import ticlEGammaTrackstersProducer as _ticlEGammaTrackstersProducer
from RecoTICL.ParticleFlow.ticlEGammaSuperClustersProducer_cfi import ticlEGammaSuperClustersProducer as _ticlEGammaSuperClustersProducer
from RecoEcal.EgammaClusterProducers.particleFlowSuperClusteringSequence_cff import particleFlowSuperClusterHGCal as _particleFlowSuperClusterHGCal
from RecoEcal.EgammaClusterProducers.particleFlowSuperClusteringSequence_cff import particleFlowSuperClusteringTask as _particleFlowSuperClusteringTask
from RecoLocalCalo.Configuration.hgcalLocalReco_cff import hgcalLocalRecoTask as _hgcalLocalRecoTask
from RecoEgamma.EgammaPhotonProducers.photonCore_cfi import photonCoreHGC as _photonCoreHGC
ticlEGammaTracksters = _ticlEGammaTrackstersProducer.clone()
# A supercluster needs a score of at least 0.5 below Et 10 GeV and 0.9 above it. The type score of each e/gamma
# trackster is the input of the photon candidate type in ticlCandidate.
ticlEGammaSuperClusters = _ticlEGammaSuperClustersProducer.clone(
    # the tracksters that the superclustering groups
    inputTracksters=_ticlV6Modules["ticlTracksterLinksSuperclusteringDNN"].tracksters_collections[0],
    scoreModel="RecoTICL/ParticleFlow/data/egammaSuperClusters/score_v1.onnx",
    minScore=0.5,
    minScoreHighEt=0.9,
    typeScoreModel="RecoTICL/ParticleFlow/data/egammaSuperClusters/type_v1.onnx")
ticl_v6.toReplaceWith(_hgcalLocalRecoTask, _hgcalLocalRecoTask.copyAndExclude([_particleFlowClusterHGCal]))
ticl_v6.toReplaceWith(_particleFlowSuperClusteringTask,
                      _particleFlowSuperClusteringTask.copyAndExclude([_particleFlowSuperClusterHGCal]))
ticl_v6.toModify(_photonCoreHGC, scIslandEndcapProducer="ticlEGammaSuperClusters")
# Electron seeds of the HGCAL superclusters: from the KF track of a TICL electron, and from the supercluster only when
# no trackster is typed electron and a track has its Electron hypothesis on one.
from RecoTICL.ParticleFlow.ticlElectronSuperClustersProducer_cfi import ticlElectronSuperClustersProducer as _ticlElectronSuperClustersProducer
from RecoTICL.ParticleFlow.ticlElectronSeedsProducer_cfi import ticlElectronSeedsProducer as _ticlElectronSeedsProducer
from RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeeds_cff import ecalDrivenElectronSeeds as _ecalDrivenElectronSeeds
from RecoParticleFlow.PFTracking.mergedElectronSeeds_cfi import electronMergedSeeds as _electronMergedSeeds
ticlElectronSuperClusters = _ticlElectronSuperClustersProducer.clone()
ticlElectronSeeds = _ticlElectronSeedsProducer.clone()
ticl_v6.toModify(ticlPFTask, lambda t: t.add(ticlEGammaTracksters, ticlEGammaSuperClusters, ticlElectronSuperClusters,
                                              ticlElectronSeeds))
ticl_v6.toModify(_ecalDrivenElectronSeeds, endcapSuperClusters="ticlElectronSuperClusters")
ticl_v6.toModify(_electronMergedSeeds, EcalBasedSeeds="ticlElectronSeeds")
# With the Mustache superclustering, the interpretation stage reads the Mustache superclusters.
(ticl_v6 & ticl_superclustering_mustache_ticl).toModify(
    ticlTracksterInterpretations, egamma_tracksters_collections=["ticlTracksterLinksSuperclusteringMustache"])
