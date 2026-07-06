import FWCore.ParameterSet.Config as cms

hltTiclTracksterInterpretations = cms.EDProducer("TracksterLinksProducer",
    arbitrationMaxSharedEnergyFraction = cms.double(0.2),
    cutTk = cms.string('1.48 < abs(eta) < 3.0 && pt > 1. && quality("highPurity") && hitPattern().numberOfLostHits("MISSING_OUTER_HITS") < 5'),
    detector = cms.string('HGCAL'),
    egammaInterpretationDescPSet = cms.PSet(
        algo_verbosity = cms.int32(0),
        delta_tk_sc = cms.double(0.1),
        eop_max = cms.double(1.5),
        eop_min = cms.double(0.5),
        min_em_fraction = cms.double(0.8),
        min_supercluster_energy = cms.double(1),
        type = cms.string('EGamma')
    ),
    egamma_tracksters_collections = cms.VInputTag(cms.InputTag("hltTiclTracksterLinks")),
    inferenceAlgo = cms.string('TracksterInferenceByPFN'),
    interpretationDescPSet = cms.PSet(
        algo_verbosity = cms.int32(0),
        delta_tk_ts_interface = cms.double(0.03),
        delta_tk_ts_layer1 = cms.double(0.02),
        energy_overshoot_fraction = cms.double(0.2),
        energy_overshoot_max = cms.double(10),
        timing_quality_threshold = cms.double(0.5),
        type = cms.string('ChargedHadron')
    ),
    jetInterpretationDescPSet = cms.PSet(
        algo_verbosity = cms.int32(0),
        delta_tk_ts = cms.double(0.1),
        min_trackster_energy = cms.double(5),
        recovery_max_eop = cms.double(3),
        recovery_min_eop = cms.double(0.2),
        type = cms.string('Jet')
    ),
    layer_clusters = cms.InputTag("hltMergeLayerClusters"),
    layer_clustersTime = cms.InputTag("hltMergeLayerClusters","timeLayerCluster"),
    linkingPSet = cms.PSet(
        algo_verbosity = cms.int32(0),
        type = cms.string('Recovery')
    ),
    mightGet = cms.optional.untracked.vstring,
    muonInterpretationDescPSet = cms.PSet(
        algo_verbosity = cms.int32(0),
        delta_tk_ts = cms.double(0.1),
        mip_energy_max = cms.double(10),
        onnx_model_path = cms.string(''),
        type = cms.string('Muon')
    ),
    muons = cms.InputTag("hltPhase2L3Muons"),
    original_masks = cms.VInputTag("hltMergeLayerClusters:InitialLayerClustersMask"),
    pluginInferenceAlgoTracksterInferenceByDNN = cms.PSet(
        algo_verbosity = cms.int32(0),
        doPID = cms.int32(1),
        doRegression = cms.int32(1),
        eid_min_cluster_energy = cms.double(1),
        eid_n_clusters = cms.int32(10),
        eid_n_layers = cms.int32(50),
        inputNames = cms.vstring('input'),
        miniBatchSize = cms.untracked.int32(64),
        onnxEnergyModelPath = cms.string(''),
        onnxPIDModelPath = cms.string(''),
        output_en = cms.vstring('enreg_output'),
        output_id = cms.vstring('pid_output'),
        type = cms.string('TracksterInferenceByDNN')
    ),
    pluginInferenceAlgoTracksterInferenceByPFN = cms.PSet(
        algo_verbosity = cms.int32(0),
        doPID = cms.int32(1),
        doRegression = cms.int32(1),
        eid_min_cluster_energy = cms.double(2.5),
        eid_n_clusters = cms.int32(10),
        eid_n_layers = cms.int32(50),
        inputNames = cms.vstring(
            'input',
            'input_tr_features'
        ),
        onnxEnergyModelPath = cms.string('RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/linking/energy_v1.onnx'),
        onnxPIDModelPath = cms.string('RecoHGCal/TICL/data/ticlv5/onnx_models/CNN/linking/id_v0.onnx'),
        output_en = cms.vstring('enreg_output'),
        output_id = cms.vstring('pid_output'),
        type = cms.string('TracksterInferenceByPFN')
    ),
    propagator = cms.string('PropagatorWithMaterial'),
    regressionAndPid = cms.bool(True),
    runInterpretation = cms.bool(True),
    timingSoA = cms.InputTag("mtdSoA"),
    tracks = cms.InputTag("hltGeneralTracks"),
    tracksters_collections = cms.VInputTag("hltTiclTracksterLinks"),
    useArbitration = cms.bool(False),
    useMTDTiming = cms.bool(False)
)

from Configuration.ProcessModifiers.ticlv5_TrackLinkingGNN_cff import ticlv5_TrackLinkingGNN
ticlv5_TrackLinkingGNN.toModify(hltTiclTracksterInterpretations,
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
