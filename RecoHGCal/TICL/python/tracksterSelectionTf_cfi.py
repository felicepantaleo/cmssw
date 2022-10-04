from PhysicsTools.TensorFlow.tfGraphDefProducer_cfi import tfGraphDefProducer as _tfGraphDefProducer
tracksterSelectionTf = _tfGraphDefProducer.clone(
    ComponentName = "tracksterSelectionTf",
    FileName = "RecoHGCal/TICL/data/tf_models/ticl_v4_model_energy_id.pb"
)
