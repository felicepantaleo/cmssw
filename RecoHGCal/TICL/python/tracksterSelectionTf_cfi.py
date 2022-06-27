from RecoTracker.FinalTrackSelectors.tfGraphDefProducer_cfi import tfGraphDefProducer as _tfGraphDefProducer
tracksterSelectionTf = _tfGraphDefProducer.clone(
    ComponentName = "tracksterSelectionTf",
    FileName = "RecoHGCal/TICL/data/tf_models/energy_id_v0.pb"
)

tracksterSelectionTfER = _tfGraphDefProducer.clone(
    ComponentName = "tracksterSelectionTfER",
    FileName = "RecoHGCal/TICL/data/tf_models/energy_regression_tracksters.pb"
)
