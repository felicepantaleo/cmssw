import FWCore.ParameterSet.Config as cms

# With ticl_v6, this modifier runs the shower-shape PID, the Recovery iteration, the Cornetto linking and the
# superclustering DNN of TICLv6 on the host (pyTICL preset v6 with heterogeneous=False).
ticl_v6_host = cms.Modifier()
