import FWCore.ParameterSet.Config as cms

# With ticl_v6, this modifier runs the Recovery iteration and the Cornetto linking of TICLv6 on the device (pyTICL
# preset v6 with heterogeneous=True). The results are converted to the host products under the same labels.
ticl_v6_heterogeneous = cms.Modifier()
