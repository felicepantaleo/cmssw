# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
#
# t-channel single-top GEN fragment (q q' -> t q'' via t-channel W). The standard relval
# matrix has no single-top sample. This fragment gives a gallery/library example for the
# 'top' selection preset with the production co-products (the recoiling spectator quark).

import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *

generator = cms.EDFilter("Pythia8ConcurrentGeneratorFilter",
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         filterEfficiency = cms.untracked.double(1.0),
                         comEnergy = cms.double(14000.0),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        processParameters = cms.vstring(
            'Top:qq2tq(t:W) = on',   # t-channel single top (and single antitop)
            '6:m0 = 175 ',
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CP5Settings',
                                    'processParameters',
                                    )
        )
                         )
ProductionFilterSequence = cms.Sequence(generator)
