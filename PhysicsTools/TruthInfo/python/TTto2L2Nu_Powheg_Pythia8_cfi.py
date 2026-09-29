# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
#
# ttbar -> dilepton (TTto2L2Nu) at NLO with POWHEG (hvq) showered by Pythia8, for a
# ttbar gallery/library example.
#
# WARNING: the gridpack, tune and PS weights are Run3 at 13.6 TeV. The GEN centre-of-mass
# energy does NOT match the Phase-2 14 TeV detector and conditions. Use it only as a
# topology demonstration. For a consistent Phase-2 sample, use a 14 TeV gridpack and a
# matching tune.

import FWCore.ParameterSet.Config as cms

externalLHEProducer = cms.EDProducer('ExternalLHEProducer',
    args = cms.vstring('/cvmfs/cms.cern.ch/phys_generator/gridpacks/PdmV/Run3Summer22/Powheg/TT/hvq_slc7_amd64_gcc10_CMSSW_12_4_8_TTto2L2Nu_powheg-pythia8.tgz'),
    nEvents = cms.untracked.uint32(5000),
    numberOfParameters = cms.uint32(1),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh'),
    generateConcurrently = cms.untracked.bool(True),
)

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunesRun3ECM13p6TeV.PythiaCP5Settings_cfi import *
from Configuration.Generator.Pythia8PowhegEmissionVetoSettings_cfi import *
from Configuration.Generator.PSweightsPythia.PythiaPSweightsSettings_cfi import *

generator = cms.EDFilter("Pythia8ConcurrentHadronizerFilter",
    PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        pythia8PowhegEmissionVetoSettingsBlock,
        pythia8PSweightsSettingsBlock,
        processParameters = cms.vstring(
        'POWHEG:nFinal = 2',
        'TimeShower:mMaxGamma = 1.0'
        ),
        parameterSets = cms.vstring(
            'pythia8CommonSettings',
            'pythia8CP5Settings',
            'pythia8PowhegEmissionVetoSettings',
            'processParameters',
            'pythia8PSweightsSettings',
        )
    ),
    comEnergy = cms.double(13600),
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    pythiaPylistVerbosity = cms.untracked.int32(1),
)
