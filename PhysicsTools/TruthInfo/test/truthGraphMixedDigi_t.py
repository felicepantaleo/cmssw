#!/usr/bin/env python3
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

"""The collapsed pileup record keeps exactly the species the reconstructable levels
stop at."""

import subprocess
import sys
import unittest

import PhysicsTools.TruthInfo.truthGraphMixedDigi_cff as digi


def reconstructable(producer):
    return list(producer.postProcessing.reconstructablePdgIds)


class TestKeptSpeciesMatchTheLevels(unittest.TestCase):
    def test_default_configuration(self):
        kept = list(digi.truthGraphAccumulator.collapsedGenKeptPdgIds)
        self.assertEqual(kept, digi.reconstructablePdgIds)
        self.assertEqual(reconstructable(digi.truthLogicalGraphProducer), kept)


# Run in a fresh interpreter: the era modifiers apply only to modules imported after the
# process exists, and this file imports the cff before any process.
_HF_FACTORS_CHECK = """
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C26I13M9_cff import Phase2C26I13M9
from Configuration.ProcessModifiers.enableTruth_cff import enableTruth
process = cms.Process("T", Phase2C26I13M9, enableTruth)
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("Configuration.StandardSequences.Digi_cff")
hcal = process.mix.digitizers.hcal
index = list(process.truthLogicalGraphHitIndexProducer.hfSamplingFactors)
digitizer = [hcal.hf1.samplingFactor.value(), hcal.hf2.samplingFactor.value()]
assert index == digitizer, (index, digitizer)
"""


class TestHFEnergyScale(unittest.TestCase):
    def test_index_uses_the_digitizer_sampling_factors(self):
        result = subprocess.run([sys.executable, "-c", _HF_FACTORS_CHECK], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
