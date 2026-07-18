#!/usr/bin/env python3
# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""TICLv6 drift detector.

Assert that the configuration pyTICL generates for the v6 preset reproduces the
live baseline ``iterTICLTask`` with the ``ticl_dev`` process modifier applied,
byte-for-byte, module by module. Keeps the preset and the modifier wiring in
``iterativeTICL_cff`` from drifting apart.
"""

import sys

import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.ticl_dev_cff import ticl_dev
from RecoTICL.Configuration import presets
from RecoTICL.Configuration.compare import diff_tasks


def build_baseline():
    p = cms.Process("TEST", ticl_dev)
    p.load("RecoHGCal.TICL.iterativeTICL_cff")
    return p, p.iterTICLTask


def build_pyticl():
    p = cms.Process("TEST")
    assembled = presets.v6().assemble()
    assembled.add_to_process(p)
    return p, p.iterTICLTask


def main():
    base_p, base_task = build_baseline()
    test_p, test_task = build_pyticl()
    diff = diff_tasks(base_p, base_task, test_p, test_task)
    if diff:
        print(diff)
        print("FAIL: pyTICL v6 does not reproduce iterTICLTask under ticl_dev")
        return 1
    n = len([m for m in base_task.moduleNames()])
    print(f"OK: pyTICL v6 reproduces iterTICLTask under ticl_dev byte-for-byte ({n} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
