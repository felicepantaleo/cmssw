#!/bin/bash

function die { echo $1: status $2 ; exit $2; }

cmsRun ${SCRAM_TEST_PATH}/writeHGCalRecoSoA_cfg.py || die "Failure using writeHGCalRecoSoA_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/readHGCalRecoSoA_cfg.py || die "Failure using readHGCalRecoSoA_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/readHGCalRecoSoA_cfg.py --rntuple || die "Failure using readHGCalRecoSoA_cfg.py --rntuple" $?
