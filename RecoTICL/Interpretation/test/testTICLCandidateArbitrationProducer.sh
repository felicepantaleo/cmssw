#!/bin/bash
# TICLCandidateArbitrationProducer is built with its global cache, and its construction throws on a bad configuration.
cfg=${SCRAM_TEST_PATH}/testTICLCandidateArbitrationProducer_cfg.py
cmsRun "$cfg" || exit 1
mustThrow() {
  if cmsRun "$cfg" "$1" > "$1.log" 2>&1; then
    echo "the construction with $1 did not throw"
    exit 1
  fi
  grep -q "$2" "$1.log" || { cat "$1.log"; exit 1; }
}
mustThrow claimEnergyWithoutClaims "useClaimTracksterEnergy requires attachClaimTracksters"
mustThrow negativeDepositFraction "claimTracksterMinDepositFraction must be at least 0"
mustThrow badAssignment "candidateTracksters.assignment must be none, nearest or energyShare"
mustThrow photonScoreWithoutProducts "photonMinScore > 0 needs eGammaScore and eGammaTracksters"
