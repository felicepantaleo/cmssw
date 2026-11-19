#!/bin/bash
# The e/gamma producers are built with the three models of TICLEGammaSuperClustersProducer, and its construction
# throws when a model file does not exist.
cfg=${SCRAM_TEST_PATH}/testTICLEGammaProducers_cfg.py
cmsRun "$cfg" || exit 1
mustThrow() {
  if cmsRun "$cfg" "$1" > "$1.log" 2>&1; then
    echo "the construction with $1 did not throw"
    exit 1
  fi
  grep -q "$2" "$1.log" || { cat "$1.log"; exit 1; }
}
mustThrow missingRegression "regression_v0.onnx"
mustThrow missingScore "score_v0.onnx"
mustThrow missingTypeScore "type_v0.onnx"
