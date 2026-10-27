"""Opt-in inference inside the existing candidate producer.

No additional EDM products/tasks are installed. Backend selection is explicit:
serial_sync, cuda_async or rocm_async; an unavailable backend is an error.
"""
import math
import FWCore.ParameterSet.Config as cms


def _execution(*, backend, batchSize, convertToFP16, warmupIterations):
    if backend not in ("serial_sync", "cuda_async", "rocm_async"):
        raise ValueError("backend must be serial_sync, cuda_async or rocm_async")
    if type(batchSize) is not int or not 0 < batchSize <= 2147483647:
        raise ValueError("batchSize must be a positive int fitting the SoA size type")
    if type(convertToFP16) is not bool:
        raise TypeError("convertToFP16 must be bool")
    if type(warmupIterations) is not int or not 0 <= warmupIterations <= 2147483647:
        raise ValueError("warmupIterations must be a non-negative int")
    return cms.PSet(backend=cms.string(backend),
                    batchSize=cms.uint32(batchSize), convertToFP16=cms.bool(convertToFP16),
                    warmupIterations=cms.int32(warmupIterations))


def makeCandidateInference(*, model="PFN", doPID=True, doRegression=True,
                           pidModel=None, regressionModel=None, minClusterEnergy=2.5,
                           backend="serial_sync", batchSize=64,
                           convertToFP16=False, warmupIterations=0):
    """Return pluginInferenceAlgoPyTorchAlpaka; model paths are candidate-stage specific."""
    if model not in ("CNN", "DNN", "PFN"):
        raise ValueError("model must be CNN, DNN or PFN")
    if type(doPID) is not bool or type(doRegression) is not bool:
        raise TypeError("doPID and doRegression must be bool")
    if model == "CNN" and doRegression:
        raise ValueError("CNN supports PID only")
    if not math.isfinite(minClusterEnergy):
        raise ValueError("minClusterEnergy must be finite")
    pidModel = "RecoTICL/Inference/data/candidate_pid_" + model.lower() + ".pt" if pidModel is None else pidModel
    regressionModel = ("RecoTICL/Inference/data/candidate_energy_" + model.lower() + ".pt"
                       if regressionModel is None else regressionModel)
    if (doPID and not pidModel) or (doRegression and not regressionModel):
        raise ValueError("Every enabled task requires a nonempty model path")
    config = _execution(backend=backend, batchSize=batchSize,
                        convertToFP16=convertToFP16, warmupIterations=warmupIterations)
    config.modelFamily = cms.string(model)
    config.algo_verbosity = cms.int32(0)
    config.doPID = cms.bool(doPID)
    config.doRegression = cms.bool(doRegression)
    config.pidModel = cms.string(pidModel)
    config.regressionModel = cms.string(regressionModel)
    config.minClusterEnergy = cms.double(minClusterEnergy)
    return config


def withCandidateInference(producer, **options):
    """Clone a TICLCandidateProducer with inference before Trackster publication."""
    return producer.clone(inferenceAlgo=cms.string("PyTorchAlpaka"), regressionAndPid=cms.bool(True),
                          pluginInferenceAlgoPyTorchAlpaka=makeCandidateInference(**options))
