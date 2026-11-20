import FWCore.ParameterSet.Config as cms


def _setNoiseTableFromJobSeed(pset):
    for name in pset.parameterNames_():
        value = getattr(pset, name)
        if isinstance(value, cms.PSet):
            _setNoiseTableFromJobSeed(value)
    if hasattr(pset, "noiseTableFromJobSeed"):
        pset.noiseTableFromJobSeed = True


def customiseEventSeeding(process):
    """Seeds the random numbers of each event from the EventID, so that they do not depend on the stream.

    The digitizers of the mixing modules that keep a noise table fill it from the job seed.
    """
    process.RandomNumberGeneratorService.seedingMode = cms.untracked.string("event")
    for label in ("mix", "mixData"):
        if hasattr(process, label):
            _setNoiseTableFromJobSeed(getattr(process, label))
    return process
