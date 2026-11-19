# Builds pfTICL with PFMuonAlgo and, for the candidates of TICLCandidateArbitrationProducer, with
# muonsFromCandidates=True: both configurations pass the validation of the module.
import FWCore.ParameterSet.Config as cms
from RecoTICL.ParticleFlow.pfTICLProducer_cfi import pfTICLProducer as _pfTICL

process = cms.Process("TEST")
process.source = cms.Source("EmptySource")
process.maxEvents.input = 0
process.pfTICL = _pfTICL.clone()
process.pfTICLFromArbitration = _pfTICL.clone(ticlCandidateSrc="ticlCandidateArbitration", muonsFromCandidates=True)
process.p = cms.Path(process.pfTICL + process.pfTICLFromArbitration)
