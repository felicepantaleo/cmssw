import FWCore.ParameterSet.Config as cms

from ..modules.hltTiclTracksterInterpretations_cfi import *
from ..modules.hltTiclCandidate_cfi import *

HLTTiclCandidateSequence = cms.Sequence(hltTiclTracksterInterpretations+hltTiclCandidate)
