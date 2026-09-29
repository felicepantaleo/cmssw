# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# Event content for the MC-truth graph, with two verbosity levels.
#
# The DIGI step builds the truth once, where the merged signal+pileup simHits exist:
# a logical graph and a per-particle per-cell sim-energy hit index. The index is
# UNRESOLVED (recHitMap=""). BranchHitAssociator matches by DetId, so the same index
# serves any stage (L1, HLT, offline RECO) that exposes its reco objects as
# (DetId, fraction).
#
#   compact (default): the logical graph, the unresolved hit index, and the raw
#     merged graph (TruthGraph_mix). The index gives the fraction numerator
#     (per-particle per-cell energy) and, summed over all particles per cell, the
#     denominator. The branch validators read the raw graph (rawSrc, for the
#     trackId->particle map), and it lets the logical graph and the index be rebuilt
#     offline (~3 MB/ev). mix:genPayload is not kept, so a rebuild sets rawGenPayload
#     to "": then a stable pileup GEN-only particle has no momentum and a pileup GEN
#     vertex that merged with no SimVertex has no position.
#
#     INVARIANT: the persisted index is COMPLETE: every contributor that leaves a hit,
#     no pdgId pruning. The per-cell denominator is the sum of the index hit energies
#     over all particles, so a dropped contributor biases every surviving fraction
#     high. Do not prune the index to selected species without also persisting a
#     separate all-contributor per-cell total map.
#
#   full: compact + the merged simHits (all contributors, with trackId and eventId).
#     Only this level rebuilds the association at a DIFFERENT granularity (e.g. L1
#     trigger cells) or with another metric. Calo simHits always, tracking simHits
#     with includeTrackingHits.

import FWCore.ParameterSet.Config as cms

# The compact, stage-independent truth: the physics graph, the unresolved
# per-particle per-cell footprint, and the raw merged CSR graph (needed by the
# branch validators' rawSrc and for offline rebuild; ~3 MB/ev).
_truthGraphKeep = [
    'keep *_truthLogicalGraphProducer_*_*',
    'keep *_truthLogicalGraphHitIndexProducer_*_*',
    'keep TruthGraph_mix_*_*',
]


def _truthSimHitsKeep(includeTrackingHits):
    """The merged (signal+pileup) simHits kept at the 'full' level. Calo always;
    tracking (tracker + muon + MTD) only with includeTrackingHits."""
    keep = [
        'keep *_mix_mergedHGCHits_*',
        'keep *_mix_mergedEcalHits_*',
        'keep *_mix_mergedHcalHits_*',
    ]
    if includeTrackingHits:
        keep += [
            'keep *_mix_mergedTrackerHits_*',
            'keep *_mix_mergedMuonHits_*',
            'keep *_mix_mergedMtdHits_*',
        ]
    return keep


# The default analysis-tier content.
truthContentCompact = cms.untracked.vstring(_truthGraphKeep)


def truthContentFull(includeTrackingHits=True):
    """The 'full' content: compact + merged simHits."""
    return cms.untracked.vstring(_truthGraphKeep + _truthSimHitsKeep(includeTrackingHits))


def truthEventContent(level='compact', includeTrackingHits=True):
    """Return the output commands for a truth verbosity level.
    level='compact' (default) or 'full'."""
    if level == 'compact':
        return cms.untracked.vstring(truthContentCompact)
    if level == 'full':
        return truthContentFull(includeTrackingHits)
    raise ValueError("unknown truth event-content level %r; choose 'compact' or 'full'" % level)


def setTruthEventContent(process, level='compact', includeTrackingHits=True):
    """Append a truth verbosity level to every output module in the process.
    level='compact' (default: graph, unresolved hit index and raw graph) or
    'full' (adds the merged simHits for re-association at another granularity)."""
    commands = truthEventContent(level, includeTrackingHits)
    for out in process.outputModules_().values():
        out.outputCommands.extend(commands)
    return process
