# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# The single place that says WHICH reco collections are associated to truth branches
# and WITH WHICH working points. The associator producers, the DQM validators, the
# harvesters and the plotting script all import from here, so a collection is added
# in one edit and cannot drift between the four.
#
# Same shape as RecoHGCal/TICL/python/iterativeTICL_cff.py: the labels live in a
# cms.PSet rather than a plain list so an era or a process modifier can retarget a
# domain with toModify, and the instance-label lists are plain Python built by
# looping over that PSet.

import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabelsPSet

# Reco collections per domain. Each entry is a module label; a collection that also
# needs an instance label is written "label:instance" and split by the producers.
truthGraphRecoLabelsPSet = cms.PSet(
    tracks=cms.vstring("generalTracks"),
    vertices=cms.vstring("offlinePrimaryVertices"),
    secondaryVertices=cms.vstring("inclusiveSecondaryVertices"),
    pfCandidates=cms.vstring("particleFlow", "pfTICL"),
    jets=cms.vstring("ak4PFJetsPuppi"),
    # Every ticlIterLabels entry that actually produces a vector<ticl::Trackster>.
    # ticlCandidate is in that list too but produces TICLCandidates, which mix a track
    # with tracksters and therefore need a two-channel match (tracker hits AND calo
    # energy) rather than this one; it gets its own domain when that exists.
    tracksters=cms.vstring(*[l for l in ticlIterLabelsPSet.labels if l != "ticlCandidate"]),
)

# The same domains reconstructed by the HLT menu. Kept as a separate PSet rather than
# extra entries in the offline one because the two are different reconstructions of the
# same event and must be compared, not pooled: they get their own producers, their own
# DQM folders and their own pages. A domain the menu does not reconstruct is left empty
# and simply produces nothing.
truthGraphHltRecoLabelsPSet = cms.PSet(
    tracks=cms.vstring("hltGeneralTracks"),
    vertices=cms.vstring("hltOfflinePrimaryVertices"),
    secondaryVertices=cms.vstring(),
    pfCandidates=cms.vstring(),
    jets=cms.vstring(),
    tracksters=cms.vstring("hltTiclTrackstersCLUE3DHigh", "hltTiclTracksterLinks"),
)

# Working points of the branch association. Fixed is the plain per-root match; the
# adaptive points differ only in how much branch spread they tolerate before
# rejecting a level, so they bracket the climb rather than sample it densely.
truthBranchWorkingPointsPSet = cms.PSet(
    names=cms.vstring("Fixed", "AdaptiveTight", "AdaptiveNominal", "AdaptiveLoose"),
    adaptiveReverseWeight=cms.vdouble(0.0, 1.0, 1.0, 1.0),
    adaptiveMaxReverseScore=cms.vdouble(0.0, 0.6, 1.0, 1.5),
)


def workingPoints():
    """(name, reverseWeight, maxReverseScore) per working point, in declaration order."""
    return list(
        zip(
            truthBranchWorkingPointsPSet.names,
            truthBranchWorkingPointsPSet.adaptiveReverseWeight,
            truthBranchWorkingPointsPSet.adaptiveMaxReverseScore,
        )
    )


def recoLabels(domain, flavour="offline"):
    """The reco collection labels configured for one domain and one reconstruction.

    flavour is "offline" or "hlt"; they are separate reconstructions of the same event,
    so they are never pooled.
    """
    pset = truthGraphRecoLabelsPSet if flavour == "offline" else truthGraphHltRecoLabelsPSet
    return list(getattr(pset, domain))


def instanceKey(label):
    """Product instance key for a collection label: label and instance joined by an
    underscore. HGCal uses no separator for product labels but an underscore for DQM
    folder names; this package uses the underscore for BOTH so a key reads the same
    wherever it appears."""
    return label.replace(":", "_")


def associatorInstances(domain):
    """Every product instance label this domain's associator emits: both directions
    for every (collection, working point) pair."""
    instances = []
    for label in recoLabels(domain):
        key = instanceKey(label)
        for wp, _, _ in workingPoints():
            instances.append(key + "ToTruthBranch" + wp)
            instances.append("TruthBranchTo" + key + wp)
    return instances


def allAssociatorInstances():
    """The union over every configured domain, for a consumer that takes them flat."""
    instances = []
    for domain in truthGraphRecoLabelsPSet.parameterNames_():
        instances.extend(associatorInstances(domain))
    return instances
