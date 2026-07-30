# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# The association producers, one per domain, all driven by the label and working-point
# lists in truthGraphAssociationLabels_cff so a collection is configured in exactly one
# place. Composite domains consume the constituent domain's maps, so the order in
# truthGraphAssociatorsTask matters only for readability: the framework resolves the
# data dependency itself.

import FWCore.ParameterSet.Config as cms

from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociationLabels_cff import (
    truthBranchWorkingPointsPSet,
    recoLabels,
)

# Shared selection of which truth branches are candidates at all. A 1 GeV floor keeps
# the maps and the efficiency denominators from being dominated by soft particles that
# no reconstruction was going to find; loosen it per domain if a study needs to.
truthBranchSelectorBlock = cms.PSet(
    ptMin=cms.double(1.0),
    ptMax=cms.double(1e100),
    etaMin=cms.double(-4.0),
    etaMax=cms.double(4.0),
    pdgIds=cms.vint32(),
    signalOnly=cms.bool(False),
    intimeOnly=cms.bool(False),
    chargedOnly=cms.bool(False),
    invertEta=cms.bool(False),
)

_workingPointArgs = dict(
    workingPointNames=cms.vstring(*truthBranchWorkingPointsPSet.names),
    adaptiveReverseWeight=cms.vdouble(*truthBranchWorkingPointsPSet.adaptiveReverseWeight),
    adaptiveMaxReverseScore=cms.vdouble(*truthBranchWorkingPointsPSet.adaptiveMaxReverseScore),
)

_truthSources = dict(
    src=cms.InputTag("truthLogicalGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
)


def _tags(domain, flavour="offline"):
    return cms.VInputTag(*[cms.InputTag(*label.split(":")) for label in recoLabels(domain, flavour)])


# Branch levels the truth-driven direction asks about, one denominator product per
# level, side by side. Only hit-based domains have levels: a composite object's truth
# target is a vertex, fixed by its resolution instead.
_truthLevels = cms.vstring(
    "stableLegsFromUpstream", "caloBoundary", "stableDecayProducts", "hardProcess"
)

# Hit-based: the object owns detector hits.
allTrackToTruthBranchAssociators = cms.EDProducer(
    "AllTrackToTruthBranchAssociatorsProducer",
    recoCollections=_tags("tracks"),
    branchSelector=truthBranchSelectorBlock.clone(),
    truthLevels=_truthLevels,
    **_truthSources,
    **_workingPointArgs,
)

# Constituent-based: a vertex has no hits of its own, its truth is aggregated from the
# tracks it is built from, which the track associator has already matched.
allVertexToTruthBranchAssociators = cms.EDProducer(
    "AllVertexToTruthBranchAssociatorsProducer",
    recoCollections=_tags("vertices"),
    branchSelector=truthBranchSelectorBlock.clone(),
    constituentAssociator=cms.string("allTrackToTruthBranchAssociators"),
    constituentCollection=cms.string("generalTracks"),
    # A primary vertex asks which INTERACTION a track came from, so a track produced in
    # a decay downstream of the vertex still counts at the vertex its chain started
    # from. Counting it at its own production vertex would call the whole decay chain of
    # the event contamination.
    vertexResolution=cms.string("interaction"),
    **_truthSources,
    **_workingPointArgs,
)

allSecondaryVertexToTruthBranchAssociators = allVertexToTruthBranchAssociators.clone(
    recoCollections=_tags("secondaryVertices"),
    # A secondary vertex IS a decay or interaction vertex, so the tracks that belong to
    # it were produced there and the immediate production vertex is the right target.
    vertexResolution="immediate",
)

# Hit-based on the CALORIMETER channel: a trackster owns energy through its layer
# clusters, so it is matched on shared energy, the same quantity the TICL trackster
# validation scores against.
truthBranchTracksterAssociators = cms.EDProducer(
    "TruthBranchTracksterAssociatorsProducer",
    recoCollections=_tags("tracksters"),
    branchSelector=truthBranchSelectorBlock.clone(),
    layerClusters=cms.InputTag("hgcalMergeLayerClusters"),
    truthLevels=_truthLevels,
    **_truthSources,
    **_workingPointArgs,
)

# The HLT menu's own reconstruction of the same event. Same producers, same working
# points, different input collections, so the two can be compared page by page.
hltTrackToTruthBranchAssociators = allTrackToTruthBranchAssociators.clone(
    recoCollections=_tags("tracks", "hlt"),
)
hltVertexToTruthBranchAssociators = allVertexToTruthBranchAssociators.clone(
    recoCollections=_tags("vertices", "hlt"),
    constituentAssociator="hltTrackToTruthBranchAssociators",
    constituentCollection="hltGeneralTracks",
)
hltTruthBranchTracksterAssociators = truthBranchTracksterAssociators.clone(
    recoCollections=_tags("tracksters", "hlt"),
    layerClusters="hltMergeLayerClusters",
)

truthGraphAssociatorsTask = cms.Task(
    allTrackToTruthBranchAssociators,
    allVertexToTruthBranchAssociators,
    allSecondaryVertexToTruthBranchAssociators,
    truthBranchTracksterAssociators,
    hltTrackToTruthBranchAssociators,
    hltVertexToTruthBranchAssociators,
    hltTruthBranchTracksterAssociators,
)
