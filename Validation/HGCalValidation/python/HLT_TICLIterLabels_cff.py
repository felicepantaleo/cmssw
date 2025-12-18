import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.ticl_v4_cff import ticl_v4

# TICLv5 HLT labels are now the default
hltTiclIterLabels = [
    "hltTiclTrackstersCLUE3DHigh",
    "hltTiclTrackstersCLUE3DHighL1Seeded",
    "hltTiclTracksterLinks",
    "hltTiclCandidate"
]

# Revert to TICLv4 labels if modifier is active
ticl_v4.toModify(
    globals(),
    lambda g: g.update({
        "hltTiclIterLabels": [
            "hltTiclTrackstersCLUE3DHigh",
            "hltTiclTrackstersCLUE3DHighL1Seeded",
            "hltTiclTrackstersMerge"
        ]
    })
)

## remove the L1Seeded iteration form the HLT Ticl labels for Scouting
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
_ngtLabels = [label for label in hltTiclIterLabels if label != "hltTiclTrackstersCLUE3DHighL1Seeded"]
ngtScouting.toModify(
    globals(), lambda g: g.update({"hltTiclIterLabels": _ngtLabels})
)