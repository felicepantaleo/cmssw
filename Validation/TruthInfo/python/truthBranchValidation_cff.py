# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# The DQM analyzers and their harvesting, both generated from the same label and
# working-point lists the associators use, so the folder names, the ME names and the
# harvester subDirs cannot drift apart.
#
# One entry in _domains below is all it takes to add a reco domain: the analyzer, the
# folder names, the harvester subDirs and every ratio string are derived from it.

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociationLabels_cff import (
    truthBranchWorkingPointsPSet,
    recoLabels,
    instanceKey,
)

_wps = list(truthBranchWorkingPointsPSet.names)

# Axis definition per x variable, shared by every domain. Built here so the booking, the
# harvester strings and the plot script all read one list.
_axes = {
    "pt": (50, 0.0, 100.0),
    "eta": (50, -4.0, 4.0),
    "phi": (36, -3.2, 3.2),
    "nhits": (40, 0.0, 40.0),
    "vertpos": (40, 0.0, 60.0),
    "zpos": (40, -30.0, 30.0),
    "dxy": (40, -5.0, 5.0),
    "dz": (40, -20.0, 20.0),
    # Graph-only axes: depth of the branch root in the graph, and the fraction of the
    # branch footprint that belongs to the root particle itself.
    "depth": (15, 0.0, 15.0),
    "root_footprint_fraction": (20, 0.0, 1.0),
}
_algoBlockArgs = {}
for _name, (_n, _lo, _hi) in _axes.items():
    _algoBlockArgs["nint_" + _name] = cms.int32(_n)
    _algoBlockArgs["min_" + _name] = cms.double(_lo)
    _algoBlockArgs["max_" + _name] = cms.double(_hi)
_algoBlockArgs.update(
    nintScore=cms.int32(50), minScore=cms.double(0.0), maxScore=cms.double(1.0),
    nintShared=cms.int32(50), minShared=cms.double(0.0), maxShared=cms.double(50.0),
    # Wide on purpose. The truth reference is the BRANCH ROOT, and a reco object matched
    # to a branch by shared hits or energy can belong to a descendant of that root, so
    # the residual has a long tail that a narrow window pushes into the overflow, leaving
    # the slice fit with a nearly flat in-range distribution and no convergence.
    nintRes=cms.int32(120), minRes=cms.double(-1.5), maxRes=cms.double(1.5),
    # Coarser than the efficiency axes on purpose: every x slice of the residual 2D gets
    # a Gaussian fit, and a slice with a handful of entries returns a meaningless width.
    nint_res_eta=cms.int32(20), min_res_eta=cms.double(-4.0), max_res_eta=cms.double(4.0),
    nint_res_pt=cms.int32(15), min_res_pt=cms.double(0.0), max_res_pt=cms.double(100.0),
)

# Truth-side variables are properties of the BRANCH, so every domain supplies all of
# them. Reco-side variables are properties of the reco object and differ by domain: a
# vertex has no momentum and no impact parameter, a trackster has no track parameters.
# Booking a variable a domain cannot fill would put a spike at zero in every reco-side
# plot and read as a real feature.
truthPlotVariables = ["pt", "eta", "phi", "nhits", "vertpos", "zpos", "dxy", "dz", "depth", "root_footprint_fraction"]

_domains = [
    dict(
        name="tracks",
        module="TruthBranchTrackValidator",
        label="truthBranchTrackValidator",
        associator="allTrackToTruthBranchAssociators",
        dirName="TruthInfo/Offline/Tracking/",
        recoVariables=["pt", "eta", "phi", "nhits", "vertpos", "zpos", "dxy", "dz"],
    ),
    dict(
        name="vertices",
        module="TruthBranchVertexValidator",
        label="truthBranchVertexValidator",
        associator="allVertexToTruthBranchAssociators",
        dirName="TruthInfo/Offline/Vertexing/",
        # A vertex has a position and a track multiplicity, and nothing else this set
        # can express. The TRUTH object here is a graph vertex, not a particle branch,
        # so pt, eta, depth and root_footprint_fraction do not exist on that side either.
        recoVariables=["nhits", "vertpos", "zpos"],
        truthVariables=["nhits", "vertpos", "zpos"],
        sharedRange=(0.0, 1.0),
        # nhits counts tracks on the reco side but PARTICLES at the truth vertex, and an
        # interaction vertex has hundreds of them: the 40-bin default put every truth
        # entry in the overflow, so the efficiency was empty in the visible range.
        axisOverrides={"nhits": (50, 0.0, 500.0)},
    ),
    dict(
        name="secondaryVertices",
        module="TruthBranchVertexValidator",
        label="truthBranchSecondaryVertexValidator",
        associator="allSecondaryVertexToTruthBranchAssociators",
        dirName="TruthInfo/Offline/SecondaryVertexing/",
        recoVariables=["nhits", "vertpos", "zpos"],
        truthVariables=["nhits", "vertpos", "zpos"],
        sharedRange=(0.0, 1.0),
    ),
    dict(
        name="tracksters",
        module="TruthBranchTracksterValidator",
        label="truthBranchTracksterValidator",
        associator="truthBranchTracksterAssociators",
        dirName="TruthInfo/Offline/Calorimetry/",
        # A trackster has a barycentre and a layer-cluster count; its pt is the raw
        # energy projected transversally along that barycentre.
        recoVariables=["pt", "eta", "phi", "nhits", "vertpos", "zpos"],
    ),
]

# The HLT menu's reconstruction of the same event, same domains and same variables. A
# domain the menu does not reconstruct has no labels and is skipped below, so nothing
# has to be commented out when the menu changes.
_hltDomains = [
    dict(_d,
         flavour="hlt",
         label="hlt" + _d["label"][0].upper() + _d["label"][1:],
         associator={"allTrackToTruthBranchAssociators": "hltTrackToTruthBranchAssociators",
                     "allVertexToTruthBranchAssociators": "hltVertexToTruthBranchAssociators",
                     "allSecondaryVertexToTruthBranchAssociators": "hltVertexToTruthBranchAssociators",
                     "truthBranchTracksterAssociators": "hltTruthBranchTracksterAssociators"}[_d["associator"]],
         dirName=_d["dirName"].replace("TruthInfo/Offline/", "TruthInfo/HLT/"))
    for _d in _domains
]
for _d in _domains:
    _d["flavour"] = "offline"
_domains = _domains + [_d for _d in _hltDomains if recoLabels(_d["name"], "hlt")]


def _algoBlock(recoVariables, truthVariables=None, sharedRange=None, axisOverrides=None):
    args = dict(_algoBlockArgs)
    for _var, (_n, _lo, _hi) in (axisOverrides or {}).items():
        args["nint_" + _var] = cms.int32(_n)
        args["min_" + _var] = cms.double(_lo)
        args["max_" + _var] = cms.double(_hi)
    if sharedRange is not None:
        # A composite domain's shared quantity is a FRACTION of the object's
        # constituents, so it lives in [0, 1]; the hit-based default of [0, 50] counts
        # hits or GeV and would put every fraction in the first bin.
        args["minShared"] = cms.double(sharedRange[0])
        args["maxShared"] = cms.double(sharedRange[1])
    return cms.PSet(
        truthVariables=cms.vstring(*(truthVariables or truthPlotVariables)),
        recoVariables=cms.vstring(*recoVariables),
        **args,
    )


# Every ratio is formed by DQMGenericClient from the num/denom names, so this package
# ships no harvesting C++. The metric set follows MultiTrackValidator (efficiency, fake,
# duplicate, pileup) plus purity from the TICL trackster validation, which asks the
# complementary question: how much of the reco object belongs to the branch it matched.
# Which direction each metric belongs to is not a style choice; it decides the
# denominator the number carries.
#
#   TRUTH to RECO, denominator the truth object: efficiency, duplicate rate, split rate.
#     The truth target is fixed a priori by the domain's resolution, so the reco-driven
#     adaptive working point plays no part in choosing it.
#   RECO to TRUTH, denominator the reco object: fake rate, pileup rate, reco purity.
#
# This is the split HGVHistoProducerAlgo already uses (maxSimToRecoScoreForPurity and
# maxSimToRecoScoreForDuplicate on one side, maxRecoToSimScoreForNonFake and
# maxRecoToSimScoreForMerge on the other) and that QuickTrackAssociatorByHits encodes as
# two separate implementations with different denominators.
def _efficiencyStrings(recoVariables, truthVariables=None):
    out = []
    for var in (truthVariables or truthPlotVariables):
        out.append(f"efficiency_vs_{var} 'Branch efficiency vs {var}' num_assoc(simToReco)_{var} num_simul_{var}")
        out.append(f"duplicate_vs_{var} 'Duplicate rate vs {var}' num_duplicate_{var} num_simul_{var}")
        out.append(f"splitrate_vs_{var} 'Split rate vs {var}' num_split_{var} num_simul_{var}")
    for var in recoVariables:
        out.append(f"fakerate_vs_{var} 'Fake rate vs {var}' num_assoc(recoToSim)_{var} num_reco_{var} fake")
        out.append(f"pileuprate_vs_{var} 'Pileup rate vs {var}' num_pileup_{var} num_reco_{var}")
        out.append(f"recopurity_vs_{var} 'Reco purity vs {var}' num_assoc(recoToSim)_{var} num_reco_{var}")
    # Efficiency and duplicate rate against the Geant4 creation process of the branch.
    # The axis is categorical, one bin per truth::VertexReason, and it exists only
    # because the graph keeps the process that made each particle.
    out.append("efficiency_vs_reason 'Branch efficiency vs creation process' "
               "num_assoc(simToReco)_reason num_simul_reason")
    out.append("duplicate_vs_reason 'Duplicate rate vs creation process' num_duplicate_reason num_simul_reason")
    return out


# Gaussian slice fits, the same mechanism MTV uses: DQMGenericClient books <prefix>_Mean
# and <prefix>_Sigma from each 2D. The string is three tokens,
# "<outputPrefix> '<title>' <sourceHistogram>"; a two-token form parses without an error
# and silently produces nothing.
_resolutions = [
    "ptres_vs_eta 'Relative p_{T} residual vs #eta' ptres_vs_eta",
    "ptres_vs_pt 'Relative p_{T} residual vs p_{T}' ptres_vs_pt",
    "etares_vs_eta '#eta residual vs #eta' etares_vs_eta",
    "phires_vs_eta '#phi residual vs #eta' phires_vs_eta",
]

truthBranchValidationSequence = cms.Sequence()
# One harvester per domain: DQMGenericClient applies one string list to all its subDirs,
# so a folder that never booked num_reco_pt must not be asked for fakerate_vs_pt.
truthBranchHarvestingSequence = cms.Sequence()

for _d in _domains:
    _analyzer = cms.EDProducer(
        _d["module"],
        src=cms.InputTag("truthLogicalGraphProducer"),
        hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
        dirName=cms.string(_d["dirName"]),
        associator=cms.string(_d["associator"]),
        recoCollections=cms.VInputTag(
            *[cms.InputTag(*l.split(":")) for l in recoLabels(_d["name"], _d["flavour"])]),
        workingPoints=cms.vstring(*_wps),
        histoProducerAlgoBlock=_algoBlock(_d["recoVariables"], _d.get("truthVariables"),
                                          _d.get("sharedRange"), _d.get("axisOverrides")),
    )
    globals()[_d["label"]] = _analyzer
    truthBranchValidationSequence += _analyzer

    _folders = [_d["dirName"] + instanceKey(_label) + "_" + _wp
                for _label in recoLabels(_d["name"], _d["flavour"]) for _wp in _wps]
    _harvester = DQMEDHarvester(
        "DQMGenericClient",
        subDirs=cms.untracked.vstring(*_folders),
        efficiency=cms.vstring(*_efficiencyStrings(_d["recoVariables"], _d.get("truthVariables"))),
        resolution=cms.vstring(*_resolutions),
        # Fit the core, not the tail: the slice fit is restricted to a window around the
        # peak, which is what makes Sigma a resolution rather than the width of the axis.
        resolutionLimitedFit=cms.untracked.bool(True),
        verbose=cms.untracked.uint32(0),
        outputFileName=cms.untracked.string(""),
    )
    globals()[_d["label"].replace("Validator", "PostProcessor")] = _harvester
    truthBranchHarvestingSequence += _harvester
