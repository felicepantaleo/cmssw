# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# The DQM analyzers and their harvesting, both generated from the same label and
# working-point lists the associators use, so the folder names, the ME names and the
# harvester subDirs cannot drift apart.
#
# One entry in _domains below is all it takes to add a reco domain: the analyzer, the
# folder names, the harvester subDirs and every ratio string are derived from it.

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

# Acceptance regions, mirroring truth::kEtaRegionFolders. Each num_* row is booked again
# in a sub-folder of the same name, with the SAME ME names, so one string list harvests
# the inclusive folder and every region.
_etaRegions = ["", "etaLt15", "eta15to30", "eta30to45"]


def _withRegions(folders):
    return [f + ("" if not r else "/" + r) for f in folders for r in _etaRegions]


from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociationLabels_cff import (
    truthBranchWorkingPointsPSet,
    recoLabels,
    instanceKey,
)

_wps = list(truthBranchWorkingPointsPSet.names)

# Branch levels of the truth graph, matching truthGraphAssociators_cff: the
# truth-driven metrics are measured once per level, in per-level folders, while the
# reco-driven metrics keep their per-working-point folders. Composite domains have no
# levels; their one truth-driven folder is named by the domain's vertex resolution.
# reconstructableFromSignal is the resonance's visible final state: the walk from each
# signal root down to the first object a detector reconstructs, pi0 included as an object
# rather than as two photons. It needs LevelFlag::Signal on the graph, which is stamped at
# DIGI, so a sample produced before that carries an EMPTY level rather than a wrong one.
# partonJets is one root per parton-initiated jet: the hard-scatter legs that are quarks
# or gluons, each standing for everything downstream of it. There is no clustering; the
# jet IS the descendant subgraph and its flavour is the parton's PDG id. The deepest-
# element rule keeps the b rather than the top above it, so a jet is never counted twice.
_truthLevels = ["stableLegsFromUpstream", "caloBoundary", "stableDecayProducts", "hardProcess",
                "reconstructableFromSignal", "underlyingEvent", "partonJets"]

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
    # Where the branch ENTERS the calorimeter, not where its root was produced. Same
    # range as eta so the two are read side by side; a branch that never reached the
    # calorimeter is filled at kNoCaloEntry and lands in the underflow of both the
    # numerator and the denominator.
    "caloeta": (50, -4.0, 4.0),
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
truthPlotVariables = ["pt", "eta", "phi", "nhits", "vertpos", "zpos", "dxy", "dz", "depth",
                      "root_footprint_fraction", "caloeta"]

# Individual-match thresholds per domain, taken from the corresponding standard
# validation rather than invented. Tracks and vertices are judged on the fraction of
# shared COMPONENTS (hits; constituent tracks), calorimetry on the fraction of shared
# ENERGY. Each value cites where it lives in the reference package.
#
# Tracks: QuickTrackAssociatorByHits with SimToRecoDenominator='reco' counts a truth
# object reconstructed when a track shares MORE THAN 75% of its own hits with it, with
# no truth-normalised cut at all (Cut_RecoToSim=0.75, Purity_SimToReco=0.75,
# Quality_SimToReco=0.5 in SimTracker/TrackAssociatorProducers/python/
# quickTrackAssociatorByHits_cfi.py:4-8, applied in plugins/
# QuickTrackAssociatorByHitsImpl.cc:234-244 and 312-326; MultiTrackValidator adds no
# further cut, plugins/MultiTrackValidator.cc:939-943).
#
# Vertices: VertexAssociatorByPositionAndTracks gates on POSITION and ships its
# shared-track-fraction cut DISABLED (sharedTrackFraction=-1.0 in SimTracker/
# VertexAssociation/plugins/VertexAssociatorByPositionAndTracksProducer.cc:72, the
# fraction branch at src/VertexAssociatorByPositionAndTracks.cc:129), so on the
# shared-components axis this framework uses, the reference criterion is any positive
# shared fraction.
#
# Calorimetry: HGCalValidator counts the three on DIFFERENT axes, and the association
# scores are the TICL ones (Validation/HGCalValidation/src/HGVHistoProducerAlgo.cc:
# 2897-2899). EFFICIENCY is a SHARED ENERGY FRACTION cut, shared energy over the truth
# branch's energy IN THE DETECTORS THE COLLECTION RECONSTRUCTS (the reference sim
# trackster exists only in HGCAL; the truth branch here also holds the barrel deposits
# of the same particles, which no endcap reco object can cover), above
# minTSTSharedEneFracEfficiency = 0.5
# (Validation/HGCalValidation/python/HGVHistoProducerAlgoBlock_cfi.py:82). PURITY and
# DUPLICATE cut the simToReco score below maxSimToRecoScoreForPurity/Duplicate = 0.2
# (cfi:72-73). FAKE and MERGE cut the recoToSim score below
# maxRecoToSimScoreForNonFake/Merge = 0.6 (cfi:70-71, applied
# HGVHistoProducerAlgo.cc:2819-2820).
_trackThresholds = dict(minTruthPurityForIndividual=0.0, minRecoPurityLoose=0.75)
_vertexThresholds = dict(minTruthPurityForIndividual=0.0, minRecoPurityLoose=0.0)
_caloThresholds = dict(minSharedEnergyFractionForIndividual=0.5,
                       maxSimToRecoScoreForDuplicate=0.2,
                       maxRecoToSimScore=0.6)

_domains = [
    dict(
        name="tracks",
        module="TruthBranchTrackValidator",
        label="truthBranchTrackValidator",
        associator="allTrackToTruthBranchAssociators",
        dirName="TruthInfo/Offline/Tracking/",
        recoVariables=["pt", "eta", "phi", "nhits", "vertpos", "zpos", "dxy", "dz"],
        thresholds=_trackThresholds,
    ),
    dict(
        name="vertices",
        module="TruthBranchVertexValidator",
        label="truthBranchVertexValidator",
        associator="allVertexToTruthBranchAssociators",
        dirName="TruthInfo/Offline/Vertexing/",
        # A primary vertex is resolved at the INTERACTION, matching the associator.
        vertexResolution="interaction",
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
        thresholds=_vertexThresholds,
    ),
    dict(
        name="secondaryVertices",
        module="TruthBranchVertexValidator",
        label="truthBranchSecondaryVertexValidator",
        associator="allSecondaryVertexToTruthBranchAssociators",
        dirName="TruthInfo/Offline/SecondaryVertexing/",
        # A secondary vertex is resolved at the IMMEDIATE production vertex.
        vertexResolution="immediate",
        recoVariables=["nhits", "vertpos", "zpos"],
        truthVariables=["nhits", "vertpos", "zpos"],
        sharedRange=(0.0, 1.0),
        thresholds=_vertexThresholds,
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
        thresholds=_caloThresholds,
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
# denominator the number carries AND the folder family it lives in.
#
#   TRUTH to RECO, denominator the truth object: efficiency, duplicate rate, split rate.
#     The truth target is fixed a priori per graph level, so these live in the
#     per-level folders and never see a working point.
#   RECO to TRUTH, denominator the reco object: fake rate, pileup rate, reco purity.
#     These live in the per-working-point folders.
#
# This is the split HGVHistoProducerAlgo already uses (maxSimToRecoScoreForPurity and
# maxSimToRecoScoreForDuplicate on one side, maxRecoToSimScoreForNonFake and
# maxRecoToSimScoreForMerge on the other) and that QuickTrackAssociatorByHits encodes as
# two separate implementations with different denominators.
# duplicate is skipped for a calorimetric domain, which does not book the numerator:
# its reco objects are built from disjoint layer clusters, so two of them cannot each
# capture most of the same branch energy. Asking for a ratio whose numerator was never
# booked produces an empty plot and a harvester warning per folder.
def _truthDrivenStrings(truthVariables=None, duplicate=True):
    out = []
    for var in (truthVariables or truthPlotVariables):
        out.append(f"efficiency_vs_{var} 'Branch efficiency vs {var}' num_assoc(simToReco)_{var} num_simul_{var}")
        # Cumulative: the truth object counts as found when all reco objects of the
        # collection together cover it, not only when a single one does.
        out.append(f"efficiency_cumulative_vs_{var} 'Cumulative branch efficiency vs {var}' "
                   f"num_assoc_cumulative_{var} num_simul_{var}")
        if duplicate:
            out.append(f"duplicate_vs_{var} 'Duplicate rate vs {var}' num_duplicate_{var} num_simul_{var}")
        out.append(f"splitrate_vs_{var} 'Split rate vs {var}' num_split_{var} num_simul_{var}")
    # Efficiency and duplicate rate against the Geant4 creation process of the branch.
    # The axis is categorical, one bin per truth::VertexReason, and it exists only
    # because the graph keeps the process that made each particle.
    out.append("efficiency_vs_reason 'Branch efficiency vs creation process' "
               "num_assoc(simToReco)_reason num_simul_reason")
    if duplicate:
        out.append("duplicate_vs_reason 'Duplicate rate vs creation process' num_duplicate_reason num_simul_reason")
    return out


# strict adds the calorimetric non-fake criterion as its OWN page. Only a calorimetric
# domain books num_assoc_strict, so asking for it elsewhere would be an empty plot.
def _recoDrivenStrings(recoVariables, strict=False):
    out = []
    for var in recoVariables:
        # A fake is an object no truth branch owns: none of the dominance antichain
        # contributes to it, or several do with no winner. The two are disjoint and
        # nocandidate below is the first of them on its own.
        out.append(f"fakerate_vs_{var} 'Fake rate vs {var}' num_dominated_{var} num_reco_{var} fake")
        out.append(f"nocandidate_vs_{var} 'No-candidate rate vs {var}' "
                   f"num_assoc(recoToSim)_{var} num_reco_{var} fake")
        # Where the dominance question is UNDEFINED: the object matched truth, but none
        # of its candidates sits at the dominance level. Deliberately not folded into
        # the fake rate, which would measure level coverage rather than reconstruction.
        out.append(f"nolevelcandidate_vs_{var} 'No dominance-level candidate vs {var}' "
                   f"num_levelcandidate_{var} num_reco_{var} fake")
        if strict:
            out.append(f"contaminated_vs_{var} 'Contaminated rate vs {var}' "
                       f"num_assoc_strict_{var} num_reco_{var} fake")
        out.append(f"pileuprate_vs_{var} 'Pileup rate vs {var}' num_pileup_{var} num_reco_{var}")
        # Its own numerator, filled with the purity as a weight. Dividing the UNWEIGHTED
        # match count by num_reco would give the matched fraction a second time.
        out.append(f"recopurity_vs_{var} 'Reco purity vs {var}' num_recopurity_{var} num_reco_{var}")
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
    # Every denominator here is an ANTICHAIN: the levels by construction, and
    # signal/signalNoSelection because the seeds are reduced to their most upstream
    # members. A set holding both a particle and its own daughter would count the same
    # energy twice, so it cannot be a truth denominator.
    # The truth-driven folder suffixes: the graph levels, the overall signal entry
    # (denominator the preset seed objects, signalSeeds) and the same seed objects with
    # no selector cut at all (signalSeedsNoSelection) for a hit-based domain, the vertex
    # resolution for a composite one.
    _truthSuffixes = ([_d["vertexResolution"]] if "vertexResolution" in _d
                      else _truthLevels + ["signal", "signalNoSelection"])
    _truthArgs = (dict(vertexResolution=cms.string(_d["vertexResolution"]))
                  if "vertexResolution" in _d else dict(truthLevels=cms.vstring(*_truthLevels)))
    _analyzer = cms.EDProducer(
        _d["module"],
        src=cms.InputTag("truthLogicalGraphProducer"),
        hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
        dirName=cms.string(_d["dirName"]),
        associator=cms.string(_d["associator"]),
        recoCollections=cms.VInputTag(
            *[cms.InputTag(*l.split(":")) for l in recoLabels(_d["name"], _d["flavour"])]),
        workingPoints=cms.vstring(*_wps),
        # Only the thresholds the domain is judged by, because each analyzer declares
        # only those: the calorimetric criteria are three cuts on two different axes,
        # the shared-component ones two cuts on one.
        **{_k: cms.double(_v) for _k, _v in _d["thresholds"].items()},
        histoProducerAlgoBlock=_algoBlock(_d["recoVariables"], _d.get("truthVariables"),
                                          _d.get("sharedRange"), _d.get("axisOverrides")),
        **_truthArgs,
    )
    globals()[_d["label"]] = _analyzer
    truthBranchValidationSequence += _analyzer

    # Two harvesters per domain because DQMGenericClient applies one string list to all
    # its subDirs: the per-WP folders carry only reco-driven MEs, the per-level folders
    # only truth-driven ones, and asking a folder for a ratio it never booked is noise.
    _wpFolders = [_d["dirName"] + instanceKey(_label) + "_" + _wp
                  for _label in recoLabels(_d["name"], _d["flavour"]) for _wp in _wps]
    _harvester = DQMEDHarvester(
        "DQMGenericClient",
        subDirs=cms.untracked.vstring(*_withRegions(_wpFolders)),
        efficiency=cms.vstring(*_recoDrivenStrings(
            _d["recoVariables"],
            strict="minSharedEnergyFractionForIndividual" in _d["thresholds"])),
        resolution=cms.vstring(*_resolutions),
        # Fit the core, not the tail: the slice fit is restricted to a window around the
        # peak, which is what makes Sigma a resolution rather than the width of the axis.
        resolutionLimitedFit=cms.untracked.bool(True),
        verbose=cms.untracked.uint32(0),
        outputFileName=cms.untracked.string(""),
    )
    globals()[_d["label"].replace("Validator", "PostProcessor")] = _harvester
    truthBranchHarvestingSequence += _harvester

    _truthFolders = [_d["dirName"] + instanceKey(_label) + "_" + _suffix
                     for _label in recoLabels(_d["name"], _d["flavour"]) for _suffix in _truthSuffixes]
    _truthHarvester = DQMEDHarvester(
        "DQMGenericClient",
        subDirs=cms.untracked.vstring(*_withRegions(_truthFolders)),
        efficiency=cms.vstring(*_truthDrivenStrings(
            _d.get("truthVariables"),
            duplicate="minSharedEnergyFractionForIndividual" not in _d["thresholds"])),
        resolution=cms.vstring(),
        verbose=cms.untracked.uint32(0),
        outputFileName=cms.untracked.string(""),
    )
    globals()[_d["label"].replace("Validator", "TruthPostProcessor")] = _truthHarvester
    truthBranchHarvestingSequence += _truthHarvester
