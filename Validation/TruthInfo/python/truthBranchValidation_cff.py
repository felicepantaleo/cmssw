# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# The DQM analyzers and their harvesting, generated from the label and working-point lists
# that the associators use, so the folder names, the ME names and the harvester subDirs agree.
# One entry in _domains adds a reco domain: the analyzer, the folders, the harvester subDirs
# and the ratio strings come from it.

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

# Acceptance regions, the same as truth::kEtaRegionFolders. Each num_* row is booked again in
# a sub-folder of the same name, with the same ME names, so one string list harvests all of them.
# Harvest only the region folders that a domain books.
_etaRegions = ["", "etaLt15", "eta15to30", "eta30to45"]
# The bands a domain books in addition to the inclusive folder, by default.
_allEtaRegions = _etaRegions[1:]


def _withRegions(folders, regions=None):
    bands = [""] + list(_allEtaRegions if regions is None else regions)
    return [f + ("" if not r else "/" + r) for f in folders for r in bands]


def _harvestedFolders(domain, suffixes):
    """The folders one harvester reads, one per collection and suffix.

    The trackster collections of a reconstruction job are found from its schedule and its
    input, which a harvesting job does not have. Their folders are therefore matched by a
    pattern, and DQMGenericClient also takes the eta-region folders below a match.
    """
    if domain["name"] == "tracksters":
        return [domain["dirName"] + "*_" + suffix + "$" for suffix in suffixes]
    folders = [domain["dirName"] + instanceKey(label) + "_" + suffix
               for label in recoLabels(domain["name"], domain["flavour"]) for suffix in suffixes]
    return _withRegions(folders, domain.get("etaRegions"))


from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociationLabels_cff import (
    truthBranchWorkingPointsPSet,
    recoLabels,
    instanceKey,
    _truthLevels,
    _signalSeedPdgIds,
    _signalSeedHadronFlavors,
)


_wps = list(truthBranchWorkingPointsPSet.names)

# Branch levels of the truth graph, from the same list as the associators, so each booked
# folder has a denominator product.
_levels = list(_truthLevels)

# Axis definition (nbins, min, max) per x variable, shared by every domain.
_axes = {
    # Symlog to 1000 GeV: a parton jet reaches several hundred GeV in ttbar and the QCD
    # flat-pT sample goes to 3000, while caloBoundary has mostly sub-GeV particles.
    "pt": (50, 0.0, 1000.0),
    # +-4.5 is the forward acceptance boundary, so the eta30to45 region fits on the axis.
    # Beam remnants beyond 4.5 go to the overflow, outside every acceptance region.
    "eta": (50, -4.5, 4.5),
    "phi": (36, -3.2, 3.2),
    # Symlog: a branch footprint has from one hit to thousands. On no-PU ttbar, 7.1% of truth
    # nhits is above 40, and a partonJets subgraph holds 961 to 3539 hits.
    "nhits": (50, 0.0, 10000.0),
    # Symlog: a heavy-flavour decay length is sub-millimetre, a nuclear interaction is at tens
    # of cm. With uniform 1.5 cm bins, 93.4% of truth SVs are in the first bin.
    "vertpos": (40, 0.0, 60.0),
    "zpos": (40, -30.0, 30.0),
    "dxy": (40, -5.0, 5.0),
    "dz": (40, -20.0, 20.0),
    # Graph-only axes: depth of the branch root in the graph, and the fraction of the
    # branch footprint that belongs to the root particle itself.
    "depth": (15, 0.0, 15.0),
    # Top edge above 1: a root that owns the full footprint has a fraction of exactly 1.0,
    # which is the overflow of a [0, 1] axis. This is 36.6% of the entries.
    "root_footprint_fraction": (21, 0.0, 1.05),
    # The species that starts the truth object, one bin each for other, d, u, s, c, b, t, g.
    # Only partonJets roots are partons, so all other levels fill bin 0.
    "flavour": (8, 0.0, 8.0),
    # Eta where the branch enters the calorimeter, on the same range as eta. A branch that
    # does not reach the calorimeter is filled at kNoCaloEntry, in the underflow of the
    # numerator and the denominator. The large underflow is intended.
    "caloeta": (50, -4.5, 4.5),
}
# Symlog binning for axes that span decades: one linear bin up to the threshold below, then
# log bins to the top. A log axis cannot hold 0, and these axes have entries at 0: on DY, 20.5%
# of the signal level has pt exactly 0 (the pre-ISR copy of the resonance), and a decay length
# of 0 means the vertex is the primary. 19% of partonJets entries have pt above 100 GeV.
_linthresh = {
    "pt": 0.1,        # GeV
    "vertpos": 0.001,  # cm, that is 10 microns
    "nhits": 1.0,      # one hit
}

_algoBlockArgs = {}
for _name, (_n, _lo, _hi) in _axes.items():
    _algoBlockArgs["nint_" + _name] = cms.int32(_n)
    _algoBlockArgs["min_" + _name] = cms.double(_lo)
    _algoBlockArgs["max_" + _name] = cms.double(_hi)
    _algoBlockArgs["linthresh_" + _name] = cms.double(_linthresh.get(_name, 0.0))
_algoBlockArgs.update(
    nintScore=cms.int32(50), minScore=cms.double(0.0), maxScore=cms.double(1.0),
    nintShared=cms.int32(50), minShared=cms.double(0.0), maxShared=cms.double(50.0),
    # Wide range: the truth reference is the branch root, and a matched reco object can belong
    # to a descendant of that root. The residual has a long tail, and the slice fit needs it in range.
    nintRes=cms.int32(120), minRes=cms.double(-1.5), maxRes=cms.double(1.5),
    # Coarser than the efficiency axes: each x slice gets a Gaussian fit, which needs entries.
    nint_res_eta=cms.int32(20), min_res_eta=cms.double(-4.0), max_res_eta=cms.double(4.0),
    nint_res_pt=cms.int32(15), min_res_pt=cms.double(0.0), max_res_pt=cms.double(100.0),
)

# Truth-side variables are properties of the branch, so every hit-based domain supplies all of
# them. Reco-side variables differ by domain: a vertex has no momentum and no impact parameter,
# a trackster has no track parameters. A variable that a domain cannot fill makes a false spike at zero.
truthPlotVariables = ["pt", "eta", "phi", "nhits", "vertpos", "zpos", "dxy", "dz", "depth",
                      "root_footprint_fraction", "caloeta", "flavour"]

# Individual-match thresholds per domain, taken from the standard validation of each domain.
# Tracks and vertices use the fraction of shared components (hits, constituent tracks).
# Calorimetry uses the fraction of shared energy.
#
# Tracks: QuickTrackAssociatorByHits (quickTrackAssociatorByHits_cfi.py) with
# SimToRecoDenominator='reco' counts a truth object as reconstructed when a track shares more
# than 75% of its own hits with it, with no truth-normalised cut. MultiTrackValidator adds no cut.
#
# Calorimetry: HGCalValidator (HGVHistoProducerAlgoBlock_cfi.py) uses three different axes.
# Efficiency: the shared energy over the truth branch energy in the detectors that the collection
# reconstructs is above minTSTSharedEneFracEfficiency = 0.5. Duplicate: the simToReco score is
# below maxSimToRecoScoreForDuplicate = 0.2. Fake: the recoToSim score is below
# maxRecoToSimScoreForNonFake = 0.6.
_trackThresholds = dict(minTruthPurityForIndividual=0.0, minRecoPurityLoose=0.75)
# Vertices: the reference association gates on a z window (absZ, sigmaZ in
# secondaryVertexAssociatorByPositionAndTracks_cfi.py) and disables its shared-track cut.
# This association has no position gate: it matches vertices through shared tracks. With both
# cuts at 0, one shared track is a match and the efficiency is 1 by construction. A cut of more
# than half of the constituents on each side replaces the position window.
_vertexThresholds = dict(minTruthPurityForIndividual=0.5, minRecoPurityLoose=0.5)
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
        # A primary vertex is resolved at the interaction, as in the associator.
        vertexResolution="interaction",
        # A vertex has a position and a track multiplicity only. The truth object is a graph
        # vertex, not a particle branch, so it has no pt, eta, depth or root_footprint_fraction.
        recoVariables=["nhits", "vertpos", "zpos"],
        truthVariables=["nhits", "vertpos", "zpos"],
        sharedRange=(0.0, 1.0),
        # A vertex has no pseudorapidity, so no eta-region folders.
        etaRegions=[],
        # nhits counts tracks on the reco side but particles at the truth vertex, and an
        # interaction vertex has hundreds of particles.
        axisOverrides={"nhits": (50, 0.0, 500.0)},
        thresholds=_vertexThresholds,
    ),
    dict(
        name="secondaryVertices",
        module="TruthBranchVertexValidator",
        label="truthBranchSecondaryVertexValidator",
        associator="allSecondaryVertexToTruthBranchAssociators",
        dirName="TruthInfo/Offline/SecondaryVertexing/",
        # A secondary vertex is resolved at the immediate production vertex.
        vertexResolution="immediate",
        recoVariables=["nhits", "vertpos", "zpos"],
        truthVariables=["nhits", "vertpos", "zpos"],
        sharedRange=(0.0, 1.0),
        # A vertex has no pseudorapidity, so no eta-region folders.
        etaRegions=[],
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
        # HGCal ranges: a trackster barycentre is at |z| of about 320 to 520 cm and at a
        # transverse radius up to about 180 cm. On the tracker ranges, 100% of reco zpos and
        # 54% of reco vertpos are outside the axis (200 no-PU ttbar events).
        recoAxisOverrides={"zpos": (60, -600.0, 600.0), "vertpos": (50, 0.0, 200.0)},
        thresholds=_caloThresholds,
    ),
    # Particle-flow clusters of the barrel PF blocks, one domain per subdetector as in the
    # associators: the efficiency uses the branch energy fraction in that detector only.
    # A barrel cluster is at a transverse radius of about 130 cm (ECAL) or 180 to 290 cm
    # (HCAL) and at |z| below about 400 cm, outside the tracker ranges.
    dict(
        name="pfClustersEcal",
        module="TruthBranchPFClusterValidator",
        label="truthBranchPFClusterEcalValidator",
        associator="truthBranchPFClusterEcalAssociators",
        dirName="TruthInfo/Offline/PFClustersECAL/",
        recoVariables=["pt", "eta", "phi", "nhits", "vertpos", "zpos"],
        recoAxisOverrides={"zpos": (60, -600.0, 600.0), "vertpos": (50, 0.0, 300.0),
                           "nhits": (50, 0.0, 100.0)},
        thresholds=_caloThresholds,
    ),
    dict(
        name="pfClustersHcal",
        module="TruthBranchPFClusterValidator",
        label="truthBranchPFClusterHcalValidator",
        associator="truthBranchPFClusterHcalAssociators",
        dirName="TruthInfo/Offline/PFClustersHCAL/",
        recoVariables=["pt", "eta", "phi", "nhits", "vertpos", "zpos"],
        recoAxisOverrides={"zpos": (60, -600.0, 600.0), "vertpos": (50, 0.0, 300.0),
                           "nhits": (50, 0.0, 100.0)},
        thresholds=_caloThresholds,
    ),
]

# The HLT reconstruction of the same event, with the same domains and variables.
# A domain that the HLT menu does not reconstruct has no labels and is skipped below.
_hltDomains = [
    dict(_d,
         flavour="hlt",
         label="hlt" + _d["label"][0].upper() + _d["label"][1:],
         associator={"allTrackToTruthBranchAssociators": "hltTrackToTruthBranchAssociators",
                     "allVertexToTruthBranchAssociators": "hltVertexToTruthBranchAssociators",
                     "allSecondaryVertexToTruthBranchAssociators": "hltVertexToTruthBranchAssociators",
                     "truthBranchTracksterAssociators": "hltTruthBranchTracksterAssociators",
                     # No HLT PF-cluster association exists. The HLT label lists are empty,
                     # so the recoLabels filter below drops these domains.
                     "truthBranchPFClusterEcalAssociators": "hltTruthBranchPFClusterEcalAssociators",
                     "truthBranchPFClusterHcalAssociators": "hltTruthBranchPFClusterHcalAssociators"}[_d["associator"]],
         dirName=_d["dirName"].replace("TruthInfo/Offline/", "TruthInfo/HLT/"))
    for _d in _domains
]
for _d in _domains:
    _d["flavour"] = "offline"
_domains = _domains + [_d for _d in _hltDomains if recoLabels(_d["name"], "hlt")]


def _algoBlock(recoVariables, truthVariables=None, sharedRange=None, axisOverrides=None,
               recoAxisOverrides=None, etaRegions=None):
    args = dict(_algoBlockArgs)
    # Reco-side only: a trackster barycentre is in HGCal, the truth production vertex is in the tracker.
    for _var, (_n, _lo, _hi) in (recoAxisOverrides or {}).items():
        args["nint_reco_" + _var] = cms.int32(_n)
        args["min_reco_" + _var] = cms.double(_lo)
        args["max_reco_" + _var] = cms.double(_hi)
        args["linthresh_reco_" + _var] = cms.double(0.0)
    for _var, (_n, _lo, _hi) in (axisOverrides or {}).items():
        args["nint_" + _var] = cms.int32(_n)
        args["min_" + _var] = cms.double(_lo)
        args["max_" + _var] = cms.double(_hi)
        # An override replaces the range, so it also drops the symlog threshold.
        args["linthresh_" + _var] = cms.double(0.0)
    if sharedRange is not None:
        # A composite domain's shared quantity is a fraction of the object's constituents,
        # in [0, 1]. The default [0, 50] counts hits or GeV.
        args["minShared"] = cms.double(sharedRange[0])
        args["maxShared"] = cms.double(sharedRange[1])
    return cms.PSet(
        truthVariables=cms.vstring(*(truthVariables or truthPlotVariables)),
        recoVariables=cms.vstring(*recoVariables),
        etaRegions=cms.vstring(*(_allEtaRegions if etaRegions is None else etaRegions)),
        **args,
    )


# DQMGenericClient forms every ratio from the num/denom names. The metrics follow
# MultiTrackValidator (efficiency, fake, duplicate, pileup), plus the reco purity of the TICL
# trackster validation. The direction of a metric sets its denominator and its folder:
#   truth to reco, denominator the truth object: efficiency, duplicate rate, split rate.
#     These are in the per-level folders and do not depend on a working point.
#   reco to truth, denominator the reco object: fake rate, pileup rate, reco purity.
#     These are in the per-working-point folders.
# A calorimetric domain does not book the duplicate numerator: its reco objects use disjoint
# layer clusters, so two of them cannot each hold most of the same branch energy.
def _truthDrivenStrings(truthVariables=None, duplicate=True):
    out = []
    for var in (truthVariables or truthPlotVariables):
        out.append(f"efficiency_vs_{var} 'Branch efficiency vs {var}' num_assoc(simToReco)_{var} num_simul_{var}")
        # Cumulative: the truth object counts as found when all reco objects of the
        # collection together cover it.
        out.append(f"efficiency_cumulative_vs_{var} 'Cumulative branch efficiency vs {var}' "
                   f"num_assoc_cumulative_{var} num_simul_{var}")
        if duplicate:
            out.append(f"duplicate_vs_{var} 'Duplicate rate vs {var}' num_duplicate_{var} num_simul_{var}")
        out.append(f"splitrate_vs_{var} 'Split rate vs {var}' num_split_{var} num_simul_{var}")
    # Efficiency and duplicate rate vs the creation process of the branch, one bin per
    # truth::VertexReason.
    out.append("efficiency_vs_reason 'Branch efficiency vs creation process' "
               "num_assoc(simToReco)_reason num_simul_reason")
    if duplicate:
        out.append("duplicate_vs_reason 'Duplicate rate vs creation process' num_duplicate_reason num_simul_reason")
    return out


# strict adds the calorimetric non-fake criterion as a separate plot. Only a calorimetric
# domain books num_assoc_strict.
def _recoDrivenStrings(recoVariables, strict=False):
    out = []
    for var in recoVariables:
        # A fake is an object no truth branch owns: none of the dominance antichain
        # contributes to it, or several do with no winner. The two are disjoint and
        # nocandidate below is the first of them on its own.
        out.append(f"fakerate_vs_{var} 'Fake rate vs {var}' num_dominated_{var} num_reco_{var} fake")
        out.append(f"nocandidate_vs_{var} 'No-candidate rate vs {var}' "
                   f"num_assoc(recoToSim)_{var} num_reco_{var} fake")
        # The object matches truth, but none of its candidates is at the dominance level.
        # This is not part of the fake rate, which measures reconstruction, not level coverage.
        out.append(f"nolevelcandidate_vs_{var} 'No dominance-level candidate vs {var}' "
                   f"num_levelcandidate_{var} num_reco_{var} fake")
        if strict:
            out.append(f"contaminated_vs_{var} 'Contaminated rate vs {var}' "
                       f"num_assoc_strict_{var} num_reco_{var} fake")
        out.append(f"pileuprate_vs_{var} 'Pileup rate vs {var}' num_pileup_{var} num_reco_{var}")
        # The numerator is filled with the purity as a weight.
        out.append(f"recopurity_vs_{var} 'Reco purity vs {var}' num_recopurity_{var} num_reco_{var}")
    return out


# Gaussian slice fits, as in MTV: DQMGenericClient books <prefix>_Mean and <prefix>_Sigma from
# each 2D. The string has three tokens, "<outputPrefix> '<title>' <sourceHistogram>".
# A two-token string gives no error and no output.
_resolutions = [
    "ptres_vs_eta 'Relative p_{T} residual vs #eta' ptres_vs_eta",
    "ptres_vs_pt 'Relative p_{T} residual vs p_{T}' ptres_vs_pt",
    "etares_vs_eta '#eta residual vs #eta' etares_vs_eta",
    "phires_vs_eta '#phi residual vs #eta' phires_vs_eta",
]

truthBranchValidationSequence = cms.Sequence()
# Harvesters per domain, because DQMGenericClient applies one string list to all its subDirs.
truthBranchHarvestingSequence = cms.Sequence()
# The HLT analyzers read HLT collections, which an offline reconstruction does not produce,
# so they are in separate sequences.
truthBranchHltValidationSequence = cms.Sequence()
truthBranchHltHarvestingSequence = cms.Sequence()

for _d in _domains:
    # Every denominator is an antichain: the levels by construction, and signal and
    # signalNoSelection because the seeds keep only their most upstream members.
    # A set with a particle and its daughter would count the same energy twice.
    # The truth-driven folder suffixes: for a hit-based domain, the graph levels, signal (the
    # preset seed objects) and signalNoSelection (the same seeds with no selector cut); for a
    # composite domain, the vertex resolution.
    _truthSuffixes = ([_d["vertexResolution"]] if "vertexResolution" in _d
                      else _levels + ["signal", "signalNoSelection"])
    # The analyzer books the signal folders from signalSeedPdgIds, which must have the same
    # value as in the associators. With no preset it is empty and no signal folder is booked.
    _truthArgs = (dict(vertexResolution=cms.string(_d["vertexResolution"]))
                  if "vertexResolution" in _d
                  else dict(truthLevels=cms.vstring(*_levels),
                            signalSeedPdgIds=cms.vint32(*_signalSeedPdgIds),
                            signalSeedHadronFlavors=cms.vint32(*_signalSeedHadronFlavors)))
    _analyzer = cms.EDProducer(
        _d["module"],
        src=cms.InputTag("truthLogicalGraphProducer"),
        hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
        dirName=cms.string(_d["dirName"]),
        associator=cms.string(_d["associator"]),
        recoCollections=cms.VInputTag(
            *[cms.InputTag(*l.split(":")) for l in recoLabels(_d["name"], _d["flavour"])]),
        workingPoints=cms.vstring(*_wps),
        # Only the thresholds of the domain, because each analyzer declares only those.
        **{_k: cms.double(_v) for _k, _v in _d["thresholds"].items()},
        histoProducerAlgoBlock=_algoBlock(_d["recoVariables"], _d.get("truthVariables"),
                                          _d.get("sharedRange"), _d.get("axisOverrides"),
                                          _d.get("recoAxisOverrides"), _d.get("etaRegions")),
        **_truthArgs,
    )
    globals()[_d["label"]] = _analyzer
    if _d["flavour"] == "hlt":
        truthBranchHltValidationSequence += _analyzer
    else:
        truthBranchValidationSequence += _analyzer

    # Two harvesters per domain, because DQMGenericClient applies one string list to all its
    # subDirs: the per-WP folders hold the reco-driven MEs, the per-level folders the truth-driven ones.
    _harvester = DQMEDHarvester(
        "DQMGenericClient",
        subDirs=cms.untracked.vstring(*_harvestedFolders(_d, _wps)),
        efficiency=cms.vstring(*_recoDrivenStrings(
            _d["recoVariables"],
            strict="minSharedEnergyFractionForIndividual" in _d["thresholds"])),
        resolution=cms.vstring(*_resolutions),
        # Fit a window around the peak, so Sigma is the resolution of the core.
        resolutionLimitedFit=cms.untracked.bool(True),
        verbose=cms.untracked.uint32(0),
        outputFileName=cms.untracked.string(""),
    )
    globals()[_d["label"].replace("Validator", "PostProcessor")] = _harvester
    if _d["flavour"] == "hlt":
        truthBranchHltHarvestingSequence += _harvester
    else:
        truthBranchHarvestingSequence += _harvester

    _truthHarvester = DQMEDHarvester(
        "DQMGenericClient",
        subDirs=cms.untracked.vstring(*_harvestedFolders(_d, _truthSuffixes)),
        efficiency=cms.vstring(*_truthDrivenStrings(
            _d.get("truthVariables"),
            duplicate="minSharedEnergyFractionForIndividual" not in _d["thresholds"])),
        resolution=cms.vstring(),
        verbose=cms.untracked.uint32(0),
        outputFileName=cms.untracked.string(""),
    )
    globals()[_d["label"].replace("Validator", "TruthPostProcessor")] = _truthHarvester
    if _d["flavour"] == "hlt":
        truthBranchHltHarvestingSequence += _truthHarvester
    else:
        truthBranchHarvestingSequence += _truthHarvester
