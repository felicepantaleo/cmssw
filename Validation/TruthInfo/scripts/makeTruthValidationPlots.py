#!/usr/bin/env python3
"""Render the truth-branch DQM output as a CMS-styled, browsable gallery.

The four branch-association working points are the comparison this validation exists
to make, so each variable becomes ONE overlay plot with a ratio panel against the
Fixed reference, rather than four isolated histograms the reader has to compare by eye.

Collections, working points and categories are DISCOVERED from the DQM folder names,
so a new collection or working point needs no edit here.

  makeTruthValidationPlots.py DQM_V0001_*.root --outputDir plots
"""
import argparse
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")  # batch backend, must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402
import ROOT  # noqa: E402

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

plt.style.use(hep.style.CMS)

# The reference working point: every ratio is taken against it, and it keeps the first
# colour of the Petroff cycle everywhere in the gallery so colour follows the entity.
REFERENCE_WP = "Fixed"
WP_ORDER = ["Fixed", "AdaptiveTight", "AdaptiveNominal", "AdaptiveLoose"]

# Which metrics we plot, and how each should be read.
METRICS = {
    "efficiency": (
        "Branch efficiency",
        "Of the truth branches that passed the selection, the fraction at least one reco object was matched to.",
        "num_assoc(simToReco) / num_simul",
    ),
    "purity": (
        "Purity",
        "Of the reco objects, the fraction whose best match is a truth branch. The complement of the fake rate, "
        "reported separately because it is the quantity the TICL trackster validation uses.",
        "num_assoc(recoToSim) / num_reco",
    ),
    "fakerate": (
        "Fake rate",
        "Of the reco objects, the fraction with no truth branch at all: reconstructed, but corresponding to "
        "nothing in the simulated event.",
        "1 - num_assoc(recoToSim) / num_reco",
    ),
    "duplicate": (
        "Duplicate rate",
        "Of the selected truth branches, the fraction matched by more than one reco object, that is one truth "
        "object reconstructed several times.",
        "num_duplicate / num_simul",
    ),
    "composition": (
        "Branch composition",
        "What the selected truth branches ARE, as fractions of the efficiency denominator, split by the Geant4 "
        "process that created the branch root. Read this page first: it says what the other pages are averaging "
        "over.",
        "num_simul_reason / sum(num_simul_reason)",
    ),
    "resolution": (
        "Residual mean and width",
        "Gaussian fit, slice by slice, of (reco - truth)/truth for the momentum and of (reco - truth) for the "
        "angles, against the TRUTH variable. Mean is the bias, Sigma the width. Read these with the reference in "
        "mind: it is the BRANCH ROOT, and a reco object matched to a branch by shared hits can correspond to a "
        "DESCENDANT of that root, so the width is dominated by the branch definition, not by tracking. It is "
        "therefore a diagnostic of the truth definition and not a tracking resolution. Slices with fewer than "
        "20 entries, and fits whose width exceeds the fit range or collapses below one bin, are not drawn.",
        "Gaussian slice fit of ptres_vs_*, etares_vs_*, phires_vs_*",
    ),
    "pileuprate": (
        "Pileup rate",
        "Of the reco objects, the fraction whose matched branch belongs to a pileup interaction rather than the "
        "signal one. The graph answers this directly because every branch carries its interaction id.",
        "num_pileup / num_reco",
    ),
}
VARIABLE_MEANING = {
    "pt": "branch root transverse momentum",
    "eta": "branch root pseudorapidity",
    "phi": "branch root azimuth",
    "nhits": "hits of the branch footprint in the truth hit index",
    "vertpos": "radius of the branch production vertex",
    "zpos": "z of the branch production vertex",
    "dxy": "transverse impact parameter of the branch",
    "dz": "longitudinal impact parameter of the branch",
    "depth": "number of ancestors of the branch root in the graph, that is how far down the event history it sits",
    "rootfrac": "fraction of the branch tracker footprint that belongs to the root particle itself rather than to "
                "its descendants; near 1 is a clean single particle",
}
# Per-domain caveats. A number that is correct but not discriminating reads as a result
# unless the page says otherwise, so the page says otherwise.
CATEGORY_NOTE = {
    "Vertexing": (
        "A vertex owns no hits, so it is associated to a truth VERTEX by aggregating the tracks it was built "
        "from: each track carries its own best-matched particle, and that particle's PRODUCTION VERTEX gets the "
        "track's weight. The leading truth vertex's share of the weight is the purity, and tracks whose particles "
        "were produced at an unrelated vertex are the remainder. Purity here is therefore a mean share, not a "
        "matched-or-not count. Note the definition it rests on: a track from a decay downstream of the vertex "
        "counts as contamination, which is why purity falls with track multiplicity. Mapping each track back to "
        "its interaction-level ancestor instead would count only genuine cross-interaction contamination, and is "
        "the natural refinement for primary vertices specifically."
    ),
    "SecondaryVertexing": (
        "Same aggregation as Vertexing, and it is the case the immediate-production-vertex definition suits best: "
        "a secondary vertex IS a decay or interaction vertex, so the tracks that belong to it were produced there."
    ),
    "Calorimetry": (
        "Tracksters are matched on SHARED ENERGY in the calorimeter channel, the same quantity the TICL trackster "
        "validation scores against. The truth denominator spans the whole selector acceptance, so the efficiency "
        "correctly falls to zero outside the HGCAL coverage rather than being renormalised to it."
    ),
}
STYLE = (
    "body{font-family:sans-serif;margin:2em;max-width:1500px;color:#222}"
    "h1{margin-bottom:.2em}h2{border-bottom:1px solid #ccc;padding-bottom:.2em;margin-top:1.6em}"
    "img{border:1px solid #ddd;margin:4px;vertical-align:top}"
    "a{color:#036;text-decoration:none}a:hover{text-decoration:underline}"
    ".grid{display:flex;flex-wrap:wrap}"
    ".def{background:#f4f6f8;border-left:4px solid #5790fc;padding:.8em 1em;margin:1em 0;max-width:62em}"
    ".f{font-family:monospace;background:#fff;padding:.2em .45em;border:1px solid #dde}"
    "ul.idx{line-height:1.8;max-width:62em}"
)
# Plots the graph makes possible that a frozen truth object cannot answer. Shown in the
# gallery so the next step is visible rather than tribal knowledge.
PROPOSED = [
    ("Merge rate by lowest common ancestor",
     "When two branches are reconstructed as one object the graph gives the LCA of the contributors, so the merge "
     "rate can be plotted against the LCA pdgId: WHICH physical object the merge corresponds to, for instance a "
     "pi0 whose two photons merged, not merely that a merge happened."),
    ("Adaptive-level agreement",
     "Fraction of reco objects whose adaptive level equals the fixed best match, versus pt and eta. The direct "
     "measure of what the adaptive climb buys, and flat by construction on single particles."),
    ("Two-channel candidate matching",
     "A TICLCandidate should be matched on calo shared energy AND tracker shared hits at once; the payload for "
     "that is the natural next extension of the shared-hits type."),
    ("Interaction-vertex association for primary vertices",
     "A vertex should be associated to the graph Interaction vertex rather than to particle branches. The present "
     "PV numbers are mechanically correct but aimed at the wrong truth object."),
]
# Metric order drives the page order, so a reader meets efficiency before its failure modes.
METRIC_ORDER = ["composition", "efficiency", "purity", "fakerate", "duplicate", "pileuprate", "resolution"]
VARIABLE_ORDER = ["pt", "eta", "phi", "nhits", "vertpos", "zpos", "dxy", "dz", "depth", "rootfrac"]
# Axes whose bins are named categories rather than numbers, drawn as grouped bars.
CATEGORICAL = ["reason"]
# Gaussian slice fits, ordered so bias is read before width for each quantity.
RESOLUTION_ORDER = [
    "ptres_vs_eta_Mean", "ptres_vs_eta_Sigma",
    "ptres_vs_pt_Mean", "ptres_vs_pt_Sigma",
    "etares_vs_eta_Mean", "etares_vs_eta_Sigma",
    "phires_vs_eta_Mean", "phires_vs_eta_Sigma",
]
_RES_RE = re.compile(r"^(?P<base>\w+res_vs_\w+)_(?P<stat>Mean|Sigma)$")
RESOLUTION_SOURCES = ["ptres_vs_eta", "ptres_vs_pt", "etares_vs_eta", "phires_vs_eta"]
# A Gaussian fitted to a slice with a handful of entries returns a width that is not a
# resolution. Below this many entries the point is dropped, not drawn.
MIN_SLICE_ENTRIES = 20
# A ratio formed from a handful of entries is noise with a large error bar, not a
# measurement. Bins whose DENOMINATOR is below this are not drawn.
MIN_DENOM_ENTRIES = 10
# Which num_* histogram is the denominator of each metric, so a bin can be dropped when
# there was nothing there to divide by.
DENOMINATOR = {
    "efficiency": "num_simul",
    "duplicate": "num_simul",
    "purity": "num_reco",
    "fakerate": "num_reco",
    "pileuprate": "num_reco",
}
CATEGORICAL_MEANING = {
    "reason": (
        "the Geant4 process that CREATED the branch root, read from the VertexReason of its production vertex. "
        "GenOnly is its own bin, not Unknown: a GEN-only production vertex has no SimVertex and therefore no "
        "Geant4 process at all. In a pileup sample it dominates, because collapsePileupGen replaces each pileup "
        "interaction with one GEN-only vertex carrying all of its stable particles. "
        "Primary means the particle came straight from the hard scatter; every other value is a secondary made "
        "in the detector material. This axis exists only because the graph keeps the process that made each "
        "particle: a frozen TrackingParticle or CaloParticle does not carry it."
    ),
}
_ME_RE = re.compile(r"^(?P<metric>efficiency|purity|fakerate|duplicate|pileuprate)_vs_(?P<var>\w+)$")


def hist_arrays(h):
    """Bin edges, contents and errors of a TH1 as numpy arrays."""
    n = h.GetNbinsX()
    edges = np.array([h.GetXaxis().GetBinLowEdge(i) for i in range(1, n + 2)])
    values = np.array([h.GetBinContent(i) for i in range(1, n + 1)])
    errors = np.array([h.GetBinError(i) for i in range(1, n + 1)])
    return edges, values, errors


def bin_labels(h):
    """The alphanumeric bin labels of a categorical axis, or None if it has none."""
    axis = h.GetXaxis()
    labels = [axis.GetBinLabel(i) for i in range(1, h.GetNbinsX() + 1)]
    return labels if any(labels) else None


def discover(tfile):
    """Yield (category, folder, TDirectory) for every directory holding histograms."""

    def walk(directory, path):
        holds = False
        for key in directory.GetListOfKeys():
            obj = key.ReadObj()
            if obj.InheritsFrom("TDirectory"):
                yield from walk(obj, path + [key.GetName()])
            elif obj.InheritsFrom("TH1"):
                holds = True
        if holds and len(path) >= 2:
            yield path[-2], path[-1], directory

    yield from walk(tfile, [])


def collect(files):
    """{category: {collection: {metric: {var: {wp: (edges, values, errors)}}}}}"""
    data = {}
    for fname in files:
        tfile = ROOT.TFile.Open(fname)
        if not tfile or tfile.IsZombie():
            print(f"cannot open {fname}", file=sys.stderr)
            continue
        for category, folder, folderDir in discover(tfile):
            # Folder is "<collection>_<workingPoint>"; split on the LAST underscore so a
            # collection label containing underscores survives.
            if "_" not in folder:
                continue
            collection, wp = folder.rsplit("_", 1)
            for key in folderDir.GetListOfKeys():
                # The per-process population is the denominator of the categorical
                # ratios; it is carried along so a bar can be dropped when the process
                # simply does not occur, rather than drawn as a zero efficiency.
                name = key.GetName()
                if name.startswith("num_simul_") or name.startswith("num_reco_"):
                    obj = key.ReadObj()
                    if obj.InheritsFrom("TH1") and not obj.InheritsFrom("TH2"):
                        _, counts, _ = hist_arrays(obj)
                        (
                            data.setdefault(category, {})
                            .setdefault(collection, {})
                            .setdefault("_denom", {})
                            .setdefault(name, {})[wp]
                        ) = counts
                if key.GetName() == "num_simul_reason":
                    obj = key.ReadObj()
                    if obj.InheritsFrom("TH1"):
                        _, counts, _ = hist_arrays(obj)
                        (
                            data.setdefault(category, {})
                            .setdefault(collection, {})
                            .setdefault("_counts", {})
                            .setdefault("reason", {})[wp]
                        ) = (counts, bin_labels(obj))
                    continue
                if key.GetName() in RESOLUTION_SOURCES:
                    obj = key.ReadObj()
                    if obj.InheritsFrom("TH2"):
                        proj = obj.ProjectionX()
                        (
                            data.setdefault(category, {})
                            .setdefault(collection, {})
                            .setdefault("_slices", {})
                            .setdefault(key.GetName(), {})[wp]
                        ) = (
                            np.array([proj.GetBinContent(i) for i in range(1, proj.GetNbinsX() + 1)]),
                            0.5 * (obj.GetYaxis().GetXmax() - obj.GetYaxis().GetXmin()),
                            (obj.GetYaxis().GetXmax() - obj.GetYaxis().GetXmin()) / obj.GetNbinsY(),
                        )
                        (
                            data.setdefault(category, {})
                            .setdefault(collection, {})
                            .setdefault("_residual", {})
                            .setdefault(key.GetName(), {})[wp]
                        ) = hist_arrays(obj.ProjectionY())
                    continue
                res = _RES_RE.match(key.GetName())
                if res:
                    obj = key.ReadObj()
                    if obj.InheritsFrom("TH1") and obj.GetEntries() > 0:
                        (
                            data.setdefault(category, {})
                            .setdefault(collection, {})
                            .setdefault("resolution", {})
                            .setdefault(key.GetName(), {})[wp]
                        ) = hist_arrays(obj)
                    continue
                match = _ME_RE.match(key.GetName())
                if not match:
                    continue
                obj = key.ReadObj()
                if not obj.InheritsFrom("TH1") or obj.GetEntries() == 0:
                    continue
                metric, var = match.group("metric"), match.group("var")
                if var in CATEGORICAL:
                    (
                        data.setdefault(category, {})
                        .setdefault(collection, {})
                        .setdefault(metric, {})
                        .setdefault(var, {})[wp]
                    ) = hist_arrays(obj) + (bin_labels(obj),)
                    continue
                (
                    data.setdefault(category, {})
                    .setdefault(collection, {})
                    .setdefault(metric, {})
                    .setdefault(var, {})[wp]
                ) = hist_arrays(obj)
        tfile.Close()
    return data


def _fit_ok(wp, values, slices, is_sigma=False, errors=None):
    """Mask of slices whose Gaussian fit can be believed.

    Three ways a slice fit is worthless: too few entries to constrain it, a width wider
    than the histogram it was fitted in, and a width narrower than one bin. The first
    two are runaway fits, the third is a fit that collapsed onto a single bin.
    """
    ok = np.ones(len(values), dtype=bool)
    if slices is None or wp not in slices:
        return ok
    counts, half_range, bin_width = slices[wp]
    if len(counts) == len(values):
        ok &= counts >= MIN_SLICE_ENTRIES
    ok &= np.abs(values) <= half_range
    if is_sigma:
        ok &= values > bin_width
    if errors is not None:
        # An error comparable to the value means the fit did not converge on anything.
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(values != 0, np.abs(errors / values), np.inf)
        ok &= rel <= 0.5
    return ok


def plot_metric(category, collection, metric, var, per_wp, outdir, index, slices=None, denom=None):
    """One overlay plot with a ratio panel against the reference working point."""
    wps = [w for w in WP_ORDER if w in per_wp] + [w for w in sorted(per_wp) if w not in WP_ORDER]
    if not wps:
        return None

    is_sigma = var.endswith("_Sigma")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Working points can lie on top of one another (the adaptive points differ by ~0.002
    # here), so vary marker AND linestyle: colour alone hides a curve completely.
    markers = ["o", "s", "^", "D", "v"]
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(10, 9), sharex=True, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.07)
    )
    fig.subplots_adjust(top=0.88, bottom=0.16)

    means = {}
    for i, wp in enumerate(wps):
        edges, values, errors = per_wp[wp]
        centers = 0.5 * (edges[:-1] + edges[1:])
        filled = values > 0
        filled = filled & _fit_ok(wp, values, slices, is_sigma, errors if metric == "resolution" else None)
        if denom is not None and wp in denom and len(denom[wp]) == len(values):
            filled = filled & (denom[wp] >= MIN_DENOM_ENTRIES)
        means[wp] = float(values[filled].mean()) if filled.any() else 0.0
        ax.errorbar(
            centers[filled],
            values[filled],
            yerr=errors[filled],
            fmt=markers[i % len(markers)],
            linestyle=styles[i % len(styles)],
            markersize=5,
            markerfacecolor="none" if i else None,
            linewidth=1.4,
            alpha=0.85,
            color=colors[i % len(colors)],
            label=wp,
        )

    label, meaning, formula = METRICS[metric]
    if metric == "resolution":
        # Residuals are not bounded to [0, 1]; a fixed range would push every point off
        # the axis. Scale to the data, keeping zero visible so a bias is readable.
        def _shown(w):
            v = per_wp[w][1]
            return v[(v != 0) & _fit_ok(w, v, slices, is_sigma)]

        allv = np.concatenate([_shown(w) for w in wps if _shown(w).size] or [np.zeros(1)])
        span = float(np.abs(allv).max()) if allv.size else 1.0
        ax.set_ylim(min(0.0, float(allv.min()) * 1.3 if allv.size else 0.0), span * 1.35 if span else 1.0)
        label = "Mean" if var.endswith("_Mean") else "Sigma"
    # The plot title stays generic. A bin-averaged summary in the title reads as a
    # conclusion the plot has not earned, so the measured numbers go in the README
    # caption instead, where they can be qualified.
    title = f"{label} vs {var}" if metric != "resolution" else var.replace("_", " ")
    ref = means.get(REFERENCE_WP)
    others = [means[w] for w in wps if w != REFERENCE_WP]
    if ref and others:
        adaptive = sum(others) / len(others)
        delta = (adaptive - ref) / ref * 100.0 if ref else 0.0
        caption = (f"{title}. Bin-averaged over filled bins: adaptive {adaptive:.2f}, "
                   f"fixed {ref:.2f} ({delta:+.0f}%).")
    else:
        caption = title

    fig.suptitle(title, fontsize=16, y=0.965)
    ax.set_ylabel(label)
    if metric != "resolution":
        ax.set_ylim(0.0, 1.15)
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=13, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.02))
    ax.tick_params(labelbottom=False)
    hep.cms.label(ax=ax, llabel="Private Work", rlabel="Phase-2 Simulation", fontsize=15)

    # Ratio panel: only where the reference has a value, so an empty reference bin does
    # not manufacture a spike.
    if REFERENCE_WP in per_wp:
        ref_edges, ref_values, _ = per_wp[REFERENCE_WP]
        ref_centers = 0.5 * (ref_edges[:-1] + ref_edges[1:])
        for i, wp in enumerate(wps):
            if wp == REFERENCE_WP:
                continue
            _, values, _ = per_wp[wp]
            ok = (ref_values > 0) & (values > 0)
            ok = ok & _fit_ok(REFERENCE_WP, ref_values, slices, is_sigma) & _fit_ok(wp, values, slices, is_sigma)
            if denom is not None:
                for w in (REFERENCE_WP, wp):
                    if w in denom and len(denom[w]) == len(values):
                        ok = ok & (denom[w] >= MIN_DENOM_ENTRIES)
            if ok.any():
                rax.plot(
                    ref_centers[ok],
                    values[ok] / ref_values[ok],
                    "o",
                    markersize=4,
                    color=colors[i % len(colors)],
                )
    rax.axhline(1.0, linestyle="--", color="gray", linewidth=1.2)
    rax.set_ylabel(f"ratio to {REFERENCE_WP}", fontsize=12)
    xvar = var.rsplit("_vs_", 1)[-1].split("_")[0] if "_vs_" in var else var
    rax.set_xlabel(xvar)
    rax.set_ylim(0.0, 2.0)
    rax.grid(alpha=0.3)

    name = f"{index:02d}_{category}_{collection}_{metric}_vs_{var}.png"
    fig.savefig(os.path.join(outdir, metric, name), dpi=150)
    plt.close(fig)
    return name, caption


def plot_categorical(category, collection, metric, var, per_wp, counts, outdir, index):
    """Grouped horizontal bars, one group per named category, one bar per working point.

    Categories the sample does not populate are dropped rather than drawn at zero: a
    process that never happened is not an inefficiency.
    """
    wps = [w for w in WP_ORDER if w in per_wp] + [w for w in sorted(per_wp) if w not in WP_ORDER]
    if not wps:
        return None
    labels = next((per_wp[w][3] for w in wps if per_wp[w][3]), None)
    if labels is None:
        return None

    population = None
    if counts:
        population = counts.get(REFERENCE_WP, next(iter(counts.values())))[0]
    keep = [i for i in range(len(labels))
            if (population is None and any(per_wp[w][1][i] > 0 for w in wps)) or
               (population is not None and population[i] > 0)]
    if not keep:
        return None
    # Most populated process first, so the rows the reader should trust come first and
    # a one-entry category cannot sit between two well-populated ones.
    if population is not None:
        keep.sort(key=lambda k: population[k], reverse=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(keep) + 3.6))
    # Margins fixed in inches, not in figure fractions: the figure height grows with the
    # number of categories, so a fractional bottom margin would shrink the label space.
    height = fig.get_figheight()
    fig.subplots_adjust(left=0.30, right=0.97, top=1 - 0.9 / height, bottom=1.6 / height)

    y = np.arange(len(keep))
    barh = 0.8 / len(wps)
    for i, wp in enumerate(wps):
        _, values, errors, _ = per_wp[wp]
        offset = (i - (len(wps) - 1) / 2.0) * barh
        ax.barh(y + offset, [values[k] for k in keep], height=barh * 0.92,
                xerr=[errors[k] for k in keep], color=colors[i % len(colors)],
                edgecolor="none", alpha=0.9, label=wp, error_kw=dict(lw=1, ecolor="0.3"))

    label, meaning, formula = METRICS[metric]
    title = f"{label} vs {var}"
    # The category population is what makes a bar readable, so it is written next to
    # the label instead of being left to the reader to guess from the error bar.
    ticks = []
    for k in keep:
        if population is not None:
            ticks.append(f"{labels[k]}  (N={int(population[k])})")
        else:
            ticks.append(labels[k])
    ax.set_yticks(y)
    ax.set_yticklabels(ticks, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel(label)
    ax.grid(axis="x", alpha=0.3)
    fig.suptitle(title, fontsize=16, y=0.965)
    # Below everything: a legend inside the axes covers the least populated rows, which
    # are still real measurements, so it goes under the x label in the reserved margin.
    handles, lbls = ax.get_legend_handles_labels()
    fig.legend(handles, lbls, fontsize=13, loc="lower center", ncol=len(lbls), frameon=False,
               bbox_to_anchor=(0.5, 0.15 / height))
    hep.cms.label(ax=ax, llabel="Private Work", rlabel="Phase-2 Simulation", fontsize=15)

    top = max(keep, key=lambda k: population[k]) if population is not None else keep[0]
    caption = (f"{title}. Most populated process: {labels[top]} "
               f"(N={int(population[top])}) at {per_wp[wps[0]][1][top]:.2f} for {wps[0]}."
               if population is not None else title)

    name = f"{index:02d}_{category}_{collection}_{metric}_vs_{var}.png"
    fig.savefig(os.path.join(outdir, metric, name), dpi=150)
    plt.close(fig)
    return name, caption


def plot_residual(category, collection, source, per_wp, outdir, index):
    """The residual distribution itself, overlaid across working points.

    The Gaussian slice fit summarises this distribution; when the distribution is not
    Gaussian the fit says nothing and only the distribution does.
    """
    wps = [w for w in WP_ORDER if w in per_wp] + [w for w in sorted(per_wp) if w not in WP_ORDER]
    if not wps:
        return None
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = ["-", "--", "-.", ":"]
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(top=0.88, bottom=0.16)

    cores = {}
    for i, wp in enumerate(wps):
        edges, values, _ = per_wp[wp]
        total = values.sum()
        if total <= 0:
            continue
        centers = 0.5 * (edges[:-1] + edges[1:])
        # Fraction inside +-10%, a scale-free statement about how peaked it is that does
        # not depend on a fit converging.
        cores[wp] = float(values[np.abs(centers) <= 0.1].sum() / total)
        hep.histplot(values / total, edges, ax=ax, label=wp, yerr=False,
                     color=colors[i % len(colors)], linestyle=styles[i % len(styles)], linewidth=1.6)

    ax.set_yscale("log")
    ax.set_xlabel("(reco - truth) / truth" if source.startswith("ptres") else "reco - truth")
    ax.set_ylabel("fraction of matched pairs per bin")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13, frameon=False)
    fig.suptitle(f"{source} residual distribution", fontsize=16, y=0.965)
    hep.cms.label(ax=ax, llabel="Private Work", rlabel="Phase-2 Simulation", fontsize=15)

    ref = cores.get(REFERENCE_WP)
    caption = (f"{source} residual distribution, area normalised. Fraction within 10%: "
               + ", ".join(f"{w} {cores[w]:.2f}" for w in wps if w in cores) + ".") if cores else source
    name = f"{index:02d}_{category}_{collection}_{source}_distribution.png"
    fig.savefig(os.path.join(outdir, "resolution", name), dpi=150)
    plt.close(fig)
    return name, caption


def plot_composition(category, collection, counts, outdir, index):
    """What the selected truth branches ARE, by the process that created them."""
    entry = counts.get(REFERENCE_WP) or next(iter(counts.values()))
    values, labels = entry
    if labels is None or values.sum() <= 0:
        return None
    keep = [i for i in range(len(labels)) if values[i] > 0]
    order = sorted(keep, key=lambda k: values[k], reverse=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(order) + 3.4))
    fig.subplots_adjust(left=0.30, right=0.97, top=1 - 0.9 / fig.get_figheight(),
                        bottom=0.9 / fig.get_figheight())
    frac = np.array([values[k] for k in order]) / values.sum()
    ax.barh(np.arange(len(order)), frac, color=colors[0], alpha=0.9)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([f"{labels[k]}  (N={int(values[k])})" for k in order], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of selected truth branches")
    ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Selected truth branches by creation process", fontsize=16, y=0.965)
    hep.cms.label(ax=ax, llabel="Private Work", rlabel="Phase-2 Simulation", fontsize=15)

    caption = ("Composition of the truth-branch denominator by the Geant4 process that created each branch root. "
               f"Leading process {labels[order[0]]} at {frac[0]*100:.0f}% of {int(values.sum())} branches.")
    name = f"{index:02d}_{category}_{collection}_composition_by_reason.png"
    fig.savefig(os.path.join(outdir, "composition", name), dpi=150)
    plt.close(fig)
    return name, caption


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--outputDir", default="plots")
    ap.add_argument("--sample", default="ttbar, no pileup, D122")
    ap.add_argument("--title", default="MC-truth graph validation")
    args = ap.parse_args()

    os.makedirs(args.outputDir, exist_ok=True)
    # One real directory per metric, so the gallery browses as folders and not as one
    # flat list of sixty files.
    for metric in METRIC_ORDER:
        os.makedirs(os.path.join(args.outputDir, metric), exist_ok=True)
    data = collect(args.files)
    if not data:
        print("no populated monitor elements found", file=sys.stderr)
        return 1

    written = []
    index = 1
    for category in sorted(data):
        for collection in sorted(data[category]):
            all_counts = data[category][collection].get("_counts", {})
            all_slices = data[category][collection].get("_slices", {})
            all_denom = data[category][collection].get("_denom", {})
            all_residual = data[category][collection].get("_residual", {})
            if "reason" in all_counts:
                result = plot_composition(category, collection, all_counts["reason"], args.outputDir, index)
                if result:
                    name, caption = result
                    written.append({"category": category, "collection": collection, "metric": "composition",
                                    "var": "reason", "png": name, "caption": caption})
                    index += 1
            for metric in METRIC_ORDER:
                per_metric = data[category][collection].get(metric, {})
                for var in VARIABLE_ORDER:
                    if var not in per_metric:
                        continue
                    result = plot_metric(
                        category, collection, metric, var, per_metric[var], args.outputDir, index,
                        denom=all_denom.get(f"{DENOMINATOR.get(metric, '')}_{var}"),
                    )
                    if result:
                        name, caption = result
                        written.append({"category": category, "collection": collection, "metric": metric,
                                        "var": var, "png": name, "caption": caption})
                        index += 1
                if metric == "resolution":
                    for source in RESOLUTION_SOURCES:
                        if source not in all_residual:
                            continue
                        result = plot_residual(
                            category, collection, source, all_residual[source], args.outputDir, index
                        )
                        if result:
                            name, caption = result
                            written.append({"category": category, "collection": collection, "metric": metric,
                                            "var": source, "png": name, "caption": caption})
                            index += 1
                    for var in RESOLUTION_ORDER:
                        if var not in per_metric:
                            continue
                        base = var.rsplit("_", 1)[0]
                        result = plot_metric(
                            category, collection, metric, var, per_metric[var], args.outputDir, index,
                            slices=all_slices.get(base),
                        )
                        if result:
                            name, caption = result
                            written.append({"category": category, "collection": collection, "metric": metric,
                                            "var": var, "png": name, "caption": caption})
                            index += 1
                    continue
                for var in CATEGORICAL:
                    if var not in per_metric:
                        continue
                    result = plot_categorical(
                        category, collection, metric, var, per_metric[var],
                        all_counts.get(var), args.outputDir, index
                    )
                    if result:
                        name, caption = result
                        written.append({"category": category, "collection": collection, "metric": metric,
                                        "var": var, "png": name, "caption": caption})
                        index += 1

    by_metric = {}
    for entry in written:
        by_metric.setdefault(entry["metric"], []).append(entry)

    # One page per metric, each opening with the definition of the quantity it shows.
    for metric, entries in by_metric.items():
        label, meaning, formula = METRICS[metric]
        with open(os.path.join(args.outputDir, f"{metric}.html"), "w") as page:
            page.write(f"<!doctype html><meta charset='utf-8'><title>{label}</title><style>{STYLE}</style>")
            page.write(f"<h1>{label}</h1><p><a href='index.html'>back to index</a></p>")
            page.write(f"<div class='def'><b>Definition.</b> {meaning}<br><br>"
                       f"<span class='f'>{formula}</span><br><br>"
                       "Each plot overlays the four branch-association working points. <b>Fixed</b> keeps every "
                       "matching branch; the <b>Adaptive</b> points keep the single graph level that best matches "
                       "the reco object, differing only in how much branch spread they tolerate. The lower panel "
                       "is the ratio to Fixed.</div>")
            page.write(f"<p>Sample: {args.sample}.</p>")
            used = [v for v in VARIABLE_ORDER + CATEGORICAL
                    if any(e["var"] == v or e["var"].endswith("_vs_" + v) for e in entries)]
            if used:
                page.write("<div class='def'><b>What is on the x axis.</b><ul>")
                for v in used:
                    meaning = VARIABLE_MEANING.get(v) or CATEGORICAL_MEANING.get(v, v)
                    page.write(f"<li><span class='f'>{v}</span> {meaning}</li>")
                page.write("</ul></div>")
            for category in sorted({e["category"] for e in entries}):
                if category in CATEGORY_NOTE:
                    page.write(f"<div class='def'><b>{category}.</b> {CATEGORY_NOTE[category]}</div>")
            for collection in sorted({e["collection"] for e in entries}):
                page.write(f"<h2>{collection}</h2><div class='grid'>")
                for e in [x for x in entries if x["collection"] == collection]:
                    href = f"{metric}/{e['png']}"
                    page.write(f"<a href='{href}'><img src='{href}' width='400'></a>")
                page.write("</div>")

    # Each folder carries its own definitions, so a reader who lands in the folder
    # without going through the index still knows what the plots in it mean.
    for metric, entries in by_metric.items():
        label, meaning, formula = METRICS[metric]
        with open(os.path.join(args.outputDir, metric, "DEFINITIONS.md"), "w") as defs:
            defs.write(f"# {label}\n\nSample: {args.sample}.\n\n## Definition\n\n{meaning}\n\n")
            defs.write(f"    {formula}\n\n## Working points\n\n")
            defs.write("Each plot overlays the four branch-association working points. Fixed keeps every matching\n"
                       "branch; the Adaptive points keep the single graph level that best matches the reco object,\n"
                       "differing only in how much branch spread they tolerate. The lower panel is the ratio to\n"
                       "Fixed.\n\n## Variables on the x axis\n\n")
            used = [v for v in VARIABLE_ORDER + CATEGORICAL
                    if any(e["var"] == v or e["var"].endswith("_vs_" + v) for e in entries)]
            for v in used:
                defs.write(f"- `{v}`: {VARIABLE_MEANING.get(v) or CATEGORICAL_MEANING.get(v, v)}\n")
            defs.write("\n## Quality cuts\n\n"
                       f"- A ratio bin is drawn only if its denominator has at least {MIN_DENOM_ENTRIES} entries.\n"
                       f"- A Gaussian slice fit is drawn only if its slice has at least {MIN_SLICE_ENTRIES} entries\n"
                       "  and the fitted width is inside the fit range and wider than one bin.\n"
                       "- A named category is drawn only if the sample populated it.\n")
            _cats = sorted({e["category"] for e in entries if e["category"] in CATEGORY_NOTE})
            if _cats:
                defs.write("\n## Caveats by domain\n\n")
                for _c in _cats:
                    defs.write(f"- **{_c}.** {CATEGORY_NOTE[_c]}\n")
            defs.write("\n## Plots in this folder\n\n")
            for e in entries:
                defs.write(f"- `{e['png']}`: {e['caption']}\n")

    with open(os.path.join(args.outputDir, "index.html"), "w") as idx:
        idx.write(f"<!doctype html><meta charset='utf-8'><title>{args.title}</title><style>{STYLE}</style>")
        idx.write(f"<h1>{args.title}</h1><p>Sample: {args.sample}.</p>")
        if os.path.exists(os.path.join(args.outputDir, "resource_cost.md")):
            idx.write("<p><a href='resource_cost.md'>Measured computing cost of the graph against the legacy "
                      "frozen truth objects</a></p>")
        idx.write("<div class='def'>Truth objects here are <b>branches</b> of the MC-truth graph, not frozen "
                  "TrackingParticles or CaloParticles. A branch is recomputed on demand from the graph, so the "
                  "same validation can be re-run against a different definition of what counts as one truth "
                  "object. Only branches passing the branch selector enter the denominators.</div>")
        idx.write("<h2>Metrics</h2><ul class='idx'>")
        for metric in METRIC_ORDER:
            if metric not in by_metric:
                continue
            label, meaning, formula = METRICS[metric]
            idx.write(f"<li><a href='{metric}.html'><b>{label}</b></a> <span class='f'>{formula}</span>"
                      f"<br>{meaning}</li>")
        idx.write("</ul><h2>Variables</h2><ul class='idx'>")
        for var in VARIABLE_ORDER:
            idx.write(f"<li><span class='f'>{var}</span> {VARIABLE_MEANING[var]}</li>")
        for var in CATEGORICAL:
            idx.write(f"<li><span class='f'>{var}</span> {CATEGORICAL_MEANING[var]}</li>")
        idx.write("</ul><h2>Quality cuts applied to every plot</h2><ul class='idx'>"
                  f"<li>A ratio bin is drawn only if its denominator has at least {MIN_DENOM_ENTRIES} entries. "
                  "A ratio built from a handful of entries is noise with a large error bar, not a measurement.</li>"
                  f"<li>A Gaussian slice fit is drawn only if its slice has at least {MIN_SLICE_ENTRIES} entries, "
                  "and only if the fitted width is inside the fit range and wider than one bin. The other two cases "
                  "are a fit that ran away and a fit that collapsed onto a single bin.</li>"
                  "<li>A named category is drawn only if the sample populated it. A process that never happened is "
                  "not an inefficiency.</li></ul>"
                  "<h2>Proposed plots, not yet implemented</h2>"
                  "<p>What the graph makes possible that a frozen truth object cannot answer.</p><ul class='idx'>")
        for title, why in PROPOSED:
            idx.write(f"<li><b>{title}.</b> {why}</li>")
        idx.write("</ul>")

    with open(os.path.join(args.outputDir, "README.md"), "w") as readme:
        readme.write(f"# {args.title}\n\nSample: {args.sample}.\n")
        for metric in METRIC_ORDER:
            entries = by_metric.get(metric, [])
            if not entries:
                continue
            label, meaning, formula = METRICS[metric]
            readme.write(f"\n## {label}\n\n{meaning}\n\n    {formula}\n\n")
            readme.write(f"Folder `{metric}/`, definitions in `{metric}/DEFINITIONS.md`.\n\n")
            for e in entries:
                readme.write(f"- `{metric}/{e['png']}`: {e['caption']}\n")
        readme.write("\n## Proposed plots, not yet implemented\n\n")
        for title, why in PROPOSED:
            readme.write(f"- **{title}.** {why}\n")

    with open(os.path.join(args.outputDir, "truth-validation.orbit"), "w") as orbit:
        json.dump({"target": "truth-validation", "title": args.title,
                   "description": f"Truth-branch association metrics ({args.sample})",
                   "icon": "chart", "access": "public"}, orbit, indent=2)

    print(f"wrote {len(written)} plots in {len(by_metric)} metric pages to {args.outputDir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
