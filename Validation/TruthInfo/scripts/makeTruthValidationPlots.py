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
    "efficiency": ("Branch efficiency", "fraction of selected branches matched"),
    "fakerate": ("Fake rate", "fraction of reco objects with no branch"),
    "duplicate": ("Duplicate rate", "branches matched more than once"),
}
_ME_RE = re.compile(r"^(?P<metric>efficiency|fakerate|duplicate)_vs_(?P<var>\w+)$")


def hist_arrays(h):
    """Bin edges, contents and errors of a TH1 as numpy arrays."""
    n = h.GetNbinsX()
    edges = np.array([h.GetXaxis().GetBinLowEdge(i) for i in range(1, n + 2)])
    values = np.array([h.GetBinContent(i) for i in range(1, n + 1)])
    errors = np.array([h.GetBinError(i) for i in range(1, n + 1)])
    return edges, values, errors


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
                match = _ME_RE.match(key.GetName())
                if not match:
                    continue
                obj = key.ReadObj()
                if not obj.InheritsFrom("TH1") or obj.GetEntries() == 0:
                    continue
                metric, var = match.group("metric"), match.group("var")
                (
                    data.setdefault(category, {})
                    .setdefault(collection, {})
                    .setdefault(metric, {})
                    .setdefault(var, {})[wp]
                ) = hist_arrays(obj)
        tfile.Close()
    return data


def plot_metric(category, collection, metric, var, per_wp, outdir, index):
    """One overlay plot with a ratio panel against the reference working point."""
    wps = [w for w in WP_ORDER if w in per_wp] + [w for w in sorted(per_wp) if w not in WP_ORDER]
    if not wps:
        return None

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

    label, meaning = METRICS[metric]
    # The plot title stays generic. A bin-averaged summary in the title reads as a
    # conclusion the plot has not earned, so the measured numbers go in the README
    # caption instead, where they can be qualified.
    title = f"{label} vs {var}"
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
    rax.set_xlabel(var)
    rax.set_ylim(0.0, 2.0)
    rax.grid(alpha=0.3)

    name = f"{index:02d}_{category}_{collection}_{metric}_vs_{var}.png"
    fig.savefig(os.path.join(outdir, name), dpi=150)
    plt.close(fig)
    return name, caption


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--outputDir", default="plots")
    ap.add_argument("--sample", default="ttbar, no pileup, D122")
    args = ap.parse_args()

    os.makedirs(args.outputDir, exist_ok=True)
    data = collect(args.files)
    if not data:
        print("no populated monitor elements found", file=sys.stderr)
        return 1

    written = []
    index = 1
    for category in sorted(data):
        for collection in sorted(data[category]):
            for metric in ["efficiency", "fakerate", "duplicate"]:
                per_metric = data[category][collection].get(metric, {})
                for var in ["pt", "eta", "phi"]:
                    if var not in per_metric:
                        continue
                    result = plot_metric(
                        category, collection, metric, var, per_metric[var], args.outputDir, index
                    )
                    if result:
                        written.append((category, collection, *result))
                        index += 1

    # README in reading order, one line per plot.
    with open(os.path.join(args.outputDir, "README.md"), "w") as readme:
        readme.write(f"# MC-truth graph validation\n\nSample: {args.sample}.\n\n")
        readme.write(
            "Each plot overlays the branch-association working points, with a ratio panel\n"
            "against the Fixed reference. Fixed keeps every matching branch; the adaptive\n"
            "points keep the single graph level that best matches the reco object.\n\n"
        )
        for category, collection, name, headline in written:
            readme.write(f"- `{name}`: {headline}\n")

    with open(os.path.join(args.outputDir, "truth-validation.orbit"), "w") as orbit:
        json.dump(
            {
                "target": "truth-validation",
                "title": "MC-truth graph validation",
                "description": f"Truth-branch association efficiency, fake and duplicate rates ({args.sample})",
                "icon": "chart",
                "access": "public",
            },
            orbit,
            indent=2,
        )

    print(f"wrote {len(written)} plots to {args.outputDir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
