#!/usr/bin/env python3
"""Render the truth-branch DQM output into a browsable gallery.

Collections and working points are DISCOVERED from the DQM folder names rather than
listed here, the way makeHGCalValidationPlots.py discovers TICL iterations, so a new
collection or working point needs no edit to this script.

  makeTruthValidationPlots.py DQM_V0001_*.root --outputDir plots

Writes one PNG per monitor element, one page per category, and a top-level index.
"""
import argparse
import os
import re
import sys

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

def discover(tfile):
    """Yield (category, folder, TDirectory) for every directory that directly holds
    histograms. Walking for content rather than matching a fixed DQMData/Run N/... path
    keeps this working when the save convention changes; category and folder are simply
    the last two path components."""

    def walk(directory, path):
        holdsHistograms = False
        for key in directory.GetListOfKeys():
            obj = key.ReadObj()
            if obj.InheritsFrom("TDirectory"):
                yield from walk(obj, path + [key.GetName()])
            elif obj.InheritsFrom("TH1") and not holdsHistograms:
                holdsHistograms = True
        if holdsHistograms and len(path) >= 2:
            yield path[-2], path[-1], directory

    yield from walk(tfile, [])


def render(hist, path, title):
    canvas = ROOT.TCanvas("c", "c", 700, 520)
    canvas.SetGrid()
    hist.SetTitle(title)
    hist.SetLineWidth(2)
    hist.SetMarkerStyle(20)
    hist.SetMarkerSize(0.7)
    # Efficiencies and rates are bounded, so pin the axis: an autoscaled 0.98 to 1.0
    # window makes a flat efficiency look like structure.
    if any(hist.GetName().startswith(p) for p in ("efficiency", "fakerate", "duplicate")):
        hist.SetMinimum(0.0)
        hist.SetMaximum(1.05)
        hist.Draw("E1")
    else:
        hist.Draw("HIST")
    canvas.SaveAs(path)
    canvas.Close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="Harvested DQM file(s)")
    ap.add_argument("--outputDir", default="plots")
    ap.add_argument("--title", default="MC-truth graph validation")
    args = ap.parse_args()

    os.makedirs(args.outputDir, exist_ok=True)
    pages = {}

    for fname in args.files:
        tfile = ROOT.TFile.Open(fname)
        if not tfile or tfile.IsZombie():
            print(f"cannot open {fname}", file=sys.stderr)
            continue
        for category, folder, folderDir in discover(tfile):
            images = []
            for key in folderDir.GetListOfKeys():
                obj = key.ReadObj()
                if not obj.InheritsFrom("TH1") or obj.InheritsFrom("TH2"):
                    continue
                if obj.GetEntries() == 0:
                    continue
                name = key.GetName()
                png = f"{category}_{folder}_{name}.png".replace("(", "").replace(")", "")
                render(obj, os.path.join(args.outputDir, png), f"{folder}: {name}")
                images.append((name, png))
            if images:
                pages.setdefault(category, {})[folder] = sorted(images)
        tfile.Close()

    if not pages:
        print("no populated monitor elements found", file=sys.stderr)
        return 1

    # One page per category, plus an index. Plain static HTML so it can be served from
    # any web area without a backend.
    style = (
        "body{font-family:sans-serif;margin:2em;max-width:1400px}"
        "h2{border-bottom:1px solid #ccc;padding-bottom:.2em;margin-top:1.6em}"
        "img{border:1px solid #ddd;margin:4px;vertical-align:top}"
        "a{color:#036;text-decoration:none}a:hover{text-decoration:underline}"
        ".grid{display:flex;flex-wrap:wrap}"
    )
    for category, folders in pages.items():
        with open(os.path.join(args.outputDir, f"{category}.html"), "w") as page:
            page.write(f"<!doctype html><meta charset='utf-8'><title>{category}</title><style>{style}</style>")
            page.write(f"<h1>{args.title}: {category}</h1><p><a href='index.html'>back to index</a></p>")
            for folder, images in sorted(folders.items()):
                page.write(f"<h2>{folder}</h2><div class='grid'>")
                for _, png in images:
                    page.write(f"<a href='{png}'><img src='{png}' width='380'></a>")
                page.write("</div>")

    with open(os.path.join(args.outputDir, "index.html"), "w") as index:
        index.write(f"<!doctype html><meta charset='utf-8'><title>{args.title}</title><style>{style}</style>")
        index.write(f"<h1>{args.title}</h1>")
        for category, folders in sorted(pages.items()):
            nplots = sum(len(v) for v in folders.values())
            index.write(f"<h2><a href='{category}.html'>{category}</a></h2>")
            index.write(f"<p>{len(folders)} working points / collections, {nplots} plots</p><ul>")
            for folder in sorted(folders):
                index.write(f"<li>{folder}</li>")
            index.write("</ul>")

    total = sum(len(v) for f in pages.values() for v in f.values())
    print(f"wrote {total} plots in {len(pages)} categories to {args.outputDir}/index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
