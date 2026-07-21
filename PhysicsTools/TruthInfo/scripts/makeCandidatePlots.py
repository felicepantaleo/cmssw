#!/usr/bin/env python3
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

# Self-contained PyROOT plotter for the TICLCandidate truth-branch validation. It
# reads one harvested legacy DQM file per species (output of
# harvestBranchCandidateDQM_cfg.py) and draws, overlaying species where useful:
#   * the reconstruction efficiency ladder (match -> +charge -> +PID -> +energy) per
#     species, vs pt / eta / energy - answers "where does the candidate lack most, and
#     at which step of the chain";
#   * efficiency / fake-rate / merge-rate / split overlaid across species;
#   * track<->calo linking consistency for the charged species;
#   * energy response distributions and response profiles;
#   * the per-candidate outcome breakdown (matched / fake / merged / track-calo mismatch).
# Writes PNGs + an index.html. Usage:
#   makeCandidatePlots.py photon.root:photon electron.root:electron ... -o outdir

import os
import sys
from argparse import ArgumentParser

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(1)

FOLDER = "DQMData/Run 1/HGCAL/Run summary/BranchValidator/TICLCandidate"
COLORS = [ROOT.kBlack, ROOT.kRed + 1, ROOT.kAzure + 1, ROOT.kGreen + 2,
          ROOT.kMagenta + 1, ROOT.kOrange + 7, ROOT.kCyan + 2, ROOT.kGray + 2]
# Charged species (track present) for the consistency plots; matched by label substring.
CHARGED_HINTS = ("electron", "muon", "pion", "charged", "proton", "kaon_ch")


BASEDIR = "DQMData/Run 1/HGCAL/Run summary/BranchValidator"


def get(f, name, folder=FOLDER):
    h = f.Get(folder + "/" + name)
    if not h:
        # Fallback: some harvesters drop the "Run summary" segment.
        h = f.Get(folder.replace("/Run summary", "") + "/" + name)
    return h if h else None


def scCompare(f, label, name, title, outpng):
    # Overlay the same metric from the CLUE3D-trackster folder (fragmented input) and
    # the superclustered-trackster folder (EM supercluster output) for one species.
    c = ROOT.TCanvas("c", "c", 800, 600)
    leg = ROOT.TLegend(0.58, 0.74, 0.89, 0.89)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    keep = []
    drawn = 0
    for folder, lab, col in [(BASEDIR + "/TracksterCLUE3D", "CLUE3D tracksters", ROOT.kAzure + 1),
                             (BASEDIR + "/TracksterSupercls", "superclustered", ROOT.kRed + 1)]:
        h = get(f, name, folder)
        if not h:
            continue
        h = h.Clone("%s_%s_%d" % (label, name, drawn))
        h.SetDirectory(0)
        style(h, col, "%s: %s" % (label, title))
        h.SetMaximum(1.05)
        h.SetMinimum(0.0)
        h.Draw("E1" if drawn == 0 else "E1 SAME")
        leg.AddEntry(h, lab, "lp")
        keep.append(h)
        drawn += 1
    if drawn:
        leg.Draw()
        c.SaveAs(outpng)
    c.Close()
    return outpng if drawn else None


def style(h, color, title=None):
    h.SetLineColor(color)
    h.SetMarkerColor(color)
    h.SetLineWidth(2)
    h.SetMarkerStyle(20)
    h.SetMarkerSize(0.7)
    if title:
        h.SetTitle(title)
    return h


def overlay(files_labels, meName, outpng, title, ymax=None, ymin=None, norm=False):
    c = ROOT.TCanvas("c", "c", 800, 600)
    leg = ROOT.TLegend(0.60, 0.72, 0.89, 0.89)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    drawn = 0
    keep = []
    for i, (f, label) in enumerate(files_labels):
        h = get(f, meName)
        if not h:
            continue
        h = h.Clone("%s_%d" % (meName, i))
        h.SetDirectory(0)
        if norm and h.Integral() > 0:
            h.Scale(1.0 / h.Integral())
        style(h, COLORS[i % len(COLORS)], title)
        if ymax is not None:
            h.SetMaximum(ymax)
        if ymin is not None:
            h.SetMinimum(ymin)
        h.Draw("E1" if drawn == 0 else "E1 SAME")
        leg.AddEntry(h, label, "lp")
        keep.append(h)
        drawn += 1
    if drawn:
        leg.Draw()
        c.SaveAs(outpng)
    c.Close()
    return outpng if drawn else None


def ladder(f, label, axis, outpng):
    steps = [("efficiency_%s" % axis, "match", ROOT.kBlack),
             ("charge_efficiency_%s" % axis, "+ charge", ROOT.kAzure + 1),
             ("pid_efficiency_%s" % axis, "+ PID", ROOT.kGreen + 2),
             ("energy_efficiency_%s" % axis, "+ energy", ROOT.kRed + 1)]
    c = ROOT.TCanvas("c", "c", 800, 600)
    leg = ROOT.TLegend(0.60, 0.72, 0.89, 0.89)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    drawn = 0
    keep = []
    for name, lab, col in steps:
        h = get(f, name)
        if not h:
            continue
        h = h.Clone("ladder_%s_%s" % (label, name))
        h.SetDirectory(0)
        style(h, col, "%s reconstruction ladder vs %s;;efficiency / correct fraction" % (label, axis))
        h.SetMaximum(1.05)
        h.SetMinimum(0.0)
        h.Draw("E1" if drawn == 0 else "E1 SAME")
        leg.AddEntry(h, lab, "lp")
        keep.append(h)
        drawn += 1
    if drawn:
        leg.Draw()
        c.SaveAs(outpng)
    c.Close()
    return outpng if drawn else None


def outcome(files_labels, outpng):
    c = ROOT.TCanvas("c", "c", 900, 600)
    leg = ROOT.TLegend(0.60, 0.72, 0.89, 0.89)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    drawn = 0
    keep = []
    for i, (f, label) in enumerate(files_labels):
        h = get(f, "cand_outcome")
        if not h:
            continue
        h = h.Clone("outcome_%d" % i)
        h.SetDirectory(0)
        if h.Integral() > 0:
            h.Scale(1.0 / h.Integral())
        style(h, COLORS[i % len(COLORS)], "Candidate outcome (normalized);;fraction of candidates")
        h.SetMaximum(1.05)
        h.SetBarWidth(0.8)
        h.Draw("HIST" if drawn == 0 else "HIST SAME")
        leg.AddEntry(h, label, "l")
        keep.append(h)
        drawn += 1
    if drawn:
        leg.Draw()
        c.SaveAs(outpng)
    c.Close()
    return outpng if drawn else None


def regVsRaw(f, label, outpng):
    # Overlay regressed vs raw energy response for one species: shows what the energy
    # regression adds on top of the raw trackster energy.
    c = ROOT.TCanvas("c", "c", 800, 600)
    leg = ROOT.TLegend(0.60, 0.74, 0.89, 0.89)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    keep = []
    drawn = 0
    for name, lab, col in [("energy_response", "regressed", ROOT.kRed + 1),
                           ("energy_response_raw", "raw", ROOT.kAzure + 1)]:
        h = get(f, name)
        if not h:
            continue
        h = h.Clone("%s_%s" % (label, name))
        h.SetDirectory(0)
        if h.Integral() > 0:
            h.Scale(1.0 / h.Integral())
        style(h, col, "%s energy response: regressed vs raw;E_{reco}/E_{truth};a.u." % label)
        h.Draw("HIST" if drawn == 0 else "HIST SAME")
        leg.AddEntry(h, lab, "l")
        keep.append(h)
        drawn += 1
    if drawn:
        leg.Draw()
        c.SaveAs(outpng)
    c.Close()
    return outpng if drawn else None


def main():
    parser = ArgumentParser()
    parser.add_argument("inputs", nargs="+", metavar="FILE:LABEL")
    parser.add_argument("-o", "--out", default="candidate_plots")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files_labels = []
    for spec in args.inputs:
        path, _, label = spec.partition(":")
        if not label:
            label = os.path.splitext(os.path.basename(path))[0]
        f = ROOT.TFile.Open(path)
        if not f or f.IsZombie():
            print("WARNING: cannot open %s" % path, file=sys.stderr)
            continue
        files_labels.append((f, label))
    if not files_labels:
        print("No readable inputs.", file=sys.stderr)
        return 1

    made = []

    def record(png):
        if png:
            made.append(os.path.basename(png))

    # Per-species reconstruction ladder, per axis.
    for f, label in files_labels:
        for axis in ("pt", "eta", "energy"):
            png = os.path.join(args.out, "ladder_%s_%s.png" % (label, axis))
            record(ladder(f, label, axis, png))

    # Cross-species overlays.
    for axis in ("pt", "eta", "energy"):
        record(overlay(files_labels, "efficiency_%s" % axis,
                       os.path.join(args.out, "efficiency_%s.png" % axis),
                       "Candidate efficiency vs %s;;efficiency" % axis, ymax=1.05, ymin=0.0))
        record(overlay(files_labels, "pid_efficiency_%s" % axis,
                       os.path.join(args.out, "pid_efficiency_%s.png" % axis),
                       "Efficiency incl. correct PID vs %s;;efficiency" % axis, ymax=1.05, ymin=0.0))
        record(overlay(files_labels, "energy_efficiency_%s" % axis,
                       os.path.join(args.out, "energy_efficiency_%s.png" % axis),
                       "Efficiency incl. correct energy vs %s;;efficiency" % axis, ymax=1.05, ymin=0.0))
        record(overlay(files_labels, "fakerate_%s" % axis,
                       os.path.join(args.out, "fakerate_%s.png" % axis),
                       "Candidate fake rate vs %s;;fake rate" % axis, ymax=1.05, ymin=0.0))
        record(overlay(files_labels, "mergerate_%s" % axis,
                       os.path.join(args.out, "mergerate_%s.png" % axis),
                       "Candidate merge rate vs %s;;merge rate" % axis, ymax=1.05, ymin=0.0))
        record(overlay(files_labels, "duplicate_%s" % axis,
                       os.path.join(args.out, "duplicate_%s.png" % axis),
                       "Candidate split rate vs %s;;split rate" % axis, ymax=1.05, ymin=0.0))

    # Track-calo linking consistency (charged species only).
    charged = [(f, l) for (f, l) in files_labels if any(h in l.lower() for h in CHARGED_HINTS)]
    for axis in ("pt", "eta"):
        record(overlay(charged, "trackcalo_consistency_%s" % axis,
                       os.path.join(args.out, "consistency_%s.png" % axis),
                       "Track-calo same branch vs %s;;consistency" % axis, ymax=1.05, ymin=0.0))

    # Energy response and profiles.
    record(overlay(files_labels, "energy_response",
                   os.path.join(args.out, "energy_response.png"),
                   "Candidate energy response;E_{reco}/E_{truth};a.u.", norm=True))
    for axis in ("energy", "eta", "pt"):
        record(overlay(files_labels, "energy_response_vs_%s" % axis,
                       os.path.join(args.out, "response_vs_%s.png" % axis),
                       "Energy response vs %s;;E_{reco}/E_{truth}" % axis, ymax=1.6, ymin=0.0))
    record(overlay(files_labels, "purity_calo", os.path.join(args.out, "purity_calo.png"),
                   "Best-branch calo purity;purity;a.u.", norm=True))
    record(overlay(files_labels, "purity_track", os.path.join(args.out, "purity_track.png"),
                   "Best-branch track purity;purity;a.u.", norm=True))

    # Candidate outcome breakdown.
    record(outcome(files_labels, os.path.join(args.out, "cand_outcome.png")))

    # Fragmentation: how many candidates one truth particle is split into.
    record(overlay(files_labels, "n_candidates_per_branch",
                   os.path.join(args.out, "fragmentation.png"),
                   "Candidates per truth particle;N candidates;fraction of particles", norm=True))

    # Regressed vs raw energy response, per species and the raw overlay across species.
    for f, label in files_labels:
        record(regVsRaw(f, label, os.path.join(args.out, "regvsraw_%s.png" % label)))
    record(overlay(files_labels, "energy_response_raw",
                   os.path.join(args.out, "energy_response_raw.png"),
                   "Raw-energy response;E_{raw}/E_{truth};a.u.", norm=True))
    record(overlay(files_labels, "energy_response_raw_vs_energy",
                   os.path.join(args.out, "response_raw_vs_energy.png"),
                   "Raw-energy response vs E;E [GeV];E_{raw}/E_{truth}", ymax=1.6, ymin=0.0))

    # Superclustering step: CLUE3D (fragmented input) vs superclustered tracksters,
    # for the EM guns. Efficiency should stay ~1 while the duplicate (fragmentation)
    # rate drops if superclustering correctly merges the shower.
    em = [(f, l) for (f, l) in files_labels if "photon" in l.lower() or "electron" in l.lower()]
    for f, label in em:
        record(scCompare(f, label, "efficiency_energy",
                         "trackster efficiency vs E;E [GeV];efficiency",
                         os.path.join(args.out, "sc_efficiency_%s.png" % label)))
        record(scCompare(f, label, "duplicate_energy",
                         "trackster fragmentation (duplicate) vs E;E [GeV];duplicate rate",
                         os.path.join(args.out, "sc_fragmentation_%s.png" % label)))

    # index.html
    with open(os.path.join(args.out, "index.html"), "w") as html:
        html.write("<html><head><title>TICLCandidate vs truth branch</title></head><body>\n")
        html.write("<h1>TICLCandidate physics performance vs truth branch</h1>\n")
        html.write("<p>Single-particle guns, flat p<sub>T</sub> 2-200 GeV, 1.5&lt;|&eta;|&lt;3.0, D120 no PU. "
                   "Reco candidates matched to the fired GEN-primary truth branch on both the calo "
                   "(trackster) and tracker (track) channels.</p>\n")
        for png in made:
            html.write('<div style="display:inline-block;margin:4px">'
                       '<img src="%s" width="440"></div>\n' % png)
        html.write("</body></html>\n")

    print("Wrote %d plots to %s" % (len(made), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
