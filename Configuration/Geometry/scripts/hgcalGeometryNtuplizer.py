#!/usr/bin/env python3

import argparse
import sys
from array import array

import ROOT
from DataFormats.FWLite import Events, Handle


def parse_triplet(text: str):
    parts = text.split(":")
    if len(parts) == 1:
        return (parts[0], "", "")
    if len(parts) == 2:
        return (parts[0], parts[1], "")
    if len(parts) == 3:
        return (parts[0], parts[1], parts[2])
    raise ValueError(f"Invalid EDM tag format: {text}")


def get_by_label(event, type_name: str, tag_text: str):
    handle = Handle(type_name)
    tag = parse_triplet(tag_text)
    ok = event.getByLabel(tag, handle)
    if not ok:
        return None
    return handle.product()


def safe_len(product):
    return 0 if product is None else len(product)


def sum_layercluster_energy(layer_clusters):
    if layer_clusters is None:
        return 0.0
    total = 0.0
    for lc in layer_clusters:
        total += lc.energy()
    return total


def sum_layercluster_energy_in_tracksters(tracksters, layer_clusters):
    if tracksters is None or layer_clusters is None:
        return 0.0

    total = 0.0
    for trk in tracksters:
        if not hasattr(trk, "vertices"):
            continue
        total += trk.raw_energy()
    return total


def main():
    parser = argparse.ArgumentParser(description="FWLite ntuplizer for HGCAL geometry comparison")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-events", type=int, default=-1)
    parser.add_argument("--verbose-every", type=int, default=0)

    parser.add_argument("--simclusters-tag", default="mix:MergedCaloTruth:HLT")
    parser.add_argument("--caloparticles-tag", default="mix:MergedCaloTruth:HLT")
    parser.add_argument("--layerclusters-tag", default="hgcalMergeLayerClusters::RECO")
    parser.add_argument("--tracksters-clue3dhigh-tag", default="ticlTrackstersCLUE3DHigh::RECO")
    parser.add_argument("--tracksters-merge-tag", default="ticlCandidate::RECO")

    args = parser.parse_args()

    ROOT.gROOT.SetBatch(True)
    ROOT.gSystem.Load("libFWCoreFWLite")
    ROOT.gSystem.Load("libDataFormatsFWLite")
    ROOT.FWLiteEnabler.enable()

    events = Events(args.input)

    simcluster_type = "std::vector<SimCluster>"
    caloparticle_type = "std::vector<CaloParticle>"
    layercluster_type = "std::vector<reco::CaloCluster>"
    trackster_type = "std::vector<ticl::Trackster>"

    fout = ROOT.TFile(args.output, "RECREATE")
    tree = ROOT.TTree("events", "events")

    run = array("i", [0])
    lumi = array("i", [0])
    event = array("L", [0])

    nSimClusters = array("i", [0])
    nCaloParticles = array("i", [0])
    nLayerClusters = array("i", [0])

    sumEnergyLayerClusters = array("f", [0.0])

    nTrackstersCLUE3DHigh = array("i", [0])
    sumEnergyLayerClustersInTrackstersCLUE3DHigh = array("f", [0.0])

    nTrackstersMerge = array("i", [0])
    sumEnergyLayerClustersInTrackstersMerge = array("f", [0.0])

    tree.Branch("run", run, "run/I")
    tree.Branch("lumi", lumi, "lumi/I")
    tree.Branch("event", event, "event/l")

    tree.Branch("nSimClusters", nSimClusters, "nSimClusters/I")
    tree.Branch("nCaloParticles", nCaloParticles, "nCaloParticles/I")
    tree.Branch("nLayerClusters", nLayerClusters, "nLayerClusters/I")
    tree.Branch("sumEnergyLayerClusters", sumEnergyLayerClusters, "sumEnergyLayerClusters/F")

    tree.Branch("nTrackstersCLUE3DHigh", nTrackstersCLUE3DHigh, "nTrackstersCLUE3DHigh/I")
    tree.Branch(
        "sumEnergyLayerClustersInTrackstersCLUE3DHigh",
        sumEnergyLayerClustersInTrackstersCLUE3DHigh,
        "sumEnergyLayerClustersInTrackstersCLUE3DHigh/F",
    )

    tree.Branch("nTrackstersMerge", nTrackstersMerge, "nTrackstersMerge/I")
    tree.Branch(
        "sumEnergyLayerClustersInTrackstersMerge",
        sumEnergyLayerClustersInTrackstersMerge,
        "sumEnergyLayerClustersInTrackstersMerge/F",
    )

    miss_counts = {}
    n_processed = 0

    def fetch(ev, type_name, tag_text):
        product = get_by_label(ev, type_name, tag_text)
        if product is None:
            miss_counts[tag_text] = miss_counts.get(tag_text, 0) + 1
        return product

    for ievt, ev in enumerate(events):
        if args.max_events >= 0 and ievt >= args.max_events:
            break

        n_processed += 1

        aux = ev.eventAuxiliary()
        run[0] = aux.run()
        lumi[0] = aux.luminosityBlock()
        event[0] = aux.event()

        simclusters = fetch(ev, simcluster_type, args.simclusters_tag)
        caloparticles = fetch(ev, caloparticle_type, args.caloparticles_tag)
        layerclusters = fetch(ev, layercluster_type, args.layerclusters_tag)
        tracksters_clue = fetch(ev, trackster_type, args.tracksters_clue3dhigh_tag)
        tracksters_merge = fetch(ev, trackster_type, args.tracksters_merge_tag)

        nSimClusters[0] = safe_len(simclusters)
        nCaloParticles[0] = safe_len(caloparticles)
        nLayerClusters[0] = safe_len(layerclusters)
        sumEnergyLayerClusters[0] = sum_layercluster_energy(layerclusters)

        nTrackstersCLUE3DHigh[0] = safe_len(tracksters_clue)
        sumEnergyLayerClustersInTrackstersCLUE3DHigh[0] = sum_layercluster_energy_in_tracksters(tracksters_clue, layerclusters)

        nTrackstersMerge[0] = safe_len(tracksters_merge)
        sumEnergyLayerClustersInTrackstersMerge[0] = sum_layercluster_energy_in_tracksters(tracksters_merge, layerclusters)

        tree.Fill()

        if args.verbose_every > 0 and (ievt + 1) % args.verbose_every == 0:
            print(f"[info] processed {ievt + 1} events")

    fout.cd()
    tree.Write()
    fout.Close()

    print("[info] wrote", args.output)
    print("[info] tags used:")
    print("  simclusters              =", args.simclusters_tag)
    print("  caloparticles            =", args.caloparticles_tag)
    print("  layerclusters            =", args.layerclusters_tag)
    print("  tracksters clue3d high   =", args.tracksters_clue3dhigh_tag)
    print("  tracksters merge         =", args.tracksters_merge_tag)

    if miss_counts:
        for tag_text, count in sorted(miss_counts.items()):
            print(
                f"[error] product missing for tag '{tag_text}' in {count}/{n_processed} events",
                file=sys.stderr,
            )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())