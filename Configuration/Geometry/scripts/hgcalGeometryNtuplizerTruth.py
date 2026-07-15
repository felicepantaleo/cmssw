#!/usr/bin/env python3

# FWLite ntuplizer for HGCAL geometry comparison, truth-graph extension.
# Adds SimCluster-hit-level observables to the standard reco set so MIP-like
# probes (muons) compare geometry through per-layer truth hits instead of
# nearly-empty layer cluster or trackster counts. Requires samples produced
# with the enableTruth process modifier.

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


def sum_trackster_energy(tracksters):
    if tracksters is None:
        return 0.0
    total = 0.0
    for trk in tracksters:
        total += trk.raw_energy()
    return total


def hgcal_layer_info(detid):
    # Returns (is_hgcal, is_scintillator, layer). DetId::Forward subdets:
    # HGCalEE=8, HGCalHSi=9, HGCalHSc=10. CE-H layers are offset so the
    # layer number is global (1..47) across CE-E + CE-H.
    detid = int(detid)
    det = (detid >> 28) & 0xF
    if det == 8:
        return True, False, ROOT.HGCSiliconDetId(detid).layer()
    if det == 9:
        return True, False, ROOT.HGCSiliconDetId(detid).layer() + 26
    if det == 10:
        return True, True, ROOT.HGCScintillatorDetId(detid).layer() + 26
    return False, False, 0


def main():
    parser = argparse.ArgumentParser(description="FWLite truth-graph ntuplizer for HGCAL geometry comparison")
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

    # Truth observables restricted to what the io_v1 SimCluster persists and
    # what cppyy can call safely: hits_and_fractions() works; hits_and_energies()
    # SEGFAULTS in FWLite and numberOfSimHits()/simEnergy() read unpersisted
    # members (always 0). Verified on 20_0_X_2026-07-15-1100 enableTruth output.
    nSimHits = array("i", [0])
    nLayersWithSimHits = array("i", [0])
    maxSimHitLayer = array("i", [0])
    nSimHitsSilicon = array("i", [0])
    nSimHitsScintillator = array("i", [0])

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

    tree.Branch("nSimHits", nSimHits, "nSimHits/I")
    tree.Branch("nLayersWithSimHits", nLayersWithSimHits, "nLayersWithSimHits/I")
    tree.Branch("maxSimHitLayer", maxSimHitLayer, "maxSimHitLayer/I")
    tree.Branch("nSimHitsSilicon", nSimHitsSilicon, "nSimHitsSilicon/I")
    tree.Branch("nSimHitsScintillator", nSimHitsScintillator, "nSimHitsScintillator/I")

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
        sumEnergyLayerClustersInTrackstersCLUE3DHigh[0] = sum_trackster_energy(tracksters_clue)
        nTrackstersMerge[0] = safe_len(tracksters_merge)
        sumEnergyLayerClustersInTrackstersMerge[0] = sum_trackster_energy(tracksters_merge)

        n_hits = 0
        n_si = 0
        n_sci = 0
        layers = set()
        if simclusters is not None:
            for sc in simclusters:
                for pair in sc.hits_and_fractions():
                    is_hgcal, is_sci, layer = hgcal_layer_info(pair.first)
                    if not is_hgcal:
                        continue
                    n_hits += 1
                    layers.add(layer)
                    if is_sci:
                        n_sci += 1
                    else:
                        n_si += 1

        nSimHits[0] = n_hits
        nLayersWithSimHits[0] = len(layers)
        maxSimHitLayer[0] = max(layers) if layers else 0
        nSimHitsSilicon[0] = n_si
        nSimHitsScintillator[0] = n_sci

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
