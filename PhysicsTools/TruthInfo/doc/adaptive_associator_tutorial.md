# Tutorial: the MC-truth graph, the branch associators, and reading the validation

A hands-on introduction to the navigable MC-truth graph and the truth-branch associators.
You will produce one sample from scratch, associate reco objects to truth branches,
harvest the DQM pages and read them without fooling yourself.

Everything here has been run end to end, and every number quoted is measured on the sample
the commands produce. Check your output against it. If a number differs by more than a few
percent, something in your chain is different and it is worth finding out what before
going on.

Budget: about 10 minutes of CPU on 16 threads and about 4 GB of disk.

---

## 0. What the graph is, in one page

The old truth objects (`TrackingParticle`, `CaloParticle`, `SimCluster`) are FLAT: each is
a frozen bundle of a particle plus its hits, and the relationships between them are gone.
The MC-truth graph keeps the event history itself as something you can walk.

Three ideas; the rest of this tutorial is mechanics.

**Particles and vertices.** `truth::Graph` is bipartite: particles connected through
production vertices. It spans GEN and SIM uniformly, so a generator particle and the
Geant4 track it became are the same object seen at two depths.

**A Branch is a view, not a product.** `truth::Branch` is a lightweight, non-owning view
of one root particle plus a closure of its descendants. Nothing is copied. A branch has an
energy, a set of hits, a production process and a pileup provenance, and you choose how
deep the closure goes.

**Levels are ANTICHAINS, and this matters more than anything else here.** A level is a set
of branches where no member is an ancestor of another. That property is what makes
counting meaningful. Against the FULL graph every ancestor trivially contains its
descendants' hits, so every reco object matches a whole nested chain and every metric
degenerates. The levels shipped are:

| Level | What its roots are |
|---|---|
| `caloBoundary` | the particle crossing into the calorimeter, the `CaloParticle` analogue |
| `stableDecayProducts` | status-1 particles with sim hits anywhere |
| `stableLegsFromUpstream` | stable legs hanging off the selection's synthetic source vertices |
| `hardProcess` | the **outgoing legs** of the hard scatter, NOT the resonance. On ttbar it is b, b~ and the W decay products, not the two tops; on H to two photons it is the photons |
| `reconstructableFromSignal` | the signal's **visible final state**: walk down from each signal root, stop at the first thing a detector reconstructs as an object. A pi0 is an object even though it decays, so the pi0 is labelled and its two photons are not; an a1 or rho is walked through. Neutrinos are dropped |
| `underlyingEvent` | the stable legs of the underlying event, the counterpart of `stableLegsFromUpstream` on the ISR side |
| `partonJets` | one root per **parton-initiated jet**: the hard-scatter legs that are quarks or gluons, each standing for everything downstream of it. No clustering and no cone; the jet IS the descendant subgraph and its flavour is the parton PDG id. It is a subset of `hardProcess`, so it inherits the deepest-element rule that keeps the b rather than the top above it, which is also the physics: the top decays before it hadronises |

Two more levels exist in the graph but are not asked as branch denominators, because
their job is to define the **secondary-vertex** truth instead (section 5b):

| Level | What its roots are |
|---|---|
| `bHadrons` | the first b hadron along each chain. A B\* radiates down to a B and both are b hadrons, so the antichain keeps the B\*: one member per physical decay, not one per generator copy |
| `cHadrons` | the same for charm. Beauty and charm are SEPARATE levels on purpose: a B decays to a D, so a single combined level would keep the B and silently drop every charm vertex |

On top of the levels come `signal` (the preset's seed objects, so the RESONANCE itself:
two tops, one Z, one Higgs, ten taus) and `signalNoSelection` (the same with no kinematic
cut).

A sample with NO resonance has no `signal` level at all. Both folders are simply **not
booked** when the configuration names no seed species, rather than booked empty: the
question has no meaning there, and an empty folder reads as an efficiency of zero. QCD is
the sample in the set where this shows. The old behaviour fell back to every selected
root, which is not an antichain and is exactly the denominator that was removed for that
reason.

**Every one of these is an ANTICHAIN**: no member is an ancestor of another, so no
efficiency counts one object twice. That is the entry requirement: a denominator holding
both a particle and its own daughter would count the same energy twice.

Since 2026-08 the levels also travel WITH the graph, as a `levelFlags` bitmask on every
`ParticleData`, so the dot dump, a job log and a consumer outside CMSSW all see them
without knowing how a level is defined. `signal` is the one bit that cannot be recomputed
from the graph alone, so the seed species that produced it are recorded on the graph and
the dumper audits every flagged particle against them.

---

## 1. Release and branch

```bash
# Pick an IB that still EXISTS: they are rotated off cvmfs after about a week, and an
# area whose base has gone fails with "Missing Release top" and a missing cmsRun.
scram list -c CMSSW | grep CMSSW_20_1_X_2026
cmsrel CMSSW_20_1_X_2026-08-01-1100     # or whatever the newest one is today
cd CMSSW_20_1_X_2026-08-01-1100/src
cmsenv
git cms-init
# One topic brings everything: truth-adaptive-associator is stacked on the
# "MC-truth graph by default for Run4" work, so this pulls in both the
# default-on truth graph and the associators.
git cms-merge-topic felicepantaleo:truth-adaptive-associator
scram b -j 16
```

Check the unit tests before trusting anything you produce:

```bash
cd PhysicsTools/TruthInfo && scram b runtests   # expect 9 passes
cd ../..
```

---

## 2. Produce the sample: ten taus, no pileup

Workflow `34887.0`, `TenTau_E_15_500_pythia8` on Run4D122: ten taus per event between 15
and 500 GeV. Chosen deliberately, because taus decay to one or three prongs, which is the
cleanest way to see the difference between a truth object reconstructed **as one object**
and one reconstructed **in pieces**.

```bash
mkdir -p tentau && cd tentau

# 2a. GEN-SIM, 200 events
cmsDriver.py TenTau_E_15_500_pythia8_cfi -s GEN,SIM -n 200 \
  --conditions auto:phase2_realistic_T35 --beamspot DBrealisticHLLHC \
  --datatier GEN-SIM --eventcontent FEVTDEBUG \
  --geometry ExtendedRun4D122 --era Phase2C26I13M9 --relval 9000,100 \
  --fileout file:step1.root --nThreads 16 \
  --python_filename gensim.py --no_exec
cmsRun gensim.py                                    # about 3 minutes

# 2b. DIGI + L1 + RAW. The truth graph and the hit index are built and persisted
#     HERE, by default: enableTruth is in the Phase2C17I13M9 era, which every
#     Phase-2 geometry era inherits. There is no extra option to pass.
cmsDriver.py step2 \
  -s DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW,HLT:@relvalRun4 \
  --conditions auto:phase2_realistic_T35 --datatier GEN-SIM-DIGI-RAW -n 200 \
  --eventcontent FEVTDEBUGHLT --geometry ExtendedRun4D122 --era Phase2C26I13M9 \
  --filein file:step1.root --fileout file:step2.root --nThreads 16 \
  --python_filename digi.py --no_exec
cmsRun digi.py                                      # about 2.5 minutes

# 2c. RECO, with the associators attached by a customise
cmsDriver.py step3 -s RAW2DIGI,L1Reco,RECO -n 200 \
  --conditions auto:phase2_realistic_T35 --datatier GEN-SIM-RECO \
  --eventcontent FEVTDEBUGHLT --geometry ExtendedRun4D122 --era Phase2C26I13M9 \
  --customise SimGeneral/TruthGraphAssociatorProducers/customiseTruthGraphAssociators.customiseTruthGraphAssociators \
  --filein file:step2.root --fileout file:step3.root --nThreads 16 \
  --python_filename reco.py --no_exec
cmsRun reco.py                                      # about 2.5 minutes
```

Confirm the graph really crossed the DIGI boundary:

```bash
edmDumpEventContent step2.root | grep -iE "truthLogicalGraph|TruthGraph"
```

You should see the logical graph, the unresolved hit index and the raw `TruthGraph_mix`.

!!! warning "Consume, do not rebuild: the GRAPH, not the associators"
    `truthLogicalGraphProducer` and `truthLogicalGraphHitIndexProducer` must **not** be
    scheduled at RECO or later. Their products come from the DIGI file, where the merged
    signal-plus-pileup simHits were live; rebuilding them later silently loses pileup.
    The ASSOCIATORS are the opposite case: they are cheap, stateless consumers of the
    graph, and section 4 re-runs them in the DQM step on purpose, so the maps always
    come from the code being tested.

### 2d. The focused selection, optional but recommended

A preset makes the logical graph a focused subgraph around the physics object you care
about, and it is what materialises the synthetic Interaction and Upstream source vertices
that `stableLegsFromUpstream` needs. Add this to `digi.py` before running it:

```python
from PhysicsTools.TruthInfo.truthGraphSelections import postProcessingPSet as _ppSet
_preset = _ppSet(name='TenTau_E_15_500_pythia8_cfi')
for _p in _preset.parameterNames_():
    setattr(process.truthLogicalGraphProducer.postProcessing, _p, getattr(_preset, _p))
```

The fragment name is resolved automatically. For a particle gun the species word is
matched, so `TenTau...` gives `seedPdgIds = [15, -15]`. Check it:

```bash
python3 -c "
from PhysicsTools.TruthInfo.truthGraphSelections import postProcessingPSet
print(postProcessingPSet(name='TenTau_E_15_500_pythia8_cfi').seedPdgIds)"
```

Skip this step and `signal` degenerates into every selected root while
`stableLegsFromUpstream` comes out empty. Both are visible in the plots, so it is a useful
thing to get wrong once on purpose.

---

## 3. The association products

```bash
edmDumpEventContent step3.root | grep -i RecoToTruth | head
```

For every reco collection you get, per **working point**, one map in each direction:

| Product | Direction | Payload |
|---|---|---|
| `<collection>RecoToTruth<WP>` | reco to truth | shared quantity, reco-normalised score |
| `<collection>TruthToReco` | truth to reco | sim-normalised fraction, truth-normalised score |

plus the denominator lists `truthToRecoTargets<Level>`, `signalSeeds` and
`signalSeedsNoSelection`, and beside every level denominator a parallel
`truthToRecoTargets<Level>Eligibility` mask saying which plotted-axis cut each target
fails, so an efficiency against pt keeps the objects the pt cut would have removed
(section 7 relies on it).

The truth-to-reco map is written **once**, not per working point, on purpose: the truth
target is fixed a priori by the level, so a reco-driven working point has no business
entering it.

### The four working points

| Name | What it does |
|---|---|
| `Fixed` | no climb. Keeps **every** matching branch root |
| `AdaptiveTight`, `AdaptiveNominal`, `AdaptiveLoose` | climb to the single graph level that best matches the object, differing in how much branch spread they tolerate |

The climb minimises `score + adaptiveReverseWeight * reverseScore`. `score` falls as the
branch climbs, since it covers more of the reco object; `reverseScore` rises as the branch
spreads into energy the object does not have. Levels above `adaptiveMaxReverseScore` are
rejected. The climb never selects or crosses a bare parton, diquark, string or cluster
node, so it stops at physical particles.

For a clean single particle the adaptive level is that particle. The climb earns its keep
on a converted photon or a decay in flight, where the right answer is the merged parent
rather than the individual daughters.

`Fixed` matters beyond being a baseline: it is the only map carrying **more than one
candidate per object**, so any measurement comparing a leading candidate against a
runner-up must read it. An adaptive point inserts only the branch it climbed to.

---

## 4. Run the validation

Two steps, because a single-process analyze-plus-harvest job does **not** save the
analyzer's monitor elements: the `DQMGlobalEDAnalyzer` per-run cache is not in the
`DQMStore` that `DQMFileSaver` walks. Step A writes DQMIO, step B harvests it. Same split
`MultiTrackValidator` and `HGCalValidator` use.

Step A, `dqmA.py`:

```python
import FWCore.ParameterSet.Config as cms
process = cms.Process("TRUTHDQM")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("DQMServices.Core.DQMStore_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.source = cms.Source("PoolSource",
                            fileNames=cms.untracked.vstring("file:step3.root"))
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(-1))
process.options = cms.untracked.PSet(numberOfThreads=cms.untracked.uint32(16),
                                     numberOfStreams=cms.untracked.uint32(16))

# Point the trackster lists at what the INPUT FILE actually holds. A DQM-only job has
# no scheduled TICL producer, so without this it falls back to the label registry,
# which can list collections your menu does not schedule. Must run BEFORE the
# associator and validation cffs are imported.
from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociationLabels_cff import (
    setTracksterLabelsFromProcess)
setTracksterLabelsFromProcess(process)

# Re-run the ASSOCIATORS here, rather than consuming the copies step3 wrote. They cost
# 4.3 ms/event, and it buys two things: the maps are made by the code you are running
# now rather than by whatever built the RECO file, and this is the one place the signal
# seeds are set. Skipping this and consuming the RECO-era products is measured below in
# section 8c: it read signal = 107 per event where the answer is exactly 10.
from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociators_cff import (
    allTrackToTruthBranchAssociators, allVertexToTruthBranchAssociators,
    allSecondaryVertexToTruthBranchAssociators, truthBranchTracksterAssociators,
    hltTrackToTruthBranchAssociators, hltVertexToTruthBranchAssociators,
    hltTruthBranchTracksterAssociators)
for _label in ("allTrackToTruthBranchAssociators", "allVertexToTruthBranchAssociators",
               "allSecondaryVertexToTruthBranchAssociators", "truthBranchTracksterAssociators",
               "hltTrackToTruthBranchAssociators", "hltVertexToTruthBranchAssociators",
               "hltTruthBranchTracksterAssociators"):
    setattr(process, _label, globals()[_label])

# The preset's seed species, to the associators AND the hit-based validators: the
# associators fill the signalSeeds denominator from it, and the validators book the
# signal folders only when it names a resonance. hasattr guards the composite
# validators, which do not declare the parameter.
from PhysicsTools.TruthInfo.truthGraphSelections import seedPdgIdsForPreset
_signalSeeds = seedPdgIdsForPreset(name="TenTau_E_15_500_pythia8_cfi")
for _label in ("allTrackToTruthBranchAssociators", "allVertexToTruthBranchAssociators",
               "allSecondaryVertexToTruthBranchAssociators", "truthBranchTracksterAssociators",
               "hltTrackToTruthBranchAssociators", "hltVertexToTruthBranchAssociators",
               "hltTruthBranchTracksterAssociators"):
    getattr(process, _label).signalSeedPdgIds = _signalSeeds

# EVERY validator the sequence contains must be attached, the HLT ones included, or
# configuration fails with "An entry in sequence truthBranchValidationSequence has no
# label". The sequence is built from the domain list, not from what you attach.
import Validation.TruthInfo.truthBranchValidation_cff as _tv
for _label in ("truthBranchTrackValidator", "truthBranchVertexValidator",
               "truthBranchSecondaryVertexValidator", "truthBranchTracksterValidator",
               "hltTruthBranchTrackValidator", "hltTruthBranchVertexValidator",
               "hltTruthBranchTracksterValidator"):
    setattr(process, _label, getattr(_tv, _label))
    _v = getattr(process, _label)
    if hasattr(_v, "signalSeedPdgIds"):
        _v.signalSeedPdgIds = _signalSeeds
process.truthBranchValidationSequence = _tv.truthBranchValidationSequence

process.DQMoutput = cms.OutputModule(
    "DQMRootOutputModule",
    fileName=cms.untracked.string("file:dqmio.root"),
    outputCommands=cms.untracked.vstring("drop *", "keep *_MEtoEDMConverter_*_*"),
    splitLevel=cms.untracked.int32(0))
process.p = cms.Path(process.allTrackToTruthBranchAssociators
                     + process.allVertexToTruthBranchAssociators
                     + process.allSecondaryVertexToTruthBranchAssociators
                     + process.truthBranchTracksterAssociators
                     + process.hltTrackToTruthBranchAssociators
                     + process.hltVertexToTruthBranchAssociators
                     + process.hltTruthBranchTracksterAssociators
                     + process.truthBranchValidationSequence)
process.e = cms.EndPath(process.DQMoutput)
```

Step B reads that with `DQMRootSource`, runs `process.truthBranchHarvestingSequence` from
the same cff and saves with `dqmSaver.workflow = "/TruthInfo/Validation/RECO"`. Then:

```bash
cmsRun dqmA.py    # about a minute: the associators run here too
cmsRun dqmB.py    # about 5 seconds
makeTruthValidationPlots.py DQM_V0001_R000000001__TruthInfo__Validation__RECO.root \
  --outputDir plots --sample "TenTau E 15-500 GeV, no pileup, D122, 200 events"
```

About 2700 plots behind one `index.html`, organised by metric and acceptance region.

### The folder layout

```
TruthInfo/Offline/{Tracking,Vertexing,SecondaryVertexing,Calorimetry}/
    <collection>_<level>     truth-driven: efficiency, split rate, composition
    <collection>_<WP>        reco-driven:  fake, contaminated, purity, pileup, resolution
TruthInfo/HLT/...            the same, for the HLT menu's own reconstruction
```

Which family a metric lives in is not a style choice: it decides the denominator.
Truth-driven metrics divide by a truth object and are binned in branch variables;
reco-driven ones divide by a reco object and are binned in the reco object's own
variables.

---

## 5. Check your output against these numbers

All from the chain above, 200 events.

| Quantity | Expected | Why |
|---|---|---|
| `signal` denominator | **10.00 / event** | ten taus, exactly. The cleanest check that the selector is not eating signal |
| `signalNoSelection` | **10.00 / event** | equal to `signal`, so the kinematic cut removed nothing |
| `reconstructableFromSignal` | 18.59 / event | the taus' visible decay products, pi0 counted as one object rather than two photons |
| `hardProcess` | **0.00** | correct, not a bug: a particle **gun** has no hard-process record, so `isHardProcess` is set on nothing |
| `underlyingEvent` | **0.00** | also correct: a gun has no underlying event. On ttbar it is 103.17 |
| `partonJets` | **0.00** | a gun has no partons either. On no-PU ttbar it is 4.80 / event: 400 b jets over 200 events, exactly two per event, plus 560 light-quark jets |
| `caloBoundary` | 35.34 / event | the taus' decay products entering the calorimeter |
| `stableDecayProducts` | 36.33 / event | the generator-stable final state |
| `stableLegsFromUpstream` | 39.90 / event | the ISR and upstream side of the interaction |
| reco tracksters | 13.2 / event | ten taus at one or three prongs |
| reco tracks | 13.8 / event | same |
| trackster fake rate | 0.0484 | identical at every working point, by construction |
| tracking fake rate | 0.2904 | at every working point. Almost all of it is "no single owner", not "matched nothing", which is 0.0000 |
| mean of `reco_purity`, tracks | 0.337 Fixed to **0.862** AdaptiveNominal | where the adaptive climb pays |
| mean of `reco_purity`, tracksters | 0.054 Fixed to **0.939** AdaptiveNominal | the same effect, stronger in the calorimeter |

If `hardProcess` is empty, that is the expected answer here, and a good illustration that
the level means what it says: on ttbar, DY and VBF it gives 5.73, 1.51 and 6.00 per event.

---

## 5b. The secondary-vertex denominator is the heavy-flavour DECAY vertices

A secondary vertex is where a b or c hadron decayed, so that is exactly what the
denominator is: the decay vertices of the `bHadrons` and `cHadrons` antichains. The
associator switches this on with `heavyFlavorOnly`.

The looser question, "does this vertex's incoming particle carry heavy flavour anywhere
in its subgraph", is true at every vertex all the way along the chain, and it inflates
the denominator about threefold. Measured on no-PU ttbar:

| Criterion | Truth SVs per event |
|---|---|
| b/c hadron decay vertices | **4.00**, measured over 200 events |
| incoming particle's subgraph carries heavy flavour | 12 and 16 |
| every graph vertex with two selected roots (no restriction at all) | 45.9 |
| what `inclusiveSecondaryVertices` actually reconstructs | **4.1** |

Only the first matches reco, and it matches it to two decimals: 801 truth secondary
vertices over 200 events against 4.1 reconstructed. The other two cap the efficiency at a third and at a tenth
respectively, however good the reconstruction is, and the cap is a property of the
denominator rather than of the detector. This is the single most common way to produce a
believable-looking efficiency that means nothing.

---

## 6. Reading the reco-side pages: six different questions

| Page | Formula | Question |
|---|---|---|
| `fakerate` | `1 - num_dominated/num_reco` | does **one** truth branch own the object |
| `nocandidate` | `1 - num_assoc(recoToSim)/num_reco` | does it correspond to **any** truth branch. A subset of `fakerate` |
| `nolevelcandidate` | `1 - num_levelcandidate/num_reco` | is the fake question even **defined** for it |
| `contaminated` | `1 - num_assoc_strict/num_reco` | does its best candidate pass the 0.6 recoToSim score, HGCalValidator's non-fake cut. Calorimetry only |
| `recopurity` | `num_recopurity/num_reco` | how much of the object belongs to the branch it matched, as a **mean** |
| `pileuprate` | `num_pileup/num_reco` | is the matched branch an overlaid interaction |

Each has its **own numerator**. Merging any two is how this package produced its two worst
bugs, both in section 8.

A **fake** is an object no truth branch owns, which happens two ways: it matched nothing
at all, or it has contributions from several different particles with none dominating.
Section 8b is where that definition is built, including the category it deliberately does
NOT call a fake.

!!! warning "`fakerate` and `contaminated` are different questions"
    The recoToSim score is normalised against the cell's **total** truth energy
    (`recoEnergy = fraction * cellTotalEnergy`), so at PU200 a cell shared with overlaid
    interactions drives it towards 1 even for a well matched object. Measured on ttbar
    PU200, `ticlCandidate` / AdaptiveNominal: **73.8%** of tracksters fail the 0.6 cut
    while only **2.2%** have no candidate at all. Say "fake" only for the second.

### Exercise 1

The tracking fake rate is **identical at all four working points**, while the mean purity
climbs from 0.58 to 0.99. Explain why before reading on.

> The climb changes **which** branch an object matches, not **whether** it matches one, so
> the adaptive gain must land on the purity page and cannot appear on the fake page. Use
> it as a control: a fake rate that moves with the working point means the association
> gained or lost candidates, and that needs explaining.

### Exercise 2

At PU200 the tracking `nocandidate` rate is 0.000 to 0.001, **lower** than the 0.003 to
0.010 with no pileup at all. Did reconstruction improve by adding 200 interactions?

> No, the metric **saturates**. With 200 interactions the truth graph is dense enough that
> nearly every reco object overlaps something, so "matched to nothing" stops
> discriminating. At PU200 read the pileup rate (0.93 to 0.95) and the purity (0.10 to
> 0.73) instead. Before quoting any rate sitting at 0 or 1, ask whether its definition can
> still vary in that regime. A saturated metric and a perfect detector draw the same plot.

### Exercise 3

On ttbar PU200 the `fakerate` is 0.027 and `nolevelcandidate` is 0.872. Why is the first
number nearly meaningless on its own?

> Because the fake question is only defined for the 12.8% of tracksters that have a
> candidate at `caloBoundary`. That level holds ~113 objects per event at PU200 against
> 99.7 with no pileup, since the selector's 1 GeV floor removes nearly all soft pileup,
> while there are thousands of tracksters. A rate measured on a small, non-random subset
> needs the subset size quoted with it every time.

---

## 7. Reading the truth-side pages: individual against cumulative

On the efficiency page each level is drawn as a pair:

- **individual**, filled and solid: a **single** reco object covered the truth object.
- **cumulative**, open and dashed: all reco objects of the collection **together** cover it.

A tau decaying to three pions is reconstructed as three tracksters, so it is individually
LOST and cumulatively FOUND. The gap between the two curves is exactly the `splitrate`
page, and the two must agree bin by bin, because the outcomes are mutually exclusive:

```
individual + duplicate + split + lost = 1     so     cumulative = individual + split
```

Split rate at `caloBoundary`, measured:

| Sample | tracking | ticlCandidate |
|---|---|---|
| **TenTau no-PU** | **0.086** | **0.071** |
| ttbar no-PU | 0.066 | 0.035 |
| DY no-PU | 0.005 | 0.016 |

TenTau opens the widest gap, which is the whole reason to run it; DY, whose Z goes to two
leptons, is flattest. If your gap and your split rate ever disagree, that is a
classification bug, not a tuning question.

!!! note "`num_duplicate` is absent from calorimetric folders"
    The duplicate outcome needs two reco objects each below
    `maxSimToRecoScoreForDuplicate` on the **same** branch, and two objects built from
    **disjoint** layer clusters have scores summing to at least one. Measured on 200 no-PU
    ttbar events, `ticlCandidate`, `ticlTrackstersCLUE3DHigh` and `ticlTracksterLinks`
    each use every layer cluster in at most one trackster. This diverges deliberately from
    HGCalValidator, which books the plot; the split rate carries the pathology instead.

---

### 7b. Always read the ACCEPTANCE REGION, not the inclusive number

Every metric is booked four times: inclusively, and again in three eta sub-folders
`etaLt15`, `eta15to30`, `eta30to45`. The plot pages carry one set per region. This is not
a refinement, it is usually the difference between a number that means something and one
that does not.

Take the tau itself, `signal`, ticlCandidate, on the sample you just produced:

| region | taus | individual | split | cumulative |
|---|---|---|---|---|
| inclusive | 2000 | 0.194 | 0.098 | 0.291 |
| abs(eta) below 1.5 | 1230 | 0.016 | 0.006 | 0.022 |
| 1.5 to 3.0 | 767 | **0.478** | **0.246** | **0.725** |
| 3.0 to 4.5 | 3 | 0.000 | 0.000 | 0.000 |

Read inclusively you would conclude the calorimeter reconstructs 19% of taus. It does not:
**61.5% of the taus enter the barrel**, where no trackster can exist, and that population
sets the inclusive number. In the endcap, where the reconstruction actually runs, a single
trackster covers the tau 48% of the time and the collection covers it 73% of the time.

The 24.6-point gap between individual and cumulative IS the split rate, and it is what a
multi-prong decay looks like: no one trackster holds the whole tau, several together do.
Compare with `caloBoundary`, a generic single particle, which splits less (0.223).

CONTROL you can run yourself: the regions must partition the sample. 1230 + 767 + 3 =
2000, and the region numerators weighted by their denominators reproduce the inclusive
value to three decimals. If they do not, something is being double counted or dropped.

### 7c. Jets: read the flavour axis, and read it cumulatively

`partonJets` gives one truth object per parton-initiated jet, so every metric can be read
against `flavour`, an axis with one named bin per initiating species (`other`, `d`, `u`,
`s`, `c`, `b`, `t`, `g`). Only this level has parton roots; on every other level the whole
distribution sits in `other` by construction, which is a feature rather than a gap.

Read it CUMULATIVELY. A jet is reconstructed as many objects, never as one, so individual
efficiency answers a question nobody asked. This is the same lesson the three-prong tau
teaches in 7b, one step further along.

Measured, 200 no-PU ttbar events, `ticlCandidate`, denominator `num_simul_flavour`:

| | denominator | endcap denominator | endcap cumulative efficiency |
|---|---|---|---|
| b jets | **400** (exactly 2 per event) | 87 | **0.59** |
| all parton jets | 960 | 249 | **0.60** |

The b count is the strongest single check in this document: a ttbar event has exactly two
b quarks, and 400 over 200 events is that statement. If your b bin is not exactly twice
your event count, the level is wrong before any efficiency is worth reading.

The inclusive cumulative efficiency is 0.22, and it is again the barrel doing it: read the
region, per 7b.

The gluon bin needs a sample whose hard scatter makes gluons. On QCD flat pT (workflow
34843.0, 200 events, no PU) `partonJets` is **400 = exactly 2.00 per event**, a dijet
having two hard-scatter legs, split g 266, u 65, d 43, s 11, c 8, b 7. Gluon dominance is
what QCD at the LHC has to give. There `partonJets` and `hardProcess` coincide bin for
bin, because every hard-scatter leg in a QCD event IS a parton; on ttbar they differ by
exactly the 186 leptonic legs (1146 minus 960). Those two statements together are a good
check that the level is the subset it claims to be.

The `t` bin is never filled by `partonJets`, and that is the deepest-element rule working:
a top always has a hard-process b below it. It IS filled by the `signal` level on ttbar,
with 2.00 per event, so a point at `t` tells you which level you are reading.

!!! warning "Jet subgraphs OVERLAP, and the roots being an antichain does not prevent it"
    The jet ROOTS are an antichain, so no jet contains another. The SUBGRAPHS are not
    disjoint. Two quarks colour connected to each other, the u and d~ of a hadronic W,
    fragment through one string, so its hadrons descend from BOTH and their hits are
    counted under both jets. Measured on no-PU ttbar: 1221 of 8096 hits shared, 0.15 of
    the union, ALL of it between that one pair; the b and b~ share nothing, and a
    dileptonic event with no hadronic W shares 0.00. This is inherent to defining a jet
    with no clustering algorithm. Assigning each hadron to exactly one jet is precisely
    what a clustering algorithm is for, and it is the reason one would add one.

## 8. How not to fool yourself

Three mistakes, all real, all expensive, all of which produced numbers that looked
perfectly plausible. This is the most useful section here.

### 8a. Three histogram calls, three different answers

- `GetEntries()` counts how many times `Fill` was called and ignores weights entirely.
- `Integral()` sums bin contents in the **visible** range and **drops** underflow and overflow.
- a bin-content sum adds **weights**, which are not counts.

**Incident one.** A DY `signal` denominator read 0.57 objects per event where it must be
exactly 1, one Z. A Z produced at rest has pt about 0 and therefore abs(eta) going to
infinity, so it sits in the **overflow** bin, which `Integral()` excludes. Four rounds
were lost and the correct hypothesis was killed twice by that one bad reading. Use
`GetEntries()` when you mean "how many objects".

**Incident two.** A fake rate read 0.83 where it is 0.003, because the matched numerator
was filled with the purity as a **weight** and then read as a count, making the fake page
one minus the mean purity. The tell was `GetEntries()` disagreeing with the bin sum: 84
unmatched objects by count against about 21000 by summed weight. A second tell was the
shape, since the rate **rose** with pt, which no fake-rate mechanism explains but low
purity at high pt does.

Habit to form: on any numerator, compare `GetEntries()` against the bin sum. For an
unweighted histogram they agree by construction, so a disagreement **is** the diagnosis.

### 8b. Antichains, again

A fake by dominance is a natural physicist's definition: an object whose hits come from
many DIFFERENT generated particles with none dominating. Two monitor elements measure it,
`leading_truth_share` and `dominance_ratio`, leading over runner-up.

They must be computed over an antichain. The selected-root set is **not** one, so a tau,
its daughter pion and that pion's descendants are candidates simultaneously with
**nested** subgraphs carrying nearly identical shared energy, and the leader gets compared
against its own child. The control is this very sample, where ten isolated taus must give
one overwhelming winner:

| candidate set | median leading share | fraction with ratio near 1 |
|---|---|---|
| every selected root, nested | 0.26 | 0.999 |
| `caloBoundary`, antichain | **0.98** | 0.064 |

A number that moves like that between two definitions is not a threshold to tune, it is a
bug to fix. The parameter is `dominanceLevel`, default `caloBoundary`.

The published criterion is therefore

```
fake = matched to nothing
       OR (has a candidate at dominanceLevel AND leading share < 0.5)
```

with the threshold in `minLeadingTruthShare`. Because dominance is read from the first
working point's map, the only one carrying every candidate, this is **identical at all
four working points** by construction, which is the control to check first.

### 8b-bis. The category that is deliberately not a fake

An object that matched truth but has **no candidate at the dominance level** is not
counted as a fake. The question is undefined for it, not answered negatively. Counting it
would measure how much of the event the level covers rather than how well the collection
reconstructs, and the difference is not small: doing so gave fake rates of 0.36 to 0.60 on
no-PU across every sample and both domains.

That was settled by measurement, and the shape of the investigation is the lesson. Three
candidate explanations were each killed with a number rather than an argument:

| suspected cause | test | result |
|---|---|---|
| the threshold is wrong | distribution where dominance IS defined | share is 1.0 for 45% of ttbar tracksters, below 0.5 for only 5.7%. Not the threshold |
| the level is wrong for tracking | config-only probe, `caloBoundary` to `stableDecayProducts` | 0.540 to 0.489, with calorimetry identical to four decimals as the control. Not the level |
| nested candidates dilute the leader | project each candidate onto its antichain ancestor | changed nothing to four decimals. Not nesting |

The real cause was the undefined category itself, 28% to 57% of objects, while only 0.3%
of tracks match nothing at all. It now has its own page, `nolevelcandidate`, and the fake
rate must be read beside it.

### 8c. Match the code that made the file

The graph is built at **DIGI**. A change to the graph producer, the pruning scope or the
sub-event id therefore cannot be repaired by re-running the validation: you must re-run
step2. A change to the selector, the levels, the associators or the validators runs in the
DQM step, so re-harvesting an existing RECO file **is** the full chain for those.

Not academic. Three PU200 samples were once compared where two carried a graph built
before a pruning fix and one after. It showed as `stableDecayProducts` reading 175 per
event in one and 7640 in the others, and tracking efficiency 0.690 against 0.008. Always
put your samples side by side before believing any single one.

The same rule covers the ASSOCIATION products inside a RECO file: they are as old as the
RECO that wrote them. Measured on this very sample, harvesting the RECO-era maps instead
of re-running the associators read `signal` = **107.00 per event** where the answer is
exactly 10.00, because the file predated a fix to how empty seed lists are treated, while
every other level agreed to the last digit. A wrong number that arrives surrounded by
right ones is the expensive kind, and it is why section 4 re-runs the associators in the
DQM step.

---

## 9. Where to look next

- `PhysicsTools/TruthInfo/interface/Branch.h` for the closure specifications:
  `subtree`, `stableLeaves`, `depth(n)`, `untilPdgId`, `predicate`.
- `PhysicsTools/TruthInfo/src/BranchHitAssociator.cc` for the scores, in particular how
  `simFraction` is normalised, which is what section 6 is about.
- `Validation/TruthInfo/python/truthBranchValidation_cff.py`: one entry in `_domains` is
  all it takes to add a reco domain. Folder names, harvester subdirectories and every
  ratio string are derived from it.
- `SimGeneral/TruthGraphAssociatorProducers/plugins/TruthBranchAssociationDumper.cc` when
  you need to know what is actually **in** a map rather than whether it exists.

## Notes and caveats

- **No pileup** in this tutorial. The truth graph is still built, from the signal
  `g4SimHits`. With pileup, `collapsePileupGen` keeps each overlaid interaction's stable
  status-1 GEN particles on one collapsed vertex **plus** the full SIM continuation; only
  the intermediate GEN decay chain is dropped.
- **Truth scope** defaults to the full detector: calo, MTD, muon and tracker. The
  calorimetric associator uses only the `Calo` channel, so the scope does not change its
  result. `customiseTruthReduced` drops the tracker, the largest sim-hit family.
- **A/B a parameter cheaply.** `maxRecoToSimScore`, `minSharedEnergyFractionForIndividual`
  and `dominanceLevel` are all **validator** parameters, so you can clone the DQM config,
  change one and re-run step A only, holding the graph, the hits and the associators
  bit-identical. Far cheaper than a rebuild, and it is how the 73.8%-against-2.2% split in
  section 6 was established.
