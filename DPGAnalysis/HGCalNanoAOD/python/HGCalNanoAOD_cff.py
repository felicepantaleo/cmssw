import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *
from DPGAnalysis.HGCalNanoAOD.hgcalTracksters_cfi import *
from DPGAnalysis.HGCalNanoAOD.hgcalTICLCandidates_cfi import *
from DPGAnalysis.HGCalNanoAOD.hgcalTICLSuperClusters_cfi import *
from DPGAnalysis.HGCalNanoAOD.hgcalLayerClusters_cfi import *
from DPGAnalysis.HGCalNanoAOD.hgcalTruthBranchTables_cfi import hgcalTruthBranchTables

######################################
# Offline HGCAL NanoAOD Tables
######################################

OfflineHGCalTables = cms.Sequence(
    hgcalTrackstersTableSequence
    + ticlCandidateTable
    + ticlCandidateExtraTable
)

# Store additional validation objects 
OfflineHGCalValidationTables = cms.Sequence(
    hgcalTiclAssociationsTableSequence
    + hgcalSimTracksterSequence
    + ticlSimCandidateTable
    + ticlSimCandidateExtraTable
    + hgcalLayerClustersTableSequence
)

######################################
# Sequences for different NanoAOD flavours
######################################

# Offline HGCAL NanoAOD (NANO:@HGCAL) - reconstruction objects only
hgcalNanoSequence = cms.Sequence(
    OfflineHGCalTables
)

# Offline HGCAL NanoAOD with validation info (NANO:@HGCALVal) - includes sim objects and scores
hgcalNanoValidationSequence = cms.Sequence(
    OfflineHGCalTables
    + OfflineHGCalValidationTables
)

# Truth-branch training tables: reads the truth graph and the trackster-to-branch
# association maps persisted by PhysicsTools/TruthInfo customiseTruthBranchTraining.
hgcalTruthBranchTablesAllLevels = hgcalTruthBranchTables.clone(
    associations=[
        cms.InputTag("allTrackstersToTruthBranchAssociationsAllLevels", "ticlTrackstersCLUE3DHighToTruthBranch"),
        cms.InputTag("allTrackstersToTruthBranchAssociationsAllLevels", "ticlTracksterInterpretationsToTruthBranch"),
    ],
    tableNames=["ticlTrackstersCLUE3DHighToTruthBranchAllLevels",
                "ticlTracksterInterpretationsToTruthBranchAllLevels"],
    branchTableName="TruthBranchAllLevels",
    computeLabels=False,
)

hgcalTruthBranchTableSequence = cms.Sequence(hgcalTruthBranchTables + hgcalTruthBranchTablesAllLevels)

hgcalNanoTruthSequence = cms.Sequence(
    hgcalNanoSequence.copy()
    + hgcalTruthBranchTableSequence
)


def hgcalNanoCustomize(process):
    """
    Customization function for offline HGCAL NanoAOD.
    This function is called when producing NanoAOD with HGCAL content.
    """
    # The candidate extra table propagates tracks to the HGCAL surfaces: a NANO-only
    # job does not schedule the reconstruction, so the propagator EventSetup modules
    # (TrackingComponentsRecord) must be loaded explicitly.
    process.load("TrackingTools.MaterialEffects.MaterialPropagator_cfi")
    process.load("TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi")
    if hasattr(process, "NANOAODSIMoutput"):
        process.NANOAODSIMoutput.outputCommands.append(
            "keep nanoaodFlatTable_*Table*_*_*"
        )

    if hasattr(process, "NANOAODoutput"):
        process.NANOAODoutput.outputCommands.append(
            "keep nanoaodFlatTable_*Table*_*_*"
        )

    return process
