import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalAssociatorProducers.simClusterToCaloParticleAssociator_cfi import simClusterToCaloParticleAssociator

SimClusterToCaloParticleAssociation = simClusterToCaloParticleAssociator.clone(
    caloParticles = "mix:MergedCaloTruth",
    simClusters = "mix:MergedCaloTruth"
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2

premix_stage2.toModify(SimClusterToCaloParticleAssociation,
    caloParticles = "mixData:MergedCaloTruth",
    simClusters = "mixData:MergedCaloTruth",
)

