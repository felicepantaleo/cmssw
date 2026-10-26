import FWCore.ParameterSet.Config as cms

# The HGCAL layer clusters merged on the device, in the order of hltMergeLayerClusters, so that the two merged
# collections index the same layer clusters. Read by the CLUEstering iteration of TICLv6.
hltHgcalSoAMergeLayerClusters = cms.EDProducer("HGCalSoAMergeLayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalLayerClustersSoA = cms.VInputTag(
        "hltHgcalSoALayerClustersProducer",
        "hltHgcalSoALayerClustersProducerHSci",
        "hltHgcalSoALayerClustersProducerHSi"
    )
)
