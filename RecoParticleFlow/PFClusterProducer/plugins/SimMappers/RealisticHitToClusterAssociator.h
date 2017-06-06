#ifndef __RecoParticleFlow_PFClusterProducer_RealisticHitToClusterAssociator_H__
#define __RecoParticleFlow_PFClusterProducer_RealisticHitToClusterAssociator_H__

#include <vector>
#include "HGCSimCluster2D.h"
#include "RealisticCluster.h"


class RealisticHitToClusterAssociator
{
        using Hit3DPosition = std::array<float,3>;
    public:
        void init(std::size_t numberOfHits, std::size_t numberOfSimClusters,
                std::size_t numberOfLayers)
        {
            hitPosition_.resize(numberOfHits);

            totalEnergy_.resize(numberOfHits);
            layerId_.resize(numberOfHits);
            MCAssociatedSimCluster_.resize(numberOfHits);
            MCEnergyFraction_.resize(numberOfHits);
            HitToRealisticSimCluster_.resize(numberOfHits);
            HitToRealisticEnergyFraction_.resize(numberOfHits);
            distance2FromOriSimTrack_.resize(numberOfHits);
            clusters2D_.resize(numberOfSimClusters);
            simTrackPosAtLayer_.resize(numberOfSimClusters);
            for (unsigned int i = 0; i < numberOfSimClusters; ++i)
            {
                clusters2D_[i].resize(numberOfLayers);
                simTrackPosAtLayer_[i].resize(numberOfLayers);
            }
            NeighboringSimClusters_.resize(numberOfSimClusters);
        }

        void insertHitPosition(float x, float y, float z, unsigned int hitIndex)
        {
            hitPosition_[hitIndex] = {{x,y,z}};

        }

        void insertLayerId(unsigned int layerId, unsigned int hitIndex)
        {
            layerId_[hitIndex] = layerId;
        }

        void insertHitEnergy(float energy, unsigned int hitIndex)
        {
            totalEnergy_[hitIndex] = energy;
        }

        void insertSimClusterIdAndFraction(unsigned int scIdx, float fraction,
                unsigned int hitIndex)
        {
            MCAssociatedSimCluster_[hitIndex].push_back(scIdx);
            MCEnergyFraction_[hitIndex].push_back(fraction);
        }

        void insertSimTrackPositionAtLayer(unsigned int scIdx, int layerId, float x, float y, float z)
        {
            simTrackPosAtLayer_[scIdx][layerId] = {{x,y,z}};
        }

        void create2dSimClusters(bool distanceFilter = false, float maxDistance2 = 0.f)
        {
            unsigned int numberOfHits = layerId_.size();
            for(unsigned int hitId = 0; hitId< numberOfHits; ++hitId)
            {
                unsigned int numberOfClusters = MCAssociatedSimCluster_[hitId].size();
                unsigned int layer = layerId_[hitId];
                distance2FromOriSimTrack_.resize(numberOfClusters);
                for(unsigned int clId = 0; clId < numberOfClusters; ++clId )
                {
                    distance2FromOriSimTrack_[hitId][clId] = distance2FromSimTrack(hitId, clId);
                    if(distanceFilter)
                    {
                        if ( distance2FromOriSimTrack_[hitId][clId]< maxDistance2)
                        {
                            clusters2D_[clId][layer].addHitAndEnergy(hitId, MCEnergyFraction_[hitId][clId]*totalEnergy_[hitId]);
                        }
                    }
                    else
                    {
                        clusters2D_[clId][layer].addHitAndEnergy(hitId, MCEnergyFraction_[hitId][clId]*totalEnergy_[hitId]);
                    }
                }
            }
        }

        float distanceFromSimTrack(unsigned int hitId, unsigned int clusterId)
        {
            auto l = layerId_[hitId];
            auto& simTrackPosition = simTrackPosAtLayer_[clusterId][l];
            float distanceSquared = std::pow((hitPosition_[hitId][0] - simTrackPosition[0]),2) + std::pow((hitPosition_[hitId][1] - simTrackPosition[1]),2);
            return std::sqrt(distanceSquared);
        }

        float distance2FromSimTrack(unsigned int hitId, unsigned int clusterId)
        {
            auto l = layerId_[hitId];
            auto& simTrackPosition = simTrackPosAtLayer_[clusterId][l];
            float distanceSquared = std::pow((hitPosition_[hitId][0] - simTrackPosition[0]),2) + std::pow((hitPosition_[hitId][1] - simTrackPosition[1]),2);
            return distanceSquared;
        }

        void computeAssociation(float energyHalfDistance2)
        {
            //if more than 90% of a hit's energy belongs to a cluster, that rechit is not counted as shared
            constexpr float exclusiveFraction = 0.7;
            unsigned int numberOfHits = layerId_.size();
            std::vector<float> partialEnergies;

            for(unsigned int hitId = 0; hitId < numberOfHits; ++hitId)
            {
                partialEnergies.clear();
                unsigned int numberOfClusters = MCAssociatedSimCluster_[hitId].size();
                unsigned int layer = layerId_[hitId];

                HitToRealisticSimCluster_[hitId].resize(numberOfClusters);
                HitToRealisticEnergyFraction_[hitId].resize(numberOfClusters);
                partialEnergies.resize(numberOfClusters);

                float sumE=0.f;
                for(unsigned int clId = 0; clId < numberOfClusters; ++clId )
                {
                    partialEnergies[clId] = clusters2D_[clId][layer].getLayerEnergy() * std::exp(-distance2FromOriSimTrack_[hitId][clId]/energyHalfDistance2) ;
                    sumE += partialEnergies[clId];
                }

                for(unsigned int clId = 0; clId < numberOfClusters; ++clId )
                {
                    unsigned int simClusterIndex = MCAssociatedSimCluster_[hitId][clId];
                    HitToRealisticSimCluster_[hitId][clId] = simClusterIndex;
                    float assignedFraction = partialEnergies[clId]/sumE;
                    HitToRealisticEnergyFraction_[hitId][clId] = assignedFraction;
                    float assignedEnergy = assignedFraction * sumE;
                    RealisticSimClusters_[simClusterIndex].increaseEnergy(assignedEnergy);
                    RealisticSimClusters_[simClusterIndex].addHitAndFraction(hitId, assignedFraction);
                    if(assignedFraction > exclusiveFraction)
                    {
                        RealisticSimClusters_[simClusterIndex].increaseExclusiveEnergy(assignedEnergy);
                    }
                }
            }
        }


        void findAndMergeInvisibleClusters()
        {
            unsigned int numberOfRealSimClusters = RealisticSimClusters_.size();
            const constexpr float invisibleFraction = 0.2;
            for(unsigned int clId= 0; clId <  numberOfRealSimClusters; ++clId)
            {

                if(RealisticSimClusters_[clId].getExclusiveEnergyFraction() < invisibleFraction)
                {
                    RealisticSimClusters_[clId].setVisible(false);
                    auto& hAndF = RealisticSimClusters_[clId].hitIdsAndFractions;
                    while (!hAndF.empty())
                    {
                        unsigned int hitId = hAndF.back().first;
                        float fraction = hAndF.back().second;
                        float correction = 1.f/(1.f-fraction);

// we have to reloop again over the hits and reassign them to the remaining clusters
                        unsigned int numberOfClusters = HitToRealisticSimCluster_[hitId].size();
                        for(unsigned int i = 0; i< numberOfClusters; ++i)
                        {
                            if(HitToRealisticSimCluster_[hitId][i] != clId && RealisticSimClusters_[HitToRealisticSimCluster_[hitId][i]].isVisible())
                            {
                                float oldEnergy = HitToRealisticEnergyFraction_[hitId][clId]*totalEnergy_[hitId];
                                HitToRealisticEnergyFraction_[hitId][clId]*=correction;

                                float newEnergy= HitToRealisticEnergyFraction_[hitId][clId]*totalEnergy_[hitId];
                                RealisticSimClusters_[HitToRealisticSimCluster_[hitId][i]].increaseEnergy(newEnergy-oldEnergy);
                                RealisticSimClusters_[HitToRealisticSimCluster_[hitId][i]].modifyFractionForHitId(HitToRealisticEnergyFraction_[hitId][clId], hitId);
                            }
                        }

                        hAndF.pop_back();
                    }

                }
            }
        }

        const std::vector< RealisticCluster > & realisticClusters() const { return RealisticSimClusters_; }

        private:

        std::vector<Hit3DPosition> hitPosition_;
        std::vector<float> totalEnergy_;
        std::vector<unsigned int> layerId_;

        // MC association: for each hit, the indices of the SimClusters and their contributed
        // fraction to the energy of the hit is stored
        std::vector< std::vector<unsigned int> > MCAssociatedSimCluster_;
        std::vector< std::vector<float> > MCEnergyFraction_;
        // For each hit, the squared distance from the propagated simTrack to the layer is calculated for every SimCluster associated
        std::vector< std::vector<float> > distance2FromOriSimTrack_;

        // each Simcluster is split in 2d clusters
        std::vector< std::vector<HGCSimCluster2D> > clusters2D_;

        // for each SimCluster and for each layer, we store the position of the propagated simTrack onto the layer
        std::vector< std::vector<Hit3DPosition> > simTrackPosAtLayer_;

        // Realistic association: for each hit, the indices of the RealisticClusters and their contributed
        // fraction to the energy of the hit is stored
        // There is one to one association between these realistic simclusters and simclusters
        std::vector< std::vector<unsigned int> > HitToRealisticSimCluster_;
        // for each hit, fractions of the energy associated to a realistic simcluster are computed
        std::vector< std::vector<float> > HitToRealisticEnergyFraction_;

        // for each RealisticSimCluster, the ids of the RealisticSimClusters with whom the cluster is sharing
        // energy and the amount of energy shared is computed
        std::vector< std::vector<unsigned int> > NeighboringSimClusters_;

        // the vector of the Realistic SimClusters
        std::vector< RealisticCluster > RealisticSimClusters_;


};

#endif
