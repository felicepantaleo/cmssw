#ifndef __RecoParticleFlow_PFClusterProducer_RealisticHitToClusterAssociator_H__
#define __RecoParticleFlow_PFClusterProducer_RealisticHitToClusterAssociator_H__

#include <vector>
#include "HGCSimCluster2D.h"
#include "RealisticCluster.h"

namespace{
    float getDecayLength (unsigned int layer)
    {
        if (layer <= 28)
             return 2.f;
         if (layer > 28 && layer <= 40)
             return 1.5f;
         if (layer > 40)
             return 1.f;

         return 0.f;
    }


}


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
            distanceFromMaxHit_.resize(numberOfHits);
            clusters2D_.resize(numberOfSimClusters);
            maxHitPosAtLayer_.resize(numberOfSimClusters);
            maxEnergyHitAtLayer_.resize(numberOfSimClusters);
            for (unsigned int i = 0; i < numberOfSimClusters; ++i)
            {
                clusters2D_[i].resize(numberOfLayers);
                maxHitPosAtLayer_[i].resize(numberOfLayers);
                maxEnergyHitAtLayer_[i].resize(numberOfLayers, 0.f);

            }
            RealisticSimClusters_.resize(numberOfSimClusters);
        }

        void insertHitPosition(float x, float y, float z, unsigned int hitIndex)
        {
            hitPosition_.at(hitIndex) =
            {
                {   x,y,z}};

        }

        void insertLayerId(unsigned int layerId, unsigned int hitIndex)
        {
            layerId_.at(hitIndex) = layerId;
        }

        void insertHitEnergy(float energy, unsigned int hitIndex)
        {
            totalEnergy_.at(hitIndex) = energy;
        }

        void insertSimClusterIdAndFraction(unsigned int scIdx, float fraction,
                unsigned int hitIndex, float associatedEnergy)
        {
            MCAssociatedSimCluster_.at(hitIndex).push_back(scIdx);
            MCEnergyFraction_.at(hitIndex).push_back(fraction);
            auto layerId = layerId_.at(hitIndex);
            if(associatedEnergy > maxEnergyHitAtLayer_.at(scIdx).at(layerId))
            {
                maxHitPosAtLayer_.at(scIdx).at(layerId) = hitPosition_.at(hitIndex);
                maxEnergyHitAtLayer_.at(scIdx).at(layerId) = associatedEnergy;
            }
        }

        void create2dSimClusters(bool distanceFilter = false, float maxDistance = 0.f)
        {
            unsigned int numberOfHits = layerId_.size();
            std::cout<< "Creating 2dSimClusters with " << numberOfHits << " hits " << std::endl;

            for(unsigned int hitId = 0; hitId< numberOfHits; ++hitId)
            {
                unsigned int numberOfClusters = MCAssociatedSimCluster_.at(hitId).size();
                if(numberOfClusters>0)
                std::cout<< "hit " << hitId << " " << hitPosition_[hitId][0] << " " << hitPosition_[hitId][1] << " " <<
                hitPosition_[hitId][2] << " has  " << numberOfClusters << " clusters associated " << std::endl;

                unsigned int layer = layerId_.at(hitId);
                distanceFromMaxHit_.at(hitId).resize(numberOfClusters);
                for(unsigned int clId = 0; clId < numberOfClusters; ++clId )
                {
                    auto simClusterId = MCAssociatedSimCluster_.at(hitId).at(clId);
                    std::cout << "\t for a fraction " << MCEnergyFraction_.at(hitId).at(clId) << " to simClusterID " << simClusterId << " with maxHitEnergy on layer  " << layer << " " << maxEnergyHitAtLayer_.at(simClusterId).at(layer)
                    << " in position " << maxHitPosAtLayer_.at(simClusterId).at(layer).at(0) << " " << maxHitPosAtLayer_.at(simClusterId).at(layer).at(1) << " " <<
                    maxHitPosAtLayer_.at(simClusterId).at(layer).at(2) << " distance: "<< XYdistanceFromMaxHit(hitId, simClusterId) << std::endl;
                    if(maxEnergyHitAtLayer_.at(simClusterId).at(layer)>0.f)
                    {
                        distanceFromMaxHit_.at(hitId).at(clId) = XYdistanceFromMaxHit(hitId, simClusterId);
                        if(distanceFilter)
                        {

                            if ( distanceFromMaxHit_.at(hitId).at(clId)< maxDistance)
                            {
                                clusters2D_.at(simClusterId).at(layer).addHitAndEnergy(hitId, MCEnergyFraction_.at(hitId).at(clId)*totalEnergy_[hitId]);
                            }
                        }
                        else
                        {
                            clusters2D_.at(simClusterId).at(layer).addHitAndEnergy(hitId, MCEnergyFraction_.at(hitId).at(clId)*totalEnergy_[hitId]);
                        }
                    }

                }
            }
        }

        float XYdistanceFromMaxHit(unsigned int hitId, unsigned int clusterId)
        {
            auto l = layerId_[hitId];
            auto& maxHitPosition = maxHitPosAtLayer_[clusterId][l];
            float distanceSquared = std::pow((hitPosition_[hitId][0] - maxHitPosition[0]),2) + std::pow((hitPosition_[hitId][1] - maxHitPosition[1]),2);
            return std::sqrt(distanceSquared);
        }

        void computeAssociation(bool useMCFractionsForExclEnergy=false)
        {
            //if more than 90% of a hit's energy belongs to a cluster, that rechit is not counted as shared
            constexpr float exclusiveFraction = 0.7;
            unsigned int numberOfHits = layerId_.size();
            std::vector<float> partialEnergies;

            for(unsigned int hitId = 0; hitId < numberOfHits; ++hitId)
            {
                partialEnergies.clear();
                unsigned int numberOfClusters = MCAssociatedSimCluster_[hitId].size();
                if(numberOfClusters>1)
                {
                    std::cout<< "computeAssociation: hit " << hitId << " " << hitPosition_[hitId][0] << " " << hitPosition_[hitId][1] << " " <<
                    hitPosition_[hitId][2] << " has  " << numberOfClusters << " clusters associated " << std::endl;
                }
                unsigned int layer = layerId_[hitId];
                HitToRealisticSimCluster_.at(hitId).resize(numberOfClusters);
                HitToRealisticEnergyFraction_.at(hitId).resize(numberOfClusters);
                partialEnergies.resize(numberOfClusters);
                float energyDecayLength = getDecayLength(layer);
                float sumE = 0.f;

                for(unsigned int clId = 0; clId < numberOfClusters; ++clId )
                {

                    auto simClusterId = MCAssociatedSimCluster_.at(hitId).at(clId);
//                    partialEnergies[clId] = clusters2D_.at(simClusterId).at(layer).getLayerEnergy() * std::exp(-distanceFromMaxHit_.at(hitId).at(clId)/energyDecayLength);
                    partialEnergies[clId] = 0.001f+maxEnergyHitAtLayer_[simClusterId][layer] * std::exp(-distanceFromMaxHit_.at(hitId).at(clId)/energyDecayLength);

                    sumE += partialEnergies[clId];
                    if(numberOfClusters>1)
                    std::cout << "\t\tcluster "<< simClusterId << " has partial energy: " <<partialEnergies[clId] << " on layer " << layer << std::endl;

                }
                if(sumE != 0.f)
                {
                    float invSumE = 1.f/sumE;
                    for(unsigned int clId = 0; clId < numberOfClusters; ++clId )
                    {

                        unsigned int simClusterIndex = MCAssociatedSimCluster_[hitId][clId];
                        HitToRealisticSimCluster_[hitId][clId] = simClusterIndex;
                        float assignedFraction = partialEnergies[clId]*invSumE;
                        HitToRealisticEnergyFraction_[hitId][clId] = assignedFraction;
                        float assignedEnergy = assignedFraction * sumE;
                        RealisticSimClusters_[simClusterIndex].increaseEnergy(assignedEnergy);
                        RealisticSimClusters_[simClusterIndex].addHitAndFraction(hitId, assignedFraction);
                        if(numberOfClusters>1)
                        std::cout << "\t for a MC fraction " << MCEnergyFraction_.at(hitId).at(clId) << " realistic fraction " <<
                                assignedFraction << "  to simClusterID " << simClusterIndex << " with maxHitEnergy on layer  " <<
                                layer << " " << maxEnergyHitAtLayer_.at(simClusterIndex).at(layer)
                                            << " in position " << maxHitPosAtLayer_.at(simClusterIndex).at(layer).at(0) <<
                                            " " << maxHitPosAtLayer_.at(simClusterIndex).at(layer).at(1) << " " <<
                                            maxHitPosAtLayer_.at(simClusterIndex).at(layer).at(2) << " distance: "<<
                                            XYdistanceFromMaxHit(hitId, simClusterIndex) << std::endl;
                        bool isExclusive = ((!useMCFractionsForExclEnergy && (assignedFraction > exclusiveFraction) )
                                || (useMCFractionsForExclEnergy && (MCEnergyFraction_.at(hitId).at(clId) >exclusiveFraction )));
                        if(isExclusive)
                        {
                            RealisticSimClusters_[simClusterIndex].increaseExclusiveEnergy(assignedEnergy);
                        }

                    }
                }
            }
        }

        void findAndMergeInvisibleClusters()
        {
            unsigned int numberOfRealSimClusters = RealisticSimClusters_.size();
            const constexpr float invisibleFraction = 0.2f;
            for(unsigned int clId= 0; clId < numberOfRealSimClusters; ++clId)
            {

                if(RealisticSimClusters_[clId].getExclusiveEnergyFraction() < invisibleFraction)
                {
                    std::cout << "cluster " << clId << " is invisible!" << std::endl;
                    RealisticSimClusters_[clId].setVisible(false);
                    auto hAndF = RealisticSimClusters_[clId].hitsIdsAndFractions();
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
                                std::cout << "\t its energy is being shared with cluster " << HitToRealisticSimCluster_[hitId][i] << std::endl;
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

        const std::vector< RealisticCluster > & realisticClusters() const
        {   return RealisticSimClusters_;}

        private:

        std::vector<Hit3DPosition> hitPosition_;
        std::vector<float> totalEnergy_;
        std::vector<unsigned int> layerId_;

        // MC association: for each hit, the indices of the SimClusters and their contributed
        // fraction to the energy of the hit is stored
        std::vector< std::vector<unsigned int> > MCAssociatedSimCluster_;
        std::vector< std::vector<float> > MCEnergyFraction_;
        // For each hit, the squared distance from the propagated simTrack to the layer is calculated for every SimCluster associated
        std::vector< std::vector<float> > distanceFromMaxHit_;

        // each Simcluster is split in 2d clusters
        std::vector< std::vector<HGCSimCluster2D> > clusters2D_;

        // for each SimCluster and for each layer, we store the position of the most energetic hit of the simcluster in the layer
        std::vector< std::vector<Hit3DPosition> > maxHitPosAtLayer_;
        std::vector< std::vector<float> > maxEnergyHitAtLayer_;
        // Realistic association: for each hit, the indices of the RealisticClusters and their contributed
        // fraction to the energy of the hit is stored
        // There is one to one association between these realistic simclusters and simclusters
        std::vector< std::vector<unsigned int> > HitToRealisticSimCluster_;
        // for each hit, fractions of the energy associated to a realistic simcluster are computed
        std::vector< std::vector<float> > HitToRealisticEnergyFraction_;

        // the vector of the Realistic SimClusters
        std::vector< RealisticCluster > RealisticSimClusters_;

    };

#endif
