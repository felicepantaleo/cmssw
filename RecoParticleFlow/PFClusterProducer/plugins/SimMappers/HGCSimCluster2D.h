#ifndef __RecoParticleFlow_PFClusterProducer_HGCSimCluster2D_H__
#define __RecoParticleFlow_PFClusterProducer_HGCSimCluster2D_H__

#include <array>
#include <vector>


class HGCSimCluster2D
{
    public:
        HGCSimCluster2D()
        {
            energyOnLayer_ = 0.f;
        }


        void addHitAndEnergy(unsigned int hitId, float energy)
        {
            hitsIdsOnLayer_.push_back(hitId);
            energyOnLayer_+= energy;
        }

        float getLayerEnergy() const
        {
            return energyOnLayer_;
        }


    private:
        float energyOnLayer_;
        std::vector<unsigned int> hitsIdsOnLayer_;


};

#endif
