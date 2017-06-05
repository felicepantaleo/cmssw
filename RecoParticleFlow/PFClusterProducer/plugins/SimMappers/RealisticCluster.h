#ifndef __RecoParticleFlow_PFClusterProducer_RealisticCluster_H__
#define __RecoParticleFlow_PFClusterProducer_RealisticCluster_H__

#include <array>
#include <vector>


class RealisticCluster
{

    public:

        RealisticCluster()
        {
            totalEnergy = 0.f;
            exclusiveEnergy = 0.f;
            visible = true;
        }
        void increaseEnergy(float value)
        {
            totalEnergy+=value;
        }
        void increaseExclusiveEnergy(float value)
        {
            exclusiveEnergy+=value;
        }

        float getExclusiveEnergyFraction() const
        {
            return exclusiveEnergy/totalEnergy;
        }

        float getEnergy() const
        {
            return totalEnergy;
        }

        bool isVisible() const
        {
            return visible;
        }

        void setVisible(bool vis)
        {
            visible = vis;
        }

        void addHitAndFraction(unsigned int hit, float fraction)
        {
            hitIdsAndFractions.emplace_back(hit,fraction);
        }
//TODO: hits are sorted, replace with binary search.
        void modifyFractionForHitId(float fraction, unsigned int hitId)
        {

            auto it = std::find_if( hitIdsAndFractions.begin(), hitIdsAndFractions.end(),
                [&hitId](const std::pair<unsigned int, float>& element){ return element.first == hitId;} );

            it->second = fraction;

        }

        std::vector<std::pair<unsigned int, float> > hitIdsAndFractions;


    private:

        float totalEnergy;
        float exclusiveEnergy;
        bool visible;
};

#endif
