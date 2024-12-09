#ifndef IOMC_ParticleGun_ManyParticleFlatEtGunProducer_H
#define IOMC_ParticleGun_ManyParticleFlatEtGunProducer_H

#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"

#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"

namespace edm
{

  class ManyParticleFlatEtGunProducer : public BaseFlatGunProducer
  {

  public:
    ManyParticleFlatEtGunProducer(const ParameterSet &);
    ~ManyParticleFlatEtGunProducer() override;

  private:

    void produce(Event & e, const EventSetup& es) override;

  protected :
    // data members
    std::vector<double> vPtMax;
    std::vector<double> vPtMin;
    std::vector<double> vEtaMin;
    std::vector<double> vEtaMax;
    std::vector<double> vPhiMin;
    std::vector<double> vPhiMax;
    std::vector<int> vPartIDs;
  };
}

#endif
