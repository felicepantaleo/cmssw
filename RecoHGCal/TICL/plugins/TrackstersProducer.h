#ifndef __RecoHGCal_TICL_TrackstersProducer_H__
#define __RecoHGCal_TICL_TrackstersProducer_H__
#include "RecoHGCal/TICL/interface/PatternRecognitionAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

class TrackstersProducer : public edm::stream::EDProducer<> {
public:
  TrackstersProducer(const edm::ParameterSet &);
  ~TrackstersProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void produce(edm::Event &, const edm::EventSetup &) override;


private:

  edm::EDGetTokenT<PatternRecognitionAlgo> patterRecognitionAlgoToken;

  edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token;
  edm::EDGetTokenT<std::vector<std::pair<unsigned int, float>>> filteredClustersMask_token;

  const PatternRecognitionAlgo theAlgo;
};


#endif
