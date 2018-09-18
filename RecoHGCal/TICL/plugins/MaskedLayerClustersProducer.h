#ifndef __RecoHGCal_TICL_MaskedLayerClustersProducer_H__
#define __RecoHGCal_TICL_MaskedLayerClustersProducer_H__

class MaskedLayerClustersProducer : public edm::stream::EDProducer<> {
public:
  MaskedLayerClustersProducer(const edm::ParameterSet &);
  ~MaskedLayerClustersProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void produce(edm::Event &, const edm::EventSetup &) override;

  std::vector<std::pair<unsigned int, float> > maskLayerClusters(const std::vector<reco::BasicCluster>&,
      const std::vector<float>& clustersMask, bool isFirstIteration);

private:
  edm::EDGetTokenT<std::vector<reco::BasicCluster>> clusters_token;
  edm::EDGetTokenT<std::vector<float>> clustersMask_token;
  const bool isFirstIteration_;
};


#endif
