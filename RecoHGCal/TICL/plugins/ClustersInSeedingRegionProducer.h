#ifndef __RecoHGCal_TICL_ClustersInSeedingRegionProducer_H__
#define __RecoHGCal_TICL_ClustersInSeedingRegionProducer_H__

class ClustersInSeedingRegionProducer : public edm::stream::EDProducer<> {
public:
  ClustersInSeedingRegionProducer(const edm::ParameterSet &);
  ~ClustersInSeedingRegionProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void produce(edm::Event &, const edm::EventSetup &) override;

  template<class T>
  std::vector<unsigned int> filterLayerClustersBySeeding(const T&, const std::vector<reco::BasicCluster>& , const std::vector<bool>& );


  std::vector<unsigned int> filterLayerClustersByPosition(const std::vector<std::array<math::XYZPoint, 2> >&, const std::vector<reco::BasicCluster>&, const std::vector<bool>& );
private:
  edm::EDGetTokenT<std::vector<reco::BasicCluster>> clusters_token;
  edm::EDGetTokenT<std::vector<bool>> clustersMask_token;

};


#endif
