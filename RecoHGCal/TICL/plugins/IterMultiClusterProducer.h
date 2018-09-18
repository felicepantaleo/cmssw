#ifndef __RecoHGCal_TICL_IterMultiClusterProducer_H__
#define __RecoHGCal_TICL_IterMultiClusterProducer_H__

#include "DataFormats/Math/interface/Point3D.h"


class IterMultiClusterProducer : public edm::stream::EDProducer<> {
public:
  IterMultiClusterProducer(const edm::ParameterSet &);
  ~IterMultiClusterProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void produce(edm::Event &, const edm::EventSetup &) override;

  template<class T>
  std::vector<unsigned int> filterLayerClustersBySeeding(const T&, const std::vector<reco::BasicCluster>& , const std::vector<bool>& );


  std::vector<unsigned int> filterLayerClustersByPosition(const std::vector<std::array<math::XYZPoint, 2> >&, const std::vector<reco::BasicCluster>&, const std::vector<bool>& );
private:
  edm::EDGetTokenT<HGCRecHitCollection> hits_fh_token;
  edm::EDGetTokenT<HGCRecHitCollection> hits_ee_token;
  edm::EDGetTokenT<HGCRecHitCollection> hits_bh_token;
  edm::EDGetTokenT<std::vector<reco::BasicCluster>> clusters_token;
  edm::EDGetTokenT<std::vector<bool>> clustersMask_token;

  std::unique_ptr<HGCal3DClustering> multicluster_algo;
};


#endif
