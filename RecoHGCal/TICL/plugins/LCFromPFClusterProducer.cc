#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

class LCfromPFClusterProducer : public edm::stream::EDProducer<> {
public:
  LCfromPFClusterProducer(const edm::ParameterSet&);
  ~LCfromPFClusterProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<std::vector<reco::PFCluster>> ecalpfcluster_token_, hcalpfcluster_token_;
};

LCfromPFClusterProducer::LCfromPFClusterProducer(const edm::ParameterSet& ps) {
  ecalpfcluster_token_ = consumes<std::vector<reco::PFCluster>>(ps.getParameter<edm::InputTag>("ecalpfclusters")),
  hcalpfcluster_token_ = consumes<std::vector<reco::PFCluster>>(ps.getParameter<edm::InputTag>("hcalpfclusters"));

  produces<std::vector<reco::CaloCluster>>();
}

void LCfromPFClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<std::vector<reco::PFCluster>> ecalpfcluster_h, hcalpfcluster_h;
  evt.getByToken(ecalpfcluster_token_, ecalpfcluster_h);
  const auto ecalpfclusters = *ecalpfcluster_h;

  evt.getByToken(hcalpfcluster_token_, hcalpfcluster_h);
  const auto hcalpfclusters = *hcalpfcluster_h;

  auto clusters = std::make_unique<std::vector<reco::CaloCluster>>();

  for (const auto& pfcl : ecalpfclusters) {
    reco::CaloCluster calocluster = pfcl;
    clusters->push_back(calocluster);
  }

  for (const auto& pfcl : hcalpfclusters) {
    reco::CaloCluster calocluster = pfcl;
    if (calocluster.seed().subdetId() == HcalEndcap)
      continue;
    clusters->push_back(calocluster);
  }

  evt.put(std::move(clusters));
}

void LCfromPFClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("ecalpfclusters", edm::InputTag("particleFlowClusterECAL"));
  desc.add<edm::InputTag>("hcalpfclusters", edm::InputTag("particleFlowClusterHBHE"));
  descriptions.add("lcFromPFClusterProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LCfromPFClusterProducer);
