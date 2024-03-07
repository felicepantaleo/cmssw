#include "RecoHGCal/TICL/plugins/TICLPFBlockAlgo.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"

/**\class TICLPFBlockProducer 
\brief Producer for particle flow blocks

This producer makes use of TICLPFBlockAlgo, the particle flow block algorithm.
Particle flow itself consists in reconstructing particles from the particle 
flow blocks This is done at a later stage, see PFProducer and PFAlgo.

\author Colin Bernet
\date   April 2007
*/

class FSimEvent;

class TICLPFBlockProducer : public edm::stream::EDProducer<> {
public:
  explicit TICLPFBlockProducer(const edm::ParameterSet&);

  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  /// verbose ?
  const bool verbose_;
  const edm::EDPutTokenT<reco::PFBlockCollection> putToken_;

  /// Particle flow block algorithm
  TICLPFBlockAlgo pfBlockAlgo_;
};

DEFINE_FWK_MODULE(TICLPFBlockProducer);

using namespace std;
using namespace edm;

TICLPFBlockProducer::TICLPFBlockProducer(const edm::ParameterSet& iConfig)
    : verbose_{iConfig.getUntrackedParameter<bool>("verbose", false)}, putToken_{produces<reco::PFBlockCollection>()} {
  bool debug_ = iConfig.getUntrackedParameter<bool>("debug", false);
  pfBlockAlgo_.setDebug(debug_);

  edm::ConsumesCollector cc = consumesCollector();
  const std::vector<edm::ParameterSet>& importers = iConfig.getParameterSetVector("elementImporters");
  pfBlockAlgo_.setImporters(importers, cc);

  const std::vector<edm::ParameterSet>& linkdefs = iConfig.getParameterSetVector("linkDefinitions");
  pfBlockAlgo_.setLinkers(linkdefs);
}

void TICLPFBlockProducer::beginLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& es) {
  pfBlockAlgo_.updateEventSetup(es);
}

void TICLPFBlockProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  pfBlockAlgo_.buildElements(iEvent);

  auto blocks = pfBlockAlgo_.findBlocks();

  if (verbose_) {
    ostringstream str;
    str << pfBlockAlgo_ << endl;
    str << "number of blocks : " << blocks.size() << endl;
    str << endl;

    for (auto const& block : blocks) {
      str << block << endl;
    }

    LogInfo("TICLPFBlockProducer") << str.str() << endl;
  }

  iEvent.emplace(putToken_, blocks);
}
