#include <map>
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"
#include "SimDataFormats/JetMatching/interface/JetFlavour.h"



//
// class decleration
//
using namespace reco;
using namespace std;
using namespace edm;
#include "DataFormats/GeometryVector/interface/VectorUtil.h"

class BTagAndFlavourFilter : public edm::EDFilter
{
  struct JetRefCompare :
       public std::binary_function<edm::RefToBase<reco::Jet>, edm::RefToBase<reco::Jet>, bool> {
    inline bool operator () (const edm::RefToBase<reco::Jet> &j1,
                             const edm::RefToBase<reco::Jet> &j2) const
    { return j1.id() < j2.id() || (j1.id() == j2.id() && j1.key() < j2.key()); }
  };
  typedef std::map<edm::RefToBase<reco::Jet>, unsigned int, JetRefCompare> FlavourMap;

	public:
		explicit BTagAndFlavourFilter(const edm::ParameterSet&);
		~BTagAndFlavourFilter();

	private:

		virtual void beginJob() ;		
		virtual bool filter(edm::Event&, const edm::EventSetup&);
		virtual void endJob() ;
	        edm::InputTag jetMCSrc;
	        edm::InputTag jetTagInputTag;		
		double threshold;
		double ptThreshold;
		bool isSignal;

};


BTagAndFlavourFilter::BTagAndFlavourFilter(const edm::ParameterSet& pSet)  :
  jetMCSrc(pSet.getParameter<edm::InputTag>("jetMCSrc")),
  jetTagInputTag(pSet.getParameter<edm::InputTag>("jetTag")),
  threshold(pSet.getParameter<double>("minDiscr")),
  ptThreshold(pSet.getParameter<double>("minPt")),
  isSignal(pSet.getParameter<bool>("isSignal"))
{

}

BTagAndFlavourFilter::~BTagAndFlavourFilter()
{
}
 

bool BTagAndFlavourFilter::filter(edm::Event& event, const edm::EventSetup& setup)
{


  edm::Handle<JetFlavourMatchingCollection> jetMC;
  FlavourMap flavours;
  event.getByLabel(jetMCSrc, jetMC);
  for (JetFlavourMatchingCollection::const_iterator iter = jetMC->begin();
       iter != jetMC->end(); ++iter) {
    unsigned int fl = std::abs(iter->second.getFlavour());
    flavours.insert(std::make_pair(iter->first, fl));
  }

    edm::Handle<reco::JetTagCollection> tagHandle;
    event.getByLabel(jetTagInputTag, tagHandle);
    const reco::JetTagCollection & tagColl = *(tagHandle.product());
    for (JetTagCollection::const_iterator tagI = tagColl.begin();
         tagI != tagColl.end(); ++tagI) {
	if(!isSignal){
	      if( (flavours[tagI->first] == 1 || flavours[tagI->first] == 2 || flavours[tagI->first] == 3 || flavours[tagI->first] == 21 ) && tagI->second > threshold && tagI->first->pt() > ptThreshold) 
	       {
		std::cout << "Jet number " << tagI->first.key() << " pt  " << tagI->first->pt() << " discr " << tagI->second << std::endl;  
               return true;
        	}
	} else {
	      if (flavours[tagI->first] == 5 && tagI->second < threshold && tagI->first->pt() > ptThreshold)
               {
                std::cout << "BJet number " << tagI->first.key() << " pt  " << tagI->first->pt() << " discr " << tagI->second <<  std::endl;
               return true;
                }

	}
    }

 return false;

}

// ------------ method called once each job just before starting event loop  ------------
void 
BTagAndFlavourFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
BTagAndFlavourFilter::endJob() {

   std::cout << std::endl;
 
} 

DEFINE_FWK_MODULE(BTagAndFlavourFilter);
