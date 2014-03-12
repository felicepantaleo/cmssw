#include <memory>
#include <set>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexSmoother.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"


class TrackCollectionCleaner : public edm::EDProducer {
    public:
	TrackCollectionCleaner(const edm::ParameterSet &params);

	virtual void produce(edm::Event &event, const edm::EventSetup &es);

    private:
	bool trackFilter(const reco::TrackRef &track) const;

	edm::InputTag				vertexCollection;
	edm::InputTag				trackCollection;
};

TrackCollectionCleaner::TrackCollectionCleaner(const edm::ParameterSet &params) :
	vertexCollection(params.getParameter<edm::InputTag>("vertices")),
	trackCollection(params.getParameter<edm::InputTag>("tracks"))
{
	produces<reco::TrackCollection>();
}

void TrackCollectionCleaner::produce(edm::Event &event, const edm::EventSetup &es)
{
        using namespace reco;
	edm::Handle<VertexCollection> vertices;
	event.getByLabel(vertexCollection, vertices);
        std::set<reco::TrackRef> pvTracks;
        for(std::vector<reco::Vertex>::const_iterator pv = vertices->begin();pv != vertices->end(); ++pv)
        {
         for(std::vector<reco::TrackBaseRef>::const_iterator iter = pv->tracks_begin();
            iter != pv->tracks_end(); iter++) {
                if (pv->trackWeight(*iter) >= 0.)
                        pvTracks.insert(iter->castTo<reco::TrackRef>());
         }
	}

	edm::Handle<TrackCollection> tracks;
	event.getByLabel(trackCollection, tracks);

	std::auto_ptr<TrackCollection> recoTracks(new TrackCollection);
	int i=0;
	for(std::vector<reco::Track>::const_iterator tr = tracks->begin();  tr != tracks->end(); ++tr,i++) {
		reco::TrackRef tref(tracks,i);
            	if(pvTracks.count(tref) < 0.5 ) {
		      recoTracks->push_back(*tref);
                }
        }


	

	event.put(recoTracks);
}
DEFINE_FWK_MODULE(TrackCollectionCleaner);
