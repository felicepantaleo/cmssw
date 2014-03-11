#include <memory>
#include <set>


#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"


#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"


#include "DataFormats/GeometryVector/interface/VectorUtil.h"


//#define VTXDEBUG

class NuclearInteractionIdentifier : public edm::EDProducer {
    public:
	NuclearInteractionIdentifier(const edm::ParameterSet &params);


	virtual void produce(edm::Event &event, const edm::EventSetup &es);

    private:
	bool trackFilter(const reco::TrackRef &track) const;

	edm::InputTag				primaryVertexCollection;
	edm::InputTag				secondaryVertexCollection;
        edm::InputTag                           beamSpotCollection;
};

NuclearInteractionIdentifier::NuclearInteractionIdentifier(const edm::ParameterSet &params) :
	primaryVertexCollection      (params.getParameter<edm::InputTag>("primaryVertices")),
	secondaryVertexCollection    (params.getParameter<edm::InputTag>("secondaryVertices")),
        beamSpotCollection           (params.getParameter<edm::InputTag>("beamSpot"))
{
	produces<reco::VertexCollection>();
}


void NuclearInteractionIdentifier::produce(edm::Event &event, const edm::EventSetup &es)
{
	using namespace reco;

	edm::Handle<VertexCollection> secondaryVertices;
	event.getByLabel(secondaryVertexCollection, secondaryVertices);
        VertexCollection theSecVertexColl = *(secondaryVertices.product());

        edm::Handle<VertexCollection> primaryVertices;
        event.getByLabel(primaryVertexCollection, primaryVertices);

	std::auto_ptr<VertexCollection> recoVertices(new VertexCollection);

        if(primaryVertices->size()!=0){ 
        const reco::Vertex &pv = (*primaryVertices)[0];
    
        edm::Handle<BeamSpot> beamSpot;
        event.getByLabel(beamSpotCollection, beamSpot);

        for(unsigned int ivtx=0; ivtx < theSecVertexColl.size(); ivtx++){
	       const reco::Vertex & sv = theSecVertexColl[ivtx];	
	       float mass=sv.p4().M();
	       float pt=sv.p4().Pt();
	       float transverseBoost=pt/mass;
               GlobalPoint ppv(pv.position().x(),pv.position().y(),pv.position().z());
               GlobalPoint ssv(sv.position().x(),sv.position().y(),sv.position().z());
               GlobalPoint zero(0,0,0);
               GlobalVector flightDir = ssv-ppv;
               GlobalVector flightDir0 = ssv-zero;
	       float flightDistance2D = flightDir.perp(); 
	       float flightDistance2D0 = flightDir0.perp(); 
	       unsigned int ntracks = sv.nTracks();
	       float z=sv.position().z();
	       float nctau=flightDistance2D/transverseBoost/0.05;  //number of c*taus
	       float nctauK=flightDistance2D/transverseBoost/2.68;  //number of c*taus
               const reco::Vertex & extVertex = sv;
               GlobalVector vtxDir = GlobalVector(extVertex.p4().X(),extVertex.p4().Y(),extVertex.p4().Z());
	       float deltaR = Geom::deltaR(extVertex.position() - pv.position(), vtxDir);
	       float phi= extVertex.position().phi(); 
//               std::cout << "Vtx id: " << ivtx <<" pos: " << extVertex.position() << " fdist: " << flightDistance2D << " nctau: " << nctau << " ntr: " << ntracks << " mass: " << mass << " nctauK: " << nctauK << " DR: " << deltaR << " charge prod: "<< (*sv.tracks_begin())->charge()*((*(sv.tracks_begin()+1))->charge()) << std::endl;
              if( (fabs(z) < 29 &&  flightDistance2D0 > 2.5 && nctau > 3 )                      
                   or (fabs(z) < 29 &&  flightDistance2D0 > 2.0 && nctau > 5 && ntracks > 2 )
	           or ( flightDistance2D0 > 2.87 and flightDistance2D0 < 3.0 && nctau > 1.5 )
		   or (fabs(z) < 29 && flightDistance2D0 > 3.65 and flightDistance2D0 < 3.75 && nctau > 1.5)
		   or (fabs(z) < 29 && flightDistance2D0 > 4.1 and flightDistance2D0 < 4.25 && nctau > 1.5)
		   or (fabs(z) < 29 && flightDistance2D0 > 4.55 and flightDistance2D0 < 4.7 && nctau > 1.5)
		   or (fabs(z) < 29 && flightDistance2D0 > 7.05 and flightDistance2D0 < 7.15 && nctau > 1.5)
		   or (fabs(z) < 29 && flightDistance2D0 > 7.45 and flightDistance2D0 < 7.6 && nctau > 1.5)
		   or (fabs(z) < 29 && flightDistance2D0 > 9.85 and flightDistance2D0 < 10. && nctau > 1.5)
		   or (fabs(z) < 29 && flightDistance2D0 > 10.35 and flightDistance2D0 < 10.45 && nctau > 1.5)
                 or (fabs(z) < 29 &&  flightDistance2D0 > 2.0 && nctau > 5 && ntracks > 2 )
                   or (fabs(z) < 29 &&  flightDistance2D0 > 2.5 && flightDistance2D0 < 4 && nctau > 2 && ntracks >= 6 && mass > 4.2)
		   or (fabs(mass-0.497) < 0.0150 && ntracks == 2 && (*sv.tracks_begin())->charge()*((*(sv.tracks_begin()+1))->charge()) <0 && nctauK < 2 && nctauK > 0.01 && deltaR < 0.01) // K0... TODO: check also opposite sign
			) {
// 			std::cout << "NI " << std::endl;
		       recoVertices->push_back(theSecVertexColl[ivtx]);
 			}

        }

	for(unsigned int ivtx=0; ivtx < theSecVertexColl.size(); ivtx++){
		for(unsigned int ivtx2=0; ivtx2 < recoVertices->size(); ivtx2++){
			if(theSecVertexColl[ivtx].chi2() == (*recoVertices)[ivtx2].chi2()) continue;
			const reco::Vertex & sv = theSecVertexColl[ivtx];
			const reco::Vertex & sv2 = (*recoVertices)[ivtx2];
                        VertexDistance3D dist;
			Measurement1D d=dist.distance(sv,sv2);
			if(d.value() < 0.8)  std::cout << "Check " << d.significance() << " "<< d.value() << std::endl;

			if( (d.significance() < 10 and d.value() < 0.3) || (d.value() < 0.15 and d.significance() < 20)  )
			{
  				std::cout << "Is NI because is near an NI " << d.significance() << " "<< d.value() << std::endl;
				recoVertices->push_back(sv);
				break;
			}


		}
	}

	}	
	event.put(recoVertices);



}

DEFINE_FWK_MODULE(NuclearInteractionIdentifier);
