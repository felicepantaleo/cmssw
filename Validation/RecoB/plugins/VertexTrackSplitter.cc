#include <map>
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
//#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include <sstream>
#include <DataFormats/VertexReco/interface/Vertex.h>
#include <DataFormats/TrackReco/interface/Track.h>
//
// class decleration
//
using namespace reco;
using namespace std;
using namespace edm;
#include "DataFormats/GeometryVector/interface/VectorUtil.h"

class VertexTrackSplitter : public edm::EDProducer
{

	public:
		typedef math::XYZPoint 	Point;
		explicit VertexTrackSplitter(const edm::ParameterSet&);
		~VertexTrackSplitter();

	private:

		virtual void beginJob() ;		
		virtual void produce(edm::Event&, const edm::EventSetup&);
		virtual void endJob() ;
		// Member data

		edm::InputTag vertices_;
		unsigned int maxVertices_;

};


VertexTrackSplitter::VertexTrackSplitter(const edm::ParameterSet& config) : 
vertices_( config.getParameter<edm::InputTag>( "vertexInputTag" ) ),
maxVertices_(config.getUntrackedParameter<unsigned int> ( "max" ))
{

for(size_t i=0;i<maxVertices_;i++)
 {
  stringstream str;
  str << "vertex" << i ;	
  produces<reco::TrackCollection>(str.str().c_str());
 }
}

VertexTrackSplitter::~VertexTrackSplitter()
{
}
 

void VertexTrackSplitter::produce(edm::Event& event, const edm::EventSetup& setup)
{
   edm::Handle<std::vector<reco::Vertex> >  verticesH;
   event.getByLabel(vertices_, verticesH);
   const std::vector<reco::Vertex> & vertices = *(verticesH.product());
   
   auto_ptr<reco::TrackCollection> tracks[maxVertices_];
   for(size_t i=0;i<maxVertices_;i++)
   {
      tracks[i].reset(new reco::TrackCollection);
   }
   
   for(unsigned int  iv = 0; iv < vertices.size() && iv < maxVertices_ ; ++iv)
    {
	  for(reco::Vertex::trackRef_iterator it= vertices[iv].tracks_begin();it!= vertices[iv].tracks_end();it++)
          {
	   tracks[iv]->push_back(**it);
   	  }
   }
 for(size_t i=0;i<maxVertices_;i++)
 {
  stringstream str;
  str << "vertex" << i ;
  event.put(tracks[i], str.str().c_str());
 }

}

// ------------ method called once each job just after ending the event loop  ------------
void 
VertexTrackSplitter::beginJob() {

   std::cout << std::endl;
} 
void 
VertexTrackSplitter::endJob() {

   std::cout << std::endl;
 
} 

DEFINE_FWK_MODULE(VertexTrackSplitter);
