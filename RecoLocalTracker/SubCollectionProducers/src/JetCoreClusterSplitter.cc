#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

using namespace std;

class JetCoreClusterSplitter : public edm::EDProducer 
{

public:
  JetCoreClusterSplitter(const edm::ParameterSet& iConfig) ;
  ~JetCoreClusterSplitter() ;
  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) ;
  
private:
  std::string pixelCPE_; 
  edm::InputTag pixelClusters_;
  edm::InputTag vertices_;

};

JetCoreClusterSplitter::JetCoreClusterSplitter(const edm::ParameterSet& iConfig):
  pixelCPE_(iConfig.getParameter<std::string>("pixelCPE")),
  pixelClusters_(iConfig.getParameter<edm::InputTag>("pixelClusters")),
  vertices_(iConfig.getParameter<edm::InputTag>("vertices"))
{
  
  produces< edmNew::DetSetVector<SiPixelCluster> >();

}



JetCoreClusterSplitter::~JetCoreClusterSplitter()
{
}

void
JetCoreClusterSplitter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  edm::ESHandle<GlobalTrackingGeometry> geometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(geometry);
/*edm::ESHandle<TrackerGeometry> tracker;
iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
const TrackerGeometry * trackerGeometry = tracker.product();*/

  Handle<edmNew::DetSetVector<SiPixelCluster> > inputPixelClusters;
  iEvent.getByLabel(pixelClusters_, inputPixelClusters);
  Handle<std::vector<reco::Vertex> > vertices; 
  iEvent.getByLabel(vertices_, vertices);
  const reco::Vertex & pv = (*vertices)[0];
  Handle<std::vector<reco::CaloJet> > jets;
  iEvent.getByLabel("ak5CaloJets", jets);
  edm::ESHandle<PixelClusterParameterEstimator> pe; 
  const PixelClusterParameterEstimator * pp ;
  iSetup.get<TkPixelCPERecord>().get(pixelCPE_ , pe );  
  pp = pe.product();

  std::auto_ptr<edmNew::DetSetVector<SiPixelCluster> > output(new edmNew::DetSetVector<SiPixelCluster>());
  
  edmNew::DetSetVector<SiPixelCluster>::const_iterator detIt=inputPixelClusters->begin();
  for(;detIt!=inputPixelClusters->end();detIt++)
  {
	edmNew::DetSetVector<SiPixelCluster>::FastFiller filler(*output,detIt->id());
	const edmNew::DetSet<SiPixelCluster> & detset= *detIt;
        const GeomDet *det = geometry->idToDet( detset.id() );
	for(edmNew::DetSet<SiPixelCluster>::const_iterator cluster=detset.begin(); cluster!=detset.end(); cluster++)
	{
	        const SiPixelCluster & aCluster =  *cluster;
	        bool hasBeenSplit = false;
                GlobalPoint cPos = det->surface().toGlobal(pp->localParametersV( aCluster,( *geometry->idToDetUnit(detIt->id())))[0].first) ;
		GlobalPoint ppv(pv.position().x(),pv.position().y(),pv.position().z());
                GlobalVector clusterDir = cPos -ppv;
	        for(std::vector<reco::CaloJet>::const_iterator jit = jets->begin() ; jit != jets->end() ; jit++)
	        {
          	     if(jit->pt() > 100)
              		{
				 float jetZOverRho = jit->momentum().Z()/jit->momentum().Rho();
                 		 GlobalVector jetDir(jit->momentum().x(),jit->momentum().y(),jit->momentum().z());
				 unsigned int maxSizeY=fabs(jetZOverRho*1.9);
				 if(maxSizeY < 1) maxSizeY=1;
		  		 if(Geom::deltaR(jetDir,clusterDir) < 0.05 && aCluster.charge() > 30000 && (aCluster.sizeX() > 2 || ((unsigned int)aCluster.sizeY()) > maxSizeY+1) )
					{
                                                std::cout << "CHECK FOR SPLITTING: charge and deltaR " <<aCluster.charge() << " " << Geom::deltaR(jetDir,clusterDir) << " size x y"<< aCluster.sizeX()  << " " << aCluster.sizeY()<< " detid " << detIt->id() << std::endl;	
						std::cout << "Original Position " << cPos << std::endl;
						std::vector<SiPixelCluster::Pixel> pixels = aCluster.pixels();
						unsigned int isize=0;
						uint16_t  adcs[100];
				                uint16_t  xpos[100];
					        uint16_t  ypos[100]; 
					        uint16_t  xmin;
						uint16_t  ymin;
						unsigned int charge=0;
						SiPixelCluster * cluster = 0;
						for(unsigned int i = 0; i < pixels.size(); i++) 
						{
//							if(pixels[i].adc>10000)
//							filler.push_back(SiPixelCluster(SiPixelCluster::PixelPos(pixels[i].x,pixels[i].y),pixels[i].adc));
					  		std::cout << (int)pixels[i].x << " " <<  pixels[i].y  << " " << pixels[i].adc << std::endl;		
							//split if: too long or too much charge or too wide or gap in Y
							if(isize > maxSizeY+1 && charge > 10000)
//							if(isize > 0  && (pixels[i].y-ymin > maxSizeY || charge/maxSizeY > 20000 || pixels[i].x-xmin > 2 || (isize>0 && pixels[i].y-pixels[i-1].y>1 ) ))  
							{
									std::cout << "split!  isize:" << isize << " > " << maxSizeY+1 << " and charge : " << charge << " Z/rho: " << jetZOverRho <<  std::endl;
									if(charge < 10000) { std::cout << " to recover" << std::endl;   }
									else 
									{
								           hasBeenSplit=1; 
									   if(cluster){
									   filler.push_back(*cluster);
									   std::vector<SiPixelCluster::Pixel> pixels2 = cluster->pixels();
									   for(unsigned int i = 0; i < pixels2.size(); i++)
                				                           {
                                                				        std::cout << "    NC " <<  (int)pixels2[i].x << " " <<  pixels2[i].y  << " " << pixels2[i].adc << std::endl;
									   }	
									   std::cout << "  NC pos: " << det->surface().toGlobal(pp->localParametersV( *cluster,( *geometry->idToDetUnit(detIt->id())))[0].first) << std::endl;
									}

						
									}
									if(cluster) delete cluster;
									cluster =0;
									isize=0; charge=0;
							}
							if(isize==0) { xmin=pixels[i].x; ymin=pixels[i].y; }
							xpos[isize]=pixels[i].x;
							ypos[isize]=pixels[i].y;
							adcs[isize]=pixels[i].adc;
							charge+=pixels[i].adc;
							if(cluster==0) cluster = new SiPixelCluster(SiPixelCluster::PixelPos(pixels[i].x,pixels[i].y),pixels[i].adc);
							else cluster->add(SiPixelCluster::PixelPos(pixels[i].x,pixels[i].y),pixels[i].adc);
							isize++;

						}
						if(cluster){
							      if(hasBeenSplit)	filler.push_back(*cluster);
							      delete cluster;
							}
					}
 
			}
		}
		if(!hasBeenSplit) filler.push_back(aCluster);	
		
	}
  }
  iEvent.put(output);
}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetCoreClusterSplitter);
