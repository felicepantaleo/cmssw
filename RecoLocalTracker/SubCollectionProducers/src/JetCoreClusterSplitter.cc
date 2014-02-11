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

#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelArrayBuffer.h"
#include <stack>
using namespace std;
	template < typename T >
std::ostream& bin(T& value, std::ostream &o)
{
	for ( T bit = 16; bit; bit >>= 1 )
	{
		o << ( ( value & bit ) ? '1' : '0' );
	}
	return o;
}


class JetCoreClusterSplitter : public edm::EDProducer 
{

	public:
		JetCoreClusterSplitter(const edm::ParameterSet& iConfig) ;
		~JetCoreClusterSplitter() ;
		void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) ;

	private:
		bool split(const SiPixelCluster & aCluster, edmNew::DetSetVector<SiPixelCluster>::FastFiller & filler, float expectedADC,int sizeY);
		void print(const SiPixelArrayBuffer & b, const SiPixelCluster & aCluster );
		std::vector<SiPixelCluster> fittingSplit(const SiPixelCluster & aCluster, float expectedADC,int sizeY);

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
	Handle<edmNew::DetSetVector<SiPixelCluster> > inputPixelClustersIDEAL;
	iEvent.getByLabel("IdealsiPixelClusters", inputPixelClustersIDEAL);

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
						std::cout << "CHECK FOR OLD SPLITTING: charge and deltaR " <<aCluster.charge() << " " << Geom::deltaR(jetDir,clusterDir) << " size x y"<< aCluster.sizeX()  << " " << aCluster.sizeY()<< " detid " << detIt->id() << std::endl;	
						if(split(aCluster,filler,sqrt(1.08+jetZOverRho*jetZOverRho)*26000,maxSizeY)) hasBeenSplit=true;
						std::cout << "IDEAL was : "  << std::endl; 
						int xmin=aCluster.minPixelRow();                        
						int ymin=aCluster.minPixelCol();                                
						int xmax=aCluster.maxPixelRow();                                
						int ymax=aCluster.maxPixelCol(); 
						int last=1;
						std::map<int,int> sh;  
						SiPixelClusterCollectionNew::const_iterator myDet =  inputPixelClustersIDEAL->find(detIt->id());
						for(int x=xmin; x<= xmax;x++){                                          
							for(int y=ymin; y<= ymax;y++)                                   
							{                                                                       
								int h=0;
								int flag=0;                                           
								for(edmNew::DetSet<SiPixelCluster>::const_iterator clusterIt = myDet->begin(); clusterIt != myDet->end() ; clusterIt++,h++)
								{

									std::vector<SiPixelCluster::Pixel> pixels = clusterIt->pixels();
									for(unsigned int j = 0; j < pixels.size(); j++)
									{                               
										if(pixels[j].x==x && pixels[j].y==y){
										 if(!sh[h]) {sh[h]=last; last++; }
										 flag|=(1<<(sh[h]-1));
										}
									}                               
								}                                       

								std::cout << " " ;  bin( flag,std::cout) ; 
								// std::setiosflags(std::ios::fixed)
								//                                << std::setprecision(0)
								//                              << std::setw(7)
								//                            << std::left ; bin( flag,std::cout);
								//                                << std::left << hex << flag;


							}
							std::cout << std::endl;         
						}
						int h=0;
						 for(edmNew::DetSet<SiPixelCluster>::const_iterator clusterIt = myDet->begin(); clusterIt != myDet->end() ; clusterIt++,h++)
                                                                {

									if(sh[h]) std::cout << "IDEAL POS: " << h << " x: "  << std::setprecision(2) << clusterIt->x() << " y: " << clusterIt->y() << " c: " << clusterIt->charge() << std::endl;
                                                                }
                       




#ifdef aDUMB_ALGORITHM
						std::cout << "Original Position " << cPos << std::endl;
						std::vector<SiPixelCluster::Pixel> pixels = aCluster.pixels();
						unsigned int isize=0;
						uint16_t  adcs[100];
						uint16_t  xpos[100];
						uint16_t  ypos[100]; 
						uint16_t  xmin;
						uint16_t  ymin;
						JetCoreClusterSplitter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
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
#endif


					}

				}
			}
			if(!hasBeenSplit)
			{
				//blowup the error
				SiPixelCluster c=aCluster;
//				c.setSplitClusterErrorX(c.sizeX()*100./3.);
//				c.setSplitClusterErrorY(c.sizeY()*150./3.);
				 filler.push_back(c);	
			}

		}
	}
	iEvent.put(output);
}



bool JetCoreClusterSplitter::split(const SiPixelCluster & aCluster, edmNew::DetSetVector<SiPixelCluster>::FastFiller & filler, float expectedADC,int sizeY)
{
	std::vector<SiPixelCluster> sp=fittingSplit(aCluster,expectedADC,sizeY);
	
//	std::multimap<float, std::pair<int,int> > distances;
//	std::map<int,bool> usedReco;
//	std::map<int,bool> usedIdeal;
//	for()

	

	for(unsigned int i = 0; i < sp.size();i++ )
	{
		std::cout << "NEW POS: " << i << " x: "  << std::setprecision(2) << sp[i].x() << " y: " << sp[i].y() << " c: " << sp[i].charge() << std::endl;
		filler.push_back(sp[i]);
	}




	int xmin=aCluster.minPixelRow();
	int ymin=aCluster.minPixelCol();
	int xmax=aCluster.maxPixelRow();
	int ymax=aCluster.maxPixelCol();
	std::cout << "Splitted clusters map:" << std::endl;
	for(int x=xmin; x<= xmax;x++){
		for(int y=ymin; y<= ymax;y++)
		{
			int flag=0;
			for(unsigned int i = 0; i < sp.size();i++ ){

				std::vector<SiPixelCluster::Pixel> pixels = sp[i].pixels();
				for(unsigned int j = 0; j < pixels.size(); j++)
				{
					if(pixels[j].x==x && pixels[j].y==y) flag|=(1<<i);
				}	
			}	

			std::cout << " " ;  bin( flag,std::cout) ; 
			// std::setiosflags(std::ios::fixed)
			//                                << std::setprecision(0)
			//                              << std::setw(7)
			//                            << std::left ; bin( flag,std::cout);
			//                                << std::left << hex << flag;


		}
		std::cout << std::endl;
	}
	return (sp.size() > 0);
}

std::vector<SiPixelCluster> JetCoreClusterSplitter::fittingSplit(const SiPixelCluster & aCluster, float expectedADC,int sizeY)
{
	unsigned int expectedClusters = floor(aCluster.charge() / expectedADC +0.5) ;
	std::vector<SiPixelCluster> output;
	if(expectedClusters==1) {
		output.push_back(aCluster);	
		return output;
	}	
	int xmin=aCluster.minPixelRow();
	int ymin=aCluster.minPixelCol();
	int xmax=aCluster.maxPixelRow();
	int ymax=aCluster.maxPixelCol();
	if(expectedClusters > 5) expectedClusters=5;
//return std::vector<SiPixelCluster>(); 
	std::vector<SiPixelCluster::Pixel> pixels = aCluster.pixels();

	int deltax=xmax-xmin+1;
	int deltay=ymax-ymin-sizeY+1;
	if (deltay < 1) deltay=1;
	float perPixel=0.25*expectedADC/(0.5+sizeY);
	int npos= pixels.size();
	//	unsigned long maxComb=pow(expectedClusters, npos);
	unsigned long maxComb=pow(npos,expectedClusters);
	float chimin=1e99;
	unsigned long  bestcomb=0;
	std::cout << "Sizey and perPixel " << sizeY << " " << perPixel << std::endl;
	if(maxComb > 100000) {std::cout << "toomany" << std::endl; return std::vector<SiPixelCluster>();
	}
	
        SiPixelArrayBuffer               theOriginalBuffer;
 	theOriginalBuffer.setSize(500,500);
        for(unsigned int i = 0; i < pixels.size(); i++)
                {
                        int x=pixels[i].x ;
                        int y=pixels[i].y ;
                        int adc=pixels[i].adc;
                        theOriginalBuffer.set_adc( x, y, adc);
                }


	for(unsigned long combination=0;combination<maxComb;combination++)	
	{
		float chi2=0;
		int clbase=1;
		SiPixelArrayBuffer               theBuffer;
		theBuffer.setSize(500,500);

		for(unsigned int i = 0; i < pixels.size(); i++)
		{
			int x=pixels[i].x ;
			int y=pixels[i].y ;
			int adc=pixels[i].adc;
			theBuffer.set_adc( x, y, adc);
		}
		//print(theBuffer,aCluster);

		//std::cout << "Combination " << combination << std::endl;
		for(unsigned int cl=0;cl<expectedClusters;cl++){
			int pi=((combination / clbase)%npos);
			int clx=pixels[pi].x;
			int cly=pixels[pi].y;
		//	std::cout << "Cluster  "<< cl << " x,y " << clx -xmin <<", " << cly-ymin<< " clbase " << clbase << std::endl;
			clbase*=npos;
			for(int x=xmin-1;x<=xmax+1;x++) {
				for(int y=ymin-1;y<=ymax+1;y++) {
					if(x<0||y<0) continue;
					int fact=0;
					if(x==clx) fact=2;
					if(x+1==clx || x-1==clx) fact=1;
					if(!(y>=cly && y < cly+sizeY)) fact=0;
					if(x==clx && (y==cly-1 || y == cly+sizeY)) fact=1;
					theBuffer.set_adc(x,y,theBuffer(x,y)-fact*perPixel);
					//std::cout << "residual in "<< x-xmin <<","<< y-ymin<< " " << res << "  fact " << fact << " exp:"<< fact*perPixel <<std::endl;
				}
			}
		}
		//print(theBuffer,aCluster);
		for(int x=xmin-1;x<=xmax+1;x++) {
			for(int y=ymin-1;y<=ymax+1;y++) {
				//				std::cout << theBuffer(x,y)/1000 << " " ;
				float res=theBuffer(x,y);
				float charge=theOriginalBuffer(x,y)-theBuffer(x,y); //charge assigned to this pixel
				if(res< 0 ) { //threshold effect
					if(res > -10000){
						if(res<-5000) res+=5000;
						else  res=0;
					}
				}
				if(res> 0 && charge > 7000 ) { //reduce weights of landau tails
					res*=0.7;
				}


				chi2+=res*res;
				//std::cout << "chi2 " << chi2 << " xy" << x<< " " << y << " res,res2,charge" << res << " ," <<res*res<< ", " <<charge  << std::endl;
			}
			//		    std::cout << std::endl;
		}

//		std::cout<<"Combination " << combination << std::endl;
//		print(theBuffer,aCluster);
//		std::cout << "chi2 " << chi2 << std::endl;

		if(chi2<chimin)
		{
			chimin=chi2;
			bestcomb=combination;
		}




	}
	SiPixelArrayBuffer               theWeights;
	theWeights.setSize(500,500);

	std::cout << "best combination chi: " << chimin << " co " << bestcomb<< std::endl;
	int clbase=1;
	SiPixelArrayBuffer               theBuffer;
	theBuffer.setSize(500,500);
	SiPixelArrayBuffer               theBufferResidual;
	theBufferResidual.setSize(500,500);

	for(unsigned int i = 0; i < pixels.size(); i++)
	{
		int x=pixels[i].x ;
		int y=pixels[i].y ;
		int adc=pixels[i].adc;
		theBuffer.set_adc( x, y, adc);
		theBufferResidual.set_adc( x, y, adc);
	}
	print(theBuffer,aCluster);
	//fill weights
	for(unsigned int cl=0;cl<expectedClusters;cl++){
		int pi=((bestcomb / clbase)%npos);
		int clx=pixels[pi].x;
		int cly=pixels[pi].y;
		clbase*=npos;
//		std::cout << "cl " << cl << " center in " <<  clx-xmin << " " << cly-ymin << std::endl;
		for(int x=xmin-1;x<=xmax+1;x++) {
			for(int y=ymin-1;y<=ymax+1;y++) {
				if(x<0||y<0) continue;
				int fact=0;
				if(x==clx) fact=2;
				if(x+1==clx || x-1==clx) fact=1;
				if(!(y>=cly && y <= cly+sizeY)) fact=0;
				if(x==clx && (y==cly-1 || y == cly+sizeY+1)) fact=1;
				if(fact)
				{	
					//			std::cout << "theWeights " << theWeights(x,y) << " fact: " << fact<< " x y " << x << " " << y<<   std::endl;
					theWeights.set_adc(x,y,theWeights(x,y)+fact);
					//			std::cout << "theWeightsAfter " << theWeights(x,y) << " fact: " << fact<< " x y " << x << " " << y<<   std::endl;
					//std::cout << "residual in "<< x-xmin <<","<< y-ymin<< " " << res << "  fact " << fact << " exp:"<< fact*perPixel <<std::endl;
				}
			}
		}

	}
	//really fill clusters
	clbase=1;
	for(unsigned int cl=0;cl<expectedClusters;cl++){
		int pi=((bestcomb / clbase)%npos);
		int clx=pixels[pi].x;
		int cly=pixels[pi].y;
		clbase*=npos;
		SiPixelCluster * cluster=0;
		for(int x=xmin-1;x<=xmax+1;x++) {
			for(int y=ymin-1;y<=ymax+1;y++) {
				if(x<0||y<0) continue;
				int fact=0;
				if(x==clx) fact=2;
				if(x+1==clx || x-1==clx) fact=1;
				if(!(y>=cly && y <= cly+sizeY)) fact=0;
				if(x==clx && (y==cly-1 || y == cly+sizeY+1)) fact=1;
				if(fact){
					//			std::cout << "HEREtheWeights " << theWeights(x,y) << " fact: " << fact<< " x y " << x << " " << y<<   std::endl;
					int charge=theBuffer(x,y)*fact/theWeights(x,y);
					if(charge>0){
						theBufferResidual.set_adc(x,y,theBufferResidual(x,y)-charge);
						if(cluster){
							SiPixelCluster::PixelPos newpix(x,y);
							cluster->add( newpix, charge);
						}
						else
						{
							SiPixelCluster::PixelPos newpix(x,y);
							cluster = new  SiPixelCluster( newpix, charge ); // create protocluster
						}
					}
				}	
			}
		}
		if(cluster){ 
			output.push_back(*cluster);
			delete cluster;
		}
	}
	std::cout << "Unused charge" << std::endl;	
	print(theBufferResidual,aCluster);

	return output;


}

void JetCoreClusterSplitter::print(const SiPixelArrayBuffer & b, const SiPixelCluster & c )
{
	int xmin=c.minPixelRow();
	int ymin=c.minPixelCol();
	int xmax=c.maxPixelRow();
	int ymax=c.maxPixelCol();

	for(int x=xmin; x<= xmax;x++){
		for(int y=ymin; y<= ymax;y++){
			std::cout << std::setiosflags(std::ios::fixed)
				<< std::setprecision(0)
				<< std::setw(4)
				<< std::left << b(x,y)/1000;
		}
		std::cout << std::endl;
	}


}
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetCoreClusterSplitter);
