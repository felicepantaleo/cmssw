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
#include <time.h>
#include <algorithm> 

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

class JetCoreClusterSplitter2 : public edm::EDProducer 
{

	public:
		JetCoreClusterSplitter2(const edm::ParameterSet& iConfig) ;
		~JetCoreClusterSplitter2() ;
		void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) ;

	private:
		bool split(const SiPixelCluster & aCluster, edmNew::DetSetVector<SiPixelCluster>::FastFiller & filler, float expectedADC,int sizeY,float jetZOverRho, const edmNew::DetSet<SiPixelCluster> & );
		float distanceCluster(const SiPixelCluster & cluster,const edmNew::DetSet<SiPixelCluster> & idealClusters);
		void print(const SiPixelArrayBuffer & b, const SiPixelCluster & aCluster, int div=1000 );
		std::vector<SiPixelCluster> fittingSplit(const SiPixelCluster & aCluster, float expectedADC,int sizeY,float jetZOverRho);
		bool nextCombination(std::vector<int> & comb,int npos);
		float pixelWeight(int clx, int cly, int x, int y,int sizeY,int direction,int bintheta);
		float pixelWeight2(int clx, int cly, int x, int y,int sizeY, int direction);
		void initCharge();

		std::string pixelCPE_; 
		edm::InputTag pixelClusters_;
		edm::InputTag vertices_;
		int mapcharge[21][5][3][20][20];
		int count[21][5][3];
		int totalcharge[21][5][3];
		int nDirections;

};

JetCoreClusterSplitter2::JetCoreClusterSplitter2(const edm::ParameterSet& iConfig):
	pixelCPE_(iConfig.getParameter<std::string>("pixelCPE")),
	pixelClusters_(iConfig.getParameter<edm::InputTag>("pixelClusters")),
	vertices_(iConfig.getParameter<edm::InputTag>("vertices"))

{
	nDirections=4;
	
	for(int a=0;a<21;a++)
	for(int b=0;b<5;b++)
	for(int e=0;e<3;e++){
	count[a][b][e]=0;
	totalcharge[a][b][e]=0;
	for(int c=0;c<20;c++)
	for(int d=0;d<20;d++)
	  mapcharge[a][b][e][c][d]=0;
	}
	initCharge();
	produces< edmNew::DetSetVector<SiPixelCluster> >();

}



JetCoreClusterSplitter2::~JetCoreClusterSplitter2()
{
}

bool SortPixels (const SiPixelCluster::Pixel& i,const SiPixelCluster::Pixel& j) { return (i.adc>j.adc); }

void JetCoreClusterSplitter2::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
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
					if(fabs(cPos.z())>30) jetZOverRho=jit->momentum().Rho()/jit->momentum().Z();
					GlobalVector jetDir(jit->momentum().x(),jit->momentum().y(),jit->momentum().z());
					unsigned int maxSizeY=fabs(sqrt(1.3*1.3+1.9*1.9*jetZOverRho*jetZOverRho));
//					unsigned int maxSizeY=fabs(jetZOverRho*1.9);
					
//					unsigned int maxSizeY=fabs(jetZOverRho*1.75)+0.5;

					if(maxSizeY < 1) maxSizeY=1;
					if(Geom::deltaR(jetDir,clusterDir) < 0.05 && aCluster.charge() > 30000 && (aCluster.sizeX() > 2 || ((unsigned int)aCluster.sizeY()) > maxSizeY+1) )
					{
						std::cout << "CHECK FOR NEW SPLITTING: charge and deltaR " <<aCluster.charge() << " " << Geom::deltaR(jetDir,clusterDir) << " size x y"<< aCluster.sizeX()  << " " << aCluster.sizeY()<< " detid " << detIt->id() << std::endl;	
						std::cout << "jetZOverRho="<< jetZOverRho << std::endl;	
						SiPixelClusterCollectionNew::const_iterator myDet =  inputPixelClustersIDEAL->find(detIt->id());
						clock_t init=clock(), final;
						const edmNew::DetSet<SiPixelCluster> & idealClusters  = (*myDet);
						if(split(aCluster,filler,sqrt(1.08+jetZOverRho*jetZOverRho)*26000,maxSizeY,jetZOverRho,idealClusters)) {hasBeenSplit=true;
						final=clock()-init; cout<<"Time used: (s) " << (double)final / ((double)CLOCKS_PER_SEC)<<endl;}
						std::cout << "IDEAL was : "  << std::endl; 
						int xmin=aCluster.minPixelRow();                        
						int ymin=aCluster.minPixelCol();                                
						int xmax=aCluster.maxPixelRow();                                
						int ymax=aCluster.maxPixelCol(); 
						int last=1;
						std::map<int,int> sh;  
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
						JetCoreClusterSplitter2::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
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



float JetCoreClusterSplitter2::distanceCluster(const SiPixelCluster & cluster,const edmNew::DetSet<SiPixelCluster> & idealClusters)
{
float minDistance=1e99;
for(edmNew::DetSet<SiPixelCluster>::const_iterator ideal=idealClusters.begin(); ideal <  idealClusters.end() ; ideal++)
{
	float distance = sqrt( (cluster.x()-ideal->x())*(cluster.x()-ideal->x())  +   (cluster.y()-ideal->y())*(cluster.y()-ideal->y())*1.5*1.5 );
	if(distance<minDistance) minDistance=distance;
}
return minDistance;
}

bool JetCoreClusterSplitter2::split(const SiPixelCluster & aCluster, edmNew::DetSetVector<SiPixelCluster>::FastFiller & filler, float expectedADC,int sizeY,float jetZOverRho,const edmNew::DetSet<SiPixelCluster> & idealClusters)
{
	std::vector<SiPixelCluster> sp=fittingSplit(aCluster,expectedADC,sizeY,jetZOverRho);
	

	for(unsigned int i = 0; i < sp.size();i++ )
	{
		float distance = JetCoreClusterSplitter2::distanceCluster(sp[i],idealClusters);
		std::cout << "NEW POS: " << i << " x: "  << std::setprecision(2) << sp[i].x() << " y: " << sp[i].y() << " c: " << sp[i].charge() << " distance=" << 100*distance << " um"  << std::endl;
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

float JetCoreClusterSplitter2::pixelWeight2(int clx, int cly, int x, int y,int sizeY,int direction)
{
 if (direction>1 || direction<0) return 0;
 float fact=0; 
                               if(x==clx &&  (y>=cly && y < cly+(sizeY+1)/2) ) fact=2;
                                if(x==clx+1 && direction  &&  (y>=cly+(sizeY+1)/2) && y < cly+sizeY ) fact=2;
                                if(x==clx-1 && ! direction  &&  (y>=cly+(sizeY+1)/2) && y < cly+sizeY ) fact=2;
                                if(x==clx &&  (y>= cly+(sizeY+1)/2)  && y < cly+sizeY ) fact=1;
                                if(x==clx+1 && direction  &&  (y>=cly && y<cly+(sizeY+1)/2)  ) fact=1;
                                if(x==clx-1 && ! direction  &&  (y>=cly && y<cly+(sizeY+1)/2) ) fact=1;
                                if(x==clx+1 && direction && y==cly+sizeY ) fact=1;
                                if(x==clx-1 &&  ! direction && y==cly+sizeY ) fact=1;
                                if(x==clx && y==cly-1 ) fact=1;
return fact/(0.5+sizeY)/4.;
//return fact/(0.5+sizeY)/4.;
}

/*float JetCoreClusterSplitter2::pixelWeight(int clx, int cly, int x, int y,int sizeY,int direction)
{
 float fact=0;
                               if(x==clx &&  (y>=cly && y < y+sizeY) ) fact=8;
                                if(x==clx+1 && direction  &&  (y>=cly+(sizeY+1)/2) && y < cly+sizeY ) fact=2;
                                if(x==clx-1 && ! direction  &&  (y>=cly+(sizeY+1)/2) && y < cly+sizeY ) fact=2;
                                if(x==clx+1 && !direction  &&  (y>=cly+(sizeY+1)/2) && y < cly+sizeY ) fact=1;
                                if(x==clx-1 && direction  &&  (y>=cly+(sizeY+1)/2) && y < cly+sizeY ) fact=1;
//                              if(x==clx &&  (y>= cly+(sizeY+1)/2)  && y < cly+sizeY ) fact=1;
                                if(x==clx+1 && direction  &&  (y>=cly && y<cly+(sizeY+1)/2)  ) fact=2;
                                if(x==clx-1 && ! direction  &&  (y>=cly && y<cly+(sizeY+1)/2) ) fact=2;
                                if(x==clx+1 && !direction  &&  (y>=cly && y<cly+(sizeY+1)/2)  ) fact=1;
                                if(x==clx-1 &&  direction  &&  (y>=cly && y<cly+(sizeY+1)/2) ) fact=1;
//                                if(x==clx+1 && direction && y==cly+sizeY ) fact=1;
                                if(x==clx-1 &&  ! direction && y==cly+sizeY ) fact=1;
                                if(x==clx && y==cly-1 ) fact=1;
return fact/(11.*sizeY+2);
}*/
float JetCoreClusterSplitter2::pixelWeight(int clx, int cly, int x, int y,int sizeY,int direction,int bintheta)
{
 
 
 if(x-clx+10<-20) return 0;
 if(y-cly+(sizeY+1)/2<0) return 0;
 if(x-clx+10>=20) return 0;
 if(y-cly+(sizeY+1)/2>=20) return 0;

//if(direction>2) {cout<<"*** BUG direction>2 *****"; return 0;}
//if(direction<0) {cout<<"*** BUG direction<0 *****"; return 0;}

if(bintheta<0) {cout<<"Forced bintheta=0. It was "<<bintheta; bintheta=0;}
if(bintheta>20) {cout<<"Forced bintheta=20. It was "<<bintheta; bintheta=20;}

int caseX=direction/2;
direction=direction%2;

direction=direction+1;

// if(x-clx<=-10) x=clx-10;
// if(y-cly<=0) y=cly;
// if(x-clx>10) x=clx+9;
// if(y-cly>20) y=cly+19;

// if(x-clx<-10) return 0;
// if(y-cly<0) return 0;
// if(x-clx>=10) return 0;
// if(y-cly>20) return 0;
 unsigned int binX = clx*5./160;
// int mapcharge[21][5][3][20][20];
sizeY=sizeY+(direction-1);
 float fact=1.*mapcharge[bintheta][binX][direction][x-clx+10+caseX][y-cly+(sizeY-1)/2]/totalcharge[bintheta][binX][direction]*count[bintheta][binX][direction];
// float fact=1.*mapcharge[bintheta][binX][direction][x-clx+10][y-cly]/totalcharge[bintheta][binX][direction]*count[bintheta][binX][direction];
 //std::cout << "bin " << bintheta <<  ", " << binX  <<  ", " << x-clx+10  <<  ", " << y-cly+10 << " map " << mapcharge[bintheta][binX][x-clx+10][y-cly+10] << " tot " << totalcharge[bintheta][binX] << " fact " << fact << std::endl;  	 	
return fact;
}



std::vector<SiPixelCluster> JetCoreClusterSplitter2::fittingSplit(const SiPixelCluster & aCluster, float expectedADC,int sizeY,float jetZOverRho)
{
	const float diecimila=10000;
	
	unsigned int expectedClusters = floor(aCluster.charge() / expectedADC +0.5) ;
	std::vector<SiPixelCluster> output;
	if(expectedClusters==0) {std::cout << "ZERO????" << std::endl;} 
	if(expectedClusters<=1) {
		output.push_back(aCluster);	
		return output;
	}	
	int xmin=aCluster.minPixelRow();
	int ymin=aCluster.minPixelCol();
	int xmax=aCluster.maxPixelRow();
	int ymax=aCluster.maxPixelCol();
	int binjetZOverRho = jetZOverRho*2 + 21/2;
	if(binjetZOverRho<0) binjetZOverRho=0;
	if(binjetZOverRho>20) binjetZOverRho=20;


	if(expectedClusters > 5) expectedClusters=5;
//return std::vector<SiPixelCluster>(); 
	std::vector<SiPixelCluster::Pixel> pixels = aCluster.pixels();
	sort(pixels.begin(),pixels.end(),SortPixels);
   // FP
	//int deltax=xmax-xmin+1;
	int deltay=ymax-ymin-sizeY+1;
	if (deltay < 1) deltay=1;
	float perPixel=expectedADC;/// become per unit weight 1./(0.5+sizeY)/4.;
	int npos= pixels.size()*nDirections;
	//	unsigned long maxComb=pow(expectedClusters, npos);
	
	unsigned long maxComb=pow(npos,expectedClusters);
	for(unsigned int i=1; i <= expectedClusters; i++)  maxComb/=i;
	float chimin=1e99;
	std::cout << "Sizey and perPixel " << sizeY << " " << perPixel << std::endl;
//	if(maxComb > 100000*8) {

//	unsigned int limitComb=10000*pow(2,expectedClusters);
	unsigned int limitComb=100000;
	std::cout << "combination=" << maxComb << " npos="<<npos << "expectedClusters="<<expectedClusters<< " elapsed time=" << 0.000437043* maxComb <<endl;
	if(maxComb>limitComb)
	{
//		std::cout << "toomany combination=" << maxComb << " npos="<<npos << "expectedClusters="<<expectedClusters<<endl;
		int pixelLimit = pow(float(limitComb),1.0/expectedClusters)/nDirections;
		npos= pixelLimit*nDirections;
		maxComb=pow(npos,expectedClusters);
		std::cout << "Forced: combination=" << maxComb << " npos="<<npos << "expectedClusters="<<expectedClusters<< " elapsed time=" << 0.000437043* maxComb <<endl;
	}
	
	if(maxComb > limitComb) {
		std::cout << "toomany" << std::endl;
		int xmin=aCluster.minPixelRow();
		int ymin=aCluster.minPixelCol();
		int xmax=aCluster.maxPixelRow();
		int ymax=aCluster.maxPixelCol();
		std::cout << "Order of hits in the TOOMANY:" << std::endl;
		for(int x=xmin; x<= xmax;x++){
			for(int y=ymin; y<= ymax;y++)
			{
					int flag=0;
					std::vector<SiPixelCluster::Pixel> pixels = aCluster.pixels();
					for(unsigned int j = 0; j < pixels.size(); j++)
					{
						if(pixels[j].x==x && pixels[j].y==y) flag=j;
					}       

				std::cout << " "<<
				 std::setiosflags(std::ios::fixed)
				                                << std::setprecision(0)
				                              << std::setw(7)
				                            << std::left << flag ;


			}
			std::cout << std::endl;
		}


 return std::vector<SiPixelCluster>();
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

	std::vector<int> comb(expectedClusters); 
	std::vector<int> bestcomb(expectedClusters); 
	while(nextCombination(comb,npos)) 
		//	for(unsigned long combination=0;combination<maxComb;combination++)	
	{
		float chi2=0;
		// FP
		//int clbase=1;
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

		//int remainingFreePositions = npos;
		float prob=0;
		for(unsigned int cl=0;cl<expectedClusters;cl++){
			int pi=comb[cl]; //((combination / clbase)%remainingFreePositions);
			int clx=pixels[pi/nDirections].x;
			int cly=pixels[pi/nDirections].y;
			int direction=pi%nDirections;
			//std::cout << "Cluster  "<< cl << " pi " << pi/2 << " dir " << direction  << " x,y " << clx -xmin <<", " << cly-ymin<< " clbase " << clbase << std::endl;
			for(int x=xmin-5;x<=xmax+5;x++) {
				for(int y=ymin-(sizeY+1)/2;y<=ymax+(sizeY+1)/2;y++) {
					if(x<0||y<0) continue;
					float fact=pixelWeight(clx,cly,x, y,sizeY,direction,binjetZOverRho);
					if(fact > 0) {
							theBuffer.set_adc(x,y,theBuffer(x,y)-fact*perPixel);
						     }	
			//		std::cout << "residual in "<< x-xmin <<","<< y-ymin<< " " << theBuffer(x,y)-fact*perPixel << "  fact " << fact << " exp:"<< fact*perPixel <<std::endl;
				}
			}
			prob+=count[binjetZOverRho][int((clx)*5./160)][direction%2+1];
		}
		//print(theBuffer,aCluster);
		for(int x=xmin-5;x<=xmax+5;x++) {
			for(int y=ymin-(sizeY+1)/2;y<=ymax+(sizeY+1)/2;y++) {
				//				std::cout << theBuffer(x,y)/1000 << " " ;
				float res=theBuffer(x,y);
				float charge=theOriginalBuffer(x,y)-theBuffer(x,y); //charge assigned to this pixel
				float chargeMeasured=theOriginalBuffer(x,y); //charge assigned to this pixel

				if(res< 0 ) { //threshold effect
					if(res > -10000){
						if(res<-5000) res+=5000;
						else  res=0;
					}
				}
//				if(res> 0 && charge > 7000 ) { //reduce weights of landau tails
//					res*=0.7;
//				}

				
				if( chargeMeasured<5000 && abs(charge)<5000 ) { //threshold effect
					res=0;
				}


				if(charge==0) charge=2000;
				chi2+=(res*res)/(charge*charge);
				//std::cout << "chi2 " << chi2 << " xy" << x<< " " << y << " res,res2,charge" << res << " ," <<res*res<< ", " <<charge  << std::endl;
			}
			//		    std::cout << std::endl;
		}
		chi2/=prob;
		//		std::cout<<"Combination " << combination << std::endl;
		//		print(theBuffer,aCluster);
		//		std::cout << "chi2 " << chi2 << std::endl;
		if(chi2<chimin)
		{
			chimin=chi2;
			bestcomb=comb;
		}




	}

	SiPixelArrayBuffer               myResidual;
	myResidual.setSize(500,500);

	// get and print the residual
	for(unsigned int i = 0; i < pixels.size(); i++)
	{
		int x=pixels[i].x ;
		int y=pixels[i].y ;
		int adc=pixels[i].adc;
		myResidual.set_adc( x, y, adc);
	}

	for(unsigned int cl=0;cl<expectedClusters;cl++){
	int pi=bestcomb[cl]; //((combination / clbase)%remainingFreePositions);
	int clx=pixels[pi/nDirections].x;
	int cly=pixels[pi/nDirections].y;
	int direction=pi%nDirections;
	for(int x=xmin-5;x<=xmax+5;x++) {
		for(int y=ymin-(sizeY+1)/2;y<=ymax+(sizeY+1)/2;y++) {
			if(x<0||y<0) continue;
			float fact=pixelWeight(clx,cly,x, y,sizeY,direction,binjetZOverRho);
			if(fact > 0) {
					myResidual.set_adc(x,y,myResidual(x,y)-fact*perPixel);
				     }	
		}
	}
	}
	
	cout<<endl<<"clx-xmin,cly-ymin = ";
	for(unsigned int cl=0;cl<expectedClusters;cl++){
	int pi=bestcomb[cl]; //((combination / clbase)%remainingFreePositions);
	int clx=pixels[pi/nDirections].x - xmin;
	int cly=pixels[pi/nDirections].y - ymin;
	cout<<clx<<","<<cly<<";";
	}
	cout<<endl;
	std::cout << "Residual:" << std::endl;	
	print(myResidual,aCluster);

	// End: get and print the residual

	SiPixelArrayBuffer               theWeights;
	theWeights.setSize(500,500);

	std::cout << "best combination chi: " << chimin << " co "  ;
	for(unsigned int ii=0;ii< bestcomb.size();ii++) {std::cout << bestcomb[ii] << " ";} std::cout<<std::endl;

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
	int remainingFreePositions = npos;
	for(unsigned int cl=0;cl<expectedClusters;cl++){
		int pi=bestcomb[cl];
		int clx=pixels[pi/nDirections].x;
		int cly=pixels[pi/nDirections].y;
		int direction=pi%nDirections;
		remainingFreePositions=pi+1; // cl_i+1 <= cl_i
		clbase*=remainingFreePositions;
		for(int x=xmin-5;x<=xmax+5;x++) {
			for(int y=ymin-(sizeY+1)/2;y<=ymax+(sizeY+1)/2+1;y++) {
				if(x<0||y<0) continue;
                                float fact=pixelWeight(clx,cly,x, y,sizeY,direction,binjetZOverRho);
//				std::cout << "Fact " << fact <<   " x,y " << x -xmin <<", " <<y-ymin<<std::endl;
				if(fact>0.05)
				{	
					//			std::cout << "theWeights " << theWeights(x,y) << " fact: " << fact<< " x y " << x << " " << y<<   std::endl;
					theWeights.set_adc(x,y,theWeights(x,y)+fact*diecimila);
					//			std::cout << "theWeightsAfter " << theWeights(x,y) << " fact: " << fact<< " x y " << x << " " << y<<   std::endl;
					//std::cout << "residual in "<< x-xmin <<","<< y-ymin<< " " << res << "  fact " << fact << " exp:"<< fact*perPixel <<std::endl;
				}
			}
		}

	}

	                for(unsigned int cl=0;cl<expectedClusters;cl++){
			std::cout << "Cluster " << cl << " weight map" << std::endl;
                        int pi=bestcomb[cl]; //((combination / clbase)%remainingFreePositions);
                        int clx=pixels[pi/nDirections].x;
                        int cly=pixels[pi/nDirections].y;
                        int direction=pi%nDirections;
                        unsigned int binX = clx*5./160;
			std::cout << "binjetZOverRho=" << binjetZOverRho << " binX=" << binX  << " direction=" << direction <<  " sizeY=" << sizeY << endl;
                        //std::cout << "Cluster  "<< cl << " pi " << pi/2 << " dir " << direction  << " x,y " << clx -xmin <<", " << cly-ymin<< " clbase " << clbase << std::endl;
                        for(int x=xmin-1;x<=xmax+1;x++) {
                                for(int y=ymin-(sizeY+1)/2;y<=ymax+(sizeY+1)/2+1;y++) { 
                                        if(x<0||y<0) continue;
                                        float fact=pixelWeight(clx,cly,x, y,sizeY,direction,binjetZOverRho);
					std::cout << std::setprecision(2) << float(fact) << " "; 
                        //              if(fact > 0)    theBuffer.set_adc(x,y,theBuffer(x,y)-fact*perPixel);
                        //              std::cout << "residual in "<< x-xmin <<","<< y-ymin<< " " << theBuffer(x,y)-fact*perPixel << "  fact " << fact << " exp:"<< fact*perPixel <<std::endl;
                                }
			 std::cout << std::endl;
                        }
                        std::setprecision(0);        
                }   
	//really fill clusters
	clbase=1;
	remainingFreePositions = npos;
	for(unsigned int cl=0;cl<expectedClusters;cl++){
		int pi=bestcomb[cl];
		int clx=pixels[pi/nDirections].x;
		int cly=pixels[pi/nDirections].y;
		int direction=pi%nDirections;
		remainingFreePositions=pi+1; // cl_i+1 <= cl_i
		SiPixelCluster * cluster=0;
		clbase*=remainingFreePositions;
		for(int x=xmin-5;x<=xmax+5;x++) {
			for(int y=ymin-(sizeY+1)/2;y<=ymax+(sizeY+1)/2+1;y++) {
				if(x<0||y<0) continue;
                                float fact=pixelWeight(clx,cly,x, y,sizeY,direction,binjetZOverRho);

				if(fact>0.05 ){
					float charge=1.*theBuffer(x,y)*fact*diecimila/theWeights(x,y);
						std::cout << std::setprecision(2) << "HEREtheWeights " << theWeights(x,y) << " fact: " << fact<< " x y " << x -xmin<< " " << y-ymin<<  " charge " << charge<<   std::endl;
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
	std::cout << "Weights" << std::endl;	
	print(theWeights,aCluster,1);
	std::cout << "Unused charge" << std::endl;	
	print(theBufferResidual,aCluster);

	return output;


}
bool JetCoreClusterSplitter2::nextCombination(std::vector<int> & comb,int npos)
{
	comb[comb.size()-1]+=1;//increment
	for(int i=comb.size()-1; i>=0 ;i--)
	{
		if(i > 0 && comb[i]>comb[i-1]) {comb[i]=0;comb[i-1]+=1;}
		if(i==0 && comb[i]>=npos) { return false; }		
	}
	return true;

}
void JetCoreClusterSplitter2::print(const SiPixelArrayBuffer & b, const SiPixelCluster & c, int div )
{
	int xmin=c.minPixelRow();
	int ymin=c.minPixelCol();
	int xmax=c.maxPixelRow();
	int ymax=c.maxPixelCol();
	for(int x=xmin-5; x<= xmax+5;x++){
	for(int y=ymin-5; y<= ymax+5;y++){
	if(x<0||y<0) continue;
	if(b(x,y)!=0 && x<xmin) xmin=x;
	if(b(x,y)!=0 && y<ymin) ymin=y;
	if(b(x,y)!=0 && x>xmax) xmax=x;
	if(b(x,y)!=0 && y>ymax) ymax=y;
	}}


	for(int x=xmin; x<= xmax;x++){
		for(int y=ymin; y<= ymax;y++){
			std::cout << std::setiosflags(std::ios::fixed)
				<< std::setprecision(0)
				<< std::setw(6)
				<< std::left << b(x,y)/div;
		}
		std::cout << std::endl;
	}


}

//#include "RecoLocalTracker/SubCollectionProducers/interface/best_charge.h"
//#include "RecoLocalTracker/SubCollectionProducers/interface/chargeNoEC.h"
//#include "RecoLocalTracker/SubCollectionProducers/interface/charge2.h"
//#include "RecoLocalTracker/SubCollectionProducers/interface/charge.h"
//#include "RecoLocalTracker/SubCollectionProducers/interface/chargeNoECXmin.h"
//#include "RecoLocalTracker/SubCollectionProducers/interface/chargeNoECXmin0p5.h"
#include "RecoLocalTracker/SubCollectionProducers/interface/chargeNewSizeY.h"
//#include "RecoLocalTracker/SubCollectionProducers/interface/chargeOK.h"
//#include "RecoLocalTracker/SubCollectionProducers/interface/chargeNoECXmin0p5NewSizeYWithXmed.h"






#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetCoreClusterSplitter2);


