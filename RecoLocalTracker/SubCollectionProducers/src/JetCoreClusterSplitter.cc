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
		bool split(const SiPixelCluster & aCluster, edmNew::DetSetVector<SiPixelCluster>::FastFiller & filler, float expectedADC);
		std::vector<SiPixelCluster> recursiveSplit(const SiPixelCluster & aCluster, float threshold, int depth);
		std::vector<SiPixelCluster> reDoClustering(const SiPixelCluster & aCluster, float threshold);
		void print(const SiPixelArrayBuffer & b, const SiPixelCluster & aCluster );

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
						std::cout << "CHECK FOR SPLITTING: charge and deltaR " <<aCluster.charge() << " " << Geom::deltaR(jetDir,clusterDir) << " size x y"<< aCluster.sizeX()  << " " << aCluster.sizeY()<< " detid " << detIt->id() << std::endl;	
						if(split(aCluster,filler,sqrt(1.08+jetZOverRho*jetZOverRho)*26000)) hasBeenSplit=true;
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
				 filler.push_back(aCluster);	

		}
	}
	iEvent.put(output);
}



bool JetCoreClusterSplitter::split(const SiPixelCluster & aCluster, edmNew::DetSetVector<SiPixelCluster>::FastFiller & filler, float expectedADC)
{
	std::vector<SiPixelCluster> sp=recursiveSplit(aCluster,expectedADC,0);
	

	for(unsigned int i = 0; i < sp.size();i++ )
	{
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


std::vector<SiPixelCluster> JetCoreClusterSplitter::recursiveSplit(const SiPixelCluster & aCluster, float expectedADC, int depth)
{
	depth++;
	unsigned int expectedClusters = floor(aCluster.charge() / expectedADC +0.5) ;	
	std::cout << "depth " << depth << " expectedADC " << expectedADC << " charge: " << aCluster.charge()  <<  "exClusters " << expectedClusters <<  std::endl;
	std::vector<SiPixelCluster> tempClusters=reDoClustering(aCluster,expectedADC*0.7);
	if(tempClusters.size() >= expectedClusters) return tempClusters;
	std::vector<SiPixelCluster> output;
	int reclusteredCharge=0;
	for(unsigned int i=0;i<tempClusters.size();i++)
	{
		reclusteredCharge+=tempClusters[i].charge();
	}
	//	int lostCharge=aCluster.charge()-reclusteredCharge;
	int newExpected=reclusteredCharge/expectedClusters;

	int overlapsOnSinglePixel=0;
	int furtherSplitted=0;
	int furtherSplittedIn=0;
	int unsplitted=0;
	std::cout << "From splitter Nclus: " << tempClusters.size() << " newExpected " << newExpected << std::endl;
	for(unsigned int i=0;i<tempClusters.size();i++)
	{
		std::cout <<  i << " cluster size and charge " << tempClusters[i].size() << " " << tempClusters[i].charge()  << std::endl;
		if(tempClusters[i].size()==1 && tempClusters[i].charge() > 1.7*newExpected)
		{
			SiPixelCluster::PixelPos pix(tempClusters[i].pixels()[0].x,tempClusters[i].pixels()[0].y);
			int n=floor(tempClusters[i].charge()/newExpected);
			int adcPerCl = tempClusters[i].charge()/n;
			for(int j=0; j < n;j++)
			{
				SiPixelCluster cluster( pix, adcPerCl );
				output.push_back(cluster);
				overlapsOnSinglePixel++;
			}
		}else if(tempClusters[i].charge() > 1.7*newExpected)
		{
			std::vector<SiPixelCluster> sp=recursiveSplit(tempClusters[i],newExpected,depth);
			output.insert(output.end(), sp.begin(), sp.end());
			furtherSplitted++;
			furtherSplittedIn+=sp.size();
		}
		else
		{
			output.push_back(tempClusters[i]);
			unsplitted++;
		}
	}	 
	std::string s(depth,' ');
	std::cout << s << "Expected : " << expectedClusters << " total : "  << output.size() << "  notsplittable : " << unsplitted << " further splitted: " << furtherSplitted << " in " << furtherSplittedIn << " overlaps on 1 pixel : " << overlapsOnSinglePixel  << std::endl; 	  
	/*    tempClusters.push_back(aCluster);	 
	      while(tempClusters.size()==1 && tempClusters[0].size() > 1)
	      { 
	      tempClusters=reDoClustering(tempClusters[0],expectedADC*0.7);
	      }	*/

	return output;



}

std::vector<SiPixelCluster>  JetCoreClusterSplitter::reDoClustering(const SiPixelCluster & aCluster, float threshold)
{
	std::vector<SiPixelCluster> output;
	SiPixelArrayBuffer               theBuffer;
	theBuffer.setSize(500,500); 
	std::vector<SiPixelCluster::PixelPos>  theSeeds;
	//Copy to buffer the content of the cluster to split, create seeds and find minimum adc
	int minAdc=100000;
	int minX=0,minY=0;
	std::vector<SiPixelCluster::Pixel> pixels = aCluster.pixels();
	for(unsigned int i = 0; i < pixels.size(); i++)
	{
		int x=pixels[i].x ;
		int y=pixels[i].y ;
		int adc=pixels[i].adc;
		theBuffer.set_adc( x, y, adc);
		if ( adc >= 5000 ) 
		{ 
			theSeeds.push_back( SiPixelCluster::PixelPos(x,y) );
		}
		if(adc < minAdc) {minAdc=adc; minX=x;minY=y;}

	}
	std::cout << "Input buffer" << std::endl;
        print(theBuffer,aCluster);

	std::cout << "Minadc " << minAdc << " at " << minX<<" , " << minY <<  std::endl;
	if(minAdc> threshold)
	{
		std::cout << "Adding this minadc " << std::endl;
		SiPixelCluster::PixelPos newpix(minX,minY);
		SiPixelCluster cluster( newpix,threshold );
		output.push_back(cluster);
		theBuffer.set_adc( minX, minY, minAdc-threshold);

	}
	float shareHolders=0;
	for(int x=minX-1;x<=minX+1;x++){
		for(int y=minY-1;y<=minY+1;y++){
			if(x<0 || x > 155 || y < 0 || y > 416) continue;

			if(theBuffer(x,y) > minAdc && (x!=minX  || y!=minY)){
				if(x!=minX && y!= minY) shareHolders++; else shareHolders+=2;
			}
		}
	}
	if(shareHolders)
	{
		int chargePerShare=theBuffer(minX,minY)/shareHolders;
		std::cout << "Nshares " << shareHolders << " charge per share " << chargePerShare << std::endl;
		for(int x=minX-1;x<=minX+1;x++){
			for(int y=minY-1;y<=minY+1;y++)
			{
				if(x<0 || x > 155 || y < 0 || y > 416) continue;
				if((theBuffer(x,y) > minAdc)&& (x!=minX  || y!=minY))
				{
					if(x!=minX && y!= minY)
					{  
						int newVal=chargePerShare+theBuffer(x,y);
						if(newVal>65535) newVal=65535;
						theBuffer.set_adc(x,y,newVal);
					}
					else { 
						int newVal=chargePerShare*2+theBuffer(x,y);
						if(newVal>65535) newVal=65535;
						theBuffer.set_adc(x,y,newVal);
					}
				}
			}
		}
	}

	std::cout << "After removal and sharing" << std::endl;
	print(theBuffer,aCluster);

	//do actual clustering for each seed
	for (unsigned int i = 0; i < theSeeds.size(); i++) {
		std::stack<SiPixelCluster::PixelPos, vector<SiPixelCluster::PixelPos> > pixel_stack;
		const SiPixelCluster::PixelPos & pix = theSeeds[i];
		int seed_adc = theBuffer(pix.row(), pix.col());
		if(seed_adc <= minAdc) continue;
//		std::cout << "processing seed " << i << " adc: " << seed_adc <<  std::endl; 
		theBuffer.set_adc( pix, 1); //mark as used
		SiPixelCluster cluster( pix, seed_adc ); // create protocluster
//		std::cout << "ccc " <<  cluster.charge() << std::endl;

		pixel_stack.push( pix);
		while ( ! pixel_stack.empty()) 
		{
			//This is the standard algorithm to find and add a pixel
			SiPixelCluster::PixelPos curpix = pixel_stack.top(); pixel_stack.pop();
			for ( int r = curpix.row()-1; r <= curpix.row()+1; ++r) 
			{
				for ( int c = curpix.col()-1; c <= curpix.col()+1; ++c) 
				{
					bool isDiagonal = (c!=curpix.col() && r != curpix.row());
					if (! isDiagonal &&  theBuffer(r,c) > minAdc) // reclustering without smallest signal 
					{
						SiPixelCluster::PixelPos newpix(r,c);
						cluster.add( newpix, theBuffer(r,c));
						theBuffer.set_adc( newpix, 1);
						pixel_stack.push( newpix);
//						std::cout << "cc " <<  cluster.charge() << std::endl;
					}
				}
			}
		}

//		std::cout << "c " <<  cluster.charge() << "  > " << threshold		<< std::endl;
		if(cluster.charge() > threshold){
//			std::cout << "Cluster: " << cluster.charge() << " #" << output.size() << std::endl;
			output.push_back(cluster);
		}

	}
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
