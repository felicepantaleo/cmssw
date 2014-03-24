
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

#include <cuda_runtime.h>
#include <cuda.h>

using namespace std;
template <typename T>
std::ostream& bin(T& value, std::ostream& o) {
  for (T bit = 16; bit; bit >>= 1) {
    o << ((value & bit) ? '1' : '0');
  }
  return o;
}

const int BinsXposition = 5;
const int BinsDirections = 4;
const int BinsX = 20;
const int BinsY = 20;

const int BinsJetOverRho = 21;
const float jetZOverRhoWidth = 0.5;




extern "C" { void cudaClusterSplitter_(int*); }

class JetCoreClusterSplitter2 : public edm::EDProducer {

 public:
  JetCoreClusterSplitter2(const edm::ParameterSet& iConfig);
  ~JetCoreClusterSplitter2();
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup);

 private:
  bool split(const SiPixelCluster& aCluster,
             edmNew::DetSetVector<SiPixelCluster>::FastFiller& filler,
             float expectedADC, int sizeY, float jetZOverRho,
             const edmNew::DetSet<SiPixelCluster>&);
  float distanceCluster(const SiPixelCluster& cluster,
                        const edmNew::DetSet<SiPixelCluster>& idealClusters);
  void print(const SiPixelArrayBuffer& b, const SiPixelCluster& aCluster,
             int div = 1000);
  std::vector<SiPixelCluster> fittingSplit(const SiPixelCluster& aCluster,
                                           float expectedADC, int sizeY,
                                           float jetZOverRho);
  bool nextCombination(std::vector<int>& comb, int npos);
  unsigned int combinations(unsigned int npos, unsigned int expectedClusters);
  float pixelWeight(int clx, int cly, int x, int y, int sizeY, int direction,
                    int bintheta);
  float pixelWeight2(int clx, int cly, int x, int y, int sizeY, int direction);
  void initCharge();
  void finalizeSplitting(std::vector<int>& bestcomb,
                         unsigned int& expectedClusters,
                         std::vector<SiPixelCluster::Pixel>& pixels,
                         const SiPixelCluster& aCluster, float expectedADC,
                         int sizeY, float jetZOverRho);

  int* mapcharge_array;
  int* gpu_mapcharge_array;
  int * gpu_originalADC;
  int * originalADC;
  void cudawClusterSplitter(int*);


  std::string pixelCPE_;
  edm::InputTag pixelClusters_;
  edm::InputTag vertices_;

  int mapcharge[BinsJetOverRho][BinsXposition][BinsDirections][BinsX][BinsY];
  int count[BinsJetOverRho][BinsXposition][BinsDirections];
  int totalcharge[BinsJetOverRho][BinsXposition][BinsDirections];
  int nDirections;
};

JetCoreClusterSplitter2::JetCoreClusterSplitter2(
    const edm::ParameterSet& iConfig)
    : pixelCPE_(iConfig.getParameter<std::string>("pixelCPE")),
      pixelClusters_(iConfig.getParameter<edm::InputTag>("pixelClusters")),
      vertices_(iConfig.getParameter<edm::InputTag>("vertices")) {
  nDirections = 4;

	cudaMallocHost((void**)&originalADC, 250000*sizeof(int));
	cudaMalloc((void**)&gpu_originalADC, 250000*sizeof(int));
	cudaMallocHost((void**)&mapcharge_array, BinsJetOverRho*BinsXposition*BinsDirections*BinsX*BinsY*sizeof(int));
	cudaMalloc(&gpu_mapcharge_array,BinsJetOverRho*BinsXposition*BinsDirections*BinsX*BinsY*sizeof(int));




  for (int a = 0; a < BinsJetOverRho; a++)
    for (int b = 0; b < BinsXposition; b++)
      for (int e = 0; e < BinsDirections; e++) {
        count[a][b][e] = 0;
        totalcharge[a][b][e] = 0;
        for (int c = 0; c < BinsX; c++)
          for (int d = 0; d < 20; d++)
        	  {
        	  	  mapcharge[a][b][e][c][d] = 0;
        	  }
      }
  initCharge();
  for (int a = 0; a < BinsJetOverRho; a++)
    for (int b = 0; b < BinsXposition; b++)
      for (int e = 0; e < BinsDirections; e++) {
        count[a][b][e] = 0;
        totalcharge[a][b][e] = 0;
        for (int c = 0; c < BinsX; c++)
          for (int d = 0; d < 20; d++)
        	  {
        	  	mapcharge_array[d+BinsY*c+BinsX*BinsY*e+BinsDirections*BinsX*BinsY*b+BinsXposition*BinsDirections*BinsX*BinsY*a] = mapcharge[a][b][e][c][d];
        	  }
      }


	cudaMemcpyAsync(gpu_mapcharge_array,mapcharge_array,BinsJetOverRho*BinsXposition*BinsDirections*BinsX*BinsY*sizeof(int), cudaMemcpyHostToDevice,0);



  produces<edmNew::DetSetVector<SiPixelCluster> >();
}

JetCoreClusterSplitter2::~JetCoreClusterSplitter2() {
	cudaFreeHost(originalADC);
	cudaFreeHost(mapcharge_array);
	cudaFree(gpu_originalADC);
	cudaFree(gpu_mapcharge_array);

}

bool SortPixels(const SiPixelCluster::Pixel& i,
                const SiPixelCluster::Pixel& j) {
  return (i.adc > j.adc);



}

void JetCoreClusterSplitter2::produce(edm::Event& iEvent,
                                      const edm::EventSetup& iSetup) {


  using namespace edm;
  edm::ESHandle<GlobalTrackingGeometry> geometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(geometry);

  Handle<edmNew::DetSetVector<SiPixelCluster> > inputPixelClusters;
  iEvent.getByLabel(pixelClusters_, inputPixelClusters);
  Handle<edmNew::DetSetVector<SiPixelCluster> > inputPixelClustersIDEAL;
  iEvent.getByLabel("IdealsiPixelClusters", inputPixelClustersIDEAL);

  Handle<std::vector<reco::Vertex> > vertices;
  iEvent.getByLabel(vertices_, vertices);
  const reco::Vertex& pv = (*vertices)[0];
  Handle<std::vector<reco::CaloJet> > jets;
  iEvent.getByLabel("ak5CaloJets", jets);
  edm::ESHandle<PixelClusterParameterEstimator> pe;
  const PixelClusterParameterEstimator* pp;
  iSetup.get<TkPixelCPERecord>().get(pixelCPE_, pe);
  pp = pe.product();

  std::auto_ptr<edmNew::DetSetVector<SiPixelCluster> > output(
      new edmNew::DetSetVector<SiPixelCluster>());
  // ogni modulo del detector ha il suo array di clusters.

  edmNew::DetSetVector<SiPixelCluster>::const_iterator detIt =
      inputPixelClusters->begin();
  // Si loopa sui moduli
  // inputPixelClusters contiene i puntatori agli array dei moduli
  for (; detIt != inputPixelClusters->end(); detIt++) {
    // detIt->id() e' l'id del detector

    edmNew::DetSetVector<SiPixelCluster>::FastFiller filler(*output,
                                                            detIt->id());
    const edmNew::DetSet<SiPixelCluster>& detset = *detIt;

    // geometria del detector vuole il codice del detector

    const GeomDet* det = geometry->idToDet(detset.id());

    // si loopa sui clusters
    for (edmNew::DetSet<SiPixelCluster>::const_iterator cluster =
             detset.begin();
         cluster != detset.end(); cluster++) {
      const SiPixelCluster& aCluster = *cluster;
      bool hasBeenSplit = false;
      // bisogna passare dalla geometria locale a quella globale

      GlobalPoint cPos = det->surface().toGlobal(
          pp->localParametersV(aCluster,
                               (*geometry->idToDetUnit(detIt->id())))[0].first);
      // ppv e' il vertice primario

      GlobalPoint ppv(pv.position().x(), pv.position().y(), pv.position().z());
      // vettore direzione del cluster

      GlobalVector clusterDir = cPos - ppv;
      for (std::vector<reco::CaloJet>::const_iterator jit = jets->begin();
           jit != jets->end(); jit++) {
        // only high pt jets are considered

        if (jit->pt() > 100) {
          float jetZOverRho = jit->momentum().Z() / jit->momentum().Rho();

          // in the endcap it is rho/z
          if (fabs(cPos.z()) > 30)
            jetZOverRho = jit->momentum().Rho() / jit->momentum().Z();
          GlobalVector jetDir(jit->momentum().x(), jit->momentum().y(),
                              jit->momentum().z());
          unsigned int maxSizeY =
              fabs(sqrt(1.3 * 1.3 + 1.9 * 1.9 * jetZOverRho * jetZOverRho));
          //					unsigned int
          //maxSizeY=fabs(jetZOverRho*1.9);

          //					unsigned int
          //maxSizeY=fabs(jetZOverRho*1.75)+0.5;

          if (maxSizeY < 1) maxSizeY = 1;
          // Only the core of the Jet is considered
          // only anomalous clusters are considered (high charge, or size y
          // higher than expected
          if (Geom::deltaR(jetDir, clusterDir) < 0.05 &&
              aCluster.charge() > 30000 &&
              (aCluster.sizeX() > 2 ||
               ((unsigned int)aCluster.sizeY()) > maxSizeY + 1)) {
            std::cout << "\n\nCHECK FOR NEW SPLITTING: charge and deltaR "
                      << aCluster.charge() << " "
                      << Geom::deltaR(jetDir, clusterDir) << " size x y "
                      << aCluster.sizeX() << " " << aCluster.sizeY()
                      << " detid " << detIt->id() << std::endl;
            std::cout << "jetZOverRho=" << jetZOverRho << std::endl;
            // MC Truth
            // inputpixelclusterideal e' un array di cluster e find ci ritorna
            // l'array di cluster che sono sul modulo
            SiPixelClusterCollectionNew::const_iterator myDet =
                inputPixelClustersIDEAL->find(detIt->id());
            const edmNew::DetSet<SiPixelCluster>& idealClusters = (*myDet);
            // jetZOverRho serve perche' se va obliquo lascia piu' carica

            int binjetZOverRho =
                jetZOverRho / jetZOverRhoWidth + BinsJetOverRho / 2;
            maxSizeY = fabs(
                sqrt(1.3 * 1.3 +
                     1.9 * 1.9 * (0.5 + binjetZOverRho - BinsJetOverRho / 2) *
                         jetZOverRhoWidth *
                         (0.5 + binjetZOverRho - BinsJetOverRho / 2) *
                         jetZOverRhoWidth));
            if (split(aCluster, filler,
                      sqrt(1.08 + jetZOverRho * jetZOverRho) * 26000, maxSizeY,
                      jetZOverRho, idealClusters)) {
              hasBeenSplit = true;
            }
            std::cout << "IDEAL was : " << std::endl;
            int xmin = aCluster.minPixelRow();
            int ymin = aCluster.minPixelCol();
            int xmax = aCluster.maxPixelRow();
            int ymax = aCluster.maxPixelCol();
            int last = 1;
            std::map<int, int> sh;
            for (int x = xmin; x <= xmax; x++) {
              for (int y = ymin; y <= ymax; y++) {
                int h = 0;
                int flag = 0;
                for (edmNew::DetSet<SiPixelCluster>::const_iterator
                         clusterIt = myDet->begin();
                     clusterIt != myDet->end(); clusterIt++, h++) {

                  std::vector<SiPixelCluster::Pixel> pixels =
                      clusterIt->pixels();
                  for (unsigned int j = 0; j < pixels.size(); j++) {
                    if (pixels[j].x == x && pixels[j].y == y) {
                      if (!sh[h]) {
                        sh[h] = last;
                        last++;
                      }
                      flag |= (1 << (sh[h] - 1));
                    }
                  }
                }

                std::cout << " ";
                bin(flag, std::cout);
                // std::setiosflags(std::ios::fixed)
                //                                << std::setprecision(0)
                //                              << std::setw(7)
                //                            << std::left ; bin(
                //                            flag,std::cout);
                //                                << std::left << hex << flag;
              }
              std::cout << std::endl;
            }
            int h = 0;
            for (edmNew::DetSet<SiPixelCluster>::const_iterator
                     clusterIt = myDet->begin();
                 clusterIt != myDet->end(); clusterIt++, h++) {
              if (sh[h])
                std::cout << "IDEAL POS: " << h
                          << " x: " << std::setprecision(2) << clusterIt->x()
                          << " y: " << clusterIt->y()
                          << " c: " << clusterIt->charge() << std::endl;
            }


          }
        }
      }
      if (!hasBeenSplit) {
        // blowup the error
        SiPixelCluster c = aCluster;

        filler.push_back(c);
      }
    }
  }
  iEvent.put(output);
}

float JetCoreClusterSplitter2::distanceCluster(
    const SiPixelCluster& cluster,
    const edmNew::DetSet<SiPixelCluster>& idealClusters) {
  float minDistance = 1e99;
  for (edmNew::DetSet<SiPixelCluster>::const_iterator ideal =
           idealClusters.begin();
       ideal < idealClusters.end(); ideal++) {
    float distance = sqrt(
        (cluster.x() - ideal->x()) * (cluster.x() - ideal->x()) +
        (cluster.y() - ideal->y()) * (cluster.y() - ideal->y()) * 1.5 * 1.5);
    if (distance < minDistance) minDistance = distance;
  }
  return minDistance;
}

bool JetCoreClusterSplitter2::split(
    const SiPixelCluster& aCluster,
    edmNew::DetSetVector<SiPixelCluster>::FastFiller& filler, float expectedADC,
    int sizeY, float jetZOverRho,
    const edmNew::DetSet<SiPixelCluster>& idealClusters) {
  std::vector<SiPixelCluster> sp =
      fittingSplit(aCluster, expectedADC, sizeY, jetZOverRho);

  for (unsigned int i = 0; i < sp.size(); i++) {
    float distance =
        JetCoreClusterSplitter2::distanceCluster(sp[i], idealClusters);
    std::cout << "NEW POS: " << i << " x: " << std::setprecision(2) << sp[i].x()
              << " y: " << sp[i].y() << " c: " << sp[i].charge()
              << " distance=" << 100 * distance << " um" << std::endl;
    filler.push_back(sp[i]);
  }

  int xmin = aCluster.minPixelRow();
  int ymin = aCluster.minPixelCol();
  int xmax = aCluster.maxPixelRow();
  int ymax = aCluster.maxPixelCol();
  std::cout << "Splitted clusters map:" << std::endl;
  for (int x = xmin; x <= xmax; x++) {
    for (int y = ymin; y <= ymax; y++) {
      int flag = 0;
      for (unsigned int i = 0; i < sp.size(); i++) {

        std::vector<SiPixelCluster::Pixel> pixels = sp[i].pixels();
        for (unsigned int j = 0; j < pixels.size(); j++) {
          if (pixels[j].x == x && pixels[j].y == y) flag |= (1 << i);
        }
      }

      std::cout << " ";
      bin(flag, std::cout);

    }
    std::cout << std::endl;
  }
  return (sp.size() > 0);
}

float JetCoreClusterSplitter2::pixelWeight2(int clx, int cly, int x, int y,
                                            int sizeY, int direction) {
  if (direction > 1 || direction < 0) return 0;
  float fact = 0;
  if (x == clx && (y >= cly && y < cly + (sizeY + 1) / 2)) fact = 2;
  if (x == clx + 1 && direction && (y >= cly + (sizeY + 1) / 2) &&
      y < cly + sizeY)
    fact = 2;
  if (x == clx - 1 && !direction && (y >= cly + (sizeY + 1) / 2) &&
      y < cly + sizeY)
    fact = 2;
  if (x == clx && (y >= cly + (sizeY + 1) / 2) && y < cly + sizeY) fact = 1;
  if (x == clx + 1 && direction && (y >= cly && y < cly + (sizeY + 1) / 2))
    fact = 1;
  if (x == clx - 1 && !direction && (y >= cly && y < cly + (sizeY + 1) / 2))
    fact = 1;
  if (x == clx + 1 && direction && y == cly + sizeY) fact = 1;
  if (x == clx - 1 && !direction && y == cly + sizeY) fact = 1;
  if (x == clx && y == cly - 1) fact = 1;
  return fact / (0.5 + sizeY) / 4.;
  // return fact/(0.5+sizeY)/4.;
}

// clx posizione in interi del centro del cluster
// x y posizione del pixel in ingresso
// sizeY attesa del cluster
// direction numero da 0 a 4 /2 da sopra o sotto
// bintheta e' la JetZOverRho espressa in bin
// clx e' la posizione x del cluster, e binx ti dice la regione da 1 a 5

float JetCoreClusterSplitter2::pixelWeight(int clx, int cly, int x, int y,
                                           int sizeY, int direction,
                                           int bintheta) {

  if (x - clx + 10 < -BinsX) return 0;
  if (y - cly + (sizeY + 1) / 2 < 0) return 0;
  if (x - clx + 10 >= BinsX) return 0;
  if (y - cly + (sizeY + 1) / 2 >= BinsY) return 0;


  if (bintheta < 0) {
    cout << "Forced bintheta=0. It was " << bintheta;
    bintheta = 0;
  }
  if (bintheta >= BinsJetOverRho) {
    cout << "Forced bintheta=BinsJetOverRho-1. It was " << bintheta;
    bintheta = BinsJetOverRho - 1;
  }

  int caseX = direction / 2;
  direction = direction % 2;

  direction = direction + 1;

  unsigned int binX = clx * BinsXposition / 160;
  sizeY = sizeY + (direction - 1);
  // fact e' la percentuale di carica attesa in quel pixel dato un cluster
  // mapcharge e' la carica media rilasciata da un cluster in quel pixel
  // count e' il numero totale di cluster su quel pixel
  float fact = 1. * mapcharge[bintheta][binX][direction][x - clx + 10 + caseX]
                             [y - cly + (sizeY - 1) / 2] /
               totalcharge[bintheta][binX][direction] *
               count[bintheta][binX][direction];

  return fact;
}
unsigned int JetCoreClusterSplitter2::combinations(unsigned int npos,
		unsigned int expectedClusters) {
  // combination with repetition ( n+k-1  over k )
	unsigned int up = npos + expectedClusters - 1;
	unsigned int down = expectedClusters;
	unsigned int fdown = 1, prod = 1;
  for (unsigned int i = npos; i <= up; i++) prod *= i;

  for (unsigned int i = 1; i <= down; i++) fdown *= i;
  return prod / fdown;
}

std::vector<SiPixelCluster> JetCoreClusterSplitter2::fittingSplit(
    const SiPixelCluster& aCluster, float expectedADC, int sizeY,
    float jetZOverRho) {
  const float diecimila = 10000;

  //	unsigned int meanExp = floor(aCluster.charge() / expectedADC +0.5) ;
  unsigned int meanExp = ceil(aCluster.charge() / expectedADC);
  // output e' una collezione di clusters

  std::vector<SiPixelCluster> output;
  if (meanExp == 0) {
    std::cout << "ZERO????" << std::endl;
  }
  if (meanExp <= 1) {
    output.push_back(aCluster);
    return output;
  }
  int xmin = aCluster.minPixelRow();
  int ymin = aCluster.minPixelCol();
  int xmax = aCluster.maxPixelRow();
  int ymax = aCluster.maxPixelCol();
  int binjetZOverRho = jetZOverRho / jetZOverRhoWidth + BinsJetOverRho / 2;
  if (binjetZOverRho < 0) binjetZOverRho = 0;
  if (binjetZOverRho > BinsJetOverRho - 1) binjetZOverRho = BinsJetOverRho - 1;
  std::vector<int> bestcomb;
  unsigned int bestExpCluster = meanExp;
  std::vector<SiPixelCluster::Pixel> pixels = aCluster.pixels();
  sort(pixels.begin(), pixels.end(), SortPixels);
  float chiN = -1;
  float chimin = 1e99;
  if (meanExp > 7) meanExp = 7;

  bool forced = false;
  for (unsigned int expectedClusters = meanExp - 1; expectedClusters <= meanExp;
       expectedClusters++) {
    float chiminlocal = 1e99;
    nDirections = 4;
    int deltay = ymax - ymin - sizeY + 1;
    if (deltay < 1) deltay = 1;

    int npos = pixels.size() * nDirections;
#ifdef COMBINATION_CUT_

    float approxComb = combinations(npos, expectedClusters);
    unsigned long maxComb = approxComb;
    unsigned int limitComb = 10000;
    if (approxComb > 1e9) maxComb = limitComb + 1;


    std::cout << "combination=" << maxComb << "(" << approxComb
              << ") npos=" << npos << "expectedClusters=" << expectedClusters
              << " elapsed time=" << 0.000437043 * maxComb << endl;
    if (maxComb > limitComb) {
      forced = true;
      npos = pixels.size() * nDirections;
      maxComb = combinations(npos, expectedClusters);


      while (combinations(npos, expectedClusters) > limitComb &&
             npos > int(nDirections)) {
        npos -= nDirections;
      }

      maxComb = combinations(npos, expectedClusters);

      std::cout << "Forced: combination=" << maxComb << " npos=" << npos
                << "expectedClusters=" << expectedClusters
                << " elapsed time=" << 0.000437043 * maxComb << endl;
    }

    if (maxComb > limitComb) {
      std::cout << "toomany" << std::endl;
      int xmin = aCluster.minPixelRow();
      int ymin = aCluster.minPixelCol();
      int xmax = aCluster.maxPixelRow();
      int ymax = aCluster.maxPixelCol();
      std::cout << "Order of hits in the TOOMANY:" << std::endl;
      for (int x = xmin; x <= xmax; x++) {
        for (int y = ymin; y <= ymax; y++) {
          int flag = 0;
          std::vector<SiPixelCluster::Pixel> pixels = aCluster.pixels();
          for (unsigned int j = 0; j < pixels.size(); j++) {
            if (pixels[j].x == x && pixels[j].y == y) flag = j;
          }

          std::cout << " " << std::setiosflags(std::ios::fixed)
                    << std::setprecision(0) << std::setw(7) << std::left
                    << flag;
        }
        std::cout << std::endl;
      }
      return std::vector<SiPixelCluster>();
    }
#endif

	// FP: GPU if expected Clusters > 2
	if(expectedClusters > 2)
	{
	    // FP: copy only the initial adc array to the GPU


	    for (unsigned int i = 0; i < pixels.size(); i++) {
	    	originalADC[pixels[i].x + 500*pixels[i].y]= pixels[i].adc;
	    }
	    cudaMemcpyAsync(gpu_originalADC,originalADC,250000*sizeof(int), cudaMemcpyHostToDevice,0);


	    //Kernel goes here
	    cudawClusterSplitter(gpu_mapcharge_array);




	}
	else
	{
	    SiPixelArrayBuffer theOriginalBuffer(500, 500);

	    for (unsigned int i = 0; i < pixels.size(); i++) {
	      int x = pixels[i].x;
	      int y = pixels[i].y;
	      int adc = pixels[i].adc;
	      theOriginalBuffer.set_adc(x, y, adc);
	    }

    std::vector<int> comb(expectedClusters);

// need to parallelize this while loop
    while (nextCombination(comb, npos))
    {

      float chi2 = 0;
      SiPixelArrayBuffer theBuffer;
      theBuffer.setSize(500, 500);
      for (unsigned int i = 0; i < pixels.size(); i++) {
        int x = pixels[i].x;
        int y = pixels[i].y;
        int adc = pixels[i].adc;
        theBuffer.set_adc(x, y, adc);
      }

      float prob = 0;
      for (unsigned int cl = 0; cl < expectedClusters; cl++) {
        int pi = comb[cl];
        // nDirections sono 4: sotto o sopra, e i due interi intorno a sizeY
        // attesa (che e' un float)

        int clx = pixels[pi / nDirections].x;
        int cly = pixels[pi / nDirections].y;
        int direction = pi % nDirections;

        for (int x = xmin - 5; x <= xmax + 5; x++) {
          for (int y = ymin - (sizeY + 1) / 2; y <= ymax + (sizeY + 1) / 2;
               y++) {
            if (x < 0 || y < 0) continue;
            float fact =
                pixelWeight(clx, cly, x, y, sizeY, direction, binjetZOverRho);
            if (fact > 0) {
              theBuffer.set_adc(x, y, theBuffer(x, y) - fact * expectedADC);
            }

          }
        }
        // Dato che testiamo due possibili posizioni in y del cluster (quello
        // atteso e
        // quello atteso + 1, dobbiamo favorire quello atteso
        // si guarda quante volte ha colpito size Y+1
        prob += count[binjetZOverRho][int((clx) * 5. / 160)][direction % 2 + 1];
      }
      // print(theBuffer,aCluster);
      for (int x = xmin - 5; x <= xmax + 5; x++) {
        for (int y = ymin - (sizeY + 1) / 2; y <= ymax + (sizeY + 1) / 2; y++) {

          float res = theBuffer(x, y);
          float charge = theOriginalBuffer(x, y) -
                         theBuffer(x, y);  // charge assigned to this pixel
          float chargeMeasured =
              theOriginalBuffer(x, y);  // charge assigned to this pixel

          //					if(res< 0 ) { //threshold effect
          //						if(res > -10000){
          //							if(res<-5000)
          //res+=5000;
          //							else  res=0;
          //						}
          //					}
          //									if(res> 0 && charge > 7000 ) {
          ////reduce weights of landau tails
          //										res*=0.7;
          //									}

          if (chargeMeasured < 5000 && abs(charge) < 5000) {  // threshold
                                                              // effect
            res = 0;
          }

          if (chargeMeasured <= 2000) chargeMeasured = 2000;
          if (fabs(charge) < 2000) charge = 2000;

          chi2 += (res * res) / (charge * charge);

        }
      }
      prob /= expectedClusters;
      chi2 /= prob;

      if (chi2 < chiminlocal) {
        chiminlocal = chi2;
      }
      if (chi2 < chimin) {
        chiN = chi2 * prob / aCluster.size();
        chimin = chi2;
        bestcomb = comb;
        bestExpCluster = expectedClusters;
      }
    }
  }
    std::cout << " chiN " << chiN << " sizeY  " << sizeY << " exADC "
              << expectedADC << std::endl;
    std::cout << " chi " << std::setprecision(7) << chiminlocal << std::endl;
  }
  if (forced)
    finalizeSplitting(bestcomb, bestExpCluster, pixels, aCluster, expectedADC,
                      sizeY, jetZOverRho);
  SiPixelArrayBuffer myResidual;
  myResidual.setSize(500, 500);
  std::cout << " Expected clusters was " << meanExp << " best comb for "
            << bestExpCluster << " " << bestcomb.size() << std::endl;
  std::cout << " Difference clusters: " << int(bestExpCluster) - int(meanExp)
            << std::endl;
  unsigned int expectedClusters = bestExpCluster;
  // get and print the residual
  for (unsigned int i = 0; i < pixels.size(); i++) {
    int x = pixels[i].x;
    int y = pixels[i].y;
    int adc = pixels[i].adc;
    myResidual.set_adc(x, y, adc);
  }

  for (unsigned int cl = 0; cl < expectedClusters; cl++) {
    int pi = bestcomb[cl];
    int clx = pixels[pi / nDirections].x;
    int cly = pixels[pi / nDirections].y;
    int direction = pi % nDirections;
    for (int x = xmin - 5; x <= xmax + 5; x++) {
      for (int y = ymin - (sizeY + 1) / 2; y <= ymax + (sizeY + 1) / 2; y++) {
        if (x < 0 || y < 0) continue;
        float fact =
            pixelWeight(clx, cly, x, y, sizeY, direction, binjetZOverRho);
        if (fact > 0) {
          myResidual.set_adc(x, y, myResidual(x, y) - fact * expectedADC);
        }
      }
    }
  }

  cout << endl << "clx-xmin,cly-ymin = ";
  for (unsigned int cl = 0; cl < expectedClusters; cl++) {
    int pi = bestcomb[cl];
    int clx = pixels[pi / nDirections].x - xmin;
    int cly = pixels[pi / nDirections].y - ymin;
    cout << clx << "," << cly << ";";
  }
  cout << endl;
  std::cout << "Residual:" << std::endl;
  print(myResidual, aCluster);

  // End: get and print the residual

  SiPixelArrayBuffer theWeights;
  theWeights.setSize(500, 500);

  std::cout << "best combination chi: " << std::setprecision(7) << chimin
            << " co ";
  for (unsigned int ii = 0; ii < bestcomb.size(); ii++) {
    std::cout << bestcomb[ii] << " ";
  }
  std::cout << std::endl;


  SiPixelArrayBuffer theBuffer;
  theBuffer.setSize(500, 500);
  SiPixelArrayBuffer theBufferResidual;
  theBufferResidual.setSize(500, 500);

  for (unsigned int i = 0; i < pixels.size(); i++) {
    int x = pixels[i].x;
    int y = pixels[i].y;
    int adc = pixels[i].adc;
    theBuffer.set_adc(x, y, adc);
    theBufferResidual.set_adc(x, y, adc);
  }
  print(theBuffer, aCluster);

  for (unsigned int cl = 0; cl < expectedClusters; cl++) {
    int pi = bestcomb[cl];
    int clx = pixels[pi / nDirections].x;
    int cly = pixels[pi / nDirections].y;
    int direction = pi % nDirections;

    for (int x = xmin - 5; x <= xmax + 5; x++) {
      for (int y = ymin - (sizeY + 1) / 2; y <= ymax + (sizeY + 1) / 2 + 1;
           y++) {
        if (x < 0 || y < 0) continue;
        float fact =
            pixelWeight(clx, cly, x, y, sizeY, direction, binjetZOverRho);

        if (fact > 0.05) {

          theWeights.set_adc(x, y, theWeights(x, y) + fact * diecimila);

        }
      }
    }
  }

  for (unsigned int cl = 0; cl < expectedClusters; cl++) {
    std::cout << "Cluster " << cl << " weight map" << std::endl;
    int pi = bestcomb[cl];  //((combination / clbase)%remainingFreePositions);
    int clx = pixels[pi / nDirections].x;
    int cly = pixels[pi / nDirections].y;
    int direction = pi % nDirections;
    unsigned int binX = clx * 5. / 160;
    std::cout << "binjetZOverRho=" << binjetZOverRho << " binX=" << binX
              << " direction=" << direction << " sizeY=" << sizeY << endl;
    // std::cout << "Cluster  "<< cl << " pi " << pi/2 << " dir " << direction
    // << " x,y " << clx -xmin <<", " << cly-ymin<< " clbase " << clbase <<
    // std::endl;
    for (int x = xmin - 1; x <= xmax + 1; x++) {
      for (int y = ymin - (sizeY + 1) / 2; y <= ymax + (sizeY + 1) / 2 + 1;
           y++) {
        if (x < 0 || y < 0) continue;
        float fact =
            pixelWeight(clx, cly, x, y, sizeY, direction, binjetZOverRho);
        std::cout << std::setprecision(2) << float(fact) << " ";
        //              if(fact > 0)
        //              theBuffer.set_adc(x,y,theBuffer(x,y)-fact*perPixel);
        //              std::cout << "residual in "<< x-xmin <<","<< y-ymin<< "
        //              " << theBuffer(x,y)-fact*perPixel << "  fact " << fact
        //              << " exp:"<< fact*perPixel <<std::endl;
      }
      std::cout << std::endl;
    }
    std::setprecision(0);
  }
  // really fill clusters
  //	clbase=1;
  //	remainingFreePositions = npos;
  for (unsigned int cl = 0; cl < expectedClusters; cl++) {
    int pi = bestcomb[cl];
    int clx = pixels[pi / nDirections].x;
    int cly = pixels[pi / nDirections].y;
    int direction = pi % nDirections;
    //		remainingFreePositions=pi+1; // cl_i+1 <= cl_i
    SiPixelCluster* cluster = 0;
    //		clbase*=remainingFreePositions;
    for (int x = xmin - 5; x <= xmax + 5; x++) {
      for (int y = ymin - (sizeY + 1) / 2; y <= ymax + (sizeY + 1) / 2 + 1;
           y++) {
        if (x < 0 || y < 0) continue;
        float fact =
            pixelWeight(clx, cly, x, y, sizeY, direction, binjetZOverRho);

        if (fact > 0.05) {
          float charge =
              1. * theBuffer(x, y) * fact * diecimila / theWeights(x, y);
          std::cout << std::setprecision(2) << "HEREtheWeights "
                    << theWeights(x, y) << " fact: " << fact << " x y "
                    << x - xmin << " " << y - ymin << " charge " << charge
                    << std::endl;
          if (charge > 0) {
            theBufferResidual.set_adc(x, y, theBufferResidual(x, y) - charge);
            if (cluster) {
              SiPixelCluster::PixelPos newpix(x, y);
              cluster->add(newpix, charge);
            } else {
              SiPixelCluster::PixelPos newpix(x, y);
              cluster =
                  new SiPixelCluster(newpix, charge);  // create protocluster
            }
          }
        }
      }
    }
    if (cluster) {
      output.push_back(*cluster);
      delete cluster;
    }
  }
  std::cout << "Weights" << std::endl;
  print(theWeights, aCluster, 1);
  std::cout << "Unused charge" << std::endl;
  print(theBufferResidual, aCluster);

  return output;
}

bool JetCoreClusterSplitter2::nextCombination(std::vector<int>& comb,
                                              int npos) {
  comb[comb.size() - 1] += 1;  // increment
  for (int i = comb.size() - 1; i >= 0; i--) {
    if (i > 0 && comb[i] > comb[i - 1]) {
      comb[i] = 0;
      comb[i - 1] += 1;
    }
    if (i == 0 && comb[i] >= npos) {
      return false;
    }
  }
  std::cout << "nposizioni: " << npos << " combinazione: " << std::endl;
  for( std::vector<int>::const_iterator i = comb.begin(); i != comb.end(); ++i)
      std::cout << *i << ' ';
  std::cout<< " "<< std::endl;
  return true;
}
void JetCoreClusterSplitter2::print(const SiPixelArrayBuffer& b,
                                    const SiPixelCluster& c, int div) {
  int xmin = c.minPixelRow();
  int ymin = c.minPixelCol();
  int xmax = c.maxPixelRow();
  int ymax = c.maxPixelCol();
  for (int x = xmin - 5; x <= xmax + 5; x++) {
    for (int y = ymin - 5; y <= ymax + 5; y++) {
      if (x < 0 || y < 0) continue;
      if (b(x, y) != 0 && x < xmin) xmin = x;
      if (b(x, y) != 0 && y < ymin) ymin = y;
      if (b(x, y) != 0 && x > xmax) xmax = x;
      if (b(x, y) != 0 && y > ymax) ymax = y;
    }
  }

  for (int x = xmin; x <= xmax; x++) {
    for (int y = ymin; y <= ymax; y++) {
      std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(0)
                << std::setw(6) << std::left << b(x, y) / div;
    }
    std::cout << std::endl;
  }
}


void JetCoreClusterSplitter2::finalizeSplitting(
    std::vector<int>& bestcomb, unsigned int& expectedClusters,
    std::vector<SiPixelCluster::Pixel>& pixels, const SiPixelCluster& aCluster,
    float expectedADC, int sizeY, float jetZOverRho) {
  //	const float diecimila=10000;
  //	unsigned int expectedClusters = floor(aCluster.charge() / expectedADC
  //+0.5) ;
  //	expectedClusters = floor(aCluster.charge() / expectedADC +0.5) ;
  if (expectedClusters == 0) {
    std::cout << "ZERO????" << std::endl;
  }
  if (bestcomb.size() < expectedClusters)
    cout << "Adding " << expectedClusters - bestcomb.size() << " clusters"
         << endl;
  for (unsigned int i = bestcomb.size(); i < expectedClusters; i++) {
    bestcomb[i] = bestcomb[i - bestcomb.size()];
  }
  //	if(bestcomb.size()<=);
  int xmin = aCluster.minPixelRow();
  int ymin = aCluster.minPixelCol();
  int xmax = aCluster.maxPixelRow();
  int ymax = aCluster.maxPixelCol();
  int binjetZOverRho = jetZOverRho / jetZOverRhoWidth + BinsJetOverRho / 2;
  if (binjetZOverRho < 0) binjetZOverRho = 0;
  if (binjetZOverRho > BinsJetOverRho) binjetZOverRho = BinsJetOverRho;

  //	std::vector<SiPixelCluster::Pixel> pixels = aCluster.pixels();
  sort(pixels.begin(), pixels.end(), SortPixels);

  float perPixel = expectedADC;  /// become per unit weight 1./(0.5+sizeY)/4.;
  float chimin = 1e99;

  SiPixelArrayBuffer theOriginalBuffer;
  theOriginalBuffer.setSize(500, 500);
  for (unsigned int i = 0; i < pixels.size(); i++) {
    int x = pixels[i].x;
    int y = pixels[i].y;
    int adc = pixels[i].adc;
    theOriginalBuffer.set_adc(x, y, adc);
  }

  std::vector<int> finalDx(expectedClusters);
  std::vector<int> finalDy(expectedClusters);

  unsigned int consecutiveNoChange = 0;
  unsigned int pixelIt = 0;
  bool first = true;
  if (expectedClusters > 1)
    while (consecutiveNoChange <= expectedClusters) {
      SiPixelArrayBuffer theBuffer;
      theBuffer.setSize(500, 500);

      for (unsigned int i = 0; i < pixels.size(); i++) {
        int x = pixels[i].x;
        int y = pixels[i].y;
        int adc = pixels[i].adc;
        theBuffer.set_adc(x, y, adc);
      }

      float prob = 0;
      int posX = -1;
      int posY = -1;
      int dir = -1;
      for (unsigned int cl = 0; cl < expectedClusters; cl++) {
        int pi =
            bestcomb[cl];  //((combination / clbase)%remainingFreePositions);
        if (cl == pixelIt) {
          posX = pixels[pi / nDirections].x + finalDx[cl];
          posY = pixels[pi / nDirections].y + finalDy[cl];
          dir = pi % nDirections;
          prob += count[binjetZOverRho][int((posX) * 5. / 160)][dir % 2 + 1];
          continue;
        }
        int clx = pixels[pi / nDirections].x + finalDx[cl];
        int cly = pixels[pi / nDirections].y + finalDy[cl];
        int direction = pi % nDirections;
        for (int x = xmin - 5; x <= xmax + 5; x++) {
          for (int y = ymin - (sizeY + 1) / 2; y <= ymax + (sizeY + 1) / 2;
               y++) {
            if (x < 0 || y < 0) continue;
            float fact =
                pixelWeight(clx, cly, x, y, sizeY, direction, binjetZOverRho);
            if (fact > 0) {
              theBuffer.set_adc(x, y, theBuffer(x, y) - fact * perPixel);
            }
          }
        }
        prob += count[binjetZOverRho][int((clx) * 5. / 160)][direction % 2 + 1];
      }
      if (posX < 0 || posY < 0 || dir < 0) cout << "*******ERROR*********";
      int bestDx = 0, bestDy = 0;
      for (int dX = -1; dX <= 1; dX++) {
        for (int dY = -1; dY <= 1; dY++) {
          if ((posX + dX) < 0 || (posY + dY) < 0) continue;

          if (dX == 0 && dY == 0 && !first) continue;
          float chi2 = 0;
          for (int x = xmin - 5; x <= xmax + 5; x++) {
            for (int y = ymin - (sizeY + 1) / 2; y <= ymax + (sizeY + 1) / 2;
                 y++) {
              //				std::cout << theBuffer(x,y)/1000
              //<< " " ;
              float res = theBuffer(x, y);
              res -= float(pixelWeight(posX + dX, posY + dY, x, y, sizeY, dir,
                                       binjetZOverRho)) *
                     perPixel;
              float charge = theOriginalBuffer(x, y) -
                             res;  // charge assigned to this pixel
              float chargeMeasured =
                  theOriginalBuffer(x, y);  // charge assigned to this pixel

              if (res < 0) {  // threshold effect
                if (res > -10000) {
                  if (res < -5000)
                    res += 5000;
                  else
                    res = 0;
                }
              }
              //				if(res> 0 && charge > 7000 ) {
              ////reduce weights of landau tails
              //					res*=0.7;
              //				}

              if (chargeMeasured < 5000 &&
                  abs(charge) < 5000) {  // threshold effect
                res = 0;
              }

              if (charge == 0) charge = 2000;
              chi2 += (res * res) / (charge * charge);
            }
          }
          //		chi2/=prob
          //	bool changed=false;
          if (chi2 < chimin) {
            chimin = chi2;
            bestDx = dX;
            bestDy = dY;
          }
        }
      }
      if (bestDx == 0 && bestDy == 0)
        consecutiveNoChange++;
      else
        consecutiveNoChange = 0;
      if (consecutiveNoChange == 0)
        cout << "AfterSplitting:  cluster=" << pixelIt
             << " in x=" << finalDx[pixelIt] + bestDx
             << " y=" << finalDy[pixelIt] + bestDy
             << " instead of x=" << finalDx[pixelIt]
             << " y=" << finalDy[pixelIt] << " chimin=" << chimin << endl;
      if (consecutiveNoChange == 0) {
        finalDx[pixelIt] += bestDx;
        finalDy[pixelIt] += bestDy;
      } else
        pixelIt++;
      if (first) first = false;
      if (pixelIt >= expectedClusters) pixelIt = 0;
    }
  //	//write bestcomb
  for (unsigned int cl = 0; cl < expectedClusters; cl++) {
    if (finalDx[cl] == 0 && finalDy[cl] == 0) continue;
    int pi = bestcomb[cl];
    int clx = pixels[pi / nDirections].x + finalDx[cl];
    int cly = pixels[pi / nDirections].y + finalDy[cl];
    int direction = pi % nDirections;
    bool found = false;
    unsigned int pix = 0;
    for (; pix < pixels.size() && !found; pix++) {
      if (clx == pixels[pix].x && cly == pixels[pix].y) {
        found = true;
      }
    }
    pix--;

    if (!found) {
      SiPixelCluster::Pixel newpixel = SiPixelCluster::Pixel();
      newpixel.x = clx;
      newpixel.y = cly;
      newpixel.adc = 1;
      pixels.push_back(newpixel);
      cout << "Added pixel in(x,y)=" << clx << " " << cly << endl;
      pix++;
    }
    pi = pix * nDirections + direction;
    bestcomb[cl] = pi;
  }
}

void JetCoreClusterSplitter2::cudawClusterSplitter(int* tex) { cudaClusterSplitter_(tex); }


#include "RecoLocalTracker/SubCollectionProducers/interface/chargeNewSizeY.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetCoreClusterSplitter2);
