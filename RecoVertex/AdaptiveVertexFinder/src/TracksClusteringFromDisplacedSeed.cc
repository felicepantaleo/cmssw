#include "RecoVertex/AdaptiveVertexFinder/interface/TracksClusteringFromDisplacedSeed.h"
//#define VTXDEBUG 1


TracksClusteringFromDisplacedSeed::TracksClusteringFromDisplacedSeed(const edm::ParameterSet &params) :
//	maxNTracks(params.getParameter<unsigned int>("maxNTracks")),
	min3DIPSignificance(params.getParameter<double>("seedMin3DIPSignificance")),
	min3DIPValue(params.getParameter<double>("seedMin3DIPValue")),
	clusterMaxDistance(params.getParameter<double>("clusterMaxDistance")),
        clusterMaxSignificance(params.getParameter<double>("clusterMaxSignificance")), //3
        clusterScale(params.getParameter<double>("clusterScale")),//10.
        clusterMinAngleCosine(params.getParameter<double>("clusterMinAngleCosine")), //0.0
        angleScalingWithGamma(params.getParameter<bool>("angleScalingWithGamma")),
        maxCosineCut(params.getParameter<double>("maxCosineCut")),
        maxPixelResidual(params.getParameter<double>("maxPixelResidual")),
        minPtForMissingPXB1(params.getParameter<double>("minPtForMissingPXB1"))


{
	
}

std::pair<std::vector<reco::TransientTrack>,GlobalPoint> TracksClusteringFromDisplacedSeed::nearTracks(const reco::TransientTrack &seed, const std::vector<reco::TransientTrack> & tracks, const  reco::Vertex & primaryVertex) const
{
      VertexDistance3D distanceComputer;
      GlobalPoint pv(primaryVertex.position().x(),primaryVertex.position().y(),primaryVertex.position().z());
      std::vector<reco::TransientTrack> result;
      TwoTrackMinimumDistance dist;
      GlobalPoint seedingPoint;
      float sumWeights=0;
      std::pair<bool,Measurement1D> ipSeed = IPTools::absoluteImpactParameter3D(seed,primaryVertex);
      float pvDistance = ipSeed.second.value();
//      float densityFactor = 2./sqrt(20.*tracks.size()); // assuming all tracks being in 2 narrow jets of cone 0.3
      float densityFactor = 2./sqrt(20.*80); // assuming 80 tracks being in 2 narrow jets of cone 0.3
      for(std::vector<reco::TransientTrack>::const_iterator tt = tracks.begin();tt!=tracks.end(); ++tt )   {

       if(*tt==seed) continue;

       std::pair<bool,Measurement1D> ip = IPTools::absoluteImpactParameter3D(*tt,primaryVertex);
       if(dist.calculate(tt->impactPointState(),seed.impactPointState()))
            {
		 GlobalPoint ttPoint          = dist.points().first;
		 GlobalError ttPointErr       = tt->impactPointState().cartesianError().position();
	         GlobalPoint seedPosition     = dist.points().second;
	         GlobalError seedPositionErr  = seed.impactPointState().cartesianError().position();
                 Measurement1D m = distanceComputer.distance(VertexState(seedPosition,seedPositionErr), VertexState(ttPoint, ttPointErr));
                 GlobalPoint cp(dist.crossingPoint()); 


                 float distanceFromPV =  (dist.points().second-pv).mag();
                 float distance = dist.distance();
		 GlobalVector trackDir2D(tt->impactPointState().globalDirection().x(),tt->impactPointState().globalDirection().y(),0.); 
		 GlobalVector seedDir2D(seed.impactPointState().globalDirection().x(),seed.impactPointState().globalDirection().y(),0.); 
#ifdef VTXDEBUG     

		 float dotprodTrackSeed2D = trackDir2D.unit().dot(seedDir2D.unit());
#endif
		/* GlobalVector vtxDir = GlobalVector(extVertex.p4().X(),extVertex.p4().Y(),extVertex.p4().Z());
                 float gamma = (extVertex.position() - pv.position()).R()/0.05;
                      float gamma1 = extVertex.p4().Gamma();
                     if(Geom::deltaR(extVertex.position() - pv.position(), vtxDir)>1./gamma) continue; //pointing angle
		 */

                 float dotprodTrack = (dist.points().first-pv).unit().dot(tt->impactPointState().globalDirection().unit());
                 float dotprodSeed = (dist.points().second-pv).unit().dot(seed.impactPointState().globalDirection().unit());

		 math::XYZTLorentzVectorD sum;
		 ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > vec;
		 vec.SetPx(tt->track().px());
		 vec.SetPy(tt->track().py());
		 vec.SetPz(tt->track().pz());
		 vec.SetM(0.137);
		 sum += vec;
		 vec.SetPx(seed.track().px());
		 vec.SetPy(seed.track().py());
		 vec.SetPz(seed.track().pz());
		 vec.SetM(0.137);
		 sum += vec;
		 float gamma1= sum.Gamma();
		 float gamma = distanceFromPV/0.05;
		 float minCos = cos((gamma>gamma1)?(3./gamma):(3./gamma1));	
		 if(minCos > maxCosineCut) minCos = maxCosineCut;
		 if(!angleScalingWithGamma) minCos = clusterMinAngleCosine;

                 float w = distanceFromPV*distanceFromPV/(pvDistance*distance);
          	 bool selected = (m.significance() < clusterMaxSignificance && 
                    dotprodSeed > clusterMinAngleCosine && //Angles between PV-PCAonSeed vectors and seed directions
                    dotprodTrack > clusterMinAngleCosine && //Angles between PV-PCAonTrack vectors and track directions
		dotprodSeed > minCos && dotprodTrack > minCos &&
//                    dotprodTrackSeed2D > clusterMinAngleCosine && //Angle between track and seed
        //      distance*clusterScale*tracks.size() < (distanceFromPV+pvDistance)*(distanceFromPV+pvDistance)/pvDistance && // cut scaling with track density
                   distance*clusterScale < densityFactor*distanceFromPV && // cut scaling with track density
                    distance < clusterMaxDistance);  // absolute distance cut

#ifdef VTXDEBUG
            	    std::cout << tt->trackBaseRef().key() << " :  " << (selected?"+":" ")<< " sig " << m.significance() << " < " << clusterMaxSignificance <<  " &&  seedCos " << 
                    dotprodSeed  << " > " <<  clusterMinAngleCosine << "  && trackCos " << 
                    dotprodTrack  << " > " <<  clusterMinAngleCosine << "  && 2D " << 
                    dotprodTrackSeed2D  << " > " <<  clusterMinAngleCosine << "  &&  density: "  << 
                    distance*clusterScale  << " < " <<  densityFactor*distanceFromPV << "  crossingtoPV: " << distanceFromPV << " dis*scal " <<  distance*clusterScale << "  <  " << densityFactor*distanceFromPV << " dist: " << distance << " < " << clusterMaxDistance <<  " cosGamma " << minCos <<  " gamma: " << gamma << " gamma1: " << gamma1 <<  std::endl; // cut scaling with track density
#endif           
                 if(selected)
                 {
                     result.push_back(*tt);
                     seedingPoint = GlobalPoint(cp.x()*w+seedingPoint.x(),cp.y()*w+seedingPoint.y(),cp.z()*w+seedingPoint.z());  
                     sumWeights+=w; 
                 }
            }
       }

   seedingPoint = GlobalPoint(seedingPoint.x()/sumWeights,seedingPoint.y()/sumWeights,seedingPoint.z()/sumWeights);
   return std::pair<std::vector<reco::TransientTrack>,GlobalPoint>(result,seedingPoint);

}





std::vector<TracksClusteringFromDisplacedSeed::Cluster> TracksClusteringFromDisplacedSeed::clusters(
	 const reco::Vertex &pv,
	 const std::vector<reco::TransientTrack> & selectedTracks
 )
{
	using namespace reco;
	std::vector<TransientTrack> seeds;
	for(std::vector<TransientTrack>::const_iterator it = selectedTracks.begin(); it != selectedTracks.end(); it++){
                std::pair<bool,Measurement1D> ip = IPTools::absoluteImpactParameter3D(*it,pv);
                if(ip.first && ip.second.value() >= min3DIPValue && ip.second.significance() >= min3DIPSignificance && 
		   ( maxPixelResidual <0 || (	fabs(it->track().residualX(0)) < maxPixelResidual &&  fabs(it->track().residualY(0)) < maxPixelResidual ) )
		   && (it->track().pt() > minPtForMissingPXB1 || it->track().hitPattern().hasValidHitInFirstPixelBarrel() || fabs(it->track().eta()) > 2.0 ) )
                  { 
#ifdef VTXDEBUG
                    std::cout << "new seed " <<  it-selectedTracks.begin() << " ref " << it->trackBaseRef().key()  << " " << ip.second.value() << " " << ip.second.significance() << " " << it->track().hitPattern().trackerLayersWithMeasurement() << " " << it->track().pt() << " " << it->track().eta() << std::endl;
#endif
                    seeds.push_back(*it);  
                  }
 
	}

        std::vector< Cluster > clusters;
        int i = 0;
	for(std::vector<TransientTrack>::const_iterator s = seeds.begin();
	    s != seeds.end(); ++s, ++i)
        {
#ifdef VTXDEBUG
		std::cout << "Seed N. "<<i <<   std::endl;
#endif // VTXDEBUG
        	std::pair<std::vector<reco::TransientTrack>,GlobalPoint>  ntracks = nearTracks(*s,selectedTracks,pv);
//	        std::cout << ntracks.first.size() << " " << ntracks.first.size()  << std::endl;
//                if(ntracks.first.size() == 0 || ntracks.first.size() > maxNTracks ) continue;
                ntracks.first.push_back(*s);
	        Cluster aCl;
                aCl.seedingTrack = *s;
                aCl.seedPoint = ntracks.second; 
	        aCl.tracks = ntracks.first; 
                clusters.push_back(aCl); 
       }
	 	
return clusters;
}

