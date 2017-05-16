#include "GenericSimClusterMapper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"

#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"


#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackPropagation/RungeKutta/interface/defaultRKPropagator.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"
#ifdef PFLOW_DEBUG
#define LOGVERB(x) edm::LogVerbatim(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) edm::LogInfo(x)
#else
#define LOGVERB(x) LogTrace(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) LogDebug(x)
#endif

void GenericSimClusterMapper::
updateEvent(const edm::Event& ev) {
  ev.getByToken(_simClusterToken,_simClusterH);
}



void GenericSimClusterMapper::
retrieveLayerZPositions()
{
    _layerZPositions.clear();
    constexpr int numberOfLayers = 52;
    DetId id;
//FIXME: magic numbers have to disappear!!!!
    for(unsigned ilayer=1; ilayer<=numberOfLayers; ++ilayer)
    {
        if (ilayer<=28) id=HGCalDetId(ForwardSubdetector::HGCEE,1,ilayer,1,2,1);
        if (ilayer>28 && ilayer<=40) id=HGCalDetId(ForwardSubdetector::HGCHEF,1,ilayer-28,1,2,1);
        if (ilayer>40) id=HcalDetId(HcalSubdetector::HcalEndcap, 50, 100, ilayer-40);
        const GlobalPoint pos = _rhtools.getPosition(id);
        _layerZPositions.push_back(pos.z());
    }
}

void GenericSimClusterMapper::
update(const edm::EventSetup& es) {
    _rhtools.getEventSetup(es);
    // get Geometry, B-field, Topology
    edm::ESHandle<MagneticField> bFieldH;
    es.get<IdealMagneticFieldRecord>().get(bFieldH);
    _bField = bFieldH.product();
    retrieveLayerZPositions();
}

void GenericSimClusterMapper::
buildClusters(const edm::Handle<reco::PFRecHitCollection>& input,
	      const std::vector<bool>& rechitMask,
	      const std::vector<bool>& seedable,
	      reco::PFClusterCollection& output) {
  const SimClusterCollection& simClusters = *_simClusterH;
  auto const& hits = *input;

  // for quick indexing back to hit energy
  std::unordered_map<uint32_t, size_t> detIdToIndex(hits.size());
  for( uint32_t i = 0; i < hits.size(); ++i ) {
    detIdToIndex[hits[i].detId()] = i;
    auto ref = makeRefhit(input,i);
  }

  for( const auto& sc : simClusters )
  {
    output.emplace_back();
    reco::PFCluster& back = output.back();
    edm::Ref<std::vector<reco::PFRecHit> > seed;
    double energy = 0.0, highest_energy = 0.0;
    auto hitsAndFractions = std::move( sc.hits_and_fractions() );
    bool hasSimTrack = !sc.g4Tracks().empty();

    if(hasSimTrack)
    {
        const SimTrack& trk = sc.g4Tracks()[0];
        auto& trkPositionAtTrackerSurface = trk.trackerSurfacePosition();
        GlobalPoint point(trkPositionAtTrackerSurface.x(), trkPositionAtTrackerSurface.y(), trkPositionAtTrackerSurface.z() );
        auto& trkMomentumAtTrackerSurface = trk.trackerSurfaceMomentum();
        GlobalVector vector( trkMomentumAtTrackerSurface.x(), trkMomentumAtTrackerSurface.y(), trkMomentumAtTrackerSurface.z() );
        auto trkCharge = sc.charge();
        defaultRKPropagator::Product prod( _bField, alongMomentum, 5.e-5);
        auto & RKProp = prod.propagator;

        // Define error matrix
        ROOT::Math::SMatrixIdentity id;
        AlgebraicSymMatrix55 C(id);
        C *= 0.01;
        CurvilinearTrajectoryError err(C);
        Plane::PlanePointer startingPlane = Plane::build( Plane::PositionType (trkPositionAtTrackerSurface.x(), trkPositionAtTrackerSurface.y(), trkPositionAtTrackerSurface.z() ), Plane::RotationType () );
        TrajectoryStateOnSurface startingStateP(GlobalTrajectoryParameters(point,vector, trkCharge, _bField), err, *startingPlane);

        std::vector<float> xp;
        std::vector<float> yp;
        std::vector<float> zp;

        std::cout << "starting propagation " << std::endl;
        for(unsigned il=0; il<_layerZPositions.size(); ++il) {
              float xp_curr=0;
              float yp_curr=0;
              float zp_curr=0;

              for (int zside = -1; zside <=1; zside+=2)
              {
                  // clearly try both sides
                  Plane::PlanePointer endPlane = Plane::build( Plane::PositionType (0,0,zside*_layerZPositions[il]), Plane::RotationType());


                  std::cout << "Trying from " << " layer " << il << " starting point "
                            << startingStateP.globalPosition() << std::endl;

                  TrajectoryStateOnSurface trackStateP = RKProp.propagate(startingStateP, *endPlane);
                    if (trackStateP.isValid())
                    {
                        xp_curr = trackStateP.globalPosition().x();
                        yp_curr = trackStateP.globalPosition().y();
                        zp_curr = trackStateP.globalPosition().z();

                         std::cout << "Succesfully finished Positive track propagation  -------------- with RK: " << trackStateP.globalPosition() << std::endl;
                    }

              }
              xp.push_back(xp_curr);
              yp.push_back(yp_curr);
              zp.push_back(zp_curr);
              std::cout << il << " " << xp_curr<< " " << yp_curr << " " << zp_curr<< std::endl;
          } // closes loop on layers


    }

//    FreeTrajectoryState fts (tpVertex, tpMomentum, tpCharge, _bField);



    for( const auto& hAndF : hitsAndFractions )
    {
      auto itr = detIdToIndex.find(hAndF.first);
      if( itr == detIdToIndex.end() ) continue; // hit wasn't saved in reco
      auto ref = makeRefhit(input,itr->second);
      const double hit_energy = hAndF.second * ref->energy();
      energy += hit_energy;
      back.addRecHitFraction(reco::PFRecHitFraction(ref, hAndF.second));
      if( hit_energy > highest_energy || highest_energy == 0.0) {
	highest_energy = hit_energy;
	seed = ref;
      }
    }
    if( back.hitsAndFractions().size() != 0 ) {
      back.setSeed(seed->detId());
      back.setEnergy(energy);
      back.setCorrectedEnergy(energy);
    } else {
      back.setSeed(-1);
      back.setEnergy(0.f);
    }
  }
}

