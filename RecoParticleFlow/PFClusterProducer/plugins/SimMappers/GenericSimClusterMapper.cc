#include "GenericSimClusterMapper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"

#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackPropagation/RungeKutta/interface/defaultRKPropagator.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"
#include <iostream>
#include "RealisticHitToClusterAssociator.h"

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

// TODO:
// 1) get planes z from geometry
// 2) propagate only on planes with rechits (once per plane)
// 3) analytical propagator
// 4) check indices (starts from 1 or 0)

void GenericSimClusterMapper::updateEvent(const edm::Event& ev)
{
    ev.getByToken(_simClusterToken, _simClusterH);
    ev.getByToken(_simVtxToken, _simVerticesHandle);

}

void GenericSimClusterMapper::retrieveLayerZPositions()
{
    _layerZPositions.clear();
    constexpr int numberOfLayers = 52;
    DetId id;
//FIXME: magic numbers have to disappear!!!!
    for (unsigned ilayer = 1; ilayer <= numberOfLayers; ++ilayer)
    {
        if (ilayer <= 28)
            id = HGCalDetId(ForwardSubdetector::HGCEE, 1, ilayer, 1, 2, 1);
        if (ilayer > 28 && ilayer <= 40)
            id = HGCalDetId(ForwardSubdetector::HGCHEF, 1, ilayer - 28, 1, 2, 1);
        if (ilayer > 40)
            id = HcalDetId(HcalSubdetector::HcalEndcap, 50, 100, ilayer - 40);
        const GlobalPoint pos = _rhtools.getPosition(id);
        _layerZPositions.push_back(pos.z());
    }
}

void GenericSimClusterMapper::update(const edm::EventSetup& es)
{
    _rhtools.getEventSetup(es);
    edm::ESHandle < MagneticField > bFieldH;
    es.get<IdealMagneticFieldRecord>().get(bFieldH);
    _bField = bFieldH.product();
    retrieveLayerZPositions();

}

void GenericSimClusterMapper::buildClusters(const edm::Handle<reco::PFRecHitCollection>& input,
        const std::vector<bool>& rechitMask, const std::vector<bool>& seedable,
        reco::PFClusterCollection& output)
{
    std::cout << "start  building clusters" << std::endl;
    bool distanceFilter = false;
    float maxDistance = 10.f; //10cm
    const SimClusterCollection& simClusters = *_simClusterH;
    auto const& hits = *input;
    RealisticHitToClusterAssociator realisticAssociator;
    //layers are counted starting from 1
    realisticAssociator.init(hits.size(), simClusters.size(), _layerZPositions.size() + 1);
    std::cout << "there are hits: " << hits.size() << " SimClusters : " << simClusters.size()
            << " layers: " << _layerZPositions.size() << std::endl;

    // for quick indexing back to hit energy
    std::unordered_map < uint32_t, size_t > detIdToIndex(hits.size());

    std::vector<float> PFRecHitOnlySimClusters;
    PFRecHitOnlySimClusters.resize(simClusters.size(), 0.f);

    unsigned int previousLayerId = 10000;
    for (uint32_t i = 0; i < hits.size(); ++i)
    {

        detIdToIndex[hits[i].detId()] = i;
        auto ref = makeRefhit(input, i);
        const auto& hitPos = _rhtools.getPosition(ref->detId());

        realisticAssociator.insertHitPosition(hitPos.x(), hitPos.y(), hitPos.z(), i);
        realisticAssociator.insertHitEnergy(ref->energy(), i);
        realisticAssociator.insertLayerId(_rhtools.getLayerWithOffset(ref->detId()), i);

        if (_rhtools.getLayerWithOffset(ref->detId()) != previousLayerId)
        {
            std::cout << "hit id : " << i << " x,y,z " << hitPos.x() << " " << hitPos.y() << " "
                    << hitPos.z() << " energy " << ref->energy() << " layer "
                    << _rhtools.getLayerWithOffset(ref->detId()) << std::endl;
            previousLayerId = _rhtools.getLayerWithOffset(ref->detId());
        }
    }
    for (unsigned int ic = 0; ic < simClusters.size(); ++ic)
    {
        const auto & sc = simClusters[ic];
        auto hitsAndFractions = std::move(sc.hits_and_fractions());
        std::cout << "simcluster " << ic << " contains " << hitsAndFractions.size() << " hits"
                << std::endl;

        for (const auto& hAndF : hitsAndFractions)
        {
            auto itr = detIdToIndex.find(hAndF.first);
            if (itr == detIdToIndex.end()){
                    std::cout << "DEBUG hit not saved in reco. DetId "<< hAndF.first << " fraction: " << hAndF.second << " layer " <<_rhtools.getLayerWithOffset(hAndF.first) << std::endl;
                continue; // hit wasn't saved in reco
            }
            unsigned int layerId = _rhtools.getLayerWithOffset(hAndF.first);

            auto hitId = itr->second;

            auto ref = makeRefhit(input, hitId);
            float fraction = hAndF.second;
            float associatedEnergy = fraction * ref->energy();
            realisticAssociator.insertSimClusterIdAndFraction(ic, fraction, hitId,
                    associatedEnergy);

            PFRecHitOnlySimClusters[ic]+=associatedEnergy;
        }




//        const SimTrack& trk = sc.g4Tracks()[0];
//        const math::XYZTLorentzVectorD & vtxPos = (*_simVerticesHandle)[trk.vertIndex()].position();
//
//        GlobalPoint point(vtxPos.X(), vtxPos.Y(), vtxPos.Z());
//
//        auto& trkMomentumAtIP = trk.momentum();
//        GlobalVector vector(trkMomentumAtIP.x(), trkMomentumAtIP.y(), trkMomentumAtIP.z());
//
//        auto trkCharge = trk.charge();
//        defaultRKPropagator::Product prod(_bField, alongMomentum, 5.e-5);
//        auto & RKProp = prod.propagator;
//
//        // Define error matrix
//        ROOT::Math::SMatrixIdentity id;
//        AlgebraicSymMatrix55 C(id);
//        C *= 0.01;
//        CurvilinearTrajectoryError err(C);
//        Plane::PlanePointer startingPlane = Plane::build(
//                Plane::PositionType(vtxPos.x(), vtxPos.y(), vtxPos.z()), Plane::RotationType());
//        TrajectoryStateOnSurface startingStateP(
//                GlobalTrajectoryParameters(point, vector, trkCharge, _bField), err, *startingPlane);
//
//        for (unsigned il = 0; il < _layerZPositions.size(); ++il)
//        {
//            float xp_curr = 0;
//            float yp_curr = 0;
//            float zp_curr = 0;
//            int zside;
//
//            //TODO: choose zside in the first iteration and keep it to avoid useless propagations backwards
//            for (zside = -1; zside <= 1; zside += 2)
//            {
//                // clearly try both sides
//                Plane::PlanePointer endPlane = Plane::build(
//                        Plane::PositionType(0, 0, zside * _layerZPositions[il]),
//                        Plane::RotationType());
//
//                TrajectoryStateOnSurface trackStateP = RKProp.propagate(startingStateP, *endPlane);
//                if (trackStateP.isValid())
//                {
//                    xp_curr = trackStateP.globalPosition().x();
//                    yp_curr = trackStateP.globalPosition().y();
//                    zp_curr = trackStateP.globalPosition().z();
//
//                }
//
//            }
//            realisticAssociator.insertSimTrackPositionAtLayer(ic, il, xp_curr, yp_curr, zp_curr);
//        } // closes loop on layers

    }
    realisticAssociator.create2dSimClusters(distanceFilter, maxDistance);
    realisticAssociator.computeAssociation();
    realisticAssociator.findAndMergeInvisibleClusters();
    auto realisticClusters = std::move(realisticAssociator.realisticClusters());
    unsigned int nClusters = realisticClusters.size();
    for (unsigned ic = 0; ic < nClusters; ++ic)
    {
        std::cout << "realistic cluster " << ic <<" pdgid " <<std::setw(5) << simClusters[ic].pdgId()  << " E " <<std::setw(8)<< realisticClusters[ic].getEnergy()<< " excl E : " <<std::setw(8)<< realisticClusters[ic].getExclusiveEnergy() << " excl fraction: " <<realisticClusters[ic].getExclusiveEnergy()/realisticClusters[ic].getEnergy()<<
                " real number of hits " <<std::setw(4)<< realisticClusters[ic].hitsIdsAndFractions().size() << "\t\t MC E " <<std::setw(8)<< simClusters[ic].energy() << ". is visible? " <<std::setw(5)<<(realisticClusters[ic].isVisible()? " true ": " false ")<<
                " MC numhits " <<std::setw(4)<<  simClusters[ic].hits_and_fractions().size()  << " PFRHSC "<<std::setw(8)<<  PFRecHitOnlySimClusters[ic]<<  std::endl;
        if (realisticClusters[ic].isVisible())
        {
            float highest_energy = 0.0f;
            output.emplace_back();
            reco::PFCluster& back = output.back();
            edm::Ref < std::vector<reco::PFRecHit> > seed;
            auto hitsIdsAndFractions = std::move(realisticClusters[ic].hitsIdsAndFractions());



            for (const auto& idAndF : hitsIdsAndFractions)
            {
                auto ref = makeRefhit(input, idAndF.first);
                back.addRecHitFraction(reco::PFRecHitFraction(ref, idAndF.second));
                const float hit_energy = idAndF.second * ref->energy();
                if (hit_energy > highest_energy || highest_energy == 0.0)
                {
                    highest_energy = hit_energy;
                    seed = ref;
                }
            }

            if (back.hitsAndFractions().size() != 0)
            {
                back.setSeed(seed->detId());
                back.setEnergy(realisticClusters[ic].getEnergy());
                back.setCorrectedEnergy(realisticClusters[ic].getEnergy());
            }
            else
            {
                back.setSeed(-1);
                back.setEnergy(0.f);
            }
        }

    }

//    for (unsigned int ic = 0; ic < simClusters.size(); ++ic)
//    {
//        const auto & sc = simClusters[ic];
//        output.emplace_back();
//        reco::PFCluster& back = output.back();
//        edm::Ref < std::vector<reco::PFRecHit> > seed;
//        double energy = 0.0, highest_energy = 0.0;
//        auto hitsAndFractions = std::move(sc.hits_and_fractions());
//
//
//
//
//        for (const auto& hAndF : hitsAndFractions)
//        {
//            bool hitIsWithinDistance = false;
//            if (distanceFilter && hasSimTrack)
//            {
//                const auto& hitPos = _rhtools.getPosition(hAndF.first);
//                int hitLayer = _rhtools.getLayerWithOffset(hAndF.first);
//                float distance = (hitPos - corePointsInLayer[hitLayer]).mag();
//                hitIsWithinDistance = distance < maxDistance;
//            }
//            if (!distanceFilter || (distanceFilter && hitIsWithinDistance))
//            {
//
//                auto itr = detIdToIndex.find(hAndF.first);
//                if (itr == detIdToIndex.end())
//                    continue; // hit wasn't saved in reco
//                auto ref = makeRefhit(input, itr->second);
//                const double hit_energy = hAndF.second * ref->energy();
//                energy += hit_energy;
//                back.addRecHitFraction(reco::PFRecHitFraction(ref, hAndF.second));
//                if (hit_energy > highest_energy || highest_energy == 0.0)
//                {
//                    highest_energy = hit_energy;
//                    seed = ref;
//                }
//            }
//        }
//        if (back.hitsAndFractions().size() != 0)
//        {
//            back.setSeed(seed->detId());
//            back.setEnergy(energy);
//            back.setCorrectedEnergy(energy);
//        }
//        else
//        {
//            back.setSeed(-1);
//            back.setEnergy(0.f);
//        }
//    }
}

