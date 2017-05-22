#include <unordered_map>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "DataFormats/Common/interface/Handle.h"
#include "CommonTools/Utils/interface/DynArray.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGenerator.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUHitsAndDoublets.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUCellularAutomaton.h"

#include "LayerQuadruplets.h"
#include "CAHitQuadrupletGenerator.h"
#include "CACell.h"

namespace
{

template<typename T>
T sqr(T x)
{
	return x * x;
}
}

using namespace std;
using namespace ctfseeding;

CAHitQuadrupletGenerator::CAHitQuadrupletGenerator(const edm::ParameterSet& cfg,
		edm::ConsumesCollector& iC) :
		theSeedingLayerToken(
				iC.consumes < SeedingLayerSetsHits
						> (cfg.getParameter < edm::InputTag > ("SeedingLayers"))), extraHitRPhitolerance(
				cfg.getParameter<double>("extraHitRPhitolerance")), //extra window in ThirdHitPredictionFromCircle range (divide by R to get phi)
		maxChi2(cfg.getParameter < edm::ParameterSet > ("maxChi2")), fitFastCircle(
				cfg.getParameter<bool>("fitFastCircle")), fitFastCircleChi2Cut(
				cfg.getParameter<bool>("fitFastCircleChi2Cut")), useBendingCorrection(
				cfg.getParameter<bool>("useBendingCorrection")), CAThetaCut(
				cfg.getParameter<double>("CAThetaCut")), CAPhiCut(
				cfg.getParameter<double>("CAPhiCut"))
{
	if (cfg.exists("SeedComparitorPSet"))
	{
		edm::ParameterSet comparitorPSet = cfg.getParameter < edm::ParameterSet
				> ("SeedComparitorPSet");
		std::string comparitorName = comparitorPSet.getParameter < std::string
				> ("ComponentName");
		if (comparitorName != "none")
		{
			theComparitor.reset(
					SeedComparitorFactory::get()->create(comparitorName,
							comparitorPSet, iC));
		}
	}
}

CAHitQuadrupletGenerator::~CAHitQuadrupletGenerator()
{
}

void CAHitQuadrupletGenerator::hitQuadruplets(const TrackingRegion& region,
		OrderedHitSeeds & result, const edm::Event& ev,
		const edm::EventSetup& es)
{
	edm::Handle < SeedingLayerSetsHits > hlayers;
	ev.getByToken(theSeedingLayerToken, hlayers);
	const SeedingLayerSetsHits& layers = *hlayers;
	if (layers.numberOfLayersInSet() != 4)
		throw cms::Exception("Configuration")
				<< "CAHitQuadrupletsGenerator expects SeedingLayerSetsHits::numberOfLayersInSet() to be 4, got "
				<< layers.numberOfLayersInSet();

	std::unordered_map < std::string, GPULayerHits > gpuHitsMap;
	std::unordered_map < std::string, GPULayerDoublets > gpuDoubletMap;

	for (unsigned int j = 0; j < layers.size(); j++)
		for (unsigned int i = 0; i < 4; ++i)
		{
			auto const & layer = layers[j][i];
			if (gpuHitsMap.find(layer.name()) == gpuHitsMap.end())
			{
				RecHitsSortedInPhi const & hits = theLayerCache(layer, region,
						ev, es);
				gpuHitsMap[layer.name()] = copy_hits_to_gpu(hits);
			}
		}

	HitPairGeneratorFromLayerPair thePairGenerator(0, 1, &theLayerCache);
	std::unordered_map < std::string, HitDoublets > doubletMap;
	std::array<const GPULayerDoublets *, 3> layersDoublets;
	for (unsigned int j = 0; j < layers.size(); j++)
	{
		for (unsigned int i = 0; i < 3; ++i)
		{
			auto const & inner = layers[j][i];
			auto const & outer = layers[j][i + 1];
			auto layersPair = inner.name() + '+' + outer.name();
			auto it = gpuDoubletMap.find(layersPair);
			if (it == gpuDoubletMap.end())
			{
				auto const & h_doublets = thePairGenerator.doublets(region, ev,
						es, inner, outer);

//        std::cout<< "numberOfDoublets" << h_doublets.size()<< " layers j "<< j << "layer i "<< i<<std::endl;
				auto const & d_doublets = copy_doublets_to_gpu(h_doublets,
						gpuHitsMap[inner.name()], gpuHitsMap[outer.name()]);
				std::tie(it, std::ignore) = gpuDoubletMap.insert(
						std::make_pair(layersPair, d_doublets));
			}
			layersDoublets[i] = &it->second;
//      std::cout << " layersDoublets " << i << " " << layersDoublets[i] << std::endl;
		}

		findQuadruplets(region, result, ev, es, layers[j], layersDoublets);
	}

	for (auto & kayval : gpuDoubletMap)
		free_gpu_doublets(kayval.second);
	for (auto & kayval : gpuHitsMap)
		free_gpu_hits(kayval.second);

	theLayerCache.clear();
}

void CAHitQuadrupletGenerator::findQuadruplets(const TrackingRegion& region,
		OrderedHitSeeds& result, const edm::Event& ev,
		const edm::EventSetup& es,
		const SeedingLayerSetsHits::SeedingLayerSet& fourLayers,
		std::array<const GPULayerDoublets *, 3> const & layersDoublets)
{
	if (theComparitor)
		theComparitor->init(ev, es);

	std::vector<std::array<int, 4>> foundQuadruplets;

	GPUCellularAutomaton < 4, 2000 > ca(region, CAThetaCut, CAPhiCut);
	ca.run(layersDoublets, foundQuadruplets);

	const QuantityDependsPtEval maxChi2Eval = maxChi2.evaluator(es);

	// re-used thoughout, need to be vectors because of RZLine interface
	std::vector<float> bc_r(4), bc_z(4), bc_errZ(4);

	declareDynArray(GlobalPoint, 4, gps);
	declareDynArray(GlobalError, 4, ges);
	declareDynArray(bool, 4, barrels);

	unsigned int numberOfFoundQuadruplets = foundQuadruplets.size();


	std::array<const RecHitsSortedInPhi  *, 4> hitsOnLayer;

	for(unsigned int i =0; i< hitsOnLayer.size(); ++i)
		hitsOnLayer[i]= &theLayerCache(fourLayers[i], region,
			ev, es);
//	std::cout << "found quadruplets " << numberOfFoundQuadruplets << std::endl;
//  std::cout << "I have found " << numberOfFoundQuadruplets << " quadruplets" << std::endl;
	// Loop over quadruplets
	for (unsigned int quadId = 0; quadId < numberOfFoundQuadruplets; ++quadId)
	{

		auto isBarrel = [](const unsigned id) -> bool
		{
			return id == PixelSubdetector::PixelBarrel;
		};

		std::array<BaseTrackerRecHit const *, 4> hits;

		for (unsigned int i = 0; i < 4; ++i)
		{
			// read hits from the GPU vectors
			auto const hit = hitsOnLayer[i]->hits()[foundQuadruplets[quadId][i]];
			hits[i] = hit;
			gps[i] = hit->globalPosition();
			ges[i] = hit->globalPositionError();
			barrels[i] = isBarrel(hit->geographicalId().subdetId());

		}
		PixelRecoLineRZ line(gps[0], gps[2]);
		ThirdHitPredictionFromCircle predictionRPhi(gps[0], gps[2],
				extraHitRPhitolerance);
		const float curvature = predictionRPhi.curvature(
				ThirdHitPredictionFromCircle::Vector2D(gps[1].x(), gps[1].y()));
		const float abscurv = std::abs(curvature);
		const float thisMaxChi2 = maxChi2Eval.value(abscurv);

		if (theComparitor)
		{
			SeedingHitSet tmpTriplet(hits[0], hits[2], hits[3]);

			if (!theComparitor->compatible(tmpTriplet, region))
			{
				continue;
			}
		}

		float chi2 = std::numeric_limits<float>::quiet_NaN();
		// TODO: Do we have any use case to not use bending correction?
		if (useBendingCorrection)
		{
			// Following PixelFitterByConformalMappingAndLine
			const float simpleCot = (gps.back().z() - gps.front().z())
					/ (gps.back().perp() - gps.front().perp());
			const float pt = 1 / PixelRecoUtilities::inversePt(abscurv, es);
			for (int i = 0; i < 4; ++i)
			{
				const GlobalPoint & point = gps[i];
				const GlobalError & error = ges[i];
				bc_r[i] = sqrt(
						sqr(point.x() - region.origin().x())
								+ sqr(point.y() - region.origin().y()));
				bc_r[i] += pixelrecoutilities::LongitudinalBendingCorrection(pt,
						es)(bc_r[i]);
				bc_z[i] = point.z() - region.origin().z();
				bc_errZ[i] =
						(barrels[i]) ?
								sqrt(error.czz()) :
								sqrt(error.rerr(point)) * simpleCot;
			}
			RZLine rzLine(bc_r, bc_z, bc_errZ);
			float cottheta, intercept, covss, covii, covsi;
			rzLine.fit(cottheta, intercept, covss, covii, covsi);
			chi2 = rzLine.chi2(cottheta, intercept);
		}
		else
		{
			RZLine rzLine(gps, ges, barrels);
			float cottheta, intercept, covss, covii, covsi;
			rzLine.fit(cottheta, intercept, covss, covii, covsi);
			chi2 = rzLine.chi2(cottheta, intercept);
		}
		if (edm::isNotFinite(chi2) || chi2 > thisMaxChi2)
		{
			continue;

		}
		// TODO: Do we have any use case to not use circle fit? Maybe
		// HLT where low-pT inefficiency is not a problem?
		if (fitFastCircle)
		{
			FastCircleFit c(gps, ges);
			chi2 += c.chi2();
			if (edm::isNotFinite(chi2))
				continue;
			if (fitFastCircleChi2Cut && chi2 > thisMaxChi2)
				continue;
		}

		result.emplace_back(hits[0], hits[1], hits[2], hits[3]);

	}
}
