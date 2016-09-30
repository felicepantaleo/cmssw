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

#include "CAGraph.h"

namespace
{

template<typename T>
T sqr(T x)
{
	return x * x;
}
}

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
				cfg.getParameter<double>("CAPhiCut")), CAHardPtCut(
				cfg.getParameter<double>("CAHardPtCut"))
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
	edm::Handle<SeedingLayerSetsHits> hlayers;
	ev.getByToken(theSeedingLayerToken, hlayers);
	const SeedingLayerSetsHits& layers = *hlayers;
	if (layers.numberOfLayersInSet() != 4)
		throw cms::Exception("Configuration")
				<< "CAHitQuadrupletsGenerator expects SeedingLayerSetsHits::numberOfLayersInSet() to be 4, got "
				<< layers.numberOfLayersInSet();

	CAGraph g;
	std::vector<HitDoublets> hitDoublets;
	std::vector<const RecHitsSortedInPhi  *> hitsOnLayer;
	std::vector<int> externalLayerPairs;

	HitPairGeneratorFromLayerPair thePairGenerator(0, 1, &theLayerCache);
	for (unsigned int i = 0; i < layers.size(); i++)
	{
		for (unsigned int j = 0; j < 4; ++j)
		{
			auto vertexIndex = 0;
			auto foundVertex = std::find(g.theLayers.begin(), g.theLayers.end(),
					layers[i][j].name());
			if (foundVertex == g.theLayers.end())
			{
				g.theLayers.emplace_back(layers[i][j].name(),
						layers[i][j].hits().size());
				vertexIndex = g.theLayers.size() - 1;
				auto const & layer = layers[i][j];
				hitsOnLayer.push_back( &theLayerCache(layer, region,ev, es));

			}
			else
			{
				vertexIndex = foundVertex - g.theLayers.begin();
			}
			if (j == 0)
			{

				if (std::find(g.theRootLayers.begin(), g.theRootLayers.end(),
						vertexIndex) == g.theRootLayers.end())
				{
					g.theRootLayers.emplace_back(vertexIndex);

				}

			}
			else
			{

				auto innerVertex = std::find(g.theLayers.begin(),
						g.theLayers.end(), layers[i][j - 1].name());

				CALayerPair tmpInnerLayerPair(innerVertex - g.theLayers.begin(),
						vertexIndex);

				if (std::find(g.theLayerPairs.begin(), g.theLayerPairs.end(),
						tmpInnerLayerPair) == g.theLayerPairs.end())
				{

					auto layerPairIndex = g.theLayerPairs.size();

					hitDoublets.emplace_back(
							thePairGenerator.doublets(region, ev, es,
									layers[i][j - 1], layers[i][j]));



					g.theLayerPairs.push_back(tmpInnerLayerPair);

					g.theLayers[vertexIndex].theInnerLayers.push_back(
							innerVertex - g.theLayers.begin());
					innerVertex->theOuterLayers.push_back(vertexIndex);
					g.theLayers[vertexIndex].theInnerLayerPairs.push_back(
							layerPairIndex);
					innerVertex->theOuterLayerPairs.push_back(layerPairIndex);

					auto & currentLayerPairRef = g.theLayerPairs[layerPairIndex];
					auto & currentInnerLayerRef =
							g.theLayers[currentLayerPairRef.theLayers[0]];
					auto & currentOuterLayerRef =
							g.theLayers[currentLayerPairRef.theLayers[1]];


					if (j == 3)
						externalLayerPairs.push_back(layerPairIndex);

				}

			}
		}
	}

	if (theComparitor)
		theComparitor->init(ev, es);
	const int numberOfHitsInNtuplet = 4;
	std::vector<std::array<std::array<int, 2 > , 3> > foundQuadruplets;

	GPUCellularAutomaton <2000> ca(region, CAThetaCut, CAPhiCut, CAHardPtCut, g.theLayers.size(), g.theLayerPairs.size(), externalLayerPairs  );
	ca.run(hitDoublets, hitsOnLayer, g, foundQuadruplets);


	const QuantityDependsPtEval maxChi2Eval = maxChi2.evaluator(es);

	// re-used thoughout, need to be vectors because of RZLine interface
	std::array<float, 4> bc_r;
	std::array<float, 4> bc_z;
	std::array<float, 4> bc_errZ2;
	std::array < GlobalPoint, 4 > gps;
	std::array < GlobalError, 4 > ges;
	std::array<bool, 4> barrels;

	unsigned int numberOfFoundQuadruplets = foundQuadruplets.size();

	std::array< HitDoublets::Hit,4> tmpQuadruplet;
	// Loop over quadruplets
	for (unsigned int quadId = 0; quadId < numberOfFoundQuadruplets; ++quadId)
	{

		auto isBarrel = [](const unsigned id) -> bool
		{
			return id == PixelSubdetector::PixelBarrel;
		};
		for (unsigned int i = 0; i < 3; ++i)
		{
			int currentLayerPairId = foundQuadruplets[quadId][i][0];
			auto & currentLayerPairRef = g.theLayerPairs[currentLayerPairId];

			int currentDoubletIdInLayerPair = foundQuadruplets[quadId][i][1];

			tmpQuadruplet[i] = hitDoublets[currentLayerPairId].hit(currentDoubletIdInLayerPair, HitDoublets::inner);

			gps[i] = tmpQuadruplet[i]->globalPosition();
			ges[i] = tmpQuadruplet[i]->globalPositionError();
			barrels[i] = isBarrel(tmpQuadruplet[i]->geographicalId().subdetId());
		}


		int currentLayerPairId = foundQuadruplets[quadId][2][0];
		auto & currentLayerPairRef = g.theLayerPairs[currentLayerPairId];
		int currentDoubletIdInLayerPair = foundQuadruplets[quadId][2][1];
		tmpQuadruplet[3] = hitDoublets[currentLayerPairId].hit(currentDoubletIdInLayerPair, HitDoublets::outer);

		gps[3] = tmpQuadruplet[3]->globalPosition();

		ges[3] = tmpQuadruplet[3]->globalPositionError();
		barrels[3] = isBarrel(tmpQuadruplet[3]->geographicalId().subdetId());

		PixelRecoLineRZ line(gps[0], gps[2]);
		ThirdHitPredictionFromCircle predictionRPhi(gps[0], gps[2],
				extraHitRPhitolerance);
		const float curvature = predictionRPhi.curvature(
				ThirdHitPredictionFromCircle::Vector2D(gps[1].x(), gps[1].y()));
		const float abscurv = std::abs(curvature);
		const float thisMaxChi2 = maxChi2Eval.value(abscurv);

		if (theComparitor)
		{
			SeedingHitSet tmpTriplet(tmpQuadruplet[0],
					tmpQuadruplet[2],
					tmpQuadruplet[3]);

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
			const float pt = 1.f / PixelRecoUtilities::inversePt(abscurv, es);
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
				bc_errZ2[i] =
						(barrels[i]) ?
								error.czz() :
								error.rerr(point) * sqr(simpleCot);
			}
			RZLine rzLine(bc_r, bc_z, bc_errZ2, RZLine::ErrZ2_tag());
			chi2 = rzLine.chi2();
		}
		else
		{
			RZLine rzLine(gps, ges, barrels);
			chi2 = rzLine.chi2();
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

		result.emplace_back(tmpQuadruplet[0],
				tmpQuadruplet[1],
				tmpQuadruplet[2],
				tmpQuadruplet[3]);
	}


	theLayerCache.clear();

}
