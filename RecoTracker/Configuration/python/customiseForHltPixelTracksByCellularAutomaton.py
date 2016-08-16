import FWCore.ParameterSet.Config as cms

def customiseForHltPixelTracksByCellularAutomaton(process):
    for module in producers_by_type(process, "PixelTrackProducer"):
        if not hasattr(module, "OrderedHitsFactoryPSet"):
            continue
	pset = getattr(module, "OrderedHitsFactoryPSet")
        if not hasattr(pset, "ComponentName"):
	    continue
	if not (pset.ComponentName == "StandardHitTripletGenerator"):
	    continue    
        # Adjust seeding layers
        seedingLayersName = module.OrderedHitsFactoryPSet.SeedingLayers.getModuleLabel()
   
        # Configure seed generator / pixel track producer
        triplets = module.OrderedHitsFactoryPSet.clone()
        from RecoPixelVertexing.PixelTriplets.CAHitTripletGenerator_cfi import CAHitTripletGenerator as _CAHitTripletGenerator

        module.OrderedHitsFactoryPSet  = _CAHitTripletGenerator.clone(
            ComponentName = cms.string("CAHitTripletGenerator"),
            extraHitRPhitolerance = triplets.GeneratorPSet.extraHitRPhitolerance,
            maxChi2 = dict(
                pt1    = 0.8, pt2    = 2,
                value1 = 50, value2 = 8,
                enabled = True,
            ),
            useBendingCorrection = True,
            SeedingLayers = cms.InputTag(seedingLayersName),
            CAThetaCut = cms.double(0.00125),
            CAPhiCut = cms.double(1),
        )

        if hasattr(triplets.GeneratorPSet, "SeedComparitorPSet"):
            pset.SeedComparitorPSet = triplets.GeneratorPSet.SeedComparitorPSet
    return process
