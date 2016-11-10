import FWCore.ParameterSet.Config as cms

def customiseForDetachedTripletStepByCellularAutomaton(process, regionPtMin, thetacut, phicut, hardptcut, chi2lowpt, chi2highpt, lowpt, highpt):
    for module in [process.detachedTripletStepSeeds]:
        if hasattr(module, "OrderedHitsFactoryPSet"):
            pset = getattr(module, "OrderedHitsFactoryPSet")
            if (hasattr(pset, "ComponentName") and (pset.ComponentName == "StandardHitTripletGenerator")):
                # Adjust seeding layers
                seedingLayersName = module.OrderedHitsFactoryPSet.SeedingLayers.getModuleLabel()
                # Configure seed generator / pixel track producer
                triplets = module.OrderedHitsFactoryPSet.clone()
                from RecoPixelVertexing.PixelTriplets.CAHitTripletGenerator_cfi import CAHitTripletGenerator as _CAHitTripletGenerator
    
                module.OrderedHitsFactoryPSet  = _CAHitTripletGenerator.clone(
                    ComponentName = cms.string("CAHitTripletGenerator"),
                    extraHitRPhitolerance = triplets.GeneratorPSet.extraHitRPhitolerance,
                    maxChi2 = dict(
                        pt1    = cms.double(lowpt), pt2    = cms.double(highpt),
                        value1 = cms.double(chi2lowpt), value2 = cms.double(chi2highpt),
                        enabled = cms.bool(True),
                    ),
                    useBendingCorrection = True,
                    SeedingLayers = cms.InputTag(seedingLayersName),
                    CAThetaCut = cms.double(thetacut),
                    CAPhiCut = cms.double(phicut),
                    CAHardPtCut = cms.double(hardptcut),
                )
                
                
                module.RegionFactoryPSet.RegionPSet.ptMin = cms.double(regionPtMin)
    
                if hasattr(triplets.GeneratorPSet, "SeedComparitorPSet"):
                    module.OrderedHitsFactoryPSet.SeedComparitorPSet = triplets.GeneratorPSet.SeedComparitorPSet
    return process
