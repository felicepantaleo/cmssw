import FWCore.ParameterSet.Config as cms

def customiseFor2017Offline_v1(process):
    from RecoTracker.Configuration.customiseForInitialStepQuadrupletsByCellularAutomaton import customiseForInitialStepQuadrupletsByCellularAutomaton
    process = customiseForInitialStepQuadrupletsByCellularAutomaton(process, 0.5 , 0.0012 , 0.2, 0, 200, 50, 0.7, 2) 
    from RecoTracker.Configuration.customiseForLowPtQuadStepQuadrupletsByCellularAutomaton import customiseForLowPtQuadStepQuadrupletsByCellularAutomaton
    process = customiseForLowPtQuadStepQuadrupletsByCellularAutomaton(process, 0.15, 0.0017, 0.3, 0, 1000, 150, 0.7,2) 
    from RecoTracker.Configuration.customiseForHighPtTripletsStepByCellularAutomaton import customiseForHighPtTripletsStepByCellularAutomaton 
    process = customiseForHighPtTripletsStepByCellularAutomaton(process, 0.55, 0.004, 0.07, 0.3, 100, 6, 0.8, 8)
    from RecoTracker.Configuration.customiseForLowPtTripletsStepByCellularAutomaton import customiseForLowPtTripletsStepByCellularAutomaton
    process = customiseForLowPtTripletsStepByCellularAutomaton(process, 0.2,0.002,0.05,0,70,8,0.8,2)
    from RecoTracker.Configuration.customiseForDetachedQuadStepByCellularAutomaton import customiseForDetachedQuadStepByCellularAutomaton
    process = customiseForDetachedQuadStepByCellularAutomaton(process, 0.3, 0.0011, 0,0,  500, 100, 0.8, 2)
    from RecoTracker.Configuration.customiseForDetachedTripletStepByCellularAutomaton import customiseForDetachedTripletStepByCellularAutomaton
    process = customiseForDetachedTripletStepByCellularAutomaton(process, 0.25,0.001,0,0.2,300,10,0.8,2)

    return process
