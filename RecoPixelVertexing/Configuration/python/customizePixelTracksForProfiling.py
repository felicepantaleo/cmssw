import FWCore.ParameterSet.Config as cms

def customizePixelTracksForProfilingGPUOnly(process):
    process.MessageLogger.cerr.FwkReport.reportEvery = 100

    process.Raw2Hit = cms.Path(process.offlineBeamSpot+process.offlineBeamSpotCUDA+process.siPixelClustersCUDAPreSplitting+process.siPixelRecHitsCUDAPreSplitting)

    process.load('RecoPixelVertexing.PixelTriplets.caHitNtupletCUDA_cfi')
    process.load('RecoPixelVertexing.PixelVertexFinding.pixelVertexCUDA_cfi')
    process.TVreco = cms.Path(process.caHitNtupletCUDA+process.pixelVertexCUDA)

    process.load('RecoPixelVertexing.PixelTrackFitting.pixelTrackSoA_cfi')
    process.load('RecoPixelVertexing.PixelVertexFinding.pixelVertexSoA_cfi')
    process.toSoA = cms.Path(process.pixelTrackSoA+process.pixelVertexSoA)

    process.schedule = cms.Schedule(process.Raw2Hit, process.TVreco)
    return process

def customizePixelTracksForProfilingEnableTransfer(process):
    process = customizePixelTracksForProfilingGPUOnly(process)
    process.schedule = cms.Schedule(process.Raw2Hit, process.TVreco, process.toSoA)
    return process

def customizePixelTracksForProfilingEnableConversion(process):
    # use old trick of output path
    process.MessageLogger.cerr.FwkReport.reportEvery = 100

    process.out = cms.OutputModule("AsciiOutputModule",
        outputCommands = cms.untracked.vstring(
            "keep *_pixelTracks_*_*",
            "keep *_pixelVertices_*_*",
        ),
        verbosity = cms.untracked.uint32(0),
    )

    process.outPath = cms.EndPath(process.out)

    process.schedule = cms.Schedule(process.raw2digi_step, process.reconstruction_step, process.outPath)

    return process

