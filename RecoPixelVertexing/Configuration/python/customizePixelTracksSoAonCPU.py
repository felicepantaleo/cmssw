def customizePixelTracksSoAonCPU(process) :

  process.load('RecoLocalTracker/SiPixelRecHits/siPixelRecHitHostSoA_cfi')
  process.load('RecoPixelVertexing.PixelTriplets.caHitNtupletCUDA_cfi')
  process.load('RecoPixelVertexing.PixelVertexFinding.pixelVertexCUDA_cfi')

  process.pixelTrackSoA = process.caHitNtupletCUDA.clone()
  process.pixelTrackSoA.onGPU = False
  process.pixelTrackSoA.pixelRecHitSrc = 'siPixelRecHitHostSoA'
  process.pixelVertexSoA = process.pixelVertexCUDA.clone()
  process.pixelVertexSoA.onGPU = False
  process.pixelVertexSoA.pixelTrackSrc = 'pixelTrackSoA'

  process.load('RecoPixelVertexing.PixelTrackFitting.pixelTrackProducerFromSoA_cfi')
  process.pixelTracks = process.pixelTrackProducerFromSoA.clone()
  process.load('RecoPixelVertexing.PixelVertexFinding.pixelVertexFromSoA_cfi')
  process.pixelVertices = process.pixelVertexFromSoA.clone()
  process.pixelTracks.pixelRecHitLegacySrc = 'siPixelRecHitsPreSplitting'

  process.reconstruction_step += process.siPixelRecHitHostSoA+process.pixelTrackSoA+process.pixelVertexSoA


  return process

