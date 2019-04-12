#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"


namespace testTrackingRecHit2D {

  void runKernels(TrackingRecHit2DSOAView * hits);

}




namespace {
  CUDAService makeCUDAService(edm::ParameterSet ps, edm::ActivityRegistry& ar) {
    auto desc = edm::ConfigurationDescriptions("Service", "CUDAService");
    CUDAService::fillDescriptions(desc);
    desc.validate(ps, "CUDAService");
    return CUDAService(ps, ar);
  }
}


int main() {


  exitSansCUDADevices();

  edm::ActivityRegistry ar;
  edm::ParameterSet ps;
  auto cs = makeCUDAService(ps, ar);

  auto current_device = cuda::device::current::get();
  auto stream = current_device.create_stream(cuda::stream::implicitly_synchronizes_with_default_stream);


  auto nHits = 200;
  TrackingRecHit2DCUDA tkhit(nHits,stream);

  testTrackingRecHit2D::runKernels(tkhit.view());

  //Fake the end-of-job signal.
  ar.postEndJobSignal_();

  return 0;

}





