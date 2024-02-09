#ifndef RecoHGCal_TICL_TracksterLinkingAlgoByGNN_H
#define RecoHGCal_TICL_TracksterLinkingAlgoByGNN_H

#include "RecoHGCal/TICL/interface/TracksterLinkingAlgoBase.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

namespace ticl {

  class TracksterLinkingbyGNN : public TracksterLinkingAlgoBase {
  public:
    TracksterLinkingbyGNN(const edm::ParameterSet& ps,
                          edm::ConsumesCollector iC,
                          cms::Ort::ONNXRuntime const* onnxRuntime = nullptr);

    ~TracksterLinkingbyGNN() override {}
    
    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

    void linkTracksters(const Inputs& input, 
                        std::vector<Trackster>& resultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedResultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) override;
                        
    void initialize(const HGCalDDDConstants* hgcons,
                    const hgcal::RecHitTools rhtools,
                    const edm::ESHandle<MagneticField> bfieldH,
                    const edm::ESHandle<Propagator> propH) override;
    
   private:
      std::string nnVersion_;       // Version identifier of the NN (either CNN or a GNN, to choose which inputs to use)
      double nnWorkingPoint_;       // Working point for neural network (above this score, consider the trackster candidate for superclustering)
      double deltaEtaWindow_;       // Delta eta window to consider trackster candidate pairs for inference
      double deltaPhiWindow_;       // Delta phi window to consider trackster candidate pairs for inference
  };

}  // namespace ticl

#endif