#ifndef RecoTICL_Superclustering_TracksterLinkingbySuperClusteringDNNHeterogeneous_h
#define RecoTICL_Superclustering_TracksterLinkingbySuperClusteringDNNHeterogeneous_h
/*
TICL plugin for electron superclustering in HGCAL with the superclustering DNN run through PyTorch on any alpaka
backend: the host step of TracksterLinkingbySuperClusteringDNN.

SuperclusteringDNNProducer (alpaka, PyTorch) builds the (seed, candidate) pairs from the device tracksters and scores
them. This plugin reads the pairs and their scores, copied to the host by the framework, and makes the superclusters
as the legacy plugin does: each candidate, in increasing pT rank, joins the best-scoring seed above the working point
that has not itself joined a supercluster. That step is sequential (a trackster that joins cannot seed any more),
hence on the host.

The pairs index the device tracksters, so the linking must read the legacy tracksters made from them, in the same
order (HGCalTrackstersFromSoAProducer), and those alone.
*/

#include <vector>

#include "DataFormats/HGCalReco/interface/TracksterFwd.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "RecoTICL/Linking/interface/TracksterLinkingAlgoBase.h"
#include "RecoTICL/Superclustering/interface/SuperclusteringDNNHostCollections.h"

namespace ticl {

  class TracksterLinkingbySuperClusteringDNNHeterogeneous : public TracksterLinkingAlgoBase {
  public:
    TracksterLinkingbySuperClusteringDNNHeterogeneous(const edm::ParameterSet& ps,
                                                      edm::ConsumesCollector iC,
                                                      const cms::Ort::ONNXRuntime* onnxRuntime = nullptr);

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

    void linkTracksters(const Inputs& input,
                        std::vector<Trackster>& resultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedResultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) override;

    void initialize(const HGCalDDDConstants* hgcons,
                    const ticlgeom::Tools rhtools,
                    const edm::ESHandle<MagneticField> bfieldH,
                    const edm::ESHandle<Propagator> propH) override {}

  private:
    bool passesPID(const Trackster& ts, float threshold) const;

    const edm::EDGetTokenT<SuperclusteringDNNPairsHost> pairsToken_;
    const edm::EDGetTokenT<SuperclusteringDNNPairScoresHost> scoresToken_;
    const float nnWorkingPoint_;
    const float seedPtThreshold_;
    const bool filterByTracksterPID_;
    const std::vector<int> tracksterPIDCategoriesToFilter_;
    const float emissionPIDThreshold_;
  };

}  // namespace ticl

#endif
