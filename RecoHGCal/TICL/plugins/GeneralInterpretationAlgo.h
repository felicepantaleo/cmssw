#ifndef RecoHGCal_TICL_GeneralInterpretationAlgo_H_
#define RecoHGCal_TICL_GeneralInterpretationAlgo_H_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "RecoHGCal/TICL/interface/TICLInterpretationAlgoBase.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace ticl {

  class GeneralInterpretationAlgo : public TICLInterpretationAlgoBase<reco::Track> {
  public:
    GeneralInterpretationAlgo(const edm::ParameterSet& conf, edm::ConsumesCollector iC)
        : TICLInterpretationAlgoBase<reco::Track>(conf, iC) {

      }

  virtual ~GeneralInterpretationAlgo() {}

  virtual void makeCandidates(const Inputs& input, std::vector<Trackster>& resultTracksters,
                                std::vector<TICLCandidate>& resultCandidate) override;

  static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {
      iDesc.add<int>("algo_verbosity", 0);
    }

  };

}  // namespace ticl

#endif
