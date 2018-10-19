// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 09/2018
// Copyright CERN

#ifndef RecoHGCal_TICL_PatternRecognitionAlgo_H__
#define RecoHGCal_TICL_PatternRecognitionAlgo_H__

#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBase.h"

#include <memory>

class PatternRecognitionAlgo {
public:
  explicit PatternRecognitionAlgo(const edm::ParameterSet& conf)
   {
           myalgo_ = std::make_unique<PatternRecognitionAlgoBase>(conf);
   }

  explicit PatternRecognitionAlgo(std::unique_ptr<PatternRecognitionAlgoBase> otherAlgo): myalgo_(std::move(otherAlgo)) {}

  void swap(PatternRecognitionAlgo& o) { std::swap(myalgo_, o.myalgo_); }

  void makeTracksters(
      const edm::Event& ev,
      const edm::EventSetup& es,
      const std::vector<reco::CaloCluster>& layerClusters,
      const std::vector<std::pair<unsigned int, float> >& mask, std::vector<Trackster>& result) const { myalgo_->makeTracksters(ev, es, layerClusters, mask, result); }

private:
  std::unique_ptr<PatternRecognitionAlgoBase> myalgo_;
};

#endif
