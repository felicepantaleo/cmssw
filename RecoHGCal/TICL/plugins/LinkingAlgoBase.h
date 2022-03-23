#ifndef RecoHGCal_TICL_LinkingAlgoBase_H__
#define RecoHGCal_TICL_LinkingAlgoBase_H__

#include <memory>
#include <vector>
#include "DataFormats/HGCalReco/interface/Common.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

namespace ticl {
  class LinkingAlgoBase {
  public:
    LinkingAlgoBase(const edm::ParameterSet& conf) {}

    virtual ~LinkingAlgoBase(){};

    virtual void initialize(const HGCalDDDConstants* hgcons,
                            const hgcal::RecHitTools rhtools,
                            const edm::ESHandle<MagneticField> bfieldH,
                            const edm::ESHandle<Propagator> propH) = 0;

    virtual void linkTracksters(const edm::Handle<std::vector<reco::Track>> tkH,
                                const StringCutObjectSelector<reco::Track> cutTk,
                                const edm::Handle<std::vector<Trackster>> tsH,
                                std::vector<TICLCandidate>& resultTracksters) = 0;

    static void fillPSetDescription(edm::ParameterSetDescription& desc){};
  };
}  // namespace ticl

#endif
