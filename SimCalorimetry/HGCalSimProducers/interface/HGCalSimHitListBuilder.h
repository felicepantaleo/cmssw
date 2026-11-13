#ifndef SimCalorimetry_HGCalSimProducers_interface_HGCalSimHitListBuilder_h
#define SimCalorimetry_HGCalSimProducers_interface_HGCalSimHitListBuilder_h

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "CondFormats/DataRecord/interface/HGCalDenseIndexInfoRcd.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalSimHitAccumSoA.h"
#include "DataFormats/HGCalDigi/interface/HGCalSimHitListHost.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

// Collects the Geant4 hits of one event, signal and pileup, into a list for the HGCAL portable digitizer.
// It keeps a hit only if its cell has a dense index and its time is in the time window of the digitizer.
class HGCalSimHitListBuilder {
public:
  struct Source {
    int32_t det;
    bool iscalib;
    double tofDelay;
    edm::InputTag tag;
    edm::EDGetTokenT<edm::PCaloHitContainer> token;
    std::unordered_map<uint32_t, int32_t> detIdToDenseIdx;
  };

  HGCalSimHitListBuilder(edm::ParameterSet const&, edm::ConsumesCollector);

  static void fillPSetDescription(edm::ParameterSetDescription&);

  // Updates the DetId map when the conditions change, and clears the list and the counters.
  void beginEvent(edm::EventSetup const&);

  // Appends the hits of one source in input order.
  void add(edm::PCaloHitContainer const&, std::size_t source);

  hgcaldigi::HGCalSimHitListHost makeList() const;

  std::vector<Source> const& sources() const { return sources_; }
  int32_t nDenseIndices() const { return nDenseIndices_; }
  hgcaldigi::HGCalSimHitAccumulatorMonitorCollection const& monitors() const { return monitors_; }

private:
  void buildDetIdMaps(hgcal::HGCalDenseIndexInfoHost const&, hgcal::HGCalMappingCellParamHost const&);

  static constexpr float nsPerBx_ = 25.0f;

  const edm::ESGetToken<hgcal::HGCalDenseIndexInfoHost, HGCalDenseIndexInfoRcd> denseIndexToken_;
  const edm::ESGetToken<hgcal::HGCalMappingCellParamHost, HGCalElectronicsMappingRcd> cellInfoToken_;
  edm::ESWatcher<HGCalDenseIndexInfoRcd> denseIndexWatcher_;
  edm::ESWatcher<HGCalElectronicsMappingRcd> cellInfoWatcher_;

  std::vector<Source> sources_;
  int32_t nDenseIndices_ = 0;

  std::vector<int32_t> denseIdx_;
  std::vector<int32_t> timeBin_;
  std::vector<float> energy_;
  std::vector<float> energyXtof_;
  hgcaldigi::HGCalSimHitAccumulatorMonitorCollection monitors_;
};

#endif  // SimCalorimetry_HGCalSimProducers_interface_HGCalSimHitListBuilder_h
