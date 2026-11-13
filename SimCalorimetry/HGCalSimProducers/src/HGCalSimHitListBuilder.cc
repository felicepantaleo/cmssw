#include <algorithm>
#include <cmath>

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCalSimHitListBuilder.h"

HGCalSimHitListBuilder::HGCalSimHitListBuilder(edm::ParameterSet const& iConfig, edm::ConsumesCollector iC)
    : denseIndexToken_{iC.esConsumes(iConfig.getParameter<edm::ESInputTag>("indexingSource"))},
      cellInfoToken_{iC.esConsumes(iConfig.getParameter<edm::ESInputTag>("cellMappingSource"))} {
  auto const& sourceConfigs = iConfig.getParameter<std::vector<edm::ParameterSet>>("g4HitsConfig");
  sources_.reserve(sourceConfigs.size());
  for (auto const& sourceConfig : sourceConfigs) {
    auto const det = sourceConfig.getParameter<int32_t>("det");
    auto const iscalib = sourceConfig.getParameter<bool>("iscalib");
    auto const duplicate = std::any_of(sources_.begin(), sources_.end(), [det, iscalib](Source const& source) {
      return source.det == det && source.iscalib == iscalib;
    });
    if (duplicate) {
      throw cms::Exception("Configuration")
          << "HGCalSimHitListBuilder: more than one source for det=" << det << " iscalib=" << iscalib;
    }
    auto const tag = sourceConfig.getParameter<edm::InputTag>("hits");
    sources_.push_back(Source{det,
                              iscalib,
                              sourceConfig.getParameter<double>("tof_delay"),
                              tag,
                              iC.consumes<edm::PCaloHitContainer>(tag),
                              {}});
  }
}

void HGCalSimHitListBuilder::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.add<edm::ESInputTag>("indexingSource", edm::ESInputTag{})->setComment("Label of the dense index of the cells");
  desc.add<edm::ESInputTag>("cellMappingSource", edm::ESInputTag{})->setComment("Label of the cell information");

  edm::ParameterSetDescription hitDesc;
  hitDesc.add<int32_t>("det", 8)->setComment("DetId detector code of the hits");
  hitDesc.add<bool>("iscalib", false)->setComment("True for the hits of calibration cells");
  hitDesc.add<double>("tof_delay", -9.)->setComment("Delay [ns] added to the hit time");
  hitDesc.add<edm::InputTag>("hits", edm::InputTag("g4SimHits", "HGCHitsEE"))->setComment("PCaloHit collection");
  edm::ParameterSet hitDefault;
  hitDefault.addParameter<int32_t>("det", 8);
  hitDefault.addParameter<bool>("iscalib", false);
  hitDefault.addParameter<double>("tof_delay", -9.);
  hitDefault.addParameter<edm::InputTag>("hits", edm::InputTag("g4SimHits", "HGCHitsEE"));
  desc.addVPSet("g4HitsConfig", hitDesc, {hitDefault})->setComment("Sources of PCaloHits");
}

void HGCalSimHitListBuilder::buildDetIdMaps(hgcal::HGCalDenseIndexInfoHost const& denseIndexInfo,
                                            hgcal::HGCalMappingCellParamHost const& cellInfo) {
  for (auto& source : sources_) {
    source.detIdToDenseIdx.clear();
  }

  auto const denseIndexView = denseIndexInfo.const_view();
  auto const cellInfoView = cellInfo.const_view();
  nDenseIndices_ = denseIndexView.metadata().size();
  for (int32_t i = 0; i < nDenseIndices_; ++i) {
    auto const row = denseIndexView[i];
    auto const chType = cellInfoView[row.cellInfoIdx()].t();
    // a negative channel type is an unconnected channel, type 0 is a calibration cell
    if (chType < 0) {
      continue;
    }
    auto const detId = row.detid();
    auto const det = DetId(detId).det();
    auto const iscalib = (chType == 0);
    for (auto& source : sources_) {
      if (source.det == det && source.iscalib == iscalib) {
        source.detIdToDenseIdx[detId] = i;
        break;
      }
    }
  }
}

void HGCalSimHitListBuilder::beginEvent(edm::EventSetup const& iSetup) {
  bool const denseIndexChanged = denseIndexWatcher_.check(iSetup);
  bool const cellInfoChanged = cellInfoWatcher_.check(iSetup);
  if (denseIndexChanged || cellInfoChanged) {
    buildDetIdMaps(iSetup.getData(denseIndexToken_), iSetup.getData(cellInfoToken_));
  }

  denseIdx_.clear();
  timeBin_.clear();
  energy_.clear();
  energyXtof_.clear();

  monitors_.clear();
  monitors_.reserve(sources_.size());
  for (auto const& source : sources_) {
    auto& monitor = monitors_.emplace_back(source.det, source.iscalib);
    monitor.ncellsMapped = source.detIdToDenseIdx.size();
  }
}

void HGCalSimHitListBuilder::add(edm::PCaloHitContainer const& hits, std::size_t sourceIdx) {
  auto const& source = sources_.at(sourceIdx);
  auto& monitor = monitors_.at(sourceIdx);
  monitor.nHits += hits.size();

  for (auto const& hit : hits) {
    float const energy = hit.energy();
    monitor.en += energy;

    auto const it = source.detIdToDenseIdx.find(hit.id());
    if (it == source.detIdToDenseIdx.end()) {
      ++monitor.nHitsUnknown;
      monitor.enUnknown += energy;
      continue;
    }

    float const tof = hit.time() + source.tofDelay;
    int32_t const timeBin = static_cast<int32_t>(std::roundl(tof / nsPerBx_)) + hgcaldigi::kInTimeBin;
    if (timeBin < 0 || timeBin >= hgcaldigi::kTimeBins) {
      ++monitor.nHitsRejected;
      monitor.enRejected += energy;
      continue;
    }

    if (timeBin == hgcaldigi::kInTimeBin) {
      ++monitor.nHitsIT;
      monitor.enIT += energy;
    } else {
      ++monitor.nHitsOOT;
      monitor.enOOT += energy;
    }

    denseIdx_.push_back(it->second);
    timeBin_.push_back(timeBin);
    // The ageing step reads the energy in keV.
    float const energyKeV = energy * 1.e6f;
    energy_.push_back(energyKeV);
    energyXtof_.push_back(energyKeV * tof);
  }
}

hgcaldigi::HGCalSimHitListHost HGCalSimHitListBuilder::makeList() const {
  auto const nHits = static_cast<int32_t>(denseIdx_.size());
  hgcaldigi::HGCalSimHitListHost list(nHits);
  auto view = list.view();
  for (int32_t i = 0; i < nHits; ++i) {
    auto row = view[i];
    row.denseIdx() = denseIdx_[i];
    row.timeBin() = timeBin_[i];
    row.energy() = energy_[i];
    row.energyXtof() = energyXtof_[i];
  }
  return list;
}
