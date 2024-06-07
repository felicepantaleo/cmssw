#ifndef CondFormats_HGCalObjects_TICLGeom_h
#define CondFormats_HGCalObjects_TICLGeom_h

#include "CondFormats/HGCalObjects/interface/TICLGeomHostCollection.h"
#include "CondFormats/HGCalObjects/interface/alpaka/TICLGeomDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include <map>
#include <cstdint>

struct TICLGeom {
  std::map<uint32_t, uint32_t> detIdToIndexMap;
  std::unique_ptr<TICLGeomHostCollection> hostCollection;

  TICLGeom() = delete;
  explicit TICLGeom(std::size_t size)
      : hostCollection(std::make_unique<TICLGeomHostCollection>(size, cms::alpakatools::host())) {}

  // Delete the copy constructor and copy assignment operator
  TICLGeom(const TICLGeom&) = delete;
  TICLGeom& operator=(const TICLGeom&) = delete;

  // Default move constructor and move assignment operator
  TICLGeom(TICLGeom&&) = default;
  TICLGeom& operator=(TICLGeom&&) = default;
};

#endif  // CondFormats_HGCalObjects_TICLGeom_h
