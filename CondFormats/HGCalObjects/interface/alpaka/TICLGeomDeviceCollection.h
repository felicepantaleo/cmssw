#ifndef CondFormats_HGCalObjects_TICLGeomDeviceCollection_h
#define CondFormats_HGCalObjects_TICLGeomDeviceCollection_h

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

using TICLGeomDeviceCollection = PortableDeviceCollection<TICLGeomSoA>;
using TICLGeomDeviceCollectionView = PortableDeviceCollection<TICLGeomSoA>::View;
using TICLGeomDeviceCollectionConstView = PortableDeviceCollection<TICLGeomSoA>::ConstView;
}

#endif  // CondFormats_HGCalObjects_TICLGeomDeviceCollection_h
