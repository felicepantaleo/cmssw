#ifndef CondFormats_HGCalObjects_HGCalGeomHostCollection_h
#define CondFormats_HGCalObjects_HGCalGeomHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "CondFormats/HGCalObjects/interface/HGCalGeomSoA.h"

using HGCalGeomHostCollection = PortableHostCollection<HGCalGeomSoA>;
using HGCalGeomHostCollectionView = PortableHostCollection<HGCalGeomSoA>::View;
using HGCalGeomHostCollectionConstView = PortableHostCollection<HGCalGeomSoA>::ConstView;

#endif