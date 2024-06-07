#ifndef CondFormats_HGCalObjects_TICLGeomHostCollection_h
#define CondFormats_HGCalObjects_TICLGeomHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomSoA.h"

using TICLGeomHostCollection = PortableHostCollection<TICLGeomSoA>;
using TICLGeomHostCollectionView = PortableHostCollection<TICLGeomSoA>::View;
using TICLGeomHostCollectionConstView = PortableHostCollection<TICLGeomSoA>::ConstView;

#endif