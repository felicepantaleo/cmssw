#ifndef Portable_Collection_SoA_Test_h
#define Portable_Collection_SoA_Test_h

#include "DataFormats/NGTSoATest/interface/SoALayoutTest.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

using PhysicsObjCollection = PortableHostCollection<PhysicsObj>;
using PhysicsObjExtraCollection = PortableHostCollection<PhysicsObjExtra>;

using CombinedPhysicsObjectCollection = PortableHostCollection<CombinedPhysicsObject>;

// using SoAHostCollection2 = PortableHostCollection2<PhysicsObj, PhysicsObjExtra>;
// using SoAHostCollectionView1 = PortableHostCollection2<PhysicsObj, PhysicsObjExtra>::View<0>;
// using SoAHostCollectionView2 = PortableHostCollection2<PhysicsObj, PhysicsObjExtra>::View<1>;

using SoAHostPositionCollectionView = PortableHostCollection<PhysicsObj>::View;
using SoAHostCollectionConstView = PortableHostCollection<PhysicsObj>::ConstView;

using SoAHostVelocityCollectionView = PortableHostCollection<PhysicsObjExtra>::View;
using SoAHostVelocityCollectionConstView = PortableHostCollection<PhysicsObjExtra>::ConstView;

#endif