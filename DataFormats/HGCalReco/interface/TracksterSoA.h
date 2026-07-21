#ifndef DataFormats_HGCalReco_interface_TracksterSoA_h
#define DataFormats_HGCalReco_interface_TracksterSoA_h

// Device-side view of ticl::Trackster: the per-trackster quantities that
// downstream TICL steps read, laid out as an SoA. It is deliberately not tied to
// any one algorithm; the columns are the ones every linking, interpretation or
// PID step needs.
//
// The layer-cluster constituents (Trackster::vertices()) are NOT here: they are
// a ragged, per-trackster list, so they belong in a companion CSR collection
// addressed by verticesOffset/nVertices. Those two columns are kept so a
// consumer that needs the constituents can find them, and so the SoA stays a
// faithful, round-trippable view rather than an algorithm scratch buffer.
//
// eta and phi are stored rather than recomputed from the barycenter: consumers
// bin on them, and keeping them here removes two transcendentals per trackster
// from every inner loop.

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(TracksterSoALayout,
                    // barycenter [cm]
                    SOA_COLUMN(float, baryX),
                    SOA_COLUMN(float, baryY),
                    SOA_COLUMN(float, baryZ),
                    // principal axis, eigenvectors(0), unit vector oriented outwards
                    SOA_COLUMN(float, axisX),
                    SOA_COLUMN(float, axisY),
                    SOA_COLUMN(float, axisZ),
                    // cached barycenter direction
                    SOA_COLUMN(float, eta),
                    SOA_COLUMN(float, phi),
                    // energies [GeV]
                    SOA_COLUMN(float, rawEnergy),
                    SOA_COLUMN(float, regressedEnergy),
                    SOA_COLUMN(float, rawPt),
                    // timing; timeError < 0 flags an invalid time
                    SOA_COLUMN(float, time),
                    SOA_COLUMN(float, timeError),
                    // PCA shape
                    SOA_COLUMN(float, eigenvalue0),
                    SOA_COLUMN(float, eigenvalue1),
                    SOA_COLUMN(float, eigenvalue2),
                    SOA_COLUMN(float, sigmaPCA0),
                    SOA_COLUMN(float, sigmaPCA1),
                    SOA_COLUMN(float, sigmaPCA2),
                    // range into the companion vertices collection
                    SOA_COLUMN(uint32_t, verticesOffset),
                    SOA_COLUMN(uint32_t, nVertices))

using TracksterSoA = TracksterSoALayout<>;

#endif  // DataFormats_HGCalReco_interface_TracksterSoA_h
