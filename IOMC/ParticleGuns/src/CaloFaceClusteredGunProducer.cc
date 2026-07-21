/** \class CaloFaceClusteredGunProducer
 *
 * Shoots NParticles from the vertex such that they arrive CLUSTERED AT THE
 * CALORIMETER FACE: all pairwise (eta,phi) distances at z = ZFace are at most
 * MaxDeltaR. Charged particles are pre-compensated for the solenoid bending
 * (the generated phi is rotated by the helix chord angle at the face radius,
 * minus-q-delta convention verified against SimTrack boundary crossings), so
 * the clustering constraint holds where the showers start, not at generation.
 * Species cycle through PartID. Built for HGCAL linking and candidate
 * fake-rate benches: maximum shower confusion with tracker tracks.
 */

#include <cmath>
#include <ostream>

#include "IOMC/ParticleGuns/src/BaseFlatGunProducer.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "FWCore/AbstractServices/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CLHEP/Random/RandFlat.h"

namespace edm {

  class CaloFaceClusteredGunProducer : public BaseFlatGunProducer {
  public:
    CaloFaceClusteredGunProducer(const ParameterSet& pset);
    ~CaloFaceClusteredGunProducer() override = default;
    void produce(Event& e, const EventSetup& es) override;

  private:
    double fMinE;
    std::vector<double> fMinEPart;
    double fMaxE;
    int fNParticles;
    double fMaxDeltaR;
    double fZFace;   // cm
    double fBField;  // T
  };

  CaloFaceClusteredGunProducer::CaloFaceClusteredGunProducer(const ParameterSet& pset) : BaseFlatGunProducer(pset) {
    ParameterSet pgun = pset.getParameter<ParameterSet>("PGunParameters");
    fMinE = pgun.getParameter<double>("MinE");
    fMinEPart = pgun.getParameter<std::vector<double>>("MinEPart");
    if (!fMinEPart.empty() && fMinEPart.size() != fPartIDs.size()) {
      throw cms::Exception("Configuration") << "MinEPart must be empty or aligned with PartID";
    }
    fMaxE = pgun.getParameter<double>("MaxE");
    fNParticles = pgun.getParameter<int>("NParticles");
    fMaxDeltaR = pgun.getParameter<double>("MaxDeltaR");
    fZFace = pgun.getParameter<double>("ZFace");
    fBField = pgun.getParameter<double>("BField");
    produces<HepMCProduct>("unsmeared");
    produces<GenEventInfoProduct>();
  }

  void CaloFaceClusteredGunProducer::produce(Event& e, const EventSetup& es) {
    edm::Service<edm::RandomNumberGenerator> rng;
    CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

    fEvt = new HepMC::GenEvent();
    HepMC::GenVertex* Vtx = new HepMC::GenVertex(HepMC::FourVector(0., 0., 0.));

    // Cluster center at the face; the scatter radius keeps every pair within
    // MaxDeltaR by construction.
    const double etaC = CLHEP::RandFlat::shoot(engine, fMinEta, fMaxEta);
    const double phiC = CLHEP::RandFlat::shoot(engine, fMinPhi, fMaxPhi);
    const double rScatter = 0.5 * fMaxDeltaR;

    int barcode = 1;
    for (int ip = 0; ip < fNParticles; ++ip) {
      // Uniform in the disk around the center: target position at the face.
      const double u = CLHEP::RandFlat::shoot(engine, 0., 1.);
      const double a = CLHEP::RandFlat::shoot(engine, 0., 2. * M_PI);
      const double rd = rScatter * std::sqrt(u);
      const double etaT = etaC + rd * std::cos(a);
      const double phiT = phiC + rd * std::sin(a);

      const int PartID = fPartIDs[ip % fPartIDs.size()];
      const HepPDT::ParticleData* PData = fPDGTable->particle(HepPDT::ParticleID(abs(PartID)));
      const double mass = PData->mass().value();
      // The PDG table is looked up by |pdgId|: antiparticles carry the opposite
      // charge, and an inverted sign flips the bend compensation by 2*delta.
      const double charge = (PartID < 0) ? -PData->charge() : PData->charge();
      // Aimability floors per species: the chord compensation cannot hold at low
      // pT for particles that change curvature mid-flight (electron
      // bremsstrahlung) or barely reach the face (near-looper hadrons), so each
      // species carries its own energy floor.
      const double eMin = fMinEPart.empty() ? fMinE : std::max(fMinE, fMinEPart[ip % fMinEPart.size()]);
      const double energy = CLHEP::RandFlat::shoot(engine, eMin, fMaxE);
      const double mom2 = energy * energy - mass * mass;
      const double mom = mom2 > 0. ? std::sqrt(mom2) : 0.;
      const double theta = 2. * std::atan(std::exp(-etaT));
      const double pt = mom * std::sin(theta);

      // Helix chord pre-compensation: rotate the generated direction so the
      // POSITION azimuth at the face radius equals the target. r_face from the
      // straight-line crossing of z = ZFace at this eta; curvature radius
      // R_c[m] = pT / (0.3 B). Loopers cannot be aimed: clamp the chord angle.
      double phiGen = phiT;
      if (charge != 0. && pt > 0.05 && fBField > 0.) {
        const double rFaceM = (fZFace / 100.) / std::abs(std::sinh(etaT));
        const double rc = pt / (0.3 * fBField);
        const double arg = std::min(0.9, rFaceM / (2. * rc));
        const double delta = std::asin(arg);
        phiGen = phiT + (charge > 0. ? delta : -delta) * (etaT > 0. ? 1. : -1.);
      }

      const double px = pt * std::cos(phiGen);
      const double py = pt * std::sin(phiGen);
      const double pz = mom * std::cos(theta);

      HepMC::FourVector p(px, py, pz, energy);
      HepMC::GenParticle* Part = new HepMC::GenParticle(p, PartID, 1);
      Part->suggest_barcode(barcode);
      ++barcode;
      Vtx->add_particle_out(Part);
    }

    fEvt->add_vertex(Vtx);
    fEvt->set_event_number(e.id().event());
    fEvt->set_signal_process_id(20);

    std::unique_ptr<HepMCProduct> BProduct(new HepMCProduct());
    BProduct->addHepMCData(fEvt);
    e.put(std::move(BProduct), "unsmeared");

    std::unique_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct(fEvt));
    e.put(std::move(genEventInfo));

    if (fVerbosity > 0)
      fEvt->print();
  }

}  // namespace edm

using edm::CaloFaceClusteredGunProducer;
DEFINE_FWK_MODULE(CaloFaceClusteredGunProducer);
