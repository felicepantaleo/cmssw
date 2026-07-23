// Round-trip check for LegacyTracksterToSoAProducer: rebuild the same
// edm::MultiSpan<Trackster> the producer flattened, and require every SoA column
// to match the corresponding trackster bit for bit. Fields are copied, not
// computed, so anything other than exact equality is a bug; eta/phi/rawPt are
// compared exactly too because the producer takes them from the same accessors
// this analyzer calls.
//
// Throws on the first mismatch so it can be used as a gate in a test job.

#include <vector>

#include "DataFormats/Common/interface/MultiSpan.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TracksterHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"

class TracksterSoAValidator : public edm::global::EDAnalyzer<> {
public:
  explicit TracksterSoAValidator(const edm::ParameterSet &ps)
      : soaToken_(consumes<TracksterHostCollection>(ps.getParameter<edm::InputTag>("tracksterSoA"))) {
    for (auto const &tag : ps.getParameter<std::vector<edm::InputTag>>("tracksters_collections"))
      trackstersTokens_.push_back(consumes<std::vector<ticl::Trackster>>(tag));
  }

  void analyze(edm::StreamID, const edm::Event &evt, const edm::EventSetup &) const override {
    edm::MultiSpan<ticl::Trackster> tracksters;
    for (const auto &token : trackstersTokens_)
      tracksters.add(evt.get(token));

    const auto view = evt.get(soaToken_).const_view();
    const int32_t n = static_cast<int32_t>(tracksters.size());
    if (view.metadata().size() != n) {
      throw edm::Exception(edm::errors::LogicError)
          << "SoA size " << view.metadata().size() << " != trackster count " << n;
    }

    uint32_t expectedOffset = 0;
    for (int32_t i = 0; i < n; ++i) {
      const auto &ts = tracksters[i];
      const auto element = view[i];
      check(i, "baryX", element.baryX(), ts.barycenter().x());
      check(i, "baryY", element.baryY(), ts.barycenter().y());
      check(i, "baryZ", element.baryZ(), ts.barycenter().z());
      check(i, "axisX", element.axisX(), ts.eigenvectors(0).x());
      check(i, "axisY", element.axisY(), ts.eigenvectors(0).y());
      check(i, "axisZ", element.axisZ(), ts.eigenvectors(0).z());
      check(i, "eta", element.eta(), static_cast<float>(ts.barycenter().eta()));
      check(i, "phi", element.phi(), static_cast<float>(ts.barycenter().phi()));
      check(i, "rawEnergy", element.rawEnergy(), ts.raw_energy());
      check(i, "regressedEnergy", element.regressedEnergy(), ts.regressed_energy());
      check(i, "rawPt", element.rawPt(), ts.raw_pt());
      check(i, "time", element.time(), ts.time());
      check(i, "timeError", element.timeError(), ts.timeError());
      check(i, "eigenvalue0", element.eigenvalue0(), ts.eigenvalues()[0]);
      check(i, "sigmaPCA0", element.sigmaPCA0(), ts.sigmasPCA()[0]);

      const uint32_t nVertices = static_cast<uint32_t>(ts.vertices().size());
      if (element.nVertices() != nVertices || element.verticesOffset() != expectedOffset) {
        throw edm::Exception(edm::errors::LogicError)
            << "trackster " << i << ": vertices range (" << element.verticesOffset() << ", " << element.nVertices()
            << ") != expected (" << expectedOffset << ", " << nVertices << ")";
      }
      expectedOffset += nVertices;
    }

    edm::LogPrint("TracksterSoAValidator") << n << " tracksters, SoA matches legacy exactly";
  }

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<edm::InputTag>>("tracksters_collections", {edm::InputTag("ticlTrackstersCLUE3DHigh")});
    desc.add<edm::InputTag>("tracksterSoA", edm::InputTag("tracksterSoACLUE3D"));
    descriptions.addWithDefaultLabel(desc);
  }

private:
  void check(int32_t i, const char *field, float got, float expected) const {
    if (got != expected) {
      throw edm::Exception(edm::errors::LogicError)
          << "trackster " << i << " field " << field << ": SoA " << got << " != legacy " << expected;
    }
  }

  std::vector<edm::EDGetTokenT<std::vector<ticl::Trackster>>> trackstersTokens_;
  const edm::EDGetTokenT<TracksterHostCollection> soaToken_;
};

DEFINE_FWK_MODULE(TracksterSoAValidator);
