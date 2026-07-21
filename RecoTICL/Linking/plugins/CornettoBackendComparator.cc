// Require two Cornetto linking backends to produce IDENTICAL components.
//
// The device path emits the minimum input index of each component as its label and
// the host union-find uses parent[max] = min, so the two are comparable element by
// element: this is a bitwise diff, not a physics comparison. Any difference is a
// determinism or floating-point-ordering bug, so it throws rather than warns.

#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"

using Links = std::vector<std::vector<unsigned int>>;

class CornettoBackendComparator : public edm::global::EDAnalyzer<> {
public:
  explicit CornettoBackendComparator(const edm::ParameterSet &ps)
      : refToken_(consumes<Links>(ps.getParameter<edm::InputTag>("reference"))),
        testToken_(consumes<Links>(ps.getParameter<edm::InputTag>("test"))) {}

  void analyze(edm::StreamID, const edm::Event &evt, const edm::EventSetup &) const override {
    const auto &ref = evt.get(refToken_);
    const auto &test = evt.get(testToken_);

    if (ref.size() != test.size()) {
      throw edm::Exception(edm::errors::LogicError)
          << "component count differs: reference " << ref.size() << " vs test " << test.size();
    }
    for (size_t c = 0; c < ref.size(); ++c) {
      if (ref[c] != test[c]) {
        std::ostringstream refStr, testStr;
        for (auto v : ref[c])
          refStr << v << " ";
        for (auto v : test[c])
          testStr << v << " ";
        throw edm::Exception(edm::errors::LogicError)
            << "component " << c << " differs: reference {" << refStr.str() << "} vs test {" << testStr.str() << "}";
      }
    }
    edm::LogPrint("CornettoBackendComparator") << "identical: " << ref.size() << " components match exactly";
  }

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("reference", edm::InputTag("ticlTracksterLinks", "linkedTracksterIdToInputTracksterId"));
    desc.add<edm::InputTag>("test", edm::InputTag("ticlCornettoLinksHost"));
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const edm::EDGetTokenT<Links> refToken_;
  const edm::EDGetTokenT<Links> testToken_;
};

DEFINE_FWK_MODULE(CornettoBackendComparator);
