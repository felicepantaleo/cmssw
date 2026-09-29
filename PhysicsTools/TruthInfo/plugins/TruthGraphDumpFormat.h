// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// Text helpers shared by the two graph dumpers.

#ifndef PhysicsTools_TruthInfo_plugins_TruthGraphDumpFormat_h
#define PhysicsTools_TruthInfo_plugins_TruthGraphDumpFormat_h

#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>

#include "DataFormats/Provenance/interface/EventID.h"

namespace truth::dump {

  // The name of a particle, in ASCII or with Unicode symbols. "pdg" when the species has
  // no name here.
  inline std::string pdgName(int pdgId, bool unicode) {
    struct Name {
      int pdgId;
      const char* ascii;
      const char* unicode;
    };
    static constexpr Name kNames[] = {
        {11, "e-", "e\u207B"},
        {-11, "e+", "e\u207A"},
        {13, "mu-", "\u03BC\u207B"},
        {-13, "mu+", "\u03BC\u207A"},
        {15, "tau-", "\u03C4\u207B"},
        {-15, "tau+", "\u03C4\u207A"},
        {12, "nu_e", "\u03BD\u2091"},
        {-12, "anti-nu_e", "\u03BD\u0304\u2091"},
        {14, "nu_mu", "\u03BD_\u03BC"},
        {-14, "anti-nu_mu", "\u03BD\u0304_\u03BC"},
        {16, "nu_tau", "\u03BD_\u03C4"},
        {-16, "anti-nu_tau", "\u03BD\u0304_\u03C4"},
        {22, "gamma", "\u03B3"},
        {21, "g", "g"},
        {23, "Z0", "Z\u2070"},
        {24, "W+", "W\u207A"},
        {-24, "W-", "W\u207B"},
        {25, "H", "H"},
        {2212, "p", "p"},
        {-2212, "anti-p", "p\u0304"},
        {2112, "n", "n"},
        {-2112, "anti-n", "n\u0304"},
        {111, "pi0", "\u03C0\u2070"},
        {211, "pi+", "\u03C0\u207A"},
        {-211, "pi-", "\u03C0\u207B"},
        {321, "K+", "K\u207A"},
        {-321, "K-", "K\u207B"},
        {130, "K0_L", "K\u2070_L"},
        {310, "K0_S", "K\u2070_S"},
    };
    for (auto const& name : kNames) {
      if (name.pdgId == pdgId)
        return unicode ? name.unicode : name.ascii;
    }
    const int ap = std::abs(pdgId);
    if (ap >= 1 && ap <= 6) {
      static const char* qname[7] = {"", "d", "u", "s", "c", "b", "t"};
      std::string s = qname[ap];
      if (pdgId < 0)
        s = "anti-" + s;
      return s;
    }
    return "pdg";
  }

  // "name (pdgId)", or "pdg(pdgId)" for a species with no name.
  inline std::string pdgLabel(int pdgId, bool unicode) {
    std::ostringstream ss;
    const std::string name = pdgName(pdgId, unicode);
    if (name == "pdg")
      ss << "pdg(" << pdgId << ")";
    else
      ss << name << " (" << pdgId << ")";
    return ss.str();
  }

  // The names of the reco::GenStatusFlags bits that are set, or "none".
  inline std::string statusFlagsLabel(uint16_t flags) {
    static constexpr const char* kNames[] = {"isPrompt",
                                             "isDecayedLeptonHadron",
                                             "isTauDecayProduct",
                                             "isPromptTauDecayProduct",
                                             "isDirectTauDecayProduct",
                                             "isDirectPromptTauDecayProduct",
                                             "isDirectHadronDecayProduct",
                                             "isHardProcess",
                                             "fromHardProcess",
                                             "isHardProcessTauDecayProduct",
                                             "isDirectHardProcessTauDecayProduct",
                                             "fromHardProcessBeforeFSR",
                                             "isFirstCopy",
                                             "isLastCopy",
                                             "isLastCopyBeforeFSR"};
    std::ostringstream ss;
    bool first = true;
    for (unsigned bit = 0; bit < std::size(kNames); ++bit) {
      if ((flags & (1u << bit)) == 0)
        continue;
      if (!first)
        ss << ", ";
      ss << kNames[bit];
      first = false;
    }
    return first ? std::string("none") : ss.str();
  }

  // The file name with _run<r>_lumi<l>_event<e> inserted before the extension.
  inline std::string appendEventIdToFilename(std::string const& filename, edm::EventID const& id) {
    const auto dotPos = filename.rfind('.');
    std::ostringstream ss;
    ss << filename.substr(0, dotPos) << "_run" << id.run() << "_lumi" << id.luminosityBlock() << "_event" << id.event();
    if (dotPos != std::string::npos)
      ss << filename.substr(dotPos);
    return ss.str();
  }

}  // namespace truth::dump

#endif
