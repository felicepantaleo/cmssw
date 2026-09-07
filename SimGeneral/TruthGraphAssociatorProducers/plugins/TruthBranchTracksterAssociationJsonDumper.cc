// Writes the trackster to truth-branch association of one event as JSON, for the
// interactive truth-graph viewer.
//
// A match names its truth branch by the root particle index of truth::Graph, which is
// the index TruthLogicalGraphDumper writes its particle nodes under, so the viewer joins
// the two files on that number alone. Both dumps must therefore run in the same job.

#include <fstream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

class TruthBranchTracksterAssociationJsonDumper : public edm::one::EDAnalyzer<> {
public:
  explicit TruthBranchTracksterAssociationJsonDumper(edm::ParameterSet const&);
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  using AssociationMap = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;

  struct Collection {
    std::string key;
    edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksters;
    // One map per working point, in the order of workingPointNames.
    std::vector<edm::EDGetTokenT<AssociationMap>> maps;
  };

  const std::string jsonFile_;
  const std::vector<std::string> workingPointNames_;
  const unsigned maxMatches_;
  std::vector<Collection> collections_;
};

TruthBranchTracksterAssociationJsonDumper::TruthBranchTracksterAssociationJsonDumper(edm::ParameterSet const& cfg)
    : jsonFile_(cfg.getParameter<std::string>("jsonFile")),
      workingPointNames_(cfg.getParameter<std::vector<std::string>>("workingPointNames")),
      maxMatches_(cfg.getParameter<unsigned>("maxMatches")) {
  const std::string associator = cfg.getParameter<std::string>("associator");
  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("recoCollections")) {
    Collection collection;
    // The associator keys its products by label, joined to the instance by an
    // underscore. Rebuild the same key so the maps of a collection are found.
    collection.key = tag.label();
    if (!tag.instance().empty()) {
      collection.key += "_" + tag.instance();
    }
    collection.tracksters = consumes<std::vector<ticl::Trackster>>(tag);
    for (auto const& name : workingPointNames_) {
      collection.maps.push_back(
          consumes<AssociationMap>(edm::InputTag(associator, collection.key + "RecoToTruth" + name)));
    }
    collections_.push_back(std::move(collection));
  }
}

namespace {
  void writeQuoted(std::ostream& out, std::string const& text) { out << '"' << text << '"'; }
}  // namespace

void TruthBranchTracksterAssociationJsonDumper::analyze(edm::Event const& event, edm::EventSetup const&) {
  std::ofstream out(jsonFile_);
  if (!out) {
    edm::LogWarning("TruthAssocJson") << "cannot write " << jsonFile_;
    return;
  }

  out << "{\n  \"run\": " << event.id().run() << ",\n  \"lumi\": " << event.luminosityBlock()
      << ",\n  \"event\": " << event.id().event() << ",\n  \"workingPoints\": [";
  for (std::size_t i = 0; i < workingPointNames_.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    writeQuoted(out, workingPointNames_[i]);
  }
  out << "],\n  \"recoObjects\": [\n";

  bool firstObject = true;
  for (auto const& collection : collections_) {
    edm::Handle<std::vector<ticl::Trackster>> tracksters;
    event.getByToken(collection.tracksters, tracksters);
    if (!tracksters.isValid()) {
      continue;
    }

    std::vector<AssociationMap const*> maps;
    for (auto const& token : collection.maps) {
      edm::Handle<AssociationMap> handle;
      event.getByToken(token, handle);
      maps.push_back(handle.isValid() ? handle.product() : nullptr);
    }

    for (std::size_t i = 0; i < tracksters->size(); ++i) {
      auto const& trackster = (*tracksters)[i];
      if (!firstObject) {
        out << ",\n";
      }
      firstObject = false;

      out << "    {\"id\": \"r_" << collection.key << "_" << i << "\", \"domain\": \"tracksters\", \"collection\": ";
      writeQuoted(out, collection.key);
      out << ", \"index\": " << i << ", \"rawEnergy\": " << trackster.raw_energy()
          << ", \"regressedEnergy\": " << trackster.regressed_energy()
          << ", \"eta\": " << trackster.barycenter().eta() << ", \"phi\": " << trackster.barycenter().phi()
          << ", \"nLayerClusters\": " << trackster.vertices().size() << ", \"matches\": {";

      for (std::size_t wp = 0; wp < maps.size(); ++wp) {
        if (wp != 0) {
          out << ", ";
        }
        writeQuoted(out, workingPointNames_[wp]);
        out << ": [";
        if (maps[wp] != nullptr) {
          auto const& map = maps[wp]->getMap();
          if (i < map.size()) {
            // Sorted best first by the producer, so the first entry is the match the
            // working point chose and the rest are the runners up.
            unsigned written = 0;
            for (auto const& entry : map[i]) {
              if (written >= maxMatches_) {
                break;
              }
              if (written != 0) {
                out << ", ";
              }
              out << "{\"node\": \"p" << entry.index() << "\", \"sharedEnergy\": " << entry.sharedEnergy()
                  << ", \"score\": " << entry.score() << "}";
              ++written;
            }
          }
        }
        out << "]";
      }
      out << "}}";
    }
  }

  out << "\n  ]\n}\n";
  edm::LogPrint("TruthAssocJson") << "wrote " << jsonFile_;
}

void TruthBranchTracksterAssociationJsonDumper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("jsonFile", "trackster_associations.json");
  desc.add<std::vector<edm::InputTag>>("recoCollections", {});
  desc.add<std::string>("associator", "truthBranchTracksterAssociators");
  desc.add<std::vector<std::string>>("workingPointNames", {"Fixed", "AdaptiveTight", "AdaptiveNominal"});
  desc.add<unsigned>("maxMatches", 5)->setComment("Ranked matches kept per object and working point");
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TruthBranchTracksterAssociationJsonDumper);
