#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>

#include "DataFormats/HGCalReco/interface/TracksterHost.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

class TracksterEnergyComparisonAnalyzer : public edm::stream::EDAnalyzer<> {
public:
  explicit TracksterEnergyComparisonAnalyzer(edm::ParameterSet const& config)
      : referenceToken_{consumes(config.getParameter<edm::InputTag>("reference"))},
        candidateToken_{consumes(config.getParameter<edm::InputTag>("candidate"))},
        absoluteTolerance_{config.getParameter<double>("absoluteTolerance")},
        relativeTolerance_{config.getParameter<double>("relativeTolerance")},
        maxMismatchesToPrint_{config.getParameter<unsigned int>("maxMismatchesToPrint")},
        failOnMismatch_{config.getParameter<bool>("failOnMismatch")},
        reportEveryEvent_{config.getParameter<bool>("reportEveryEvent")} {
    if (absoluteTolerance_ < 0. || relativeTolerance_ < 0.) {
      throw cms::Exception("TracksterEnergyComparison")
          << "absoluteTolerance and relativeTolerance must be non-negative.";
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription description;
    description.add<edm::InputTag>("reference", edm::InputTag("ticlTrackstersToSoAProducer"));
    description.add<edm::InputTag>("candidate", edm::InputTag("ticlTracksterInferenceProducer"));
    description.add<double>("absoluteTolerance", 1.e-5);
    description.add<double>("relativeTolerance", 1.e-4);
    description.add<unsigned int>("maxMismatchesToPrint", 10);
    description.add<bool>("failOnMismatch", false);
    description.add<bool>("reportEveryEvent", true);
    descriptions.addWithDefaultLabel(description);
  }

private:
  void analyze(edm::Event const& event, edm::EventSetup const&) override {
    auto const& reference = event.get(referenceToken_);
    auto const& candidate = event.get(candidateToken_);

    auto const referenceView = reference.const_view();
    auto const candidateView = candidate.const_view();
    auto const nReference = static_cast<std::size_t>(referenceView.tracksters().metadata().size());
    auto const nCandidate = static_cast<std::size_t>(candidateView.tracksters().metadata().size());
    if (nReference != nCandidate) {
      throw cms::Exception("TracksterEnergyComparison")
          << "The collections cannot be compared row-by-row: reference contains " << nReference
          << " Tracksters, while candidate contains " << nCandidate << '.';
    }

    double sumAbsoluteDifference = 0.;
    double sumSquaredDifference = 0.;
    double maxAbsoluteDifference = 0.;
    double maxRelativeDifference = 0.;
    std::size_t mismatchCount = 0;
    std::size_t printedMismatchCount = 0;
    std::ostringstream mismatchReport;
    mismatchReport << std::setprecision(9);

    for (std::size_t tracksterIndex = 0; tracksterIndex < nReference; ++tracksterIndex) {
      auto const expected = static_cast<double>(referenceView.tracksters()[tracksterIndex].regressed_energy());
      auto const observed = static_cast<double>(candidateView.tracksters()[tracksterIndex].regressed_energy());
      auto const absoluteDifference = std::abs(observed - expected);
      auto const relativeDifference =
          absoluteDifference / std::max(std::abs(expected), std::numeric_limits<double>::epsilon());
      sumAbsoluteDifference += absoluteDifference;
      sumSquaredDifference += absoluteDifference * absoluteDifference;
      maxAbsoluteDifference = std::max(maxAbsoluteDifference, absoluteDifference);
      maxRelativeDifference = std::max(maxRelativeDifference, relativeDifference);

      auto const finite = std::isfinite(expected) && std::isfinite(observed);
      auto const tolerance =
          absoluteTolerance_ + relativeTolerance_ * std::max(std::abs(expected), std::abs(observed));
      if (!finite || absoluteDifference > tolerance) {
        ++mismatchCount;
        if (printedMismatchCount < maxMismatchesToPrint_) {
          mismatchReport << "\n  trackster=" << tracksterIndex << " reference=" << expected << " candidate=" << observed
                         << " absDiff=" << absoluteDifference << " tolerance=" << tolerance;
          ++printedMismatchCount;
        }
      }
    }

    if (reportEveryEvent_ || mismatchCount != 0) {
      auto const denominator = static_cast<double>(nReference);
      auto const meanAbsoluteDifference = nReference == 0 ? 0. : sumAbsoluteDifference / denominator;
      auto const rootMeanSquaredDifference = nReference == 0 ? 0. : std::sqrt(sumSquaredDifference / denominator);
      std::ostringstream report;
      report << std::setprecision(7) << "Compared regressed energy for " << nReference << " Tracksters: "
             << mismatchCount << " values outside tolerance [atol=" << absoluteTolerance_
             << ", rtol=" << relativeTolerance_ << "]."
             << "\n  meanAbs=" << meanAbsoluteDifference << ", rms=" << rootMeanSquaredDifference
             << ", maxAbs=" << maxAbsoluteDifference << ", maxRel=" << maxRelativeDifference
             << mismatchReport.str();
      if (mismatchCount == 0) {
        edm::LogPrint("TracksterEnergyComparison") << report.str();
      } else {
        edm::LogWarning("TracksterEnergyComparison") << report.str();
      }
    }

    if (failOnMismatch_ && mismatchCount != 0) {
      throw cms::Exception("TracksterEnergyMismatch")
          << mismatchCount << " Trackster regressed energies differ beyond the configured tolerances.";
    }
  }

  edm::EDGetTokenT<::ticl::TracksterHost> const referenceToken_;
  edm::EDGetTokenT<::ticl::TracksterHost> const candidateToken_;
  double const absoluteTolerance_;
  double const relativeTolerance_;
  unsigned int const maxMismatchesToPrint_;
  bool const failOnMismatch_;
  bool const reportEveryEvent_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TracksterEnergyComparisonAnalyzer);
