#include "RecoHGCal/TICL/interface/TracksterInferenceByTransformer.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoFactory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace ticl {

  TracksterInferenceByTransformer::TracksterInferenceByTransformer(const edm::ParameterSet& conf,
                                                                   TICLONNXGlobalCache const* cache)
      : TracksterInferenceAlgoBase(conf, cache),
        inputNames_(conf.getParameter<std::vector<std::string>>("inputNames")),
        outputNames_(conf.getParameter<std::vector<std::string>>("outputNames")),
        nChannels_(conf.getParameter<int>("nChannels")),
        nLayers_(conf.getParameter<int>("nLayers")),
        gridH_(conf.getParameter<int>("gridH")),
        gridW_(conf.getParameter<int>("gridW")),
        windowU_(conf.getParameter<double>("windowU")),
        windowV_(conf.getParameter<double>("windowV")),
        minSigma_(conf.getParameter<double>("minSigma")),
        nGlobal_(conf.getParameter<int>("nGlobal")),
        nClasses_(conf.getParameter<int>("nClasses")),
        eidMinClusterEnergy_(conf.getParameter<double>("eid_min_cluster_energy")),
        doPID_(conf.getParameter<int>("doPID")),
        miniBatchSize_(conf.getUntrackedParameter<int>("miniBatchSize", 64)) {
    const std::string modelPath = conf.getParameter<std::string>("onnxModelPath");
    if (cache_ != nullptr && !modelPath.empty()) {
      onnxSession_ = cache_->getByModelPathString(modelPath);
    }
    enabled_ = (doPID_ != 0 && onnxSession_ != nullptr);
  }

  void TracksterInferenceByTransformer::buildFeatures(const Trackster& ts,
                                                      const std::vector<reco::CaloCluster>& layerClusters,
                                                      const hgcal::RecHitTools& rhtools,
                                                      float* grid,
                                                      float* globals) const {
    // --- globals (layout matches the training builder globals_from_row) ---
    const auto& bary = ts.barycenter();
    const double xb = bary.x(), yb = bary.y(), zb = bary.z();
    const double Rb = std::hypot(xb, yb);
    const double phib = std::atan2(yb, xb);
    const double eta = bary.eta();
    const double logE = std::log1p(static_cast<double>(ts.raw_energy()));
    const int nLc = static_cast<int>(ts.vertices().size());

    globals[0] = static_cast<float>(logE);
    globals[1] = static_cast<float>(logE);
    globals[2] = static_cast<float>(Rb / 300.0);
    globals[3] = static_cast<float>(zb / 500.0);
    globals[4] = static_cast<float>(eta / 3.0);
    globals[5] = static_cast<float>(std::log1p(static_cast<double>(nLc)));
    // globals[6,7,8,11] (density/event-context) are filled in runInference, which builds
    // the per-event LC tile once; the rest (9,10 shape, 12-17 timing/em/PCA) stay 0.

    // --- grid: energy-conserving Gaussian splat, barycenter (u,v) frame ---
    // Bin centers of linspace(-window, window, N+1), as in the training builder. The
    // splat sums/normalizes over the full grid to stay bit-faithful to training; at
    // no-PU and PU200 this is a small fraction of the module time (the ONNX inference
    // dominates), so a windowed approximation is not worth the fidelity cost.
    const double stepU = 2.0 * windowU_ / gridH_;
    const double stepV = 2.0 * windowV_ / gridW_;
    const double s = minSigma_;  // sharp splat: fixed sigma, matches the deployed model
    const double inv2s2 = 1.0 / (2.0 * s * s);
    const size_t hw = static_cast<size_t>(gridH_) * gridW_;
    std::vector<double> w(hw);

    for (int k = 0; k < nLc; ++k) {
      const unsigned int v = ts.vertices(k);
      const auto& cl = layerClusters[v];
      const int layer = static_cast<int>(rhtools.getLayerWithOffset(cl.hitsAndFractions()[0].first)) - 1;
      if (layer < 0 || layer >= nLayers_) {
        continue;
      }
      const double R = std::hypot(cl.x(), cl.y());
      const double phi = std::atan2(cl.y(), cl.x());
      double dphi = std::fmod(phi - phib + M_PI, 2.0 * M_PI);
      if (dphi < 0.0) {
        dphi += 2.0 * M_PI;  // match python modulo, which is always non-negative
      }
      dphi -= M_PI;
      const double u = R - Rb;
      const double vv = Rb * dphi;
      const double e = static_cast<double>(cl.energy()) / static_cast<double>(ts.vertex_multiplicity(k));

      double wsum = 0.0;
      for (int i = 0; i < gridH_; ++i) {
        const double uc = -windowU_ + stepU * (i + 0.5);
        const double du2 = (uc - u) * (uc - u);
        for (int j = 0; j < gridW_; ++j) {
          const double vc = -windowV_ + stepV * (j + 0.5);
          const double dv2 = (vc - vv) * (vc - vv);
          const double ww = std::exp(-(du2 + dv2) * inv2s2);
          w[static_cast<size_t>(i) * gridW_ + j] = ww;
          wsum += ww;
        }
      }
      if (wsum <= 0.0) {
        continue;
      }
      // channel 0: energy-weighted; channel 1: density (normalized occupancy).
      const size_t base0 = static_cast<size_t>(layer) * hw;
      const size_t base1 = (static_cast<size_t>(nLayers_) + layer) * hw;
      for (size_t idx = 0; idx < hw; ++idx) {
        const double wn = w[idx] / wsum;
        grid[base0 + idx] += static_cast<float>(e * wn);
        if (nChannels_ > 1) {
          grid[base1 + idx] += static_cast<float>(wn);
        }
      }
    }
  }

  void TracksterInferenceByTransformer::runInference(const std::vector<reco::CaloCluster>& layerClusters,
                                                     std::vector<Trackster>& tracksters,
                                                     const hgcal::RecHitTools& rhtools) const {
    if (!enabled_ || tracksters.empty()) {
      return;
    }

    // Select tracksters above the energy floor; zero their probabilities first.
    std::vector<int> indices;
    indices.reserve(tracksters.size());
    for (int i = 0; i < static_cast<int>(tracksters.size()); ++i) {
      float sumClusterEnergy = 0.f;
      for (const unsigned int& vertex : tracksters[i].vertices()) {
        sumClusterEnergy += static_cast<float>(layerClusters[vertex].energy());
        if (sumClusterEnergy >= eidMinClusterEnergy_) {
          tracksters[i].zeroProbabilities();
          indices.push_back(i);
          break;
        }
      }
    }

    const int total = static_cast<int>(indices.size());
    if (total == 0) {
      return;
    }

    // Per-event LC (eta,phi) occupancy tile for the density globals (6,7,8,11). This
    // mirrors the training dumper TracksterFeatureFlatTableProducer bit-for-bit: build
    // the tile once over all LCs (O(N_LC)), then each trackster reads a +/-2-bin window
    // around its barycenter (O(W^2)). Cheap relative to the ONNX forward pass.
    constexpr float kEtaMin = -3.3f, kDEta = 0.05f, kDPhi = 0.05f;
    constexpr int kNEta = 132;
    const int kNPhi = static_cast<int>(2.0 * M_PI / kDPhi) + 1;
    std::vector<float> tileE(static_cast<size_t>(kNEta) * kNPhi, 0.f);
    std::vector<int> tileN(static_cast<size_t>(kNEta) * kNPhi, 0);
    double evtE = 0.0;
    int evtN = 0;
    // A/B hook: TTPID_NO_DENSITY forces the density globals to zero, so the same binary
    // can serve the density-off model (leg A) matched to how it was trained.
    const bool noDensity = (std::getenv("TTPID_NO_DENSITY") != nullptr);
    auto phiBin = [&](float phi) {
      while (phi < -static_cast<float>(M_PI))
        phi += 2.f * static_cast<float>(M_PI);
      while (phi >= static_cast<float>(M_PI))
        phi -= 2.f * static_cast<float>(M_PI);
      const int ip = static_cast<int>((phi + static_cast<float>(M_PI)) / kDPhi);
      return std::min(std::max(ip, 0), kNPhi - 1);
    };
    for (auto const& lc : layerClusters) {
      const int ie = static_cast<int>((lc.eta() - kEtaMin) / kDEta);
      if (ie >= 0 && ie < kNEta) {
        tileE[static_cast<size_t>(ie) * kNPhi + phiBin(lc.phi())] += static_cast<float>(lc.energy());
        tileN[static_cast<size_t>(ie) * kNPhi + phiBin(lc.phi())] += 1;
      }
      evtE += lc.energy();
      ++evtN;
    }

    const int mb = std::max(1, miniBatchSize_);
    const size_t gridSize = static_cast<size_t>(nChannels_) * nLayers_ * gridH_ * gridW_;

    TracksterInferenceAlgoBase::OrtScratch ortScratch;
    ortScratch.inputs.resize(2);
    ortScratch.input_shapes.resize(2);
    ortScratch.clearPerEvent();

    for (int start = 0; start < total; start += mb) {
      const int n = std::min(mb, total - start);

      ortScratch.input_shapes[0] = {n, nChannels_, nLayers_, gridH_, gridW_};
      ortScratch.input_shapes[1] = {n, nGlobal_};

      auto& gridTensor = ortScratch.inputs[0];
      auto& globTensor = ortScratch.inputs[1];
      gridTensor.assign(static_cast<size_t>(n) * gridSize, 0.f);
      globTensor.assign(static_cast<size_t>(n) * nGlobal_, 0.f);

      for (int bi = 0; bi < n; ++bi) {
        const Trackster& ts = tracksters[indices[start + bi]];
        float* gptr = globTensor.data() + static_cast<size_t>(bi) * nGlobal_;
        buildFeatures(ts, layerClusters, rhtools, gridTensor.data() + static_cast<size_t>(bi) * gridSize, gptr);

        // density globals: window sum around the barycenter, then the same log1p/5 and
        // log1p/8 normalizations as the training builder globals_from_row.
        const float phib = std::atan2(static_cast<float>(ts.barycenter().y()), static_cast<float>(ts.barycenter().x()));
        float locE = 0.f;
        int locN = 0;
        constexpr int W = 2;
        const int ie0 = static_cast<int>((ts.barycenter().eta() - kEtaMin) / kDEta);
        const int ip0 = phiBin(phib);
        for (int de = -W; de <= W; ++de) {
          const int ie = ie0 + de;
          if (ie < 0 || ie >= kNEta)
            continue;
          for (int dp = -W; dp <= W; ++dp) {
            const int ip = ((ip0 + dp) % kNPhi + kNPhi) % kNPhi;
            locE += tileE[static_cast<size_t>(ie) * kNPhi + ip];
            locN += tileN[static_cast<size_t>(ie) * kNPhi + ip];
          }
        }
        if (!noDensity) {
          gptr[6] = static_cast<float>(std::log1p(static_cast<double>(locN)) / 5.0);
          gptr[7] = static_cast<float>(std::log1p(static_cast<double>(locE)) / 5.0);
          gptr[8] = static_cast<float>(std::log1p(static_cast<double>(evtN)) / 8.0);
          gptr[11] = static_cast<float>(std::log1p(static_cast<double>(evtE)) / 8.0);
        }
      }

      ortScratch.outputs.clear();
      onnxSession_->runInto(
          inputNames_, ortScratch.inputs, ortScratch.input_shapes, outputNames_, ortScratch.outputs, {}, n);

      if (ortScratch.outputs.empty() || outputNames_.empty()) {
        continue;
      }

      // outputs[0] = adaptive logits [n, nClasses_]; softmax then map 6 -> 8 ParticleType.
      float* logits = ortScratch.outputs[0].data();
      for (int bi = 0; bi < n; ++bi, logits += nClasses_) {
        double mx = logits[0];
        for (int c = 1; c < nClasses_; ++c) {
          mx = std::max(mx, static_cast<double>(logits[c]));
        }
        double norm = 0.0;
        std::vector<double> p(nClasses_);
        for (int c = 0; c < nClasses_; ++c) {
          p[c] = std::exp(static_cast<double>(logits[c]) - mx);
          norm += p[c];
        }
        for (int c = 0; c < nClasses_; ++c) {
          p[c] /= norm;
        }

        // adaptive classes: 0 em, 1 mip, 2 hadronic, 3 merged_em, 4 merged_hadron, 5 fake.
        // ParticleType: 0 photon, 1 electron, 2 muon, 3 neutral_pion, 4 charged_hadron,
        //               5 neutral_hadron, 6 ambiguous, 7 unknown.
        // EM fraction (photon+electron) drives isHadronic(); keep that mapping intact.
        std::array<float, 8> probs8{};
        probs8[0] = static_cast<float>(p[0] + p[3]);  // photon  <- em + merged_em
        probs8[2] = static_cast<float>(p[1]);         // muon    <- mip
        probs8[5] = static_cast<float>(p[2] + p[4]);  // neutral_hadron <- hadronic + merged_hadron
        probs8[7] = static_cast<float>(p[5]);         // unknown <- fake

        tracksters[indices[start + bi]].setProbabilities(probs8.data());
      }
    }
  }

  void TracksterInferenceByTransformer::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    TracksterInferenceAlgoBase::fillPSetDescription(iDesc);

    iDesc.add<std::string>("onnxModelPath", "RecoHGCal/TICL/data/ticlv5/onnx_models/Transformer/ttpid_model.onnx")
        ->setComment("Path to the trackster-transformer ONNX PID model. If empty, PID is skipped.");

    iDesc.add<std::vector<std::string>>("inputNames", {"grid", "globals"});
    iDesc.add<std::vector<std::string>>("outputNames", {"logits_adaptive"});

    // Grid geometry, locked to the exported model (see ttpid_meta.json).
    iDesc.add<int>("nChannels", 2);
    iDesc.add<int>("nLayers", 48);
    iDesc.add<int>("gridH", 12);
    iDesc.add<int>("gridW", 12);
    iDesc.add<double>("windowU", 12.0);
    iDesc.add<double>("windowV", 12.0);
    iDesc.add<double>("minSigma", 0.5);
    iDesc.add<int>("nGlobal", 19);
    iDesc.add<int>("nClasses", 6);

    iDesc.add<double>("eid_min_cluster_energy", 1.0);
    iDesc.add<int>("doPID", 1);

    iDesc.addUntracked<int>("miniBatchSize", 64)
        ->setComment("Mini-batch size for inference to limit peak memory usage.");
  }

}  // namespace ticl
