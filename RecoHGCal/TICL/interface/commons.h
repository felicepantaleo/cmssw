#ifndef RecoHGCal_TICL_interface_commons_h
#define RecoHGCal_TICL_interface_commons_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace ticl {

  inline Trackster::ParticleType tracksterParticleTypeFromPdgId(int pdgId, int charge) {
    if (pdgId == 111) {
      return Trackster::ParticleType::neutral_pion;
    } else {
      pdgId = std::abs(pdgId);
      if (pdgId == 22) {
        return Trackster::ParticleType::photon;
      } else if (pdgId == 11) {
        return Trackster::ParticleType::electron;
      } else if (pdgId == 13) {
        return Trackster::ParticleType::muon;
      } else {
        bool isHadron = (pdgId > 100 and pdgId < 900) or (pdgId > 1000 and pdgId < 9000);
        if (isHadron) {
          if (charge != 0) {
            return Trackster::ParticleType::charged_hadron;
          } else {
            return Trackster::ParticleType::neutral_hadron;
          }
        } else {
          return Trackster::ParticleType::unknown;
        }
      }
    }
  }

  static void addTrackster(
      const int& index,
      const std::vector<std::pair<edm::Ref<reco::CaloClusterCollection>, std::pair<float, float>>>& lcVec,
      const std::vector<float>& inputClusterMask,
      const float& fractionCut_,
      const float& energy,
      const int& pdgId,
      const int& charge,
      const edm::ProductID& seed,
      const Trackster::IterationIndex iter,
      std::vector<float>& output_mask,
      std::vector<Trackster>& result) {
    if (lcVec.empty())
      return;

    Trackster tmpTrackster;
    tmpTrackster.zeroProbabilities();
    tmpTrackster.vertices().reserve(lcVec.size());
    tmpTrackster.vertex_multiplicity().reserve(lcVec.size());
    for (auto const& [lc, energyScorePair] : lcVec) {
      if (inputClusterMask[lc.index()] > 0) {
        double fraction = energyScorePair.first / lc->energy();
        if (fraction < fractionCut_)
          continue;
        tmpTrackster.vertices().push_back(lc.index());
        output_mask[lc.index()] -= fraction;
        tmpTrackster.vertex_multiplicity().push_back(1. / fraction);
      }
    }

    tmpTrackster.setIdProbability(tracksterParticleTypeFromPdgId(pdgId, charge), 1.f);
    tmpTrackster.setRegressedEnergy(energy);
    tmpTrackster.setIteration(iter);
    tmpTrackster.setSeed(seed, index);
    result.emplace_back(tmpTrackster);
  }

  enum LayerType {

    CE_E_120 = 0,
    CE_E_200 = 1,
    CE_E_300 = 2,
    CE_H_120_F = 3,
    CE_H_200_F = 4,
    CE_H_300_F = 5,
    CE_H_120_C = 6,
    CE_H_200_C = 7,
    CE_H_300_C = 8,
    CE_H_SCINT_C = 9,
    EnumSize = 10

  };

  inline int returnIndex(DetId& lc_seed, const hgcal::RecHitTools& rhtools_) {
    auto layer_number = rhtools_.getLayerWithOffset(lc_seed);
    auto thickness = rhtools_.getSiThickIndex(lc_seed);
    auto isEELayer = (layer_number <= rhtools_.lastLayerEE(false));
    auto isScintillator = rhtools_.isScintillator(lc_seed);
    auto isFine = (layer_number <= rhtools_.lastLayerEE(false) + 7);

    if (isEELayer) {
      if (thickness == 0) {
        return CE_E_120;
      } else if (thickness == 1) {
        return CE_E_200;
      } else if (thickness == 2) {
        return CE_E_300;
      }
    } else if (!isEELayer) {
      if (isScintillator) {
        return CE_H_SCINT_C;
      } else {
        if (isFine) {
          if (thickness == 0) {
            return CE_H_120_F;
          } else if (thickness == 1) {
            return CE_H_200_F;
          } else if (thickness == 2) {
            return CE_H_300_F;
          }
        } else {
          if (thickness == 0) {
            return CE_H_120_C;
          } else if (thickness == 1) {
            return CE_H_200_C;
          }
            else if (thickness == 2) {
              return CE_H_300_C;
          }
        }
      }
    }
    return -1;
  };

}  // namespace ticl

#endif
