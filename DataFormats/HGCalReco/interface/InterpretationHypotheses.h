#ifndef DataFormats_HGCalReco_InterpretationHypotheses_h
#define DataFormats_HGCalReco_InterpretationHypotheses_h

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "DataFormats/Math/interface/Vector3D.h"

namespace ticl {
  namespace io_v1 {
    // The competing hypotheses of the TICLv6 interpretation stage and their conflicts, the input of its solver. The
    // solver accepts an independent set of the conflict graph that maximises the sum of solverWeight (greedy above its
    // exact component size). With these inputs an analysis can solve the event again with other weights or another
    // method.
    class InterpretationHypotheses {
    public:
      enum class Type : std::uint8_t {
        Muon,
        Electron,
        Photon,
        ChargedHadron,
        NeutralHadron,
        Jet,
        RecoveryChargedHadron
      };
      typedef math::XYZVectorF Vector;

      InterpretationHypotheses() = default;

      // Adds a hypothesis and returns its index. tracks are indices into the track collection of the interpretation
      // stage; a footprint of zero raw energy means no footprint.
      unsigned int add(Type type,
                       std::span<const int> tracks,
                       float weight,
                       bool accepted,
                       float rawEnergy,
                       float rawEmEnergy,
                       const Vector &barycenter) {
        type_.push_back(static_cast<std::uint8_t>(type));
        if (trackOffsets_.empty())
          trackOffsets_.push_back(0);
        tracks_.insert(tracks_.end(), tracks.begin(), tracks.end());
        trackOffsets_.push_back(static_cast<std::uint32_t>(tracks_.size()));
        weight_.push_back(weight);
        accepted_.push_back(accepted ? 1 : 0);
        rawEnergy_.push_back(rawEnergy);
        rawEmEnergy_.push_back(rawEmEnergy);
        x_.push_back(barycenter.x());
        y_.push_back(barycenter.y());
        z_.push_back(barycenter.z());
        return static_cast<unsigned int>(type_.size() - 1);
      }
      // Adds the conflict between hypotheses i and j, i < j.
      void addConflict(unsigned int i, unsigned int j) {
        conflictFirst_.push_back(i);
        conflictSecond_.push_back(j);
      }
      void setTrackBonus(float bonus) { trackBonus_ = bonus; }

      unsigned int size() const { return type_.size(); }
      Type type(unsigned int i) const { return static_cast<Type>(type_[i]); }
      std::span<const int> tracks(unsigned int i) const {
        return {tracks_.data() + trackOffsets_[i], tracks_.data() + trackOffsets_[i + 1]};
      }
      // Calibrated logit of the hypothesis model; the fixed weight of a track-only hypothesis; -infinity: vetoed.
      float weight(unsigned int i) const { return weight_[i]; }
      // The weight that the solver compares: weight plus trackBonus per track.
      float solverWeight(unsigned int i) const {
        return weight_[i] + trackBonus_ * static_cast<float>(trackOffsets_[i + 1] - trackOffsets_[i]);
      }
      float trackBonus() const { return trackBonus_; }
      bool accepted(unsigned int i) const { return accepted_[i] != 0; }
      float rawEnergy(unsigned int i) const { return rawEnergy_[i]; }
      float rawEmEnergy(unsigned int i) const { return rawEmEnergy_[i]; }
      Vector barycenter(unsigned int i) const { return Vector(x_[i], y_[i], z_[i]); }

      unsigned int nConflicts() const { return conflictFirst_.size(); }
      std::pair<unsigned int, unsigned int> conflict(unsigned int k) const {
        return {conflictFirst_[k], conflictSecond_[k]};
      }

    private:
      std::vector<std::uint8_t> type_;
      std::vector<std::uint32_t> trackOffsets_;
      std::vector<int> tracks_;
      std::vector<float> weight_;
      std::vector<std::uint8_t> accepted_;
      std::vector<float> rawEnergy_;
      std::vector<float> rawEmEnergy_;
      std::vector<float> x_;
      std::vector<float> y_;
      std::vector<float> z_;
      std::vector<std::uint32_t> conflictFirst_;
      std::vector<std::uint32_t> conflictSecond_;
      float trackBonus_ = 0.f;
    };
  }  // namespace io_v1
  using InterpretationHypotheses = io_v1::InterpretationHypotheses;
}  // namespace ticl
#endif
