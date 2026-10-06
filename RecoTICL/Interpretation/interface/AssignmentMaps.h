#ifndef RecoTICL_Interpretation_AssignmentMaps_h
#define RecoTICL_Interpretation_AssignmentMaps_h

namespace ticl {

  // Values of the trackMode map of TICLInterpretationProducer: what the accepted hypothesis of each track is.
  enum class TrackMode : int {
    kNotSelected = -1,
    kMuon = 1,
    kChargedHadron = 2,
    kElectron = 3,
    kJetMember = 4,
    kRecovery = 5
  };

}  // namespace ticl

#endif
