#ifndef CUDADATAFORMATS_HGCCONSTANT_H
#define CUDADATAFORMATS_HGCCONSTANT_H 1

#include <vector>

class HGCConstantVectorData {
 public:
  std::vector<double> fCPerMIP;
  std::vector<double> cce;
  std::vector<double> noise_fC;
  std::vector<double> rcorr;
  std::vector<double> weights;
  std::vector<int> waferTypeL;
};

class HGCeeUncalibratedRecHitConstantData {
 public:
  double hgcEE_keV2DIGI_;
  double hgceeUncalib2GeV_;
  double *hgcEE_fCPerMIP_;  
  double *hgcEE_cce_;
  double *hgcEE_noise_fC_;
  double *rcorr_;
  double *weights_;
  int *waferTypeL_;
  float xmin_;
  float xmax_;
  float aterm_;
  float cterm_;
  uint32_t rangeMatch_;
  uint32_t rangeMask_;
  bool hgcEE_isSiFE_;
  int nbytes;
  int ndelem;
  int nfelem;
  int nuelem;
  int nielem;
  int nbelem;
  int s_hgcEE_fCPerMIP_;
  int s_hgcEE_cce_;
  int s_hgcEE_noise_fC_;
  int s_rcorr_;
  int s_weights_;
  int s_waferTypeL_;
};

class HGChefUncalibratedRecHitConstantData {
 public:
  double hgcHEF_keV2DIGI_;
  double hgchefUncalib2GeV_;
  double *hgcHEF_fCPerMIP_;
  double *hgcHEF_cce_;
  double *hgcHEF_noise_fC_;
  double *rcorr_;
  double *weights_;
  int *waferTypeL_;
  float xmin_;
  float xmax_;
  float aterm_;
  float cterm_;
  uint32_t rangeMatch_;
  uint32_t rangeMask_;
  uint32_t fhOffset_;
  bool hgcHEF_isSiFE_;
  int nbytes;
  int ndelem;
  int nfelem;
  int nuelem;
  int nielem;
  int nbelem;
  int s_hgcHEF_fCPerMIP_;
  int s_hgcHEF_cce_;
  int s_hgcHEF_noise_fC_;
  int s_rcorr_;
  int s_weights_;
  int s_waferTypeL_;
};

class HGChebUncalibratedRecHitConstantData {
 public:
  double hgcHEB_keV2DIGI_;
  double hgchebUncalib2GeV_;
  double hgcHEB_noise_MIP_;
  double *weights_;
  uint32_t rangeMatch_;
  uint32_t rangeMask_;
  uint32_t fhOffset_;
  bool hgcHEB_isSiFE_;
  int nbytes;
  int ndelem;
  int nfelem;
  int nuelem;
  int nielem;
  int nbelem;
  int s_weights_;
};

#endif //CUDADATAFORMATS_HGCCONSTANT_H
