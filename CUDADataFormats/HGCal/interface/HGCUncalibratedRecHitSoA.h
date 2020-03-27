#ifndef CUDADATAFORMATS_HGCUNCALIBRATEDRECHITSOA_H
#define CUDADATAFORMATS_HGCUNCALIBRATEDRECHITSOA_H 1

class HGCUncalibratedRecHitSoA {
public:
  float *amplitude;
  float *pedestal;
  float *jitter;
  float *chi2;
  float *OOTamplitude;
  float *OOTchi2;
  uint32_t *flags;
  uint32_t *aux;
  uint32_t *id;
  int nbytes;
};

#endif //CUDADATAFORMATS_HGCUNCAIBRATEDRECHITSOA_H
