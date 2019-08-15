// Author: Felice Pantaleo,Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 11/2018

#ifndef __RecoHGCal_TICL_HGCDoublet_H__
#define __RecoHGCal_TICL_HGCDoublet_H__

#include <cmath>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

class HGCDoublet {
public:
  using HGCntuplet = std::vector<unsigned int>;

  HGCDoublet(const int innerClusterId,
             const int outerClusterId,
             const int doubletId,
             const std::vector<reco::CaloCluster> *layerClusters)
      : layerClusters_(layerClusters),
        theDoubletId_(doubletId),
        innerClusterId_(innerClusterId),
        outerClusterId_(outerClusterId),
        innerR_((*layerClusters)[innerClusterId].position().r()),
        outerR_((*layerClusters)[outerClusterId].position().r()),
        innerX_((*layerClusters)[innerClusterId].x()),
        outerX_((*layerClusters)[outerClusterId].x()),
        innerY_((*layerClusters)[innerClusterId].y()),
        outerY_((*layerClusters)[outerClusterId].y()),
        innerZ_((*layerClusters)[innerClusterId].z()),
        outerZ_((*layerClusters)[outerClusterId].z()),
        alreadyVisited_(false) {}

  float innerX() const { return innerX_; }

  float outerX() const { return outerX_; }

  float innerY() const { return innerY_; }

  float outerY() const { return outerY_; }

  float innerZ() const { return innerZ_; }

  float outerZ() const { return outerZ_; }

  float innerR() const { return innerR_; }

  float outerR() const { return outerZ_; }

  int innerClusterId() const { return innerClusterId_; }

  int outerClusterId() const { return outerClusterId_; }

  void tagAsOuterNeighbor(unsigned int otherDoublet) { outerNeighbors_.push_back(otherDoublet); }

  void tagAsInnerNeighbor(unsigned int otherDoublet) { innerNeighbors_.push_back(otherDoublet); }

  bool checkCompatibilityAndTag(
      std::vector<HGCDoublet> &, const std::vector<int> &, float, float minCosPointing = 1., bool debug = false);

  int areAligned(float xi,
                 float yi,
                 float zi,
                 float xo,
                 float yo,
                 float zo,
                 float minCosTheta,
                 float minCosPointing,
                 bool debug = false) const;

  void findNtuplets(std::vector<HGCDoublet> &, HGCntuplet &);

private:
  const std::vector<reco::CaloCluster> *layerClusters_;
  std::vector<int> outerNeighbors_;
  std::vector<int> innerNeighbors_;

  const int theDoubletId_;
  const int innerClusterId_;
  const int outerClusterId_;

  const float innerR_;
  const float outerR_;
  const float innerX_;
  const float outerX_;
  const float innerY_;
  const float outerY_;
  const float innerZ_;
  const float outerZ_;
  bool alreadyVisited_;
};

#endif /*HGCDoublet_H_ */
