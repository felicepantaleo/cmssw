#ifndef SimDataFormats_Associations_TICLAssociationMap_h
#define SimDataFormats_Associations_TICLAssociationMap_h
#include <vector>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Framework/interface/Event.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Framework/interface/Event.h"
namespace ticl {

// Define the possible map types
using mapWithFraction = std::vector< std::vector< std::pair<unsigned int, float> > >;
using mapWithFractionAndScore = std::vector< std::vector< std::pair<unsigned int, std::pair<float, float> > > >;
using oneToOneMapWithFraction = std::vector< std::pair<unsigned int, float> >;
using oneToOneMapWithFractionAndScore = std::vector< std::pair<unsigned int, std::pair<float, float> > >;

template<typename MapType, typename Collection1, typename Collection2>
class AssociationMap {
public:
    AssociationMap()
        : collectionIDs(edm::ProductID(), edm::ProductID()) {}

    AssociationMap(const edm::ProductID& id1, const edm::ProductID& id2, const edm::Event& event)
        : collectionIDs(std::make_pair(id1, id2)) {
        resize(event);
    }

    MapType& getMap() {
        return map_;
    }

    const MapType& getMap() const {
        return map_;
    }

    edm::Ref<Collection1> getRefFirst(unsigned int index) const {
        return edm::Ref<Collection1>(collectionIDs.first, index);
    }

    edm::Ref<Collection2> getRefSecond(unsigned int index) const {
        return edm::Ref<Collection2>(collectionIDs.second, index);
    }

    std::pair<const edm::ProductID&, const edm::ProductID&> getCollectionIDs() const {
        return collectionIDs;
    }

    void insert(unsigned int index1, unsigned int index2, float fraction, float score = 0.0f) {
        if constexpr (std::is_same<MapType, mapWithFraction>::value) {
            if (index1 >= map_.size()) {
                map_.resize(index1 + 1);
            }
            auto& vec = map_[index1];
            auto it = std::find_if(vec.begin(), vec.end(), [&](const auto& pair) {
                return pair.first == index2;
            });
            if (it != vec.end()) {
                it->second += fraction;
            } else {
                vec.emplace_back(index2, fraction);
            }
        } else if constexpr (std::is_same<MapType, mapWithFractionAndScore>::value) {
            if (index1 >= map_.size()) {
                map_.resize(index1 + 1);
            }
            auto& vec = map_[index1];
            auto it = std::find_if(vec.begin(), vec.end(), [&](const auto& pair) {
                return pair.first == index2;
            });
            if (it != vec.end()) {
                it->second.first += fraction;
                it->second.second += score;
            } else {
                vec.emplace_back(index2, std::make_pair(fraction, score));
            }
        } else if constexpr (std::is_same<MapType, oneToOneMapWithFraction>::value) {
            auto it = std::find_if(map_.begin(), map_.end(), [&](const auto& pair) {
                return pair.first == index1;
            });
            if (it != map_.end()) {
                it->second += fraction;
            } else {
                map_.emplace_back(index1, fraction);
            }
        } else if constexpr (std::is_same<MapType, oneToOneMapWithFractionAndScore>::value) {
            auto it = std::find_if(map_.begin(), map_.end(), [&](const auto& pair) {
                return pair.first == index1;
            });
            if (it != map_.end()) {
                it->second.first += fraction;
                it->second.second += score;
            } else {
                map_.emplace_back(index1, std::make_pair(fraction, score));
            }
        }
    }

    void insert(const edm::Ref<Collection1>& ref1, const edm::Ref<Collection2>& ref2, float fraction, float score = 0.0f) {
        insert(ref1.key(), ref2.key(), fraction, score);
    }

    void sort(bool byScore = false) {
        static_assert(!std::is_same_v<MapType, oneToOneMapWithFraction> && 
                      !std::is_same_v<MapType, oneToOneMapWithFractionAndScore>, 
                      "Sort is not applicable for one-to-one maps");
                      
        if constexpr (std::is_same_v<MapType, mapWithFraction>) {
            for (auto& vec : map_) {
                std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
                    return a.second > b.second;
                });
            }
        } else if constexpr (std::is_same_v<MapType, mapWithFractionAndScore>) {
            for (auto& vec : map_) {
                if (byScore) {
                    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
                        return a.second.second > b.second.second;
                    });
                } else {
                    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
                        return a.second.first > b.second.first;
                    });
                }
            }
        }
    }

    auto& operator[](unsigned int index1) {
        if constexpr (std::is_same_v<MapType, mapWithFraction> || std::is_same_v<MapType, mapWithFractionAndScore> ){
            if (index1 >= map_.size()) {
                throw std::out_of_range("Index out of range");
            }
            return map_[index1];
        } else if constexpr (std::is_same_v<MapType, oneToOneMapWithFraction> || std::is_same_v<MapType, oneToOneMapWithFractionAndScore>) {
            auto it = std::find_if(map_.begin(), map_.end(), [&](const auto& pair) {
                return pair.first == index1;
            });
            if (it == map_.end()) {
                throw std::out_of_range("Index not found");
            }
            return *it;
        }
    }

    const auto& operator[](unsigned int index1) const {
        if constexpr (std::is_same_v<MapType, mapWithFraction> || std::is_same_v<MapType, mapWithFractionAndScore>) {
            if (index1 >= map_.size()) {
                throw std::out_of_range("Index out of range");
            }
            return map_[index1];
        } else if constexpr (std::is_same_v<MapType, oneToOneMapWithFraction> || std::is_same_v<MapType, oneToOneMapWithFractionAndScore>) {
            auto it = std::find_if(map_.begin(), map_.end(), [&](const auto& pair) {
                return pair.first == index1;
            });
            if (it == map_.end()) {
                throw std::out_of_range("Index not found");
            }
            return *it;
        }
    }

private:
    void resize(const edm::Event& event) {
        edm::Handle<Collection1> handle1;
        event.get(collectionIDs.first, handle1);
        edm::Handle<Collection2> handle2;
        event.get(collectionIDs.second, handle2);
        map_.resize(handle1->size());
    }

    std::pair<edm::ProductID, edm::ProductID> collectionIDs;
    MapType map_;  // Store the map directly
};

}  // namespace ticl

#endif