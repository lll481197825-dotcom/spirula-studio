#pragma once

#include "sfm/core/Model.h"
#include <map>
#include <set>
#include <stdexcept>

namespace sfm::regional::raw_lineage {

using FeatureKey = uint64_t;
inline FeatureKey featureKey(const TrackElement& row) {
    return (uint64_t(row.image_id) << 32) | row.point2D_idx;
}

struct Multiset {
    std::map<FeatureKey, uint64_t> required;
    uint64_t observations = 0;
};

// A repeated source row is a separate observation, even when its feature key
// and current canonical point are the same as those of another source row.
inline Multiset originalMultiset(const Reconstruction& original, const Reconstruction& current) {
    Multiset result;
    for (const auto& item : original.images) if (item.second.registered) {
        const auto at = current.images.find(item.first);
        if (at == current.images.end() || !at->second.registered ||
            at->second.points2D.size() != item.second.points2D.size() ||
            item.second.id != item.first || at->second.id != item.first ||
            item.second.name != at->second.name || item.second.camera_id != at->second.camera_id)
            throw std::runtime_error("raw lineage lost original image/feature identity");
    }
    for (const auto& item : original.points3D) for (const auto& row : item.second.track) {
        const auto& before = original.images.at(row.image_id);
        const auto at = current.images.find(row.image_id);
        if (!before.registered || at == current.images.end() || !at->second.registered ||
            row.point2D_idx >= before.points2D.size() || row.point2D_idx >= at->second.points2D.size() ||
            row.point2D_idx >= at->second.point3D_ids.size())
            throw std::runtime_error("raw lineage lost original observation feature");
        const auto xy = before.points2D.at(row.point2D_idx), actual = at->second.points2D.at(row.point2D_idx);
        if (xy.x != actual.x || xy.y != actual.y || !std::isfinite(xy.x) || !std::isfinite(xy.y))
            throw std::runtime_error("raw lineage changed original observation coordinate");
        if (result.observations == UINT64_MAX || result.required[featureKey(row)] == UINT64_MAX)
            throw std::overflow_error("raw lineage observation count overflow");
        ++result.observations; ++result.required[featureKey(row)];
    }
    return result;
}

inline std::map<FeatureKey, uint64_t> currentMultiplicities(const Multiset& original, const Reconstruction& current) {
    std::set<uint64_t> assigned;
    for (const auto& item : original.required) {
        const auto& image = current.images.at(uint32_t(item.first >> 32));
        const uint32_t feature = uint32_t(item.first);
        const auto point = image.point3D_ids.at(feature);
        if (point == kInvalidPoint3D || !current.points3D.count(point))
            throw std::runtime_error("raw lineage original observation became unassigned");
        assigned.insert(point);
    }
    std::map<FeatureKey, uint64_t> result;
    for (const auto point : assigned) for (const auto& row : current.points3D.at(point).track) {
        const auto key = featureKey(row);
        if (!original.required.count(key)) continue;
        const auto& image = current.images.at(row.image_id);
        if (row.point2D_idx >= image.point3D_ids.size() || image.point3D_ids.at(row.point2D_idx) != point)
            throw std::runtime_error("raw lineage current track/feature association inconsistent");
        ++result[key];
    }
    return result;
}

inline uint64_t verify(const Reconstruction& original, const Reconstruction& current) {
    const auto lineage = originalMultiset(original, current);
    const auto actual = currentMultiplicities(lineage, current);
    for (const auto& item : lineage.required) {
        const auto at = actual.find(item.first);
        if (at == actual.end() || at->second < item.second)
            throw std::runtime_error("raw lineage lost original duplicate observation multiplicity: image="+
                std::to_string(uint32_t(item.first>>32))+" feature="+std::to_string(uint32_t(item.first))+
                " required="+std::to_string(item.second)+" actual="+std::to_string(at==actual.end()?0:at->second));
    }
    return lineage.observations;
}

// Only restores the missing original row multiplicity at its already assigned
// canonical point. It changes neither XYZ nor image pose nor feature identity.
inline uint64_t preserveAssignedMultiplicity(const Reconstruction& original, Reconstruction& current) {
    const auto lineage = originalMultiset(original, current);
    const auto actual = currentMultiplicities(lineage, current);
    for (const auto& item : lineage.required) {
        const auto at = actual.find(item.first);
        const auto have = at == actual.end() ? uint64_t(0) : at->second;
        auto& image = current.images.at(uint32_t(item.first >> 32));
        auto& point = current.points3D.at(image.point3D_ids.at(uint32_t(item.first)));
        for (uint64_t row = have; row < item.second; ++row)
            point.track.push_back({image.id, uint32_t(item.first)});
    }
    return verify(original, current);
}

} // namespace sfm::regional::raw_lineage
