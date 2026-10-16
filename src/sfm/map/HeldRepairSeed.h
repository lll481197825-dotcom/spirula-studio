#pragma once
#include "sfm/core/Model.h"
#include <cmath>
#include <cstdint>
#include <map>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace sfm {

// The borrowed source outlives a continuation. Its point IDs are authoritative:
// this contract protects structure and raw associations as well as camera poses.
class HeldRepairSeedContract {
public:
    explicit HeldRepairSeedContract(const Reconstruction& seed) : seed_(&seed) {
        if (!seed.rigs.empty() || !seed.rig_detached.empty())
            throw std::runtime_error("held repair seed requires an explicit rig geometry contract");
        for (const auto& point : seed.points3D) {
            if (point.first == kInvalidPoint3D || !finite(point.second.xyz) ||
                point.first >= seed.next_point3D_id || point.second.track.empty())
                throw std::runtime_error("held repair seed point identity or geometry invalid");
            points_.insert(point.first);
            for (const auto& row : point.second.track) {
                const auto at = seed.images.find(row.image_id);
                if (at == seed.images.end() || !at->second.registered ||
                    row.point2D_idx >= at->second.points2D.size() ||
                    row.point2D_idx >= at->second.point3D_ids.size() ||
                    at->second.point3D_ids[row.point2D_idx] != point.first)
                    throw std::runtime_error("held repair seed raw association invalid");
            }
        }
        if (points_.empty()) throw std::runtime_error("held repair seed has no fixed visual geometry");
        for (const auto& image : seed.images) if (image.second.registered) {
            if (image.second.id != image.first || !seed.cameras.count(image.second.camera_id) ||
                image.second.points2D.size() != image.second.point3D_ids.size())
                throw std::runtime_error("held repair seed image measurement identity invalid");
            for (uint64_t point : image.second.point3D_ids)
                if (point != kInvalidPoint3D && !contains(point))
                    throw std::runtime_error("held repair seed feature references a missing point");
        }
        requireUnchanged(seed);
    }
    const std::set<uint64_t>& points() const { return points_; }
    bool contains(uint64_t point) const { return points_.count(point) != 0; }
    bool originalObservation(uint64_t point, TrackElement row) const {
        const auto image = seed_->images.find(row.image_id);
        return point != kInvalidPoint3D && image != seed_->images.end() &&
            row.point2D_idx < image->second.point3D_ids.size() &&
            image->second.point3D_ids[row.point2D_idx] == point;
    }
    static bool sameCamera(const Camera& a, const Camera& b) {
        return a.id == b.id && a.model == b.model && a.width == b.width && a.height == b.height &&
            a.fx == b.fx && a.fy == b.fy && a.cx == b.cx && a.cy == b.cy &&
            a.k1 == b.k1 && a.k2 == b.k2 && a.p1 == b.p1 && a.p2 == b.p2 &&
            a.k3 == b.k3 && a.k4 == b.k4 && a.k5 == b.k5 && a.k6 == b.k6 &&
            a.sx1 == b.sx1 && a.sy1 == b.sy1 && a.pixel_scale == b.pixel_scale;
    }
    void requireUnchanged(const Reconstruction& candidate, const char* stage = "held.unspecified") const {
        requireMapped(candidate,[](uint32_t image) { return image; },
            [](uint32_t,uint32_t feature) { return feature; },
            [&](uint32_t image) { return seed_->images.at(image).points2D.size(); },stage);
    }
    template<class FeatureCount>
    void requireRemappedUnchanged(const Reconstruction& candidate, const std::vector<uint32_t>& global,
                                 const std::vector<std::vector<uint32_t>>& original, FeatureCount count,
                                 const char* stage = "held.remapped") const {
        requireMapped(candidate,[&](uint32_t image) { return global.at(image); },
            [&](uint32_t image,uint32_t feature) { return original.at(image).at(feature); },
            [&](uint32_t image) { return size_t(count(global.at(image))); },stage);
    }
private:
    template<class ImageMap,class FeatureMap,class FeatureCount>
    void requireMapped(const Reconstruction& candidate, ImageMap imageMap, FeatureMap featureMap,
                       FeatureCount featureCount, const char* stage) const {
        for (const auto& image : seed_->images) if (image.second.registered) {
            const auto globalImage = imageMap(image.first);
            const auto found = candidate.images.find(globalImage);
            if (found == candidate.images.end() || !found->second.registered ||
                found->second.id != globalImage || found->second.camera_id != image.second.camera_id ||
                found->second.name != image.second.name ||
                found->second.pose.R != image.second.pose.R || !sameXYZ(found->second.pose.t,image.second.pose.t) ||
                found->second.exif_orientation != image.second.exif_orientation ||
                found->second.points2D.size() != featureCount(image.first) ||
                found->second.point3D_ids.size() != featureCount(image.first))
                throw std::runtime_error("held repair changed seed image frame or feature identity");
            const auto camera = candidate.cameras.find(image.second.camera_id);
            if (camera == candidate.cameras.end() ||
                !sameCamera(camera->second,seed_->cameras.at(image.second.camera_id)))
                throw std::runtime_error("held repair changed seed camera calibration");
            for (size_t feature = 0; feature < image.second.points2D.size(); ++feature) {
                const auto mapped = featureMap(image.first,uint32_t(feature));
                const auto before = image.second.points2D.at(feature), after = found->second.points2D.at(mapped);
                const auto assignment = image.second.point3D_ids.at(feature);
                if (before.x != after.x || before.y != after.y ||
                    (assignment != kInvalidPoint3D && found->second.point3D_ids.at(mapped) != assignment)) {
                    std::ostringstream detail; detail << std::setprecision(17)
                        << "held repair changed a seed feature measurement or original assignment; stage=" << stage
                        << "; image=" << globalImage << "; source_image=" << image.first
                        << "; feature=" << mapped << "; source_feature=" << feature
                        << "; before_xy=" << before.x << ',' << before.y << "; after_xy=" << after.x << ',' << after.y
                        << "; before_point=" << assignment << "; after_point=" << found->second.point3D_ids.at(mapped);
                    throw std::runtime_error(detail.str());
                }
            }
        }
        for (const auto& point : seed_->points3D) {
            const auto found = candidate.points3D.find(point.first);
            if (found == candidate.points3D.end() || !sameXYZ(found->second.xyz,point.second.xyz))
                throw std::runtime_error("held repair moved, removed, or renumbered a seed point");
            std::map<uint64_t,uint64_t> actual;
            for (const auto& row : found->second.track)
                ++actual[(uint64_t(row.image_id)<<32)|row.point2D_idx];
            for (const auto& row : point.second.track) {
                const auto& expected = seed_->images.at(row.image_id);
                const uint32_t globalImage = imageMap(row.image_id), globalFeature = featureMap(row.image_id,row.point2D_idx);
                const auto& image = candidate.images.at(globalImage);
                const auto before = expected.points2D.at(row.point2D_idx), after = image.points2D.at(globalFeature);
                const uint64_t key = (uint64_t(globalImage)<<32)|globalFeature;
                if (before.x != after.x || before.y != after.y ||
                    image.point3D_ids.at(globalFeature) != point.first || !actual[key])
                    throw std::runtime_error("held repair changed original seed raw coordinate or point-feature association");
                --actual[key];
            }
        }
        if (candidate.next_point3D_id < seed_->next_point3D_id)
            throw std::runtime_error("held repair reused the reserved seed point ID space");
    }
    static bool finite(Vec3 p) { return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z); }
    static bool sameXYZ(Vec3 a,Vec3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
    const Reconstruction* seed_;
    std::set<uint64_t> points_;
};
}
