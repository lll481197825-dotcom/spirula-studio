#pragma once

#include "sfm/map/RegionalRecovery.h"
#include "sfm/map/RegionalQuality.h"
#include "sfm/map/RegionalLandmarks.h"
#include "sfm/map/RawLineage.h"

namespace sfm::regional {

inline Reconstruction anchoredRepairSeed(const Reconstruction& donor, const Region& context,
                                         const Reconstruction& poses, double gate = 3, const PriorSource* gps = nullptr,
                                         bool retain_fixed_geometry = false) {
    Reconstruction seed; seed.cameras = poses.cameras; seed.next_point3D_id = donor.next_point3D_id;
    const std::set<uint32_t> allowed(context.images.begin(), context.images.end());
    for (const auto& kv : donor.images) if (kv.second.registered && allowed.count(kv.first)) {
        const auto at = poses.images.find(kv.first);
        if (at == poses.images.end() || !at->second.registered) continue;
        auto im = kv.second; im.pose = at->second.pose;
        im.point3D_ids.assign(im.points2D.size(), kInvalidPoint3D); seed.images.emplace(kv.first, std::move(im));
    }
    for (const auto& kv : donor.points3D) {
        auto p = kv.second; p.track.clear();
        for (const auto& e : kv.second.track) if (seed.images.count(e.image_id)) p.track.push_back(e);
        if (p.track.size() < (retain_fixed_geometry ? 1u : 2u)) continue;
        for (const auto& e : p.track) seed.images.at(e.image_id).point3D_ids.at(e.point2D_idx) = kv.first;
        seed.points3D.emplace(kv.first, std::move(p));
    }
    std::map<uint32_t, PhotoQuality> quality;
    for (const auto& kv : seed.points3D) for (const auto& e : kv.second.track) {
        const auto& im = seed.images.at(e.image_id); auto& q = quality[e.image_id]; bool valid = true;
        const auto& cam = seed.cameras.at(im.camera_id);
        const double error = reprojErrorAt(cam, im.pose, im.points2D.at(e.point2D_idx), kv.second.xyz, &valid) / cam.pixel_scale;
        ++q.observations; q.inliers += valid && error <= gate; q.invalid_projections += !valid; q.error += error;
    }
    for (auto at = seed.images.begin(); at != seed.images.end();) {
        Vec3 position;
        const bool located = !gps || (gps->position(at->first, position) && (cameraCenter(at->second.pose) - position).norm() <= 5);
        if (!located || !photoQualityAccepted(quality[at->first], gate)) at = seed.images.erase(at); else ++at;
    }
    for (auto at = seed.points3D.begin(); at != seed.points3D.end();) {
        auto& track = at->second.track;
        track.erase(std::remove_if(track.begin(), track.end(), [&](const TrackElement& e) { return !seed.images.count(e.image_id); }), track.end());
        if (track.size() < (retain_fixed_geometry ? 1u : 2u)) {
            for (const auto& e : track) seed.images.at(e.image_id).point3D_ids.at(e.point2D_idx) = kInvalidPoint3D;
            at = seed.points3D.erase(at);
        } else ++at;
    }
    return seed;
}

inline uint64_t preserveRepairObservations(const Reconstruction& original, Reconstruction& repaired, double gate) {
    uint64_t observations = 0;
    for (const auto& kv : original.images) if (kv.second.registered) {
        const auto at = repaired.images.find(kv.first);
        if (at == repaired.images.end() || !at->second.registered ||
            at->second.points2D.size() != kv.second.points2D.size())
            throw std::runtime_error("regional repair did not restore every original image/feature identity");
    }
    for (const auto& kv : original.points3D) {
        Point3D orphan = kv.second; orphan.track.clear(); std::vector<Observation> rays;
        for (const auto& e : kv.second.track) {
            const auto& before = original.images.at(e.image_id); auto& after = repaired.images.at(e.image_id);
            const auto xy = before.points2D.at(e.point2D_idx), actual = after.points2D.at(e.point2D_idx);
            if (xy.x != actual.x || xy.y != actual.y)
                throw std::runtime_error("regional repair changed an original observation coordinate");
            ++observations; const auto point = after.point3D_ids.at(e.point2D_idx);
            if (point == kInvalidPoint3D) orphan.track.push_back(e);
            else {
                const auto& p = repaired.points3D.at(point);
                if (std::none_of(p.track.begin(), p.track.end(), [&](const TrackElement& row) {
                    return row.image_id == e.image_id && row.point2D_idx == e.point2D_idx;
                })) throw std::runtime_error("regional repair observation/track association inconsistent");
                orphan.xyz = p.xyz;
            }
            rays.push_back({0, uint64_t(e.image_id) << 32, xy});
        }
        if (orphan.track.empty()) continue;
        orphan.xyz = refineUnifiedPoint(orphan.xyz, rays, repaired, gate);
        while (repaired.points3D.count(repaired.next_point3D_id) || repaired.next_point3D_id == kInvalidPoint3D)
            ++repaired.next_point3D_id;
        const auto id = repaired.next_point3D_id++;
        for (const auto& e : orphan.track) repaired.images.at(e.image_id).point3D_ids.at(e.point2D_idx) = id;
        repaired.points3D.emplace(id, std::move(orphan));
    }
    const auto verified = raw_lineage::preserveAssignedMultiplicity(original, repaired);
    if (verified != observations) throw std::runtime_error("regional repair raw lineage count changed");
    return verified;
}

inline void mergeAnchoredRepairSeed(Reconstruction& seed, const Reconstruction& extra) {
    if (seed.images.empty()) seed.cameras = extra.cameras;
    for (const auto& kv : extra.images) {
        auto at = seed.images.find(kv.first);
        if (at == seed.images.end()) {
            auto im = kv.second; im.point3D_ids.assign(im.points2D.size(), kInvalidPoint3D);
            seed.images.emplace(kv.first, std::move(im));
        } else if (at->second.camera_id != kv.second.camera_id || at->second.pose.R != kv.second.pose.R ||
                   (at->second.pose.t - kv.second.pose.t).norm() > 1e-10)
            throw std::runtime_error("regional repair donors disagree with canonical anchor pose");
    }
    for (const auto& kv : extra.points3D) {
        auto p = kv.second; p.track.clear();
        for (const auto& e : kv.second.track) if (seed.images.at(e.image_id).point3D_ids.at(e.point2D_idx) == kInvalidPoint3D)
            p.track.push_back(e);
        if (p.track.size() < 2) continue;
        while (seed.points3D.count(seed.next_point3D_id)) ++seed.next_point3D_id;
        const auto id = seed.next_point3D_id++;
        for (const auto& e : p.track) seed.images.at(e.image_id).point3D_ids.at(e.point2D_idx) = id;
        seed.points3D.emplace(id, std::move(p));
    }
}

inline uint64_t verifyOriginalObservations(const Reconstruction& original, const Reconstruction& current) {
    return raw_lineage::verify(original, current);
}

inline SharedIndex retainSharedModels(SharedIndex input, const std::set<size_t>& deferred,
                                     const std::set<uint32_t>& photos) {
    std::vector<uint32_t> remap(input.models, UINT32_MAX); size_t retained = 0;
    for (size_t m = 0; m < input.models; ++m) if (!deferred.count(m)) remap[m] = uint32_t(retained++);
    auto filter = [&](std::vector<SharedLandmark>& points) {
        std::vector<SharedLandmark> keep;
        for (auto& p : points) {
            p.members.erase(std::remove_if(p.members.begin(), p.members.end(), [&](const SharedMember& m) {
                return deferred.count(m.model);
            }), p.members.end());
            if (p.members.size() < 2) continue;
            for (auto& m : p.members) m.model = remap.at(m.model);
            p.observations.erase(std::remove_if(p.observations.begin(), p.observations.end(), [&](const SharedObservation& o) {
                return !photos.count(uint32_t(o.key >> 32));
            }), p.observations.end());
            keep.push_back(std::move(p));
        }
        points = std::move(keep);
    };
    filter(input.landmarks); filter(input.alignment_reference);
    input.models = retained; input.rebuild(); return input;
}

} // namespace sfm::regional
