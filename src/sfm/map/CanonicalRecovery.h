#pragma once
#include "sfm/map/SceneTopology.h"
#include "sfm/core/Features.h"
#include "sfm/map/Merge.h"
#include "core/HostMemory.h"

namespace sfm::regional {
class CanonicalRepairBindingRejected : public std::runtime_error {
public:
    explicit CanonicalRepairBindingRejected(const std::string& reason) : std::runtime_error(reason) {}
};
struct RestoredCanonicalFeatureBindings {
    uint64_t matched_features = 0, reassociated_features = 0, moved_raw_rows = 0;
    uint64_t borrowed_points = 0, removed_empty_private_points = 0, raw_rows = 0;
};

// Accepted observation identity stays authoritative when a photo pose is repaired.
// This isolated incoming model retains every row/duplicate and invents no old row.
// Discard this candidate if allocation or validation throws.
inline RestoredCanonicalFeatureBindings restoreAcceptedFeatureBindings(
    const ImmutableSceneTopology& topology, const CanonicalScene& scene,
    Reconstruction& candidate, size_t scratch_budget = 0, double gate = 3) {
    if (!std::isfinite(gate) || gate <= 0) throw std::invalid_argument("canonical feature restoration requires a finite positive visual gate");
    if (scene.editing() || !scene.initialized() || topology.identity() != scene.binding().topology ||
        topology.pointSlots() != scene.pointSlots() || topology.observationCount() != scene.observations())
        throw std::logic_error("canonical feature restoration requires matching accepted topology and parameters");
    if (candidate.next_point3D_id < scene.pointSlots())
        throw std::runtime_error("incoming recovery did not reserve accepted point identity space");
    if (!scratch_budget) {
        const auto available = spirula::availableRamBytes();
        if (!available) throw std::runtime_error("canonical feature restoration needs actual RAM headroom");
        scratch_budget = available - std::min<size_t>(size_t(4) << 30, available / 4);
    }
    auto guard = [&](uint64_t extra) {
        if (extra > scratch_budget) throw std::length_error("canonical feature restoration exceeds scratch allowance");
    };
    auto same = [](Vec3 a, Vec3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; };
    std::map<uint64_t, uint64_t> raw;
    RestoredCanonicalFeatureBindings result;
    for (const auto& point : candidate.points3D) {
        if (point.first < scene.pointSlots()) {
            if (!scene.isCanonical(point.first)) throw std::runtime_error("incoming borrowed canonical ID is not a root");
            const auto accepted = scene.landmark(point.first);
            if (accepted.kind != LandmarkKind::FiniteXYZ) throw CanonicalRepairBindingRejected("incoming finite recovery borrowed an accepted nonmetric root; typed depth admission required");
            if (!same(point.second.xyz, accepted.value)) throw std::runtime_error("incoming borrowed canonical point differs from accepted XYZ");
        }
        for (const auto& row : point.second.track) {
            const auto image = candidate.images.find(row.image_id);
            if (image == candidate.images.end() || !image->second.registered || row.point2D_idx >= image->second.points2D.size() ||
                row.point2D_idx >= image->second.point3D_ids.size() || image->second.point3D_ids[row.point2D_idx] != point.first)
                throw std::runtime_error("incoming canonical restoration raw reciprocity invalid");
            const auto key = (uint64_t(row.image_id) << 32) | row.point2D_idx;
            if (!raw.count(key)) guard(storage::scaleBytes(raw.size() + 1, 80));
            if (raw[key] == UINT64_MAX || result.raw_rows == UINT64_MAX) throw std::overflow_error("incoming canonical raw count overflow");
            ++raw[key]; ++result.raw_rows;
        }
    }
    struct Action { uint32_t image, feature; uint64_t oldPoint, root, rows; };
    std::vector<Action> actions;
    std::map<uint64_t, Point3D> borrowed;
    std::map<uint64_t, uint64_t> additions;
    struct Anchor { uint64_t root; bool multiple = false; };
    std::map<uint64_t, Anchor> anchors;
    for (const auto& item : candidate.images) {
        const auto accepted = scene.poses().images.find(item.first);
        if (accepted == scene.poses().images.end() || !accepted->second.registered || !item.second.registered) continue;
        const auto& image = item.second;
        if (image.id != item.first || image.camera_id != accepted->second.camera_id || image.exif_orientation != accepted->second.exif_orientation ||
            image.name != accepted->second.name || image.points2D.size() != image.point3D_ids.size())
            throw std::runtime_error("canonical feature restoration photo identity differs");
        const auto& camera = candidate.cameras.at(image.camera_id);
        const auto& oldCamera = scene.poses().cameras.at(image.camera_id);
        if (camera.model != oldCamera.model || camera.width != oldCamera.width || camera.height != oldCamera.height || camera.pixel_scale != oldCamera.pixel_scale)
            throw std::runtime_error("canonical feature restoration measurement frame differs");
        std::map<uint32_t, uint64_t> bindings;
        topology.visitPhoto(item.first, [&](const ImmutableSceneTopology::Row& row) {
            if (row.image != item.first || scene.canonical(row.point) != row.canonical || row.feature >= image.points2D.size())
                throw std::runtime_error("accepted canonical raw feature identity differs");
            if (image.point3D_ids[row.feature] == kInvalidPoint3D) return;
            if (scene.landmark(row.canonical).kind != LandmarkKind::FiniteXYZ)
                throw CanonicalRepairBindingRejected("incoming finite recovery track overlaps an accepted nonmetric constraint; typed depth admission required");
            const auto xy = image.points2D[row.feature];
            if (xy.x != row.xy.x || xy.y != row.xy.y) throw std::runtime_error("incoming canonical feature XY changed");
            if (!bindings.count(row.feature)) guard(storage::addBytes(storage::scaleBytes(raw.size(), 80),
                storage::addBytes(storage::scaleBytes(bindings.size() + 1, 64), storage::scaleBytes(actions.capacity(), sizeof(Action)))));
            const auto inserted = bindings.emplace(row.feature, row.canonical);
            if (!inserted.second && inserted.first->second != row.canonical)
                throw std::runtime_error("accepted feature has competing canonical roots");
        });
        guard(storage::addBytes(storage::scaleBytes(raw.size(), 80), storage::scaleBytes(bindings.size(), 64)));
        for (const auto& binding : bindings) {
            const auto current = image.point3D_ids.at(binding.first); if (current == kInvalidPoint3D) continue;
            const auto key = (uint64_t(item.first) << 32) | binding.first;
            const auto incoming = raw.find(key);
            if (incoming == raw.end() || !incoming->second || !candidate.points3D.count(current))
                throw std::runtime_error("assigned incoming canonical feature lacks its raw row");
            ++result.matched_features;
            const auto root = binding.second; const auto xyz = scene.pointXYZ(root);
            const auto existing = candidate.points3D.find(root);
            if (existing != candidate.points3D.end() && !same(existing->second.xyz, xyz))
                throw std::runtime_error("canonical destination would overwrite borrowed XYZ");
            const auto anchor = anchors.emplace(current, Anchor{root, false});
            if (!anchor.second && anchor.first->second.root != root) anchor.first->second.multiple = true;
            guard(storage::addBytes(storage::scaleBytes(raw.size(), 80), storage::scaleBytes(anchors.size(), 80)));
            if (current == root) continue;
            if (existing == candidate.points3D.end() && !borrowed.count(root)) {
                Point3D point = candidate.points3D.at(current); point.xyz = xyz; point.track.clear(); borrowed.emplace(root, std::move(point));
            }
            if (actions.size() == actions.capacity()) {
                const auto capacity = std::max<size_t>(16, actions.size() * 2);
                guard(storage::addBytes(storage::scaleBytes(raw.size(), 80), storage::addBytes(storage::scaleBytes(capacity, sizeof(Action)),
                    storage::scaleBytes(borrowed.size(), sizeof(Point3D) + 128))));
                actions.reserve(capacity);
            }
            actions.push_back({item.first, binding.first, current, root, incoming->second});
            additions[root] = storage::addBytes(additions[root], incoming->second);
            guard(storage::addBytes(storage::scaleBytes(raw.size(), 80), storage::addBytes(storage::scaleBytes(actions.capacity(), sizeof(Action)),
                storage::addBytes(storage::scaleBytes(borrowed.size(), sizeof(Point3D) + 128), storage::scaleBytes(additions.size(), 80)))));
        }
    }
    std::map<uint64_t, uint64_t> wholeTracks;
    for (const auto& item : anchors) {
        if (item.first < scene.pointSlots()) continue;
        const auto& incoming = candidate.points3D.at(item.first);
        const bool hasNewPhoto = std::any_of(incoming.track.begin(), incoming.track.end(), [&](const auto& row) {
            const auto accepted = scene.poses().images.find(row.image_id);
            return accepted == scene.poses().images.end() || !accepted->second.registered;
        });
        if (!hasNewPhoto) continue;
        if (item.second.multiple)
            throw CanonicalRepairBindingRejected("incoming track with new photos bridges multiple accepted canonical roots: point=" + std::to_string(item.first));
        const auto root = item.second.root; const auto xyz = scene.pointXYZ(root);
        std::set<uint32_t> support;
        for (const auto& row : incoming.track) {
            const auto& image = candidate.images.at(row.image_id); const auto& camera = candidate.cameras.at(image.camera_id);
            const auto xy = image.points2D.at(row.point2D_idx);
            if (!std::isfinite(xy.x) || !std::isfinite(xy.y) || !std::isfinite(camera.pixel_scale) || camera.pixel_scale <= 0)
                throw std::runtime_error("canonical incoming measurement frame or XY nonfinite");
            bool valid = true; const double error = reprojErrorAt(camera, image.pose, xy, xyz, &valid) / camera.pixel_scale;
            if (!valid || !std::isfinite(error) || error > gate)
                throw CanonicalRepairBindingRejected("incoming track lacks full visual support at accepted canonical XYZ: point=" +
                    std::to_string(item.first) + " root=" + std::to_string(root) + " image=" + std::to_string(row.image_id) +
                    " feature=" + std::to_string(row.point2D_idx) + " error=" + std::to_string(error));
            support.insert(row.image_id);
            guard(storage::addBytes(storage::scaleBytes(raw.size(), 80), storage::addBytes(storage::scaleBytes(anchors.size() + wholeTracks.size(), 80), storage::scaleBytes(support.size(), 64))));
        }
        if (support.size() < 2)
            throw CanonicalRepairBindingRejected("incoming canonical track requires two distinct inlier photos: point=" + std::to_string(item.first));
        wholeTracks.emplace(item.first, root);
    }
    actions.erase(std::remove_if(actions.begin(), actions.end(), [&](const Action& action) {
        if (!wholeTracks.count(action.oldPoint)) return false;
        auto& count = additions.at(action.root);
        if (count < action.rows) throw std::logic_error("canonical whole-track action accounting underflow");
        count -= action.rows; return true;
    }), actions.end());
    for (const auto& item : wholeTracks)
        additions[item.second] = storage::addBytes(additions[item.second], candidate.points3D.at(item.first).track.size());
    uint64_t extraRows = 0; for (const auto& entry : additions) extraRows = storage::addBytes(extraRows, entry.second);
    guard(storage::addBytes(storage::scaleBytes(raw.size(), 80), storage::addBytes(storage::scaleBytes(actions.capacity(), sizeof(Action)),
        storage::addBytes(storage::scaleBytes(borrowed.size(), sizeof(Point3D) + 128), storage::addBytes(storage::scaleBytes(anchors.size() + wholeTracks.size(), 80), storage::scaleBytes(extraRows, sizeof(TrackElement)))))));
    // All identity/XYZ conflicts were checked before changing any association.
    for (auto& point : borrowed) candidate.points3D.emplace(point.first, std::move(point.second));
    result.borrowed_points = borrowed.size();
    for (const auto& entry : additions) {
        auto& track = candidate.points3D.at(entry.first).track;
        track.reserve(size_t(storage::addBytes(track.size(), entry.second)));
    }
    std::set<uint64_t> emptied;
    for (const auto& action : actions) {
        auto& before = candidate.points3D.at(action.oldPoint).track;
        const auto size = before.size();
        before.erase(std::remove_if(before.begin(), before.end(), [&](const TrackElement& row) {
            return row.image_id == action.image && row.point2D_idx == action.feature;
        }), before.end());
        if (size - before.size() != action.rows) throw std::runtime_error("canonical incoming row multiplicity changed during restoration");
        auto& after = candidate.points3D.at(action.root).track;
        for (uint64_t row = 0; row < action.rows; ++row) after.push_back({action.image, action.feature});
        candidate.images.at(action.image).point3D_ids.at(action.feature) = action.root;
        result.moved_raw_rows = storage::addBytes(result.moved_raw_rows, action.rows); ++result.reassociated_features;
        if (before.empty() && action.oldPoint >= scene.pointSlots()) emptied.insert(action.oldPoint);
    }
    for (const auto& item : wholeTracks) {
        auto& before = candidate.points3D.at(item.first).track;
        auto& after = candidate.points3D.at(item.second).track;
        for (const auto& row : before) {
            auto& assignment = candidate.images.at(row.image_id).point3D_ids.at(row.point2D_idx);
            if (assignment != item.second) { assignment = item.second; ++result.reassociated_features; }
            after.push_back(row); ++result.moved_raw_rows;
        }
        before.clear(); emptied.insert(item.first);
    }
    for (auto point : emptied) if (candidate.points3D.at(point).track.empty()) { candidate.points3D.erase(point); ++result.removed_empty_private_points; }
    uint64_t afterRows = 0;
    for (const auto& point : candidate.points3D) for (const auto& row : point.second.track) {
        const auto& image = candidate.images.at(row.image_id);
        if (image.point3D_ids.at(row.point2D_idx) != point.first) throw std::runtime_error("canonical restored raw reciprocity invalid");
        const auto key = (uint64_t(row.image_id) << 32) | row.point2D_idx;
        const auto at = raw.find(key);
        if (at == raw.end() || !at->second) throw std::runtime_error("canonical restoration invented or duplicated a raw row");
        --at->second; ++afterRows;
    }
    if (afterRows != result.raw_rows || std::any_of(raw.begin(), raw.end(), [](const auto& row) { return row.second != 0; }))
        throw std::runtime_error("canonical restoration lost incoming raw multiplicity");
    return result;
}

inline Reconstruction gatherCanonicalSeed(const ImmutableSceneTopology& topology, const CanonicalScene& scene,
                                         const std::set<uint32_t>& anchorPhotos, const std::function<std::filesystem::path(uint32_t)>& featurePath,
                                         size_t budget) {
    if (scene.editing() || !scene.initialized()) throw std::logic_error("canonical recovery requires accepted scene parameters");
    if (!featurePath) throw std::invalid_argument("canonical recovery requires explicit input feature identities");
    if (topology.identity() != scene.binding().topology || topology.pointSlots() != scene.pointSlots() || topology.observationCount() != scene.observations())
        throw std::runtime_error("canonical recovery topology/parameter identity differs");
    if (!budget) {
        const auto available = spirula::availableRamBytes(), physical = spirula::physicalRamBytes();
        if (!available || !physical) throw std::runtime_error("canonical recovery auto budget needs actual memory");
        budget = std::min<size_t>(physical / 5 * 4, available - std::min<size_t>(size_t(4) << 30, available / 4));
    }
    if (budget < (8u << 20)) throw std::length_error("canonical recovery requires 8 MiB");
    Reconstruction out; uint64_t resident = modelResidentBytes(out);
    auto guard = [&](uint64_t extra) {
        if (storage::addBytes(resident, storage::addBytes(extra, 4u << 20)) > budget) throw std::length_error("canonical recovery context exceeds allowance");
        const auto available = spirula::availableRamBytes();
        if (available && extra > available - std::min<size_t>(size_t(4) << 30, available / 4)) throw std::length_error("canonical recovery context exceeds physical RAM headroom");
    };
    for (uint32_t id : anchorPhotos) {
        cancel::check(); const auto& accepted = scene.poses().images.at(id); const auto& camera = scene.poses().cameras.at(accepted.camera_id);
        if (!accepted.registered) throw std::runtime_error("canonical recovery anchor is not visually registered");
        const auto path = featurePath(id); uint32_t count = 0;
        if (!peekFeatures(path.u8string(), count)) throw std::runtime_error("canonical recovery feature file is missing or truncated");
        const uint64_t temporary = storage::scaleBytes(count, sizeof(Keypoint) + 4 * sizeof(float) + 3 + sizeof(float) + sizeof(Vec2) + sizeof(uint64_t));
        guard(storage::addBytes(temporary, accepted.name.capacity() + sizeof(Image) + sizeof(Camera) + 256));
        auto feature = readFeatures(path.u8string(), false);
        if (feature.count() != count || feature.width != camera.width || feature.height != camera.height || feature.exif_orientation != accepted.exif_orientation ||
            !std::isfinite(feature.pixelScale()) || std::abs(feature.pixelScale() - camera.pixel_scale) > 1e-12 * std::max(1.0, camera.pixel_scale))
            throw std::runtime_error("canonical recovery feature/camera measurement frame differs");
        Image image = accepted; image.points2D.resize(count); image.point3D_ids.assign(count, kInvalidPoint3D);
        for (size_t i = 0; i < count; ++i) {
            const auto& keypoint = feature.keypoints[i]; if (!std::isfinite(keypoint.x) || !std::isfinite(keypoint.y)) throw std::runtime_error("canonical recovery feature XY nonfinite");
            image.points2D[i] = {double(keypoint.x), double(keypoint.y)};
        }
        out.images.emplace(id, std::move(image)); out.cameras.emplace(camera.id, camera);
        resident = modelResidentBytes(out);
    }
    for (uint32_t id : anchorPhotos) {
        cancel::check(); auto& image = out.images.at(id);
        topology.visitPhoto(id, [&](const ImmutableSceneTopology::Row& row) {
            if (row.image != id || row.feature >= image.points2D.size() || scene.canonical(row.point) != row.canonical)
                throw std::runtime_error("canonical recovery raw lineage differs");
            const auto xy = image.points2D[row.feature];
            if (row.xy.x != xy.x || row.xy.y != xy.y) throw std::runtime_error("canonical recovery raw/original feature XY differs");
            if (scene.landmark(row.canonical).kind != LandmarkKind::FiniteXYZ) return;
            auto& assignment = image.point3D_ids[row.feature];
            if (assignment != kInvalidPoint3D) {
                if (assignment != row.canonical) throw std::runtime_error("canonical recovery feature has competing canonical points");
                return;
            }
            auto point = out.points3D.find(row.canonical);
            if (point == out.points3D.end()) {
                guard(sizeof(std::pair<const uint64_t, Point3D>) + 64 + 2 * sizeof(TrackElement));
                Point3D value; value.xyz = scene.pointXYZ(row.canonical); point = out.points3D.emplace(row.canonical, std::move(value)).first;
                resident = storage::addBytes(resident, sizeof(std::pair<const uint64_t, Point3D>) + 64);
            }
            auto& track = point->second.track;
            if (track.size() == track.capacity()) {
                const size_t capacity = std::max<size_t>(2, track.size() * 2); guard(storage::scaleBytes(capacity, sizeof(TrackElement)));
                const size_t previous = track.capacity(); track.reserve(capacity); resident = storage::addBytes(resident, storage::scaleBytes(track.capacity() - previous, sizeof(TrackElement)));
            }
            track.push_back({id, row.feature}); assignment = row.canonical;
        });
    }
    out.next_point3D_id = scene.pointSlots();
    if (modelResidentBytes(out) + (4u << 20) > budget) throw std::length_error("canonical recovery actual context exceeds allowance");
    return out;
}
}
