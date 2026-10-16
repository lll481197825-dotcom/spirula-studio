#pragma once
#include "sfm/map/CanonicalScene.h"

namespace sfm::regional {
inline Reconstruction canonicalPreview(const CanonicalScene& scene, uint64_t limit = 50000) {
    Reconstruction preview = scene.poses(); preview.points3D.clear();
    if (!limit || !scene.pointSlots()) return preview;
    const auto stride = scene.pointSlots() / limit + (scene.pointSlots() % limit != 0);
    for (uint64_t id = 0; id < scene.pointSlots();) {
        cancel::check();
        if (scene.isCanonical(id)) {
            const auto value = scene.landmark(id);
            if (value.kind == LandmarkKind::FiniteXYZ) { Point3D point; point.xyz = value.value; preview.points3D.emplace(id, std::move(point)); }
        }
        if (stride >= scene.pointSlots() - id) break;
        id += stride;
    }
    return preview;
}
}
