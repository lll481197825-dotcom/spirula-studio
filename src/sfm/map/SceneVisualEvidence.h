#pragma once
#include "sfm/map/SceneQuality.h"
#include <numeric>

namespace sfm::regional {
struct SceneVisualEvidence {
    std::vector<PhotoQuality> photos;
    std::vector<uint64_t> landmarks;
    std::vector<uint32_t> component;
    uint64_t uniqueObservations = 0, retained_observations = 0;
    uint32_t components = 0, largest = UINT32_MAX;
};
inline SceneVisualEvidence sceneVisualEvidence(const ImmutableSceneTopology& index, const CanonicalScene& scene,
    double gate, size_t allowance, int threads = 0, ImmutableSceneTopology::Progress progress = {}) {
    if (!std::isfinite(gate) || gate <= 0 || !allowance) throw std::invalid_argument("distinct visual evidence requires a finite positive gate and allowance");
    progress = scene_index_detail::throttledProgress(std::move(progress));
    struct View { const Image* image = nullptr; const Camera* camera = nullptr; };
    const uint64_t fixed = storage::addBytes(storage::scaleBytes(index.imageSlots(), sizeof(PhotoQuality) + sizeof(uint32_t) + sizeof(uint64_t) + sizeof(View)), 4u << 20);
    if (fixed >= allowance) throw std::length_error("distinct visual evidence photo tables exceed allowance");
    std::vector<View> views(size_t(index.imageSlots()));
    for (const auto& item : scene.poses().images) if (item.first < views.size()) views[item.first] = {&item.second, &scene.poses().cameras.at(item.second.camera_id)};
    SceneVisualEvidence out; out.photos.resize(size_t(index.imageSlots())); out.component.resize(out.photos.size()); out.landmarks.resize(out.photos.size());
    std::iota(out.component.begin(), out.component.end(), 0);
    auto root = [&](uint32_t id) { while (out.component[id] != id) { out.component[id] = out.component[out.component[id]]; id = out.component[id]; } return id; };
    uint64_t previousPoint = UINT64_MAX, lastPoint = UINT64_MAX; uint32_t previousImage = UINT32_MAX, previousFeature = UINT32_MAX;
    std::vector<uint32_t> support; auto connect = [&] {
        if (support.size() < 2) return;
        for (uint32_t image : support) ++out.landmarks.at(image);
        const uint32_t first = root(support[0]); for (size_t i = 1; i < support.size(); ++i) out.component[root(support[i])] = first;
    };
    auto& pool = bacpu::Pool::get(); const int workers = threads > 0 ? std::min(threads, pool.size()) : pool.size();
    visitSceneRawBatches(index, scene, (allowance - size_t(fixed)) / 3, [&](const std::vector<SceneRawValue>& raw) {
        std::vector<SceneRawValue> unique; unique.reserve(raw.size());
        for (const auto& row : raw) {
            if (row.row.canonical == previousPoint && row.row.image == previousImage && row.row.feature == previousFeature) continue;
            previousPoint = row.row.canonical; previousImage = row.row.image; previousFeature = row.row.feature; unique.push_back(row);
        }
        struct Residual { double error = 0; bool valid = false; }; std::vector<Residual> errors(unique.size());
        const int jobs = int((unique.size() + 2047) / 2048);
        pool.runChecked(jobs, workers, [&](int job, int) {
            const size_t end = std::min(unique.size(), (size_t(job) + 1) * 2048);
            for (size_t i = size_t(job) * 2048; i < end; ++i) {
                const auto& value = unique[i]; const auto& view = views.at(value.row.image);
                if (!view.image || !view.camera || !view.image->registered || !(view.camera->pixel_scale > 0)) throw std::runtime_error("distinct visual observation has no valid global pose/camera");
                const auto& image = *view.image; const auto& camera = *view.camera;
                if (value.kind == LandmarkKind::RetainedObservation) continue;
                bool valid = true; const double error = reprojLandmarkError(camera, image.pose, value.row.xy, {value.xyz, value.kind}, &valid) / camera.pixel_scale;
                if (!std::isfinite(error)) throw std::runtime_error("nonfinite distinct visual residual"); errors[i] = {error, valid};
            }
        });
        for (size_t i = 0; i < unique.size(); ++i) {
            const auto& row = unique[i].row; const auto& e = errors[i]; auto& photo = out.photos.at(row.image);
            if (lastPoint != row.canonical) { connect(); support.clear(); lastPoint = row.canonical; }
            ++out.uniqueObservations;
            if (unique[i].kind == LandmarkKind::RetainedObservation) { ++photo.retained_observations; ++out.retained_observations; continue; }
            ++photo.observations; photo.inliers += e.valid && e.error <= gate; photo.invalid_projections += !e.valid; photo.error += e.error;
            if (unique[i].kind == LandmarkKind::FiniteXYZ && e.valid && e.error <= gate && (support.empty() || support.back() != row.image)) support.push_back(row.image);
        }
        if (progress) progress("distinct-visual-evidence", out.uniqueObservations, index.observationCount());
    });
    connect(); std::map<uint32_t, uint64_t> sizes;
    for (const auto& image : scene.poses().images) if (image.second.registered) {
        const uint32_t component = root(image.first); out.component.at(image.first) = component; ++sizes[component];
    }
    out.components = uint32_t(sizes.size()); uint64_t count = 0;
    for (const auto& item : sizes) if (item.second > count) { out.largest = item.first; count = item.second; }
    if (progress) progress("distinct-visual-evidence-complete", out.uniqueObservations, out.uniqueObservations);
    return out;
}
}
