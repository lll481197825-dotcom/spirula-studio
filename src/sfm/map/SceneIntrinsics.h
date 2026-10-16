#pragma once
#include "sfm/map/SceneQuality.h"
#include "sfm/map/Bundle.h"
#include "sfm/ba/CpuCamera.h"
#include "sfm/ba/CpuDense.h"
#include <array>

namespace sfm::regional {
struct SceneIntrinsicResult { uint64_t observations = 0, valid = 0, retained_observations = 0; uint32_t camerasUpdated = 0; };
inline SceneIntrinsicResult refineSceneIntrinsics(const ImmutableSceneTopology& index, CanonicalScene& scene,
                                                  const BundleOptions& options, double gate, size_t allowance,
                                                  double damping = 1e-3, ImmutableSceneTopology::Progress progress = {}) {
    progress = scene_index_detail::throttledProgress(std::move(progress));
    if (!scene.editing() || !(damping > 0) || !std::isfinite(damping)) throw std::invalid_argument("global intrinsic update requires a damped candidate");
    SceneIntrinsicResult result; if (!options.refine_intrinsics) return result;
    struct Group { Camera camera; uint32_t model, free; std::array<double, 12> intr{}; uint64_t photos = 0; };
    struct Normal { std::array<double, 144> h{}; std::array<double, 12> g{}; uint64_t valid = 0; };
    struct View { uint32_t group = UINT32_MAX; const Pose* globalPose = nullptr; double pose[6]{}; };
    std::vector<Group> groups; std::map<uint32_t, uint32_t> groupIds;
    std::vector<View> views(size_t(index.imageSlots()));
    for (const auto& item : scene.poses().images) if (item.second.registered) {
        const auto& camera = scene.poses().cameras.at(item.second.camera_id); auto inserted = groupIds.emplace(camera.id, uint32_t(groups.size()));
        if (inserted.second) { Group group; group.camera = camera; group.model = uint32_t(camBaModel(camera.model)); packIntrinsics(camera, group.intr.data()); groups.push_back(group); }
        auto& group = groups.at(inserted.first->second); ++group.photos;
        auto& view = views.at(item.first); view.group = inserted.first->second; view.globalPose = &item.second.pose; bundle_detail::packPose(item.second.pose, view.pose);
    }
    for (auto& group : groups) group.free = uint32_t(camNumFreeParams(group.camera.model,
        options.refine_principal_point && options.refine_extra_params && group.photos >= options.pp_min_images, options.refine_extra_params));
    const uint64_t scratch = storage::scaleBytes(groups.size(), sizeof(Normal) * 33 + sizeof(Group)) + storage::scaleBytes(views.size(), sizeof(View));
    if (scratch >= allowance / 2) throw std::length_error("global intrinsic normal tables exceed allowance");
    std::vector<Normal> total(groups.size()); auto& pool = bacpu::Pool::get();
    const int workers = options.threads > 0 ? std::min(options.threads, pool.size()) : pool.size();
    visitSceneRawBatches(index, scene, allowance - size_t(scratch), [&](const std::vector<SceneRawValue>& batch) {
        const int tasks = int((batch.size() + 2047) / 2048); std::vector<std::vector<Normal>> sums(size_t(tasks), std::vector<Normal>(groups.size()));
        pool.runChecked(tasks, workers, [&](int task, int) {
            const size_t end = std::min(batch.size(), (size_t(task) + 1) * 2048);
            for (size_t i = size_t(task) * 2048; i < end; ++i) {
                const auto& row = batch[i]; if (row.kind == LandmarkKind::RetainedObservation) continue; const auto& view = views.at(row.row.image); if (view.group == UINT32_MAX) throw std::runtime_error("global intrinsic row has no pose");
                const auto& group = groups[view.group]; if (!group.free) continue;
                bool valid = true; reprojLandmarkError(group.camera, *view.globalPose, row.row.xy, {row.xyz, row.kind}, &valid); if (!valid) continue;
                const double xyz[3] = {row.xyz.x, row.xyz.y, row.xyz.z}, xy[2] = {row.row.xy.x, row.row.xy.y};
                double residual[2], jc[36], jp[6];
                bacpu::withModel(group.model, [&](auto model) { using M = decltype(model); bacpu::jacobian<M>(view.pose, group.intr.data(), xyz, xy, residual, jc, jp, row.kind == LandmarkKind::UnitDirection);
                    const double scale = group.camera.pixel_scale; if (!(scale > 0)) throw std::runtime_error("invalid intrinsic pixel scale");
                    const double r0 = residual[0] / scale, r1 = residual[1] / scale, norm = std::hypot(r0, r1);
                    if (!std::isfinite(norm)) throw std::runtime_error("nonfinite intrinsic raw Jacobian");
                    const double weight = norm <= gate ? 1 : gate / norm; auto& normal = sums[task][view.group]; ++normal.valid;
                    for (uint32_t a = 0; a < group.free; ++a) {
                        const double j0 = jc[6 + a] / scale, j1 = jc[6 + M::kNumIntr + 6 + a] / scale;
                        normal.g[a] += weight * (j0 * r0 + j1 * r1);
                        for (uint32_t b = 0; b <= a; ++b) { const double k0 = jc[6 + b] / scale, k1 = jc[6 + M::kNumIntr + 6 + b] / scale; normal.h[a * 12 + b] += weight * (j0 * k0 + j1 * k1); }
                    }
                });
            }
        });
        for (const auto& task : sums) for (size_t group = 0; group < groups.size(); ++group) { auto& out = total[group]; const auto& in = task[group]; out.valid += in.valid; for (size_t i = 0; i < 144; ++i) out.h[i] += in.h[i]; for (size_t i = 0; i < 12; ++i) out.g[i] += in.g[i]; }
        for (const auto& row : batch) result.retained_observations += row.kind == LandmarkKind::RetainedObservation;
        result.observations += batch.size(); if (progress) progress("global-intrinsic-fullraw", result.observations, index.observationCount());
    });
    for (size_t group = 0; group < groups.size(); ++group) {
        auto& info = groups[group]; auto& normal = total[group]; result.valid += normal.valid;
        if (!info.free || !normal.valid) continue; bacpu::DenseSpd matrix; matrix.init(info.free);
        for (uint32_t a = 0; a < info.free; ++a) { for (uint32_t b = 0; b <= a; ++b) matrix.row(a)[b] = normal.h[a * 12 + b]; matrix.row(a)[a] += damping * std::max(1.0, normal.h[a * 12 + a]); }
        matrix.factorSolve(normal.g.data(), pool, 1);
        for (uint32_t i = 0; i < info.free; ++i) { info.intr[i] -= normal.g[i]; if (!std::isfinite(info.intr[i])) throw std::runtime_error("nonfinite global intrinsic update"); }
        Camera camera = info.camera; unpackIntrinsics(camera, info.intr.data()); scene.setCamera(camera); ++result.camerasUpdated;
    }
    if (result.observations != index.observationCount()) throw std::runtime_error("global intrinsic fullraw coverage incomplete");
    return result;
}
}
