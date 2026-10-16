#pragma once

#include "sfm/Pipeline.h"
#include "sfm/map/RegionalWindows.h"
#include "sfm/map/RegionalAlignment.h"
#include "sfm/map/RegionalQuality.h"
#include "sfm/map/RegionalPriors.h"

namespace sfm::regional {
using sfm::slog::Tag;
struct RegionalCosts { double initial = 0, final = 0; uint64_t observations = 0; };
struct CameraEstimate { Camera camera; std::vector<std::pair<std::array<double, 12>, uint64_t>> rows; };
using CameraEstimates = std::map<uint32_t, CameraEstimate>;
inline void addCameraEstimates(CameraEstimates& groups, const Reconstruction& rec) {
    std::map<uint32_t, uint64_t> counts;
    for (const auto& kv : rec.images) if (kv.second.registered) counts[kv.second.camera_id] += kv.second.numPoint3D();
    for (const auto& kv : rec.cameras) if (counts[kv.first]) {
        auto& group = groups[kv.first]; group.camera = kv.second; std::array<double, 12> params{};
        packIntrinsics(kv.second, params.data()); group.rows.emplace_back(params, counts[kv.first]);
    }
}
inline void applyCameraEstimates(Reconstruction& out, const CameraEstimates& groups) {
    for (const auto& kv : groups) {
        Camera camera = kv.second.camera; double values[12]{};
        for (int p = 0; p < camNumParams(camera.model); ++p) {
            std::vector<std::pair<double, uint64_t>> sorted; uint64_t total = 0;
            for (const auto& row : kv.second.rows) { sorted.emplace_back(row.first[p], row.second); total += row.second; }
            std::sort(sorted.begin(), sorted.end()); uint64_t weight = 0;
            for (const auto& row : sorted) { weight += row.second; values[p] = row.first; if (weight >= (total + 1) / 2) break; }
        }
        unpackIntrinsics(camera, values); out.cameras[kv.first] = camera;
    }
}

inline void synchronizeRegionalCameras(Reconstruction& rec, const Reconstruction& consensus) {
    for (auto& kv : rec.images) if (kv.second.registered) {
        const auto& image = consensus.images.at(kv.first);
        if (!image.registered || image.camera_id != kv.second.camera_id)
            throw std::runtime_error("regional consensus camera identity changed");
        kv.second.pose = image.pose;
        rec.cameras.at(kv.second.camera_id) = consensus.cameras.at(kv.second.camera_id);
    }
}

template<class ReadModel> inline void refineRegionalPoses(size_t models, ReadModel read,
    Reconstruction& consensus, const std::set<uint32_t>& targets, const fs::path& scratch,
    const SfmConfig& cfg, size_t budget, VkContext& context, PriorSource* gps = nullptr,
    const RegionalPoseAnchors* anchors = nullptr,
    const std::function<bool(size_t, const std::set<uint32_t>&)>& affects = {}) {
    if (targets.empty()) return;
    auto movable = targets; if (anchors) for (const auto& kv : *anchors) movable.erase(kv.first);
    if (movable.empty()) return;
    const double started = now(); double solve_seconds = 0; size_t batches = 0;
    fs::create_directories(scratch);
    const size_t cache = std::min<size_t>(64u << 20, budget / 32);
    PagedArray<Vec3> xyz(scratch / "points.bin", cache);
    fs::path records = scratch / "observations.bin";
    {
        std::ofstream out(records, std::ios::binary);
        for (size_t m = 0; m < models; ++m) {
            if (affects && !affects(m, movable)) continue;
            cancel::check(); auto rec = read(m);
            for (const auto& p : rec.points3D) {
                uint64_t id = UINT64_MAX;
                for (const auto& e : p.second.track) if (movable.count(e.image_id)) {
                    if (id == UINT64_MAX) id = xyz.append(p.second.xyz);
                    record(out, Observation{(uint64_t(e.image_id) << 32) | e.point2D_idx, id,
                        rec.images.at(e.image_id).points2D.at(e.point2D_idx)});
                }
            }
        }
    }
    xyz.flush(); const double extraction_seconds = now() - started;
    const double sort_started = now(); auto sorted = sortObservations(records, scratch / "sort", cache);
    const double sort_seconds = now() - sort_started;
    std::ifstream in(sorted, std::ios::binary); Reconstruction sub; sub.cameras = consensus.cameras;
    std::set<uint64_t> fixed;
    const size_t capacity = std::min<size_t>(500000, budget / 8192);
    auto solve = [&]() {
        if (sub.points3D.empty()) return;
        BundleOptions opt; opt.real = realCfgFromName(cfg.mapper.ba_real); opt.max_iters = 30;
        opt.loss = cfg.mapper.ba_loss; opt.loss_param = float(cfg.mapper.ba_loss_param);
        opt.device_selector = cfg.mapper.device_selector; opt.device = cfg.mapper.device;
        opt.threads = cfg.threads; opt.solver = "cg"; opt.refine_intrinsics = false;
        opt.shared_ctx = &context; opt.fixed_points = &fixed;
        const auto priors = gps ? regionalGpsPriors(sub, *gps) : PosePriors{};
        opt.priors = gps ? &priors : nullptr;
        opt.host_budget_bytes = budget / 3; opt.over_budget_throws = true;
        const double solve_started = now();
        SolverStats stats; opt.stats = &stats; runGlobalBA(sub, opt);
        solve_seconds += now() - solve_started; ++batches;
        if (!std::isfinite(stats.final_cost) || stats.final_cost > stats.initial_cost * (1 + 1e-8) + 1e-8)
            throw std::runtime_error("regional full-observation pose refinement increased cost");
        for (const auto& kv : sub.images) consensus.images.at(kv.first).pose = kv.second.pose;
        slog::diag(Tag::Map, "[regional] full-observation pose refinement: %zu photos, %zu observations, cost %.9g -> %.9g\n",
            sub.images.size(), sub.points3D.size(), stats.initial_cost, stats.final_cost);
        sub.images.clear(); sub.points3D.clear(); fixed.clear();
    };
    Observation row; uint32_t previous = UINT32_MAX;
    while (nextRecord(in, row)) {
        cancel::check(); uint32_t image = uint32_t(row.key >> 32);
        if (anchors && anchors->count(image)) continue;
        if (image != previous && (sub.images.size() >= 128 || sub.points3D.size() >= capacity / 2)) solve();
        if (sub.points3D.size() >= capacity) throw std::length_error("regional pose observation batch exceeds working budget");
        auto found = sub.images.find(image);
        if (found == sub.images.end()) {
            Image im; im.id = image; im.camera_id = consensus.images.at(image).camera_id;
            im.pose = consensus.images.at(image).pose; im.registered = true; found = sub.images.emplace(image, std::move(im)).first;
        }
        auto& im = found->second; uint32_t feature = uint32_t(im.points2D.size());
        uint64_t point = sub.points3D.size(); im.points2D.push_back(row.xy); im.point3D_ids.push_back(point);
        Point3D p; p.xyz = xyz.get(row.point); p.track.push_back({image, feature});
        sub.points3D.emplace(point, std::move(p)); fixed.insert(point); previous = image;
    }
    solve();
    slog::diag(Tag::Map, "[regional] pose timing: extraction %.3fs, sort %.3fs, solve %.3fs, batches %zu, total %.3fs\n",
        extraction_seconds, sort_seconds, solve_seconds, batches, now() - started);
}

inline std::set<uint32_t> regionalQualityTargets(const Reconstruction& poses,
    const std::vector<PhotoQuality>& photos, double gate) {
    std::set<uint32_t> result;
    for (const auto& kv : poses.images)
        if (kv.second.registered && !photoQualityAccepted(photos.at(kv.first), gate)) result.insert(kv.first);
    return result;
}

template<class ReadModel, class WriteModel, class Validate> inline Quality refineRegionalQuality(
    size_t models, ReadModel read, WriteModel write, Validate validate,
    Reconstruction& poses, SharedIndex& points, const std::set<uint32_t>& targets,
    std::vector<PhotoQuality>& photos, const fs::path& scratch,
    const SfmConfig& cfg, size_t budget, VkContext& context, PriorSource* gps = nullptr,
    const RegionalPoseAnchors* anchors = nullptr,
    const std::function<std::set<uint32_t>(Reconstruction&)>& protect = {},
    const std::function<Quality(size_t, const Reconstruction&, const Reconstruction&, std::vector<PhotoQuality>*)>& validate_model = {},
    const std::function<Reconstruction(size_t)>& read_target = {},
    const std::function<bool(size_t, const std::set<uint32_t>&)>& affects = {},
    const std::function<void(size_t)>& retain_unchanged = {}) {
    const double started = now(); double read_seconds = 0, triangulation_seconds = 0, write_seconds = 0, validation_seconds = 0;
    std::map<uint32_t, Pose> before_poses; for (uint32_t id : targets) before_poses.emplace(id, poses.images.at(id).pose);
    refineRegionalPoses(models, [&](size_t m) {
        auto rec = read_target ? read_target(m) : read(m); points.inject(m, rec); return rec;
    }, poses, targets, scratch, cfg, budget, context, gps, anchors, affects);
    auto affected = targets;
    if (protect) {
        const auto repaired = protect(poses);
        affected.insert(repaired.begin(), repaired.end());
    }
    if (affects) for (auto at = affected.begin(); at != affected.end();) {
        const auto old = before_poses.find(*at); const auto& pose = poses.images.at(*at).pose;
        if (old != before_poses.end() && old->second.R == pose.R && (old->second.t - pose.t).norm() == 0) at = affected.erase(at);
        else ++at;
    }
    std::vector<Vec3> previous_points; for (const auto& p : points.landmarks) previous_points.push_back(p.xyz);
    triangulateShared(points, poses, cfg.max_error, cfg.threads, affects ? &affected : nullptr);
    std::set<size_t> changed_models;
    for (size_t p = 0; p < points.landmarks.size(); ++p) if ((points.landmarks[p].xyz - previous_points[p]).norm() > 0)
        for (const auto& member : points.landmarks[p].members) changed_models.insert(member.model);
    std::fill(photos.begin(), photos.end(), PhotoQuality{});
    Quality quality; quality.finite_cost = 0;
    for (size_t m = 0; m < models; ++m) {
        cancel::check(); const double read_started = now(); auto rec = read(m);
        read_seconds += now() - read_started;
        const auto fixed = points.inject(m, rec);
        const bool dirty = !affects || affects(m, affected) || changed_models.count(m);
        if (dirty) {
            const double triangulation_started = now(); synchronizeRegionalCameras(rec, poses);
            triangulatePrivate(rec, fixed, cfg.max_error, cfg.threads, &affected);
            triangulation_seconds += now() - triangulation_started;
            const double write_started = now(); write(m, rec); write_seconds += now() - write_started;
        } else if (retain_unchanged) retain_unchanged(m);
        if (validate_model) {
            const double validation_started = now(); accumulateQuality(quality, validate_model(m, rec, poses, &photos));
            validation_seconds += now() - validation_started;
        }
    }
    slog::diag(Tag::Map, "[regional] quality timing: models %zu, read wait %.3fs, triangulation %.3fs, write %.3fs, fused validation %.3fs, total %.3fs\n",
        models, read_seconds, triangulation_seconds, write_seconds, validation_seconds, now() - started);
    if (validate_model) return quality;
    return validate(poses, points, &photos);
}

inline RegionalCosts boundedRegionalBA(Reconstruction& rec, const Reconstruction& consensus, const SfmConfig& cfg,
                               const std::set<uint64_t>& shared_fixed, size_t budget, VkContext& context,
                               bool measured_gauge = false, PriorSource* gps = nullptr,
                               const RegionalPoseAnchors* anchors = nullptr) {
    RegionalCosts costs; CameraEstimates estimates;
    const double started = now(); double prepare_seconds = 0, solve_seconds = 0; size_t windows = 0;
    regional::WindowGraph graph(rec, budget / 32);
    const double graph_seconds = now() - started;
    auto remaining = graph.remaining();
    size_t window = std::min<size_t>(512, std::max<size_t>(16, budget / (16u << 20)));
    while (!remaining.empty()) {
        const double prepare_started = now();
        cancel::check(); auto group = graph.select(remaining, window, window / 4 + 8);
        std::set<uint64_t> fixed = shared_fixed;
        Reconstruction sub = regional::windowModel(rec, group.images, fixed, cfg.threads);
        for (auto& kv : sub.images) {
            auto pose = consensus.images.find(kv.first);
            if (pose != consensus.images.end() && pose->second.registered) kv.second.pose = pose->second.pose;
        }
        for (auto& kv : sub.cameras) { auto common = consensus.cameras.find(kv.first); if (common != consensus.cameras.end()) kv.second = common->second; }
        BundleOptions opt; opt.real = realCfgFromName(cfg.mapper.ba_real); opt.max_iters = 15;
        opt.loss = cfg.mapper.ba_loss; opt.loss_param = float(cfg.mapper.ba_loss_param);
        opt.device_selector = cfg.mapper.device_selector; opt.device = cfg.mapper.device;
        opt.threads = cfg.threads; opt.solver = "cg"; opt.refine_intrinsics = true;
        opt.shared_ctx = &context;
        const auto priors = gps ? regionalGpsPriors(sub, *gps) : PosePriors{};
        opt.priors = gps ? &priors : nullptr;
        opt.refine_principal_point = cfg.mapper.refine_principal_point; opt.refine_extra_params = cfg.mapper.refine_extra_params;
        SolverStats stats; opt.stats = &stats; opt.fixed_points = &fixed;
        auto held_images = group.support;
        if (anchors) for (auto& kv : sub.images) {
            const auto held = anchors->find(kv.first);
            if (held == anchors->end()) continue;
            kv.second.pose = held->second; held_images.insert(kv.first);
        }
        if (measured_gauge && held_images.empty() && fixed.empty()) {
            held_images = regional::measuredGaugeAnchors(sub);
            if (held_images.empty()) throw std::runtime_error("regional measured BA cannot preserve an observable camera baseline");
        }
        opt.fixed_images = &held_images;
        opt.host_budget_bytes = budget / 3; opt.over_budget_throws = true;
        slog::diag(Tag::Map, "[regional] preparing BA window %zu: %zu cameras, %zu points; host solver budget %.1f MiB\n",
            windows + 1, sub.images.size(), sub.points3D.size(), opt.host_budget_bytes / double(1u << 20));
        prepare_seconds += now() - prepare_started; const double solve_started = now();
        try { runGlobalBA(sub, opt); }
        catch (const BAOverBudget&) { if (window <= 16) throw; window /= 2; continue; }
        catch (const BAIndexCapacity&) { if (window <= 16) throw; window /= 2; continue; }
        solve_seconds += now() - solve_started; ++windows;
        slog::diag(Tag::Map, "[regional] BA window %zu: %zu active cameras, %zu support, %zu points, %.3fs; cost %.9g -> %.9g\n",
            windows, group.core.size(), group.support.size(), sub.points3D.size(), now() - solve_started, stats.initial_cost, stats.final_cost);
        if (!std::isfinite(stats.final_cost) || stats.final_cost > stats.initial_cost * (1 + 1e-8) + 1e-8)
            throw std::runtime_error("regional coordination: invalid fixed-landmark BA cost");
        costs.initial += stats.initial_cost; costs.final += stats.final_cost;
        for (const auto& p : sub.points3D) costs.observations += p.second.track.size();
        addCameraEstimates(estimates, sub);
        for (const auto& kv : sub.images)
            if (group.core.count(kv.first)) rec.images.at(kv.first).pose = kv.second.pose;
        for (const auto& kv : sub.points3D) {
            auto& p = rec.points3D.at(kv.first);
            p.xyz = kv.second.xyz;
        }
        for (uint32_t id : group.core) remaining.erase(id);
    }
    applyCameraEstimates(rec, estimates);
    slog::diag(Tag::Map, "[regional] BA timing: %zu windows, graph %.3fs, prepare %.3fs, solve %.3fs, total %.3fs\n",
               windows, graph_seconds, prepare_seconds, solve_seconds, now() - started);
    return costs;
}

} // namespace sfm::regional
