#pragma once
#include "sfm/map/SceneTopology.h"
#include "sfm/map/RegionalQuality.h"
#include "sfm/map/CanonicalLandmarkGeometry.h"
#include "sfm/map/CanonicalParameterView.h"
#include "sfm/map/GpuRawQuality.h"
#include <future>

namespace sfm::regional {
inline bool sceneFiniteImprovementSmall(const Quality& before, const Quality& after, double tolerance) {
    if (before.observations != after.observations || !std::isfinite(tolerance) || tolerance < 0 ||
        !std::isfinite(before.finite_cost) || !std::isfinite(after.finite_cost) || before.finite_cost < 0 || after.finite_cost < 0)
        throw std::invalid_argument("scene convergence requires one complete finite objective cohort");
    if (before.retained_observations != after.retained_observations || before.directional_observations != after.directional_observations || before.invalid_projections != after.invalid_projections || after.retained_observations > after.observations || after.invalid_projections >= after.observations - after.retained_observations) return false;
    const double improvement = before.finite_cost - after.finite_cost;
    return improvement >= 0 && improvement <= tolerance * std::max(1.0, before.finite_cost);
}
struct SceneRawValue { ImmutableSceneTopology::Row row; Vec3 xyz; LandmarkKind kind = LandmarkKind::FiniteXYZ; };

struct SceneRawBatchStatistics { uint64_t flatBytes = 0, aliasRows = 0, landmarkRoots = 0, rawRows = 0; bool flatParameters = false; };

template<class Visit> void visitSceneRawBatches(const ImmutableSceneTopology& index, const CanonicalScene& scene,
                                               size_t allowance, Visit visit, bool allowFlatParameters = true, SceneRawBatchStatistics* statistics = nullptr, size_t flatCredit = SIZE_MAX, size_t batchLimit = 65536) {
    if (index.identity() != scene.binding().topology || index.pointSlots() != scene.pointSlots() || index.observationCount() != scene.observations())
        throw std::runtime_error("scene raw/parameter identity differs");
    if (allowance < (2u << 20)) throw std::length_error("scene raw evaluation requires 2 MiB");
    const size_t capacity = std::min<size_t>({65536, allowance / (4 * sizeof(SceneRawValue)), batchLimit});
    if (!capacity) throw std::length_error("scene raw batch capacity is zero");
    std::vector<SceneRawValue> batch; batch.reserve(capacity);
    CanonicalParameterView parameters(scene, allowFlatParameters ? size_t(std::min<uint64_t>(allowance / 2, flatCredit)) : 0);
    if (statistics) {
        *statistics = {}; statistics->flatParameters = parameters.flat(); statistics->flatBytes = parameters.bytes();
        if (parameters.flat()) { statistics->aliasRows = scene.pointSlots(); statistics->landmarkRoots = scene.canonicalPoints(); }
    }
    uint64_t root = UINT64_MAX, rows = 0; uint32_t retainedImage = UINT32_MAX; CanonicalLandmark point{}; std::vector<RegionalWorkingStore::PointKey> checked;
    index.visitRaw([&](const ImmutableSceneTopology::Row& row) {
        if (row.canonical != root) {
            if (root != UINT64_MAX && row.canonical < root) throw std::runtime_error("scene raw canonical ordering differs");
            root = row.canonical;
            point = parameters.landmark(root);
            retainedImage = row.image; checked.clear();
        }
        if (parameters.flat()) {
            if (parameters.canonical(row.point) != root) throw std::runtime_error("scene CSR alias mapping differs");
        } else if (std::find(checked.begin(), checked.end(), row.point) == checked.end()) {
            if (parameters.canonical(row.point) != root) throw std::runtime_error("scene CSR alias mapping differs");
            if (checked.size() >= allowance / (8 * sizeof(RegionalWorkingStore::PointKey))) throw std::length_error("scene track alias validation exceeds allowance");
            checked.push_back(row.point);
        }
        if (!std::isfinite(row.xy.x) || !std::isfinite(row.xy.y)) throw std::runtime_error("scene immutable raw coordinate is nonfinite");
        if (point.kind == LandmarkKind::RetainedObservation && row.image != retainedImage) throw std::runtime_error("retained observation root has multiple immutable supporting photos; typed admission required");
        batch.push_back({row, point.value, point.kind}); ++rows;
        if (batch.size() == capacity) { visit(batch); batch.clear(); }
    });
    if (!batch.empty()) visit(batch);
    if (rows != scene.observations()) throw std::runtime_error("scene raw validation coverage incomplete");
    if (statistics) statistics->rawRows = rows;
}

struct SceneQualityEvaluation {
    Quality quality;
    bool complete = false;
    std::string rejected;
    uint64_t gpuRows = 0, cpuRows = 0;
    uint64_t screenedGpuRows = 0;
    uint64_t rawBatchSwaps = 0;
};
namespace scene_quality_detail {
struct Rejected {};
struct BatchQuality {
    std::vector<Quality> sums;
    std::vector<std::map<uint32_t, PhotoQuality>> photos;
    uint64_t gpuRows = 0, cpuRows = 0;
};
inline void validateBaseline(const Quality& baseline, uint64_t observations) {
    if (baseline.observations != observations || !std::isfinite(baseline.cost) || baseline.cost < 0 ||
        !std::isfinite(baseline.finite_cost) || baseline.finite_cost < 0 || baseline.retained_observations > observations ||
        baseline.directional_observations > observations - baseline.retained_observations ||
        baseline.invalid_projections > observations - baseline.retained_observations)
        throw std::invalid_argument("bounded scene evaluation requires one complete valid baseline cohort");
}
}

inline SceneQualityEvaluation evaluateSceneQuality(const ImmutableSceneTopology& index, const CanonicalScene& scene, double gate,
                            size_t allowance, int threads = 0, std::vector<PhotoQuality>* photos = nullptr,
                            ImmutableSceneTopology::Progress progress = {}, const Quality* baseline = nullptr, GpuRawQuality* gpu = nullptr) {
    progress = scene_index_detail::throttledProgress(std::move(progress));
    if (!(gate > 0) || !std::isfinite(gate)) throw std::invalid_argument("invalid scene projection gate");
    if (baseline) scene_quality_detail::validateBaseline(*baseline, scene.observations());
    const auto& poses = scene.poses();
    struct View { const Image* image = nullptr; const Camera* camera = nullptr; };
    const uint64_t fixed = storage::addBytes(storage::scaleBytes(index.imageSlots(), sizeof(View) + (photos ? sizeof(PhotoQuality) : 0)), (1u << 20) + (4u << 16));
    const uint64_t reserve = storage::addBytes(fixed, allowance / 4);
    if (allowance < (2u << 20) || reserve >= allowance) throw std::length_error("scene quality buffers exceed allowance");
    size_t perRow = 2 * sizeof(SceneRawValue) + (photos ? sizeof(PhotoQuality) + 128 : 0);
    if (gpu && !photos && allowance - reserve < perRow + sizeof(GpuQualityRow)) gpu = nullptr;
    if (gpu && !photos) perRow += sizeof(GpuQualityRow);
    size_t capacity = std::min<size_t>({65536, allowance / (4 * sizeof(SceneRawValue)), size_t((allowance - reserve) / perRow)});
    if (capacity >= 2048) capacity -= capacity % 2048;
    if (!capacity) throw std::length_error("scene quality cannot fit one observation batch");
    const uint64_t otherWork = storage::addBytes(fixed, storage::scaleBytes(capacity, perRow));
    std::vector<View> views(size_t(index.imageSlots()));
    for (const auto& item : poses.images) if (item.first < views.size()) views[item.first] = {&item.second, &poses.cameras.at(item.second.camera_id)};
    if (photos) photos->assign(views.size(), {});
    auto& pool = bacpu::Pool::get(); const int workers = threads > 0 ? std::min(threads, pool.size()) : pool.size();
    const bool pipeline = workers > 1;
    const int residualWorkers = pipeline ? workers - 1 : 1;
    const bool gpuReady = gpu && !photos && gpu->begin(poses, gate) == GpuRawQualityStatus::Ready;
    SceneQualityEvaluation result; result.quality.finite_cost = 0;
    auto compute = [&](const std::vector<SceneRawValue>& batch) {
        const int tasks = int((batch.size() + 2047) / 2048);
        scene_quality_detail::BatchQuality output;
        output.sums.resize(tasks); for (auto& q : output.sums) q.finite_cost = 0;
        output.photos.resize(photos ? tasks : 0);
        if (gpuReady && batch.size() <= gpu->capacity()) {
            std::vector<GpuQualityRow> rows; rows.reserve(batch.size());
            for (const auto& value : batch) {
                const auto& view = views.at(value.row.image);
                if (!view.image || !view.camera || !view.image->registered || !(view.camera->pixel_scale > 0) || !std::isfinite(view.camera->pixel_scale)) throw std::runtime_error("scene raw observation has no valid global pose/camera");
                rows.push_back({value.xyz, value.row.xy, value.row.image, uint32_t(value.kind)});
            }
            const auto device = gpu->evaluate(rows);
            if (device.ready()) {
                if (device.partials.size() != size_t(tasks)) throw std::runtime_error("GPU scene quality partial coverage differs");
                for (int task = 0; task < tasks; ++task) {
                    const auto& part = device.partials[size_t(task)]; auto& q = output.sums[size_t(task)];
                    const size_t first = size_t(task) * 2048, end = std::min(batch.size(), first + 2048);
                    uint64_t retained = 0, directional = 0;
                    for (size_t i = first; i < end; ++i) { retained += batch[i].kind == LandmarkKind::RetainedObservation; directional += batch[i].kind == LandmarkKind::UnitDirection; }
                    if (part.observations != end - first || part.retained_observations != retained || part.directional_observations != directional ||
                        part.fallback_rows || part.invalid_projections > part.observations - retained || !std::isfinite(part.cost) || part.cost < 0 || !std::isfinite(part.finite_cost) || part.finite_cost < 0)
                        throw std::runtime_error("GPU scene quality typed partial framing differs");
                    q.observations = part.observations; q.invalid_projections = part.invalid_projections; q.retained_observations = retained;
                    q.directional_observations = directional; q.cost = part.cost; q.finite_cost = part.finite_cost;
                }
                output.gpuRows = batch.size(); return output;
            }
        }
        output.cpuRows = batch.size();
        pool.runChecked(tasks, residualWorkers, [&](int task, int) {
            cancel::check(); auto& q = output.sums[task]; const size_t end = std::min(batch.size(), (size_t(task) + 1) * 2048);
            for (size_t i = size_t(task) * 2048; i < end; ++i) {
                const auto& value = batch[i]; const auto& view = views.at(value.row.image);
                if (!view.image || !view.camera || !view.image->registered || !(view.camera->pixel_scale > 0) || !std::isfinite(view.camera->pixel_scale)) throw std::runtime_error("scene raw observation has no valid global pose/camera");
                ++q.observations;
                if (value.kind == LandmarkKind::RetainedObservation) { ++q.retained_observations; if (photos) ++output.photos[task][value.row.image].retained_observations; continue; }
                q.directional_observations += value.kind == LandmarkKind::UnitDirection;
                bool valid = true; const double error = reprojLandmarkError(*view.camera, view.image->pose, value.row.xy, {value.xyz, value.kind}, &valid) / view.camera->pixel_scale;
                if (!std::isfinite(error)) throw std::runtime_error("nonfinite scene raw residual");
                const double cost = error <= gate ? error * error : gate * (2 * error - gate);
                q.cost += cost; if (valid) q.finite_cost += cost; else ++q.invalid_projections;
                if (photos) { auto& photo = output.photos[task][value.row.image]; ++photo.observations; photo.inliers += valid && error <= gate; photo.error += error; photo.invalid_projections += !valid; }
            }
        });
        return output;
    };
    auto consume = [&](scene_quality_detail::BatchQuality output) {
        for (const auto& q : output.sums) accumulateQuality(result.quality, q);
        result.gpuRows += output.gpuRows; result.cpuRows += output.cpuRows;
        if (photos) for (const auto& local : output.photos) for (const auto& item : local) {
            auto& target = photos->at(item.first); target.observations += item.second.observations; target.retained_observations += item.second.retained_observations;
            target.inliers += item.second.inliers; target.error += item.second.error; target.invalid_projections += item.second.invalid_projections;
        }
        if (progress) progress("fullraw-validation", result.quality.observations, index.observationCount());
        if (baseline) {
            const auto& q = result.quality;
            if (q.invalid_projections > baseline->invalid_projections) result.rejected = "invalid_projections_increased";
            else if (q.finite_cost > baseline->finite_cost * (1 + 1e-8) + 1e-8) result.rejected = "finite_visual_cost_increased";
            else if (q.cost > baseline->cost * (1 + 1e-8) + 1e-8) result.rejected = "visual_cost_increased";
            if (!result.rejected.empty()) throw scene_quality_detail::Rejected{};
        }
    };
    std::vector<SceneRawValue> workingBatch;
    if (pipeline) workingBatch.reserve(capacity);
    std::future<scene_quality_detail::BatchQuality> pending;
    try {
        visitSceneRawBatches(index, scene, allowance, [&](std::vector<SceneRawValue>& batch) {
            if (!pipeline) { consume(compute(batch)); return; }
            if (pending.valid()) consume(pending.get());
            batch.swap(workingBatch); ++result.rawBatchSwaps;
            pending = std::async(std::launch::async, [&] { return compute(workingBatch); });
        }, false, nullptr, 0, capacity);
        if (pending.valid()) consume(pending.get());
        result.complete = true;
    } catch (const scene_quality_detail::Rejected&) {
        if (result.gpuRows && result.rejected != "invalid_projections_increased") {
            auto confirmed = evaluateSceneQuality(index, scene, gate, allowance, threads, photos, progress, baseline, nullptr);
            confirmed.screenedGpuRows += result.gpuRows; return confirmed;
        }
        result.complete = false;
    }
    if (result.quality.retained_observations > result.quality.observations || result.quality.directional_observations > result.quality.observations - result.quality.retained_observations || result.quality.invalid_projections > result.quality.observations - result.quality.retained_observations)
        throw std::runtime_error("scene raw type coverage differs");
    if (!std::isfinite(result.quality.cost) || !std::isfinite(result.quality.finite_cost)) throw std::runtime_error("scene raw cost overflow");
    if (result.gpuRows + result.cpuRows != result.quality.observations) throw std::runtime_error("scene quality backend row coverage differs");
    if (result.complete && result.quality.observations != scene.observations()) throw std::runtime_error("scene raw evaluation coverage incomplete");
    return result;
}

inline Quality sceneQuality(const ImmutableSceneTopology& index, const CanonicalScene& scene, double gate,
                            size_t allowance, int threads = 0, std::vector<PhotoQuality>* photos = nullptr,
                            ImmutableSceneTopology::Progress progress = {}) {
    const auto result = evaluateSceneQuality(index, scene, gate, allowance, threads, photos, std::move(progress));
    if (!result.complete) throw std::logic_error("complete scene quality returned a bounded rejection");
    return result.quality;
}

inline uint64_t triangulateScene(const ImmutableSceneTopology& index, CanonicalScene& scene, double gate,
                                 size_t allowance, int threads = 0, ImmutableSceneTopology::Progress progress = {}) {
    progress = scene_index_detail::throttledProgress(std::move(progress));
    if (!scene.editing()) throw std::logic_error("scene triangulation requires a candidate");
    if (index.identity() != scene.binding().topology) throw std::runtime_error("scene triangulation identity differs");
    struct Job { uint64_t root; CanonicalLandmark before, after; std::vector<Observation> observations; };
    const auto& poses = scene.poses(); auto& pool = bacpu::Pool::get();
    const int workers = threads > 0 ? std::min(threads, pool.size()) : pool.size();
    std::vector<Job> jobs; uint64_t bytes = 0, done = 0, total = scene.canonicalPoints();
    auto flush = [&] {
        if (jobs.empty()) return;
        pool.runChecked(int(jobs.size()), workers, [&](int i, int) { cancel::check(); jobs[i].after = refineLandmark(jobs[i].before, jobs[i].observations, poses, gate); });
        std::vector<std::pair<uint64_t, CanonicalLandmark>> updates; updates.reserve(jobs.size());
        for (const auto& job : jobs) updates.push_back({job.root, job.after});
        scene.applyLandmarkUpdates(updates); done += jobs.size(); jobs.clear(); bytes = 0;
        if (progress) progress("fullsupport-triangulation", done, total);
    };
    scene.visitCanonicalLandmarks([&](uint64_t root, CanonicalLandmark xyz) {
        if (xyz.kind == LandmarkKind::RetainedObservation) { validateLandmark(xyz); ++done; if (progress) progress("fullsupport-retained-observation", done, total); return; }
        const auto range = index.pointRange(root); const uint64_t need = storage::addBytes(sizeof(Job) + sizeof(std::pair<uint64_t, CanonicalLandmark>),
            storage::scaleBytes(range.count, sizeof(Observation) + sizeof(RegionalWorkingStore::PointKey) + sizeof(PointView) + sizeof(Vec3) + 64));
        if (need > allowance / 2) throw std::length_error("scene canonical fullsupport track exceeds allowance");
        if (bytes + need > allowance / 2 || jobs.size() >= 4096) flush();
        Job job{root, xyz, xyz, {}}; job.observations.reserve(size_t(range.count)); std::vector<RegionalWorkingStore::PointKey> checked;
        index.visitPoint(root, [&](const auto& row) {
            if (std::find(checked.begin(), checked.end(), row.point) == checked.end()) { if (scene.canonical(row.point) != root) throw std::runtime_error("scene triangulation alias differs"); checked.push_back(row.point); }
            job.observations.push_back(Observation{root, (uint64_t(row.image) << 32) | row.feature, row.xy});
        });
        jobs.push_back(std::move(job)); bytes += need;
    });
    flush(); if (done != total) throw std::runtime_error("scene triangulation canonical coverage incomplete"); return done;
}
}
