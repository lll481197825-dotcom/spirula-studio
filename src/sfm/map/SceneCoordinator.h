#pragma once
#include "sfm/map/CanonicalBundle.h"
#include "sfm/map/SceneIntrinsics.h"

namespace sfm::regional {
inline CanonicalScene::PointProducer completeSceneSeed(const ImmutableSceneTopology& index, const Reconstruction& poses,
                                                       double gate, size_t allowance, int threads = 0,
                                                       ImmutableSceneTopology::Progress progress = {},
                                                       std::shared_ptr<std::vector<std::pair<uint64_t, Vec3>>> directions = {}, double minTriAngleDegrees = 0,
                                                       std::shared_ptr<std::vector<uint64_t>> retained = {}) {
    progress = scene_index_detail::throttledProgress(std::move(progress));
    return [&index, &poses, gate, allowance, threads, progress, directions, minTriAngleDegrees, retained](const CanonicalScene::PointEmit& emit) {
        if (directions) directions->clear(); if (retained) retained->clear();
        struct Job { uint64_t root; Vec3 before, after; std::vector<Observation> track; bool direction = false, retained = false; };
        std::vector<Job> jobs; uint64_t bytes = 0, done = 0, total = 0;
        index.visitSeeds([&](const auto&) { ++total; });
        auto& pool = bacpu::Pool::get(); const int workers = threads > 0 ? std::min(threads, pool.size()) : pool.size();
        auto flush = [&] {
            if (jobs.empty()) return;
            pool.runChecked(int(jobs.size()), workers, [&](int i, int) {
                cancel::check(); auto& job = jobs[i];
                if (isRetainedObservationTrack(job.track)) {
                    if (!retained) throw std::logic_error("initial singleton support requires retained-observation classification output");
                    job.after = {}; job.retained = true; return;
                }
                job.after = refineUnifiedPoint(job.before, job.track, poses, gate);
                if (directions && minTriAngleDegrees > 0 && landmark_geometry_detail::objective({job.after},job.track,poses,gate).invalid) {
                    const auto fitted = fitUnitDirection(job.track,poses,gate,minTriAngleDegrees);
                    if (fitted.accepted) {job.after=fitted.value;job.direction=true;}
                }
            });
            for (const auto& job : jobs) {
                emit(job.root, job.after);
                if (job.retained) {
                    const auto need = storage::addBytes(storage::scaleBytes(retained->size() + 1, sizeof(uint64_t)), directions ? storage::scaleBytes(directions->size(), sizeof(directions->front())) : 0);
                    if (need > allowance / 8) throw std::length_error("initial retained evidence exceeds allowance");
                    retained->push_back(job.root);
                }
                if(job.direction) {
                    if(storage::addBytes(storage::scaleBytes(directions->size()+1,sizeof(directions->front())), retained ? storage::scaleBytes(retained->size(),sizeof(uint64_t)) : 0)>allowance/8) throw std::length_error("initial direction evidence exceeds allowance");
                    directions->push_back({job.root,job.after});
                }
            }
            done += jobs.size(); jobs.clear(); bytes = 0; if (progress) progress("initial-fullsupport-seed", done, total);
        };
        index.visitSeeds([&](const ImmutableSceneTopology::Seed& seed) {
            cancel::check(); const auto range = index.pointRange(seed.root);
            const uint64_t need = storage::addBytes(sizeof(Job), storage::scaleBytes(range.count, sizeof(Observation) + sizeof(Vec3) + sizeof(PointView) + 96));
            if (need > allowance / 2) throw std::length_error("scene initial complete support exceeds allowance");
            if (bytes + need > allowance / 2 || jobs.size() >= 4096) flush();
            Job job{seed.root, seed.xyz, seed.xyz, {}}; job.track.reserve(size_t(range.count));
            index.visitPoint(seed.root, [&](const auto& row) { job.track.push_back({seed.root, (uint64_t(row.image) << 32) | row.feature, row.xy}); });
            jobs.push_back(std::move(job)); bytes += need;
        });
        flush();
    };
}

struct SceneRoundResult {
    Quality before, after;
    bool accepted = false;
    double alpha = 0;
    CanonicalScene::Snapshot snapshot;
    std::vector<std::string> rejected;
    std::vector<std::filesystem::path> preserved;
    SceneIntrinsicResult intrinsics;
    struct Statistics {
        double baselineSeconds = 0, blockSeconds = 0, intrinsicSeconds = 0, triangulationSeconds = 0;
        double candidateQualitySeconds = 0, gatesSeconds = 0, snapshotSeconds = 0, cpuConfirmationSeconds = 0;
        uint64_t qualityRows = 0, completeEvaluations = 0, boundedRejections = 0, gateRejections = 0, baselineCacheHits = 0;
        uint64_t gpuQualityRows = 0, cpuQualityRows = 0;
    } statistics;
};

class SceneCoordinator {
public:
    using Gates = std::function<std::vector<std::string>(const Reconstruction&, const Reconstruction&)>;
    using ExecuteBlocks = std::function<void(std::vector<CanonicalBundle>&, CanonicalScene&, const BundleOptions&)>;
    struct Options { BundleOptions bundle; double gate = 3; size_t allowance = 0; int backtrackSteps = 12; uint64_t roundOrdinal = 0; GpuRawQuality* qualityGpu = nullptr; };
    SceneCoordinator(CanonicalScene& scene, const ImmutableSceneTopology& topology, Options options,
                     Gates gates, ImmutableSceneTopology::Progress progress = {})
        : scene_(scene), topology_(topology), options_(std::move(options)), gates_(std::move(gates)), progress_(std::move(progress)) {
        if (!gates_ || !(options_.gate > 0) || options_.allowance < (16u << 20) || options_.backtrackSteps < 0) throw std::invalid_argument("scene coordinator requires complete gates and resources");
        options_.bundle.loss = "huber"; options_.bundle.loss_param = float(options_.gate);
        if (!scene.initialized() || topology.identity() != scene.binding().topology) throw std::runtime_error("scene coordinator accepted topology differs");
    }
    static void requireIndependentTasks(const std::vector<CanonicalBundle>& tasks, size_t allowance) {
        std::vector<uint32_t> images; std::vector<uint64_t> points;
        for (const auto& task : tasks) {
            for (size_t i = 0; i < task.images.size(); ++i) if (!task.problem.fixed_frames.at(i)) images.push_back(task.images[i]);
            for (size_t i = 0; i < task.points.size(); ++i) if (!task.problem.fixed_points.at(i)) points.push_back(task.points[i]);
            if (storage::scaleBytes(images.capacity(), sizeof(uint32_t)) + storage::scaleBytes(points.capacity(), sizeof(uint64_t)) > allowance / 8) throw std::length_error("scene task ownership ledger exceeds allowance");
        }
        std::sort(images.begin(), images.end()); std::sort(points.begin(), points.end());
        if (std::adjacent_find(images.begin(), images.end()) != images.end() || std::adjacent_find(points.begin(), points.end()) != points.end()) throw std::runtime_error("parallel scene tasks overlap mutable parameters");
    }
    void setRoundOrdinal(uint64_t ordinal) { options_.roundOrdinal = ordinal; }
    SceneRoundResult round(std::vector<CanonicalBundle>& tasks, ExecuteBlocks execute = {}) {
        requireIndependentTasks(tasks, options_.allowance);
        using Clock = std::chrono::steady_clock;
        auto elapsed = [](const auto& start) { return std::chrono::duration<double>(Clock::now() - start).count(); };
        SceneRoundResult result;
        auto started = Clock::now();
        const bool cacheHit = !cachedQualityIdentity_.empty() && cachedQualityIdentity_ == scene_.identity();
        if (cacheHit) {
            if (scene_.inspectRetainedSnapshot(scene_.acceptedSnapshot().directory).identity != cachedQualityIdentity_) throw std::runtime_error("cached fullraw baseline snapshot changed");
            result.before = cachedQuality_; ++result.statistics.baselineCacheHits;
        }
        else result.before = sceneQuality(topology_, scene_, options_.gate, options_.allowance, options_.bundle.threads, nullptr, progress_);
        result.statistics.baselineSeconds = elapsed(started);
        if (!cacheHit) {
            result.statistics.qualityRows = result.before.observations; result.statistics.cpuQualityRows = result.before.observations; ++result.statistics.completeEvaluations;
        }
        const Reconstruction beforePoses = scene_.poses();
        auto assess = [&]() {
            const auto gatesStarted = Clock::now(); auto failed = gates_(beforePoses, scene_.poses());
            result.statistics.gatesSeconds += elapsed(gatesStarted);
            if (!failed.empty()) { ++result.statistics.gateRejections; return failed; }
            const auto qualityStarted = Clock::now();
            const auto evaluation = evaluateSceneQuality(topology_, scene_, options_.gate, options_.allowance,
                options_.bundle.threads, nullptr, progress_, &result.before, options_.qualityGpu);
            result.statistics.candidateQualitySeconds += elapsed(qualityStarted);
            result.statistics.qualityRows += evaluation.quality.observations + evaluation.screenedGpuRows;
            result.statistics.gpuQualityRows += evaluation.gpuRows + evaluation.screenedGpuRows; result.statistics.cpuQualityRows += evaluation.cpuRows;
            if (!evaluation.complete) { ++result.statistics.boundedRejections; return std::vector<std::string>{evaluation.rejected}; }
            ++result.statistics.completeEvaluations; result.after = evaluation.quality;
            if (evaluation.gpuRows) {
                const auto confirmationStarted = Clock::now();
                result.after = sceneQuality(topology_, scene_, options_.gate, options_.allowance, options_.bundle.threads, nullptr, progress_);
                result.statistics.cpuConfirmationSeconds += elapsed(confirmationStarted);
                result.statistics.qualityRows += result.after.observations; result.statistics.cpuQualityRows += result.after.observations;
                ++result.statistics.completeEvaluations;
            }
            return qualityReasons(result.before, result.after);
        };
        try {
            started = Clock::now(); scene_.beginCandidate(); result.statistics.snapshotSeconds += elapsed(started);
            started = Clock::now();
            if (execute) execute(tasks, scene_, options_.bundle);
            else for (size_t i = 0; i < tasks.size(); ++i) {
                cancel::check(); auto& task = tasks[i]; task.reload(scene_);
                if(!task.problem.num_obs) {if(progress_)progress_("block-retained-only",i+1,tasks.size());continue;}
                task.solve(options_.bundle, options_.bundle.shared_ctx); task.apply(scene_);
                if (progress_) progress_("block-solve", i + 1, tasks.size());
            }
            result.statistics.blockSeconds = elapsed(started); started = Clock::now();
            result.intrinsics = refineSceneIntrinsics(topology_, scene_, options_.bundle, options_.gate, options_.allowance, 1e-3, progress_);
            result.statistics.intrinsicSeconds = elapsed(started); started = Clock::now();
            triangulateScene(topology_, scene_, options_.gate, options_.allowance, options_.bundle.threads, progress_);
            result.statistics.triangulationSeconds = elapsed(started);
            result.rejected = assess();
            if (result.rejected.empty()) { started = Clock::now(); result.snapshot = scene_.commit(roundPhase("canonical-scene-round-fullraw-v1")); result.statistics.snapshotSeconds += elapsed(started); result.accepted = true; result.alpha = 1; rememberQuality(result); return result; }
            started = Clock::now();
            auto proposal = scene_.sealCandidate("complete-coupled-proposal"); proposal.directory = scene_.abort("fullstep-rejected"); result.preserved.push_back(proposal.directory);
            result.statistics.snapshotSeconds += elapsed(started);
            for (int level = 1; level <= options_.backtrackSteps; ++level) {
                cancel::check(); const double alpha = std::ldexp(1.0, -level); started = Clock::now(); scene_.beginInterpolatedCandidate(proposal, alpha);
                result.statistics.snapshotSeconds += elapsed(started);
                auto failed = assess();
                if (failed.empty()) { started = Clock::now(); result.snapshot = scene_.commit(roundPhase("canonical-scene-round-fullraw-step-v1")); result.statistics.snapshotSeconds += elapsed(started); result.accepted = true; result.alpha = alpha; rememberQuality(result); return result; }
                started = Clock::now();
                result.preserved.push_back(scene_.preserveInterpolatedRejection(proposal, alpha, "partialstep-rejected-" + std::to_string(level)));
                result.statistics.snapshotSeconds += elapsed(started);
                for (const auto& reason : failed) result.rejected.push_back("alpha=" + std::to_string(alpha) + ":" + reason);
                if (std::filesystem::exists(result.preserved.back() / "points.bin")) {
                    result.rejected.push_back("native_step_archive_refused_full_candidate_retained");
                    if (progress_) progress_("evidence-archive-refused-preserved", level, options_.backtrackSteps);
                    result.after = result.before; return result;
                }
            }
            result.after = result.before; return result;
        } catch (...) { if (scene_.editing()) scene_.abort("round-exception"); throw; }
    }
private:
    void rememberQuality(const SceneRoundResult& result) { cachedQualityIdentity_ = result.snapshot.identity; cachedQuality_ = result.after; }
    std::string roundPhase(std::string phase) const { if (options_.roundOrdinal) phase += ".cohort-round=" + std::to_string(options_.roundOrdinal); return phase; }
    std::vector<std::string> qualityReasons(const Quality& before, const Quality& after) const {
        std::vector<std::string> result;
        if (!after.observations || after.observations != before.observations || after.retained_observations!=before.retained_observations ||
            after.directional_observations!=before.directional_observations || after.retained_observations>after.observations ||
            after.directional_observations>after.observations-after.retained_observations || !std::isfinite(after.cost) || !std::isfinite(after.finite_cost)) result.push_back("fullraw_coverage_or_nonfinite");
        if (after.invalid_projections > before.invalid_projections) result.push_back("invalid_projections_increased");
        if (after.finite_cost > before.finite_cost * (1 + 1e-8) + 1e-8) result.push_back("finite_visual_cost_increased");
        if (after.cost > before.cost * (1 + 1e-8) + 1e-8) result.push_back("visual_cost_increased");
        const bool descent = after.invalid_projections < before.invalid_projections || after.finite_cost < before.finite_cost - (std::abs(before.finite_cost) * 1e-12 + 1e-12) || after.cost < before.cost - (std::abs(before.cost) * 1e-12 + 1e-12);
        if (!descent) result.push_back("no_complete_raw_descent"); return result;
    }
    CanonicalScene& scene_; const ImmutableSceneTopology& topology_; Options options_; Gates gates_; ImmutableSceneTopology::Progress progress_;
    std::string cachedQualityIdentity_; Quality cachedQuality_;
};
}
