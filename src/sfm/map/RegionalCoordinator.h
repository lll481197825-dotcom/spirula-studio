#pragma once

#include "sfm/map/RegionalTopology.h"
#include "sfm/map/PrivateGeometry.h"
#include "sfm/map/RegionalAdjustment.h"
#include "sfm/core/Events.h"
#include <optional>

namespace sfm::regional {

inline Quality workingQuality(RegionalWorkingStore& store, const RegionalTopology& topology,
    double gate, int threads, std::vector<PhotoQuality>* photos = nullptr) {
    if (!(gate > 0) || !std::isfinite(gate)) throw std::invalid_argument("invalid working projection gate");
    using Key = RegionalWorkingStore::PointKey;
    using Raw = RegionalWorkingStore::RawObservation;
    struct Job { Vec3 xyz; std::vector<Raw> track; };
    std::vector<Job> jobs; jobs.reserve(4096);
    Quality result; result.finite_cost = 0;
    auto& pool = bacpu::Pool::get();
    const int workers = threads > 0 ? std::min(threads, pool.size()) : pool.size();
    const auto& poses = store.poses();
    auto flush = [&] {
        const size_t grain = 128, tasks = (jobs.size() + grain - 1) / grain;
        std::vector<Quality> costs(tasks); for (auto& cost : costs) cost.finite_cost = 0;
        std::vector<std::map<uint32_t, PhotoQuality>> perPhoto(photos ? tasks : 0);
        std::exception_ptr failure; std::mutex mutex;
        pool.run(int(tasks), workers, [&](int task, int) {
            try {
                cancel::check(); auto& quality = costs.at(size_t(task));
                const size_t end = std::min(jobs.size(), (size_t(task) + 1) * grain);
                for (size_t p = size_t(task) * grain; p < end; ++p) for (const auto& row : jobs[p].track) {
                    const auto& image = poses.images.at(row.image); const auto& camera = poses.cameras.at(image.camera_id);
                    if (!image.registered || !(camera.pixel_scale > 0)) throw std::runtime_error("working validation image lacks visual pose");
                    bool visible = true; const double error = reprojErrorAt(camera, image.pose, row.xy, jobs[p].xyz, &visible) / camera.pixel_scale;
                    if (!std::isfinite(error)) throw std::runtime_error("nonfinite working validation residual");
                    const double cost = error <= gate ? error * error : gate * (2 * error - gate);
                    ++quality.observations; quality.cost += cost;
                    if (visible) quality.finite_cost += cost; else ++quality.invalid_projections;
                    if (photos) {
                        auto& photo = perPhoto.at(size_t(task))[row.image];
                        ++photo.observations; photo.inliers += error <= gate; photo.invalid_projections += !visible; photo.error += error;
                    }
                }
            } catch (...) { std::lock_guard<std::mutex> lock(mutex); if (!failure) failure = std::current_exception(); }
        });
        if (failure) std::rethrow_exception(failure);
        for (const auto& cost : costs) accumulateQuality(result, cost);
        if (photos) for (const auto& task : perPhoto) for (const auto& item : task) {
            auto& photo = photos->at(item.first); photo.observations += item.second.observations;
            photo.inliers += item.second.inliers; photo.invalid_projections += item.second.invalid_projections; photo.error += item.second.error;
        }
        jobs.clear();
    };
    uint64_t observations = 0;
    for (size_t source = 0; source < store.sourceCount(); ++source)
        store.visitSourcePoints(source, [&](Key key, Vec3 xyz, const std::vector<Raw>& track) {
            cancel::check();
            if (!std::isfinite(xyz.norm())) throw std::runtime_error("nonfinite working validation point");
            if (!jobs.empty() && (jobs.size() == 4096 || observations + track.size() > 262144)) { flush(); observations = 0; }
            if (track.size() > 262144) throw std::length_error("working validation track exceeds bounded batch");
            jobs.push_back({topology.xyz(key, xyz), track}); observations += track.size();
        });
    if (!jobs.empty()) flush();
    if (!result.observations || !std::isfinite(result.cost) || !std::isfinite(result.finite_cost))
        throw std::runtime_error("invalid complete working projection ledger");
    return result;
}

struct CoordinationResult {
    Quality quality;
    double baseline = 0;
    size_t rounds = 0;
    bool converged = false;
    std::vector<std::string> rejected;
};

inline std::optional<uint64_t> acceptedCommonGeneration(const RegionalWorkingStore& store) {
    const auto& phase = store.generationPhase();
    if (phase.empty()) return {};
    const size_t split = phase.find(':');
    if (split == std::string::npos) throw std::runtime_error("unknown working parameter acceptance phase");
    const auto kind = phase.substr(0, split);
    if (kind != "joint-round-v1" && kind != "initial-seed-v1" &&
        kind != "joint-round-private-v2" && kind != "initial-seed-private-v2")
        throw std::runtime_error("unknown working parameter acceptance phase");
    const auto number = phase.substr(split + 1);
    if (number.empty() || number.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("invalid accepted common parameter generation");
    const auto generation = std::stoull(number);
    if (generation > store.generation()) throw std::runtime_error("common parameter checkpoint exceeds accepted working head");
    return generation;
}

inline std::set<uint32_t> changedProjectionImages(const Reconstruction& before, const Reconstruction& after) {
    std::set<uint32_t> cameras, images;
    for (const auto& item : after.cameras) {
        const auto& prior = before.cameras.at(item.first); const auto& next = item.second;
        std::array<double, 12> previous{}, current{};
        packIntrinsics(prior, previous.data()); packIntrinsics(next, current.data());
        if (prior.model != next.model || prior.width != next.width || prior.height != next.height ||
            prior.pixel_scale != next.pixel_scale || previous != current) cameras.insert(item.first);
    }
    for (const auto& item : after.images) {
        const auto& prior = before.images.at(item.first); const auto& next = item.second;
        if (!next.registered) continue;
        if (!prior.registered || prior.camera_id != next.camera_id || cameras.count(next.camera_id) ||
            prior.pose.R != next.pose.R || prior.pose.t.x != next.pose.t.x ||
            prior.pose.t.y != next.pose.t.y || prior.pose.t.z != next.pose.t.z) images.insert(item.first);
    }
    return images;
}

class RegionalCoordinator {
public:
    using Gates = std::function<std::vector<std::string>(const Reconstruction&, const Reconstruction&)>;
    RegionalCoordinator(RegionalWorkingStore& store, RegionalTopology& topology, const SfmConfig& cfg,
        PriorSource* gps, const RegionalPoseAnchors& held, size_t budget, const fs::path& directory, size_t taskPhotoLimit = 0)
        : store_(store), topology_(topology), cfg_(cfg), gps_(gps), held_(held), budget_(budget), directory_(directory), taskPhotoLimit_(taskPhotoLimit) {
        fs::create_directories(directory_);
    }
    CoordinationResult run(const std::vector<Region>& regions, const Gates& gates) {
        CoordinationResult result;
        const auto parameterPath = [&](uint64_t generation) { return directory_ / ("common-" + std::to_string(generation) + ".bin"); };
        if (const auto accepted = acceptedCommonGeneration(store_)) topology_.restoreAcceptedParameters(parameterPath(*accepted));
        reportPhase("validating_initial_observations");
        result.quality = workingQuality(store_, topology_, cfg_.max_error, cfg_.threads);
        const double baseline = result.quality.cost;
        result.baseline = baseline;
        VkContext context;
        for (size_t round = 0; round < size_t(cfg_.regional_iterations); ++round) {
            currentRound_ = round + 1; totalBlocks_ = regions.size(); currentBlock_ = 0;
            dynamicInterfaces_.resize(size_t(topology_.points())); dynamicInterfaceCount_ = 0;
            cancel::check(); const auto before = store_.poses(); const auto previous = result.quality;
            roundReferenceBytes_ = modelResidentBytes(before);
            const auto acceptedPath = parameterPath(store_.generation()); topology_.saveParameters(acceptedPath);
            const auto generation = store_.generation() + 1;
            const double generationStarted = now();
            store_.resetCounters(); store_.beginGeneration(); refreshHostBase();
            const double generationSeconds = now() - generationStarted;
            bool editing = true;
            try {
                std::string acceptedStepReport;
                CameraEstimates estimates;
                events::stage_begin(Stage::Refine, regions.size());
                const double started = now(); double prepareSeconds = 0, solveSeconds = 0; exchangeSeconds_ = 0; failedTasks_ = 0;
                size_t tasks = 0;
                for (size_t block = 0; block < regions.size(); ++block) {
                    currentBlock_ = block + 1;
                    std::set<uint32_t> images;
                    for (auto id : regions[block].core) {
                        const auto at = store_.poses().images.find(id);
                        if (at != store_.poses().images.end() && at->second.registered) images.insert(id);
                    }
                    solveBlock(images, context, estimates, prepareSeconds, solveSeconds, tasks);
                    events::progress(Stage::Refine, block + 1, regions.size());
                    slog::diag(slog::Tag::Map, "[regional-working] block %zu/%zu round %zu: %zu visual photos, %zu solver tasks\n",
                        block + 1, regions.size(), round + 1, images.size(), tasks);
                }
                {
                    Reconstruction cameras = store_.poses(); applyCameraEstimates(cameras, estimates);
                    for (const auto& item : cameras.cameras) store_.setCamera(item.second);
                }
                refreshHostBase();
                const double triangulationStarted = now();
                triangulateDynamicInterfaces();
                reportPhase("shared_triangulation");
                topology_.triangulate(store_.poses(), cfg_.max_error, cfg_.threads, [&](uint64_t done, uint64_t total) {
                    reportPhase("shared_triangulation", done, total);
                });
                topology_.synchronize(store_);
                const auto affectedImages = changedProjectionImages(before, store_.poses());
                refreshHostBase();
                const uint64_t affectedBytes = storage::scaleBytes(affectedImages.size(), sizeof(uint32_t) + 64);
                const uint64_t privateBase = storage::addBytes(hostBaseBytes_, storage::addBytes(affectedBytes, std::min<uint64_t>(128u << 20, budget_ / 16)));
                if (privateBase >= budget_) throw std::length_error("private geometry has no remaining aggregate host budget");
                privateProgress_ = {}; reportPhase("private_triangulation");
                const double privateStarted = now();
                const auto privateResult = refinePrivateGeometry(store_, topology_, store_.poses(), cfg_.max_error, cfg_.threads,
                    budget_ - privateBase, directory_ / "private-support", &affectedImages, [&](const PrivateGeometryProgress& progress) {
                        privateProgress_ = progress; reportPhase("private_triangulation");
                    });
                slog::diag(slog::Tag::Map, "[regional-working] private geometry %.3fs: %zu affected photos, %llu changed source points, %llu unresolved alias groups; allocation peak bound %llu bytes, complete affected raw ledger %s\n",
                    now() - privateStarted, affectedImages.size(), (unsigned long long)privateResult.changed,
                    (unsigned long long)privateResult.unresolved_alias_groups, (unsigned long long)privateResult.peakMemoryBytes,
                    privateResult.ledger.u8string().c_str());
                const double triangulationSeconds = now() - triangulationStarted;
                const double validationStarted = now();
                reportPhase("validating_original_observations");
                auto candidate = workingQuality(store_, topology_, cfg_.max_error, cfg_.threads);
                result.rejected = gates(store_.poses(), before);
                if (!privateGeometryAccepted(privateResult)) result.rejected.push_back("affected_private_original_observation_regression");
                uint32_t stop = qualityStop(candidate, previous, baseline, cfg_.regional_cost_tolerance,
                                           round + 1 == size_t(cfg_.regional_iterations));
                if (candidate.invalid_projections > previous.invalid_projections) result.rejected.push_back("invalid_projection_count_increased");
                if (candidate.finite_cost > previous.finite_cost * (1 + 1e-8) + 1e-8) result.rejected.push_back("finite_visual_cost_increased");
                if (candidate.cost > previous.cost * (1 + 1e-8) + 1e-8) result.rejected.push_back("total_original_observation_cost_increased");
                const auto prefix = directory_ / ("generation-" + std::to_string(generation));
                writeProjectionQuality(prefix.string() + "-projection.bin", candidate);
                atomicModel(prefix.string() + "-poses.bin", store_.poses());
                std::ostringstream report;
                for (const auto& reason : result.rejected) report << reason << '\n';
                const auto io = store_.counters();
                slog::diag(slog::Tag::Map, "[regional-working] round %zu: prepare %.3fs solve-attempts %.3fs exchange %.3fs triangulate %.3fs validate %.3fs total %.3fs, retried tasks %zu; logical raw %llu index %llu parameter bytes %llu, original model deserializations %llu\n",
                    round + 1, prepareSeconds, solveSeconds, exchangeSeconds_, triangulationSeconds, now() - validationStarted, now() - started, failedTasks_,
                    (unsigned long long)io.raw_bytes, (unsigned long long)io.index_bytes,
                    (unsigned long long)io.parameter_bytes, (unsigned long long)io.source_full_reads);
                if (!result.rejected.empty() || stop == 2) {
                    if (result.rejected.empty()) { result.rejected.push_back("projection_acceptance_failed"); report << result.rejected.back() << '\n'; }
                    resume::store(prefix.string() + "-rejected.txt", report.str());
                    const fs::path candidateCommon = prefix.string() + "-candidate-common.bin";
                    topology_.saveParameters(candidateCommon);
                    writeProjectionQuality(prefix.string() + "-full-step-projection.bin", candidate);
                    const auto retained = store_.abortGeneration(); editing = false;
                    topology_.loadParameters(acceptedPath);
                    bool acceptedStep = false;
                    if (privateGeometryAccepted(privateResult)) for (size_t attempt = 1; attempt <= 12; ++attempt) {
                        cancel::check(); const double alpha = std::ldexp(1., -int(attempt));
                        const auto stepPrefix = prefix.string() + "-step-" + std::to_string(attempt);
                        store_.beginGeneration(); editing = true;
                        store_.interpolateGeneration(retained, alpha);
                        topology_.interpolateParameters(acceptedPath, candidateCommon, alpha);
                        // Shared canonical parameters are also the coordinates materialized by the native Reader.
                        topology_.synchronize(store_);
                        refreshHostBase(); reportPhase("backtracking_complete_observations");
                        const auto trial = workingQuality(store_, topology_, cfg_.max_error, cfg_.threads);
                        const auto trialStop = qualityStop(trial, previous, baseline, cfg_.regional_cost_tolerance,
                            round + 1 == size_t(cfg_.regional_iterations));
                        auto reasons = gates(store_.poses(), before);
                        if (trial.invalid_projections > previous.invalid_projections) reasons.push_back("invalid_projection_count_increased");
                        if (trial.finite_cost > previous.finite_cost * (1 + 1e-8) + 1e-8) reasons.push_back("finite_visual_cost_increased");
                        if (trial.cost > previous.cost * (1 + 1e-8) + 1e-8) reasons.push_back("total_original_observation_cost_increased");
                        if (!(trial.cost < previous.cost && trial.finite_cost < previous.finite_cost))
                            reasons.push_back("no_complete_visual_descent");
                        writeProjectionQuality(stepPrefix + "-projection.bin", trial);
                        atomicModel(stepPrefix + "-poses.bin", store_.poses());
                        topology_.saveParameters(stepPrefix + "-common.bin");
                        slog::diag(slog::Tag::Map, "[regional-working] full-observation step %.9g: %llu raw, invalid %llu -> %llu, finite %.17g -> %.17g, total %.17g -> %.17g, gate failures %zu\n",
                            alpha, (unsigned long long)trial.observations,
                            (unsigned long long)previous.invalid_projections, (unsigned long long)trial.invalid_projections,
                            previous.finite_cost, trial.finite_cost, previous.cost, trial.cost, reasons.size());
                        if (reasons.empty() && trialStop != 2) {
                            candidate = trial; stop = trialStop; result.rejected.clear(); acceptedStep = true;
                            acceptedStepReport = std::to_string(alpha) + "\n" + retained.u8string() + "\n";
                            resume::store(prefix.string() + "-selected-step.txt", acceptedStepReport);
                            writeProjectionQuality(prefix.string() + "-projection.bin", candidate);
                            atomicModel(prefix.string() + "-poses.bin", store_.poses());
                            break;
                        }
                        std::ostringstream rejected;
                        for (const auto& reason : reasons) rejected << reason << '\n';
                        resume::store(stepPrefix + "-rejected.txt", rejected.str());
                        store_.abortGeneration(); editing = false; topology_.loadParameters(acceptedPath);
                    }
                    if (!acceptedStep) {
                        result.rejected.push_back("complete_objective_backtracking_exhausted");
                        reportPhase("round_rejected"); events::stage_end(Stage::Refine); return result;
                    }
                }
                topology_.saveParameters(parameterPath(generation)); store_.commitGeneration("joint-round-private-v2:" + std::to_string(generation));
                editing = false;
                if (!acceptedStepReport.empty()) resume::store(prefix.string() + "-accepted-step.txt", acceptedStepReport);
                result.quality = candidate; ++result.rounds; result.converged = stop == 1;
                reportPhase("round_accepted");
                slog::diag(slog::Tag::Map, "[regional-working] accepted generation %llu: copy/start %.3fs, entire transaction %.3fs, accepted complete observations %llu\n",
                    (unsigned long long)generation, generationSeconds, now() - generationStarted, (unsigned long long)candidate.observations);
                events::stage_end(Stage::Refine);
                if (result.converged) break;
            } catch (...) {
                const auto failure = std::current_exception();
                if (editing) {
                    try { topology_.saveParameters(directory_ / ("generation-" + std::to_string(generation) + "-error-common.bin")); }
                    catch (const std::exception& error) { slog::diag(slog::Tag::Map, "[regional-working] common candidate preservation error: %s\n", error.what()); }
                    store_.abortGeneration(); topology_.loadParameters(acceptedPath);
                }
                reportPhase("coordination_error");
                events::stage_end(Stage::Refine); std::rethrow_exception(failure);
            }
        }
        return result;
    }
private:
    void triangulateDynamicInterfaces() {
        if (!dynamicInterfaceCount_) return;
        using Key = RegionalWorkingStore::PointKey;
        using Raw = RegionalWorkingStore::RawObservation;
        struct Seed { Vec3 xyz; uint64_t observations = 0; };
        reportPhase("dynamic_interface_triangulation");
        std::vector<Key> keys;
        topology_.visitCanonicalMembers(dynamicInterfaces_, store_, [&](Key key) {
            if (keys.size() == keys.capacity()) {
                const size_t cap = std::max<size_t>(4096, keys.size() * 2);
                requireHostMemory(storage::scaleBytes(keys.capacity() + cap, sizeof(Key)));
                keys.reserve(cap);
            }
            keys.push_back(key);
        });
        const auto directory = directory_ / ("dynamic-" + std::to_string(store_.generation() + 1));
        fs::create_directories(directory); const auto rawPath = directory / "observations.bin";
        std::map<uint64_t, Seed> seeds;
        {
            std::ofstream out(rawPath, std::ios::binary);
            store_.visitSelectedPoints(keys, [&](Key key, Vec3 value, const std::vector<Raw>& track) {
                const auto root = topology_.canonical(key); auto& seed = seeds[root];
                requireHostMemory(storage::scaleBytes(keys.capacity(), sizeof(Key)) + storage::scaleBytes(seeds.size(), sizeof(std::pair<const uint64_t, Seed>) + 64) + budget_ / 32);
                const auto previous = seed.observations;
                seed.observations = storage::addBytes(seed.observations, track.size());
                const double fraction = double(track.size()) / std::max<uint64_t>(1, seed.observations);
                seed.xyz = previous ? seed.xyz * (1 - fraction) + value * fraction : value;
                for (const auto& row : track) record(out, Observation{root, (uint64_t(row.image) << 32) | row.feature, row.xy});
            });
            out.flush(); if (!out) throw std::runtime_error("cannot save complete dynamic interface observations");
        }
        const auto sorted = sortObservations(rawPath, directory / "by-point", budget_ / 32);
        std::ifstream in(sorted, std::ios::binary); Observation row; bool have = nextRecord(in, row);
        struct Job { uint64_t root; std::vector<Observation> track; };
        std::vector<Job> jobs; uint64_t bytes = 0;
        auto flush = [&] {
            auto& pool = bacpu::Pool::get(); const int workers = cfg_.threads > 0 ? std::min(cfg_.threads, pool.size()) : pool.size();
            pool.runChecked(int(jobs.size()), workers, [&](int job, int) {
                cancel::check(); const auto& item = jobs.at(size_t(job));
                auto& value = seeds.at(item.root).xyz; value = refineUnifiedPoint(value, item.track, store_.poses(), cfg_.max_error);
            });
            jobs.clear(); bytes = 0;
        };
        while (have) {
            Job job{row.key, {}}; uint64_t previous = kInvalidPoint3D; Vec2 previousXY{};
            do {
                if (row.point != previous) { job.track.push_back(row); previous = row.point; previousXY = row.xy; }
                else if (row.xy.x != previousXY.x || row.xy.y != previousXY.y)
                    throw std::runtime_error("dynamic interface has conflicting original feature coordinates");
                if (job.track.size() * sizeof(Observation) > budget_ / 64) throw std::length_error("dynamic interface track exceeds working budget");
                have = nextRecord(in, row);
            } while (have && row.key == job.root);
            const auto needed = storage::scaleBytes(job.track.capacity(), sizeof(Observation));
            if (!jobs.empty() && bytes + needed > budget_ / 64) flush();
            bytes += needed; jobs.push_back(std::move(job));
        }
        if (!jobs.empty()) flush();
        requireHostMemory(storage::scaleBytes(keys.capacity(), sizeof(Key)) + storage::scaleBytes(seeds.size(), sizeof(std::pair<const uint64_t, Seed>) + 64) + storage::scaleBytes(keys.size(), sizeof(std::pair<Key, Vec3>)));
        std::vector<std::pair<Key, Vec3>> updates; updates.reserve(keys.size());
        for (const auto& key : keys) updates.emplace_back(key, seeds.at(topology_.canonical(key)).xyz);
        store_.applyPointUpdates(updates);
        slog::diag(slog::Tag::Map, "[regional-working] dynamic interfaces: %zu canonical points, %zu original source aliases, complete supporting observations retained\n",
            seeds.size(), keys.size());
    }
    void reportPhase(const char* phase, uint64_t groupsDone = 0, uint64_t groupsTotal = 0) const {
        uint64_t observations = 0;
        for (size_t source = 0; source < store_.sourceCount(); ++source)
            observations = storage::addBytes(observations, store_.sourceObservationCount(source));
        const auto& io = store_.counters(); std::ostringstream out;
        out << "{\"phase\":\"" << phase << "\",\"generation\":" << store_.generation()
            << ",\"round\":" << currentRound_ << ",\"block\":" << currentBlock_ << ",\"blocks\":" << totalBlocks_
            << ",\"groups_done\":" << groupsDone << ",\"groups_total\":" << groupsTotal
            << ",\"private_phase\":\"" << privateProgress_.phase << "\",\"private_source\":" << privateProgress_.source
            << ",\"private_sources\":" << privateProgress_.sourceTotal << ",\"private_points_done\":" << privateProgress_.pointsDone
            << ",\"private_points_total\":" << privateProgress_.pointsTotal << ",\"private_groups_done\":" << privateProgress_.groupsDone
            << ",\"private_groups_total\":" << privateProgress_.groupsTotal
            << ",\"source_models\":" << store_.sourceCount() << ",\"stage_registered_poses\":" << store_.poses().numRegistered()
            << ",\"original_observations\":" << observations << ",\"logical_raw_bytes\":" << io.raw_bytes
            << ",\"logical_index_bytes\":" << io.index_bytes << ",\"logical_parameter_bytes\":" << io.parameter_bytes
            << ",\"photo_cache_builds\":" << io.photo_cache_builds << ",\"photo_cache_build_seconds\":" << io.photo_cache_build_seconds
            << ",\"photo_cache_input_bytes\":" << io.photo_cache_input_bytes << ",\"photo_cache_written_bytes\":" << io.photo_cache_written_bytes
            << ",\"host_budget_bytes\":" << budget_ << ",\"host_phase_base_bytes\":" << hostBaseBytes_
            << ",\"host_available_bytes\":" << spirula::availableRamBytes()
            << ",\"full_capture_verified\":false}\n";
        resume::store(directory_ / "working-progress.json", out.str());
    }
    void solveBlock(const std::set<uint32_t>& images, VkContext& context, CameraEstimates& estimates,
        double& prepareSeconds, double& solveSeconds, size_t& tasks) {
        if (images.empty()) return;
        if (images.size() > 1 && ((taskPhotoLimit_ && images.size() > taskPhotoLimit_) || forecastBytes(images) > budget_ / 6)) {
            splitBlock(images, context, estimates, prepareSeconds, solveSeconds, tasks); return;
        }
        const double started = now();
        reportPhase("gathering_block");
        RegionalWorkingStore::Gather gathered;
        try { gathered = store_.gather(images, {}, false); }
        catch (const std::length_error&) { prepareSeconds += now() - started; ++failedTasks_; splitBlock(images, context, estimates, prepareSeconds, solveSeconds, tasks); return; }
        if (modelResidentBytes(gathered.model) > budget_ / 6) {
            gathered = {}; prepareSeconds += now() - started; ++failedTasks_; splitBlock(images, context, estimates, prepareSeconds, solveSeconds, tasks); return;
        }
        if (!fitsHostMemory(mergeMemory(gathered))) {
            gathered = {}; prepareSeconds += now() - started; ++failedTasks_;
            splitBlock(images, context, estimates, prepareSeconds, solveSeconds, tasks); return;
        }
        Reconstruction model = std::move(gathered.model); std::set<uint64_t> fixed; std::set<uint32_t> fixedImages;
        std::map<uint64_t, std::vector<RegionalWorkingStore::PointKey>> keys;
        std::map<uint64_t, Point3D> points;
        while (!model.points3D.empty()) {
            auto node = model.points3D.extract(model.points3D.begin()); const auto id = node.key();
            const auto key = gathered.pointKeys.at(id); const uint64_t canonical = topology_.canonical(key);
            const auto at = points.find(canonical);
            if (at == points.end()) {
                node.key() = canonical; node.mapped().xyz = topology_.xyz(key, node.mapped().xyz);
                points.insert(std::move(node));
            } else {
                auto& merged = at->second.track; const auto& track = node.mapped().track;
                merged.insert(merged.end(), track.begin(), track.end());
            }
            keys[canonical].push_back(key);
            if (topology_.shared(key) || gathered.boundaryPoints.count(id)) fixed.insert(canonical);
        }
        gathered.pointKeys.clear(); gathered.boundaryPoints.clear();
        for (auto& item : points) {
            auto& track = item.second.track;
            std::sort(track.begin(), track.end(), [](const TrackElement& a, const TrackElement& b) { return std::tie(a.image_id, a.point2D_idx) < std::tie(b.image_id, b.point2D_idx); });
            track.erase(std::unique(track.begin(), track.end(), [](const TrackElement& a, const TrackElement& b) { return a.image_id == b.image_id && a.point2D_idx == b.point2D_idx; }), track.end());
            for (const auto& observation : track) model.images.at(observation.image_id).point3D_ids.at(observation.point2D_idx) = item.first;
        }
        model.points3D = std::move(points);
        for (auto& item : model.images) {
            const auto at = held_.find(item.first);
            if (at != held_.end()) { item.second.pose = at->second; fixedImages.insert(item.first); }
        }
        BundleOptions options; options.real = realCfgFromName(cfg_.mapper.ba_real); options.max_iters = 20;
        options.loss = cfg_.mapper.ba_loss; options.loss_param = float(cfg_.mapper.ba_loss_param);
        options.solver = "cg"; options.device_selector = cfg_.mapper.device_selector; options.device = cfg_.mapper.device;
        options.threads = cfg_.threads; options.shared_ctx = &context; options.fixed_points = &fixed; options.fixed_images = &fixedImages;
        options.refine_intrinsics = model.numRegistered() >= 2;
        options.refine_principal_point = cfg_.mapper.refine_principal_point; options.refine_extra_params = cfg_.mapper.refine_extra_params;
        options.host_budget_bytes = budget_ / 3; options.device_budget_bytes = budget_ / 3; options.over_budget_throws = true;
        const auto priors = gps_ ? regionalGpsPriors(model, *gps_) : PosePriors{}; options.priors = gps_ ? &priors : nullptr;
        SolverStats stats; options.stats = &stats;
        prepareSeconds += now() - started; const double solveStarted = now();
        reportPhase("block_ba");
        auto retry = [&] {
            solveSeconds += now() - solveStarted; ++failedTasks_;
            model = {}; gathered = {}; keys.clear(); fixed.clear(); fixedImages.clear();
            splitBlock(images, context, estimates, prepareSeconds, solveSeconds, tasks);
        };
        try { runGlobalBA(model, options); }
        catch (const BAOverBudget&) { retry(); return; }
        catch (const BAIndexCapacity&) { retry(); return; }
        solveSeconds += now() - solveStarted; ++tasks;
        slog::diag(slog::Tag::Map, "[regional-working] solver task: %zu photos, %zu points, %.1f MiB device buffers, host/device budgets %.1f/%.1f MiB\n",
            images.size(), model.points3D.size(), stats.vram_mb, options.host_budget_bytes / double(1u << 20), options.device_budget_bytes / double(1u << 20));
        for (const auto& item : keys) if (fixed.count(item.first) && !topology_.shared(item.second.front()) && !dynamicInterfaces_.at(size_t(item.first))) {
            dynamicInterfaces_.at(size_t(item.first)) = true; ++dynamicInterfaceCount_;
        }
        if (!std::isfinite(stats.final_cost) || stats.final_cost > stats.initial_cost * (1 + 1e-8) + 1e-8)
            throw std::runtime_error("working block BA did not decrease its combined visual/prior objective");
        const double exchangeStarted = now();
        for (const auto& item : model.images) store_.setPose(item.first, item.second.pose);
        std::vector<std::pair<RegionalWorkingStore::PointKey, Vec3>> updates;
        for (const auto& item : model.points3D) if (!fixed.count(item.first))
            for (const auto& key : keys.at(item.first)) updates.emplace_back(key, item.second.xyz);
        store_.applyPointUpdates(updates); addCameraEstimates(estimates, model);
        exchangeSeconds_ += now() - exchangeStarted;
    }
    void refreshHostBase() {
        hostBaseBytes_ = storage::addBytes(storage::addBytes(topology_.memoryBytes(), store_.residentBytes()),
            storage::addBytes(dynamicInterfaces_.bytes(), roundReferenceBytes_));
    }
    bool fitsHostMemory(uint64_t extra) const {
        const uint64_t margin = std::min<uint64_t>(128u << 20, budget_ / 16);
        return storage::addBytes(hostBaseBytes_, storage::addBytes(extra, margin)) <= budget_;
    }
    void requireHostMemory(uint64_t extra) const {
        if (!fitsHostMemory(extra)) throw std::length_error("regional phase exceeds aggregate host budget");
    }
    uint64_t mergeMemory(const RegionalWorkingStore::Gather& gathered) const {
        const uint64_t count = gathered.model.points3D.size(); uint64_t tracks = 0;
        for (const auto& point : gathered.model.points3D) tracks = storage::addBytes(tracks, point.second.track.capacity());
        using Key = RegionalWorkingStore::PointKey;
        const uint64_t maps = sizeof(std::pair<const uint64_t, Key>) + 64 + sizeof(std::pair<const uint64_t, std::vector<Key>>) + 64 + 2 * sizeof(Key) + sizeof(uint64_t) + 64;
        return storage::addBytes(modelResidentBytes(gathered.model), storage::addBytes(storage::scaleBytes(count, maps),
            storage::addBytes(storage::scaleBytes(tracks, 2 * sizeof(TrackElement)), 2 * (budget_ / 3))));
    }
    uint64_t forecastBytes(const std::set<uint32_t>& images) const {
        // Metadata-only forecast avoids gathering a known oversized task. Actual capacity guards remain authoritative.
        uint64_t bytes = sizeof(Reconstruction), observations = 0;
        double points = 0;
        for (auto image : images) {
            uint64_t features = 0;
            for (size_t source = 0; source < store_.sourceCount(); ++source)
                features = std::max(features, store_.sourcePhotoFeatureCount(source, image));
            const auto& value = store_.poses().images.at(image);
            bytes = storage::addBytes(bytes, storage::scaleBytes(features, sizeof(Vec2) + sizeof(uint64_t)));
            bytes = storage::addBytes(bytes, sizeof(std::pair<const uint32_t, Image>) + 64 + value.name.capacity() + 1);
        }
        for (size_t source = 0; source < store_.sourceCount(); ++source) {
            uint64_t selected = 0;
            for (auto image : images) selected = storage::addBytes(selected, store_.sourcePhotoObservationCount(source, image));
            observations = storage::addBytes(observations, selected);
            points += double(selected) * double(store_.sourcePointCount(source)) / std::max<uint64_t>(1, store_.sourceObservationCount(source));
        }
        bytes = storage::addBytes(bytes, storage::scaleBytes(observations, sizeof(TrackElement)));
        const uint64_t predicted = uint64_t(std::ceil(points));
        return storage::addBytes(bytes, storage::scaleBytes(predicted, sizeof(std::pair<const uint64_t, Point3D>) + 64));
    }
    void splitBlock(const std::set<uint32_t>& images, VkContext& context, CameraEstimates& estimates,
        double& prepareSeconds, double& solveSeconds, size_t& tasks) {
        if (images.size() <= 1) throw std::length_error("regional working subgraph cannot fit the configured solver budget");
        auto middle = images.begin(); std::advance(middle, images.size() / 2);
        solveBlock(std::set<uint32_t>(images.begin(), middle), context, estimates, prepareSeconds, solveSeconds, tasks);
        solveBlock(std::set<uint32_t>(middle, images.end()), context, estimates, prepareSeconds, solveSeconds, tasks);
    }
    RegionalWorkingStore& store_; RegionalTopology& topology_; const SfmConfig& cfg_; PriorSource* gps_;
    const RegionalPoseAnchors& held_; size_t budget_; fs::path directory_;
    size_t currentRound_ = 0, currentBlock_ = 0, totalBlocks_ = 0, taskPhotoLimit_ = 0, failedTasks_ = 0;
    double exchangeSeconds_ = 0;
    topology_storage::Bits dynamicInterfaces_;
    PrivateGeometryProgress privateProgress_{};
    uint64_t dynamicInterfaceCount_ = 0, roundReferenceBytes_ = 0, hostBaseBytes_ = 0;
};

} // namespace sfm::regional
