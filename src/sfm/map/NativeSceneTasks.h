#pragma once
#include "sfm/map/SceneCoordinator.h"
#include "sfm/map/RegionalPlan.h"
#include "sfm/map/AtomScheduling.h"
#include "core/HostMemory.h"
#include <thread>

namespace sfm::regional {
class NativeSceneTasks {
public:
    struct Statistics { uint64_t prepares = 0, cacheHits = 0, evictions = 0, splits = 0, completed = 0, gpuCompleted = 0, contextCreates = 0, contextEvictions = 0, retainedOnlyTasks = 0, retainedOnlyRows = 0; uint32_t peakWorkers = 0, peakContexts = 0; uint64_t peakTemplates = 0, peakHostCredit = 0, peakDeviceCredit = 0;
        uint64_t lastFixed = 0, lastTemplates = 0, lastLiveContexts = 0, lastFreeRam = 0, lastFreeCommit = 0, lastSafety = 0, lastCapacity = 0, lastFutureSolver = 0, lastFutureContexts = 0;
        double prepareSeconds = 0, contextSeconds = 0, reloadSeconds = 0, applySeconds = 0, workerSeconds = 0, solverInitSeconds = 0, solverComputeSeconds = 0, waveSeconds = 0; };
    NativeSceneTasks(const ImmutableSceneTopology& index, CanonicalScene& scene, const std::vector<Region>& plan,
                     const std::map<uint32_t, Pose>& sourceAnchors, BundleOptions options, size_t budget,
                     ImmutableSceneTopology::Progress progress = {}, uint64_t externalReserve = 0)
        : index_(index), scene_(scene), options_(std::move(options)), budget_(budget), progress_(std::move(progress)), externalReserve_(externalReserve) {
        progress_ = scene_index_detail::throttledProgress(std::move(progress_));
        if (budget < (32u << 20)) throw std::length_error("native scene tasks require 32 MiB aggregate budget");
        std::vector<uint32_t> owners(size_t(index.imageSlots()), UINT32_MAX);
        for (size_t block = 0; block < plan.size(); ++block) for (uint32_t image : plan[block].core) { if (image >= owners.size() || owners[image] != UINT32_MAX) throw std::runtime_error("native scene task photo ownership is not unique"); owners[image] = uint32_t(block); }
        for (const auto& image : scene.poses().images) if (image.second.registered && (image.first >= owners.size() || owners[image.first] == UINT32_MAX)) throw std::runtime_error("registered scene photo has no block owner");
        for (size_t block = 0; block < plan.size(); ++block) {
            Descriptor task; task.id = nextId_++; task.owner = uint32_t(block);
            for (uint32_t image : plan[block].core) { const auto found = scene.poses().images.find(image); if (found != scene.poses().images.end() && found->second.registered && !sourceAnchors.count(image)) task.core.push_back(image); }
            for (uint32_t image : plan[block].images) { const auto found = scene.poses().images.find(image); if (found != scene.poses().images.end() && found->second.registered && (owners.at(image) != block || sourceAnchors.count(image))) task.anchors.insert(image); }
            if (!task.core.empty()) descriptors_.push_back(std::move(task));
        }
    }
    SceneCoordinator::ExecuteBlocks execute() { return [this](auto& supplied, auto& scene, const auto& options) {
        if (&scene != &scene_) throw std::runtime_error("native task executor scene changed");
        if (!supplied.empty()) throw std::runtime_error("native task executor requires descriptor ownership, not unrelated supplied tasks");
        options_.loss = options.loss; options_.loss_param = options.loss_param; executeRound();
    }; }
    const Statistics& statistics() const { return stats_; }
private:
    struct Descriptor { uint64_t id = 0; uint32_t owner = 0; std::vector<uint32_t> core; std::set<uint32_t> anchors; };
    struct Cached { std::unique_ptr<CanonicalBundle> task; uint64_t tick = 0; };
    struct Job { size_t descriptor = 0; CanonicalBundle* task = nullptr; atom_scheduling::Job forecast; RealCfg solvedReal = RealCfg::CPU; BundleRun run; double workerSeconds = 0; };
    using Clock = std::chrono::steady_clock;
    static double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
    static constexpr uint64_t contextHostReserve = 640ull << 20;
    uint64_t cachedBytes() const { uint64_t bytes = 0; for (const auto& item : cache_) bytes = storage::addBytes(bytes, item.second.task->residentBytes()); return bytes; }
    uint64_t metadataBytes() const { uint64_t bytes = sizeof(*this); for (const auto& task : descriptors_) bytes = storage::addBytes(bytes, storage::scaleBytes(task.core.capacity(), sizeof(uint32_t)) + storage::scaleBytes(task.anchors.size(), 64)); return bytes; }
    uint64_t baseBytes() const { return storage::addBytes(storage::addBytes(scene_.residentBytes(), index_.residentBytes()), storage::addBytes(metadataBytes() + (8u << 20), externalReserve_)); }
    uint64_t fixedBytes() const { return storage::addBytes(baseBytes(), storage::scaleBytes(contexts_.size(), contextHostReserve)); }
    uint64_t hostCapacity() {
        uint64_t capacity = budget_; uint64_t available = spirula::availableRamBytes();
        stats_.lastFreeRam = available; stats_.lastFreeCommit = 0;
#ifdef _WIN32
        MEMORYSTATUSEX memory{}; memory.dwLength = sizeof(memory);
        if (GlobalMemoryStatusEx(&memory) && memory.ullAvailPageFile) { stats_.lastFreeCommit = memory.ullAvailPageFile; available = available ? std::min<uint64_t>(available, memory.ullAvailPageFile) : memory.ullAvailPageFile; }
#endif
        stats_.lastFixed = baseBytes(); stats_.lastTemplates = cachedBytes(); stats_.lastLiveContexts = storage::scaleBytes(contexts_.size(), contextHostReserve);
        stats_.lastSafety = available ? std::min<uint64_t>(available / 8, 512u << 20) : 0;
        const uint64_t tracked = storage::addBytes(storage::addBytes(stats_.lastFixed, stats_.lastTemplates), stats_.lastLiveContexts);
        if (available) capacity = std::min<uint64_t>(capacity, storage::addBytes(tracked, available - stats_.lastSafety));
        stats_.lastCapacity = capacity;
        return capacity;
    }
    void evictTo(uint64_t desired, const std::set<uint64_t>& held) {
        while (cachedBytes() > desired) {
            auto victim = cache_.end(); for (auto it = cache_.begin(); it != cache_.end(); ++it) if (!held.count(it->first) && (victim == cache_.end() || it->second.tick < victim->second.tick)) victim = it;
            if (victim == cache_.end()) break; cache_.erase(victim); ++stats_.evictions;
        }
    }
    void split(size_t at) {
        auto original = descriptors_.at(at); if (original.core.size() < 2) throw std::length_error("one-photo complete native task exceeds physical resource capacity");
        cache_.erase(original.id); const auto middle = original.core.begin() + original.core.size() / 2;
        Descriptor a = original, b = original; a.id = nextId_++; b.id = nextId_++; a.core.assign(original.core.begin(), middle); b.core.assign(middle, original.core.end());
        descriptors_[at] = std::move(a); descriptors_.insert(descriptors_.begin() + at + 1, std::move(b)); ++stats_.splits;
        if (progress_) progress_("task-capacity-split", stats_.splits, descriptors_.size());
    }
    CanonicalBundle& prepare(size_t at, uint64_t capacity, const std::set<uint64_t>& held) {
        const auto& descriptor = descriptors_.at(at); auto found = cache_.find(descriptor.id);
        if (found != cache_.end()) { found->second.tick = ++tick_; ++stats_.cacheHits; return *found->second.task; }
        const uint64_t fixed = fixedBytes(); if (fixed >= capacity) throw std::length_error("scene authority/index consume native task budget");
        evictTo((capacity - fixed) / 3, held);
        const uint64_t current = cachedBytes(); if (current >= capacity - fixed) throw std::length_error("held native templates consume task allowance");
        if (progress_) progress_("native-task-prepare", 0, descriptor.core.size());
        slog::diag(slog::Tag::Map, "[native-scene] prepare task %llu: %zu mutable photos, %zu fixed support photos, allowance %.1f MiB\n", (unsigned long long)descriptor.id, descriptor.core.size(), descriptor.anchors.size(), (capacity - fixed - current) / 1048576.);
        const auto started = Clock::now(); auto task = std::make_unique<CanonicalBundle>(); task->prepare(index_, scene_, descriptor.core, descriptor.owner, descriptor.anchors, options_, size_t(capacity - fixed - current));
        const double elapsed = seconds(started); stats_.prepareSeconds += elapsed;
        slog::diag(slog::Tag::Map, "[native-scene] layout ready task %llu: %u photos/%u geometric points/%u geometric raw rows, %llu retained rows/%llu authority rows, %.3fs\n", (unsigned long long)descriptor.id, task->problem.num_images, task->problem.num_points, task->problem.num_obs, (unsigned long long)task->retained_observations, (unsigned long long)task->authority_observations, elapsed);
        auto& inserted = cache_[descriptor.id]; inserted.task = std::move(task); inserted.tick = ++tick_; ++stats_.prepares;
        stats_.peakTemplates = std::max(stats_.peakTemplates, cachedBytes()); return *inserted.task;
    }
    void executeRound() {
        const uint32_t hardware = std::max<uint32_t>(1, std::thread::hardware_concurrency());
        uint32_t workerCeiling = options_.threads > 0 ? std::min<uint32_t>(hardware, uint32_t(options_.threads)) : hardware;
        const auto capabilities = options_.real == RealCfg::CPU ? VkDeviceCaps{} : VkContext::probeCaps(deviceOnlyOpt(options_.device, options_.device_selector));
        options_.real = pickRealForDevice(options_.real, capabilities); const bool gpu = options_.real != RealCfg::CPU;
        if (!gpu) workerCeiling = 1;
        VkContextOptions contextOptions; contextOptions.selector = options_.device_selector; contextOptions.deviceIndex = options_.device;
        contextOptions.needFloat64 = options_.real == RealCfg::F64; contextOptions.needFloatAtomics = options_.real != RealCfg::CPU && options_.real != RealCfg::DF64; contextOptions.needInt64Atomics = options_.real == RealCfg::DF64;
        while (contexts_.size() > (gpu ? workerCeiling : 0)) { contexts_.pop_back(); ++stats_.contextEvictions; }
        size_t next = 0; uint64_t completed = 0;
        while (next < descriptors_.size()) {
            cancel::check(); uint64_t capacity = hostCapacity(); evictTo(capacity > fixedBytes() ? (capacity - fixedBytes()) / 3 : 0, {});
            std::vector<Job> jobs; std::set<uint64_t> held; uint64_t deviceUsed = 0, solverHost = 0;
            VkDeviceMemoryCapacity device; if (!contexts_.empty()) device = contexts_.front()->deviceMemoryCapacity();
            for (size_t at = next; at < descriptors_.size() && jobs.size() < workerCeiling;) {
                CanonicalBundle* task = nullptr;
                try { task = &prepare(at, capacity, held); }
                catch (const BAOverBudget&) { if (!jobs.empty()) break; if (contexts_.size() > 1) { contexts_.pop_back(); ++stats_.contextEvictions; capacity = hostCapacity(); device = contexts_.front()->deviceMemoryCapacity(); continue; } split(at); continue; }
                catch (const std::length_error&) { if (!jobs.empty()) break; if (contexts_.size() > 1) { contexts_.pop_back(); ++stats_.contextEvictions; capacity = hostCapacity(); device = contexts_.front()->deviceMemoryCapacity(); continue; } split(at); continue; }
                if (!task->problem.num_obs) {
                    if (!jobs.empty()) break;
                    ++stats_.completed; ++stats_.retainedOnlyTasks; stats_.retainedOnlyRows = storage::addBytes(stats_.retainedOnlyRows, task->retained_observations);
                    ++completed; ++next; ++at;
                    slog::diag(slog::Tag::Map, "[native-scene] task has no geometric constraints; retained %llu raw rows without a BA solve\n", (unsigned long long)task->retained_observations);
                    if (progress_) progress_("native-retained-only", completed, descriptors_.size()); continue;
                }
                const uint64_t arrays = task->residentBytes();
                const auto footprint = BundleSolver::forecastDeviceBuffers(task->problem, bundleSolverOptions(options_));
                const uint64_t host = storage::addBytes(std::max<uint64_t>(1u << 20, task->forecastHostBytes > arrays ? task->forecastHostBytes - arrays : 0), footprint.hostTables);
                if (gpu && contexts_.empty()) {
                    if (storage::addBytes(fixedBytes() + cachedBytes(), storage::addBytes(host, contextHostReserve)) > capacity) { split(at); continue; }
                    if (progress_) progress_("native-context-init", 0, 1); const auto started = Clock::now();
                    auto context = std::make_unique<VkContext>(); context->init(contextOptions); contexts_.push_back(std::move(context)); ++stats_.contextCreates; stats_.contextSeconds += seconds(started);
                    if (progress_) progress_("native-context-ready", 1, 1);
                    device = contexts_.front()->deviceMemoryCapacity(); capacity = hostCapacity();
                }
                uint64_t deviceCapacity = gpu && device.available_bytes > device.reserve_bytes ? device.available_bytes - device.reserve_bytes : 0;
                if (gpu && options_.device_budget_bytes) deviceCapacity = std::min<uint64_t>(deviceCapacity, options_.device_budget_bytes);
                const size_t newContexts = gpu && jobs.size() + 1 > contexts_.size() ? jobs.size() + 1 - contexts_.size() : 0;
                const uint64_t contextHost = storage::scaleBytes(newContexts, contextHostReserve);
                const uint64_t contextDevice = storage::scaleBytes(newContexts, 128u << 20);
                const bool fits = storage::addBytes(fixedBytes() + cachedBytes(), solverHost + host + contextHost) <= capacity &&
                    (!gpu || storage::addBytes(deviceUsed, footprint.upper + contextDevice) <= deviceCapacity);
                if (!fits) { if (!jobs.empty()) break; if (contexts_.size() > 1) { contexts_.pop_back(); ++stats_.contextEvictions; capacity = hostCapacity(); device = contexts_.front()->deviceMemoryCapacity(); continue; } split(at); continue; }
                Job job; job.descriptor = at; job.task = task; job.forecast.credit = {host, footprint.upper}; job.forecast.solver_host = host; job.forecast.solver_device = footprint.upper;
                jobs.push_back(job); held.insert(descriptors_[at].id); solverHost += host; deviceUsed += footprint.upper; ++at;
            }
            if (jobs.empty()) { if (next == descriptors_.size()) break; throw std::length_error("native scene resource admission made no progress"); }
            stats_.lastFixed = baseBytes(); stats_.lastTemplates = cachedBytes(); stats_.lastLiveContexts = storage::scaleBytes(contexts_.size(), contextHostReserve); stats_.lastFutureSolver = solverHost;
            stats_.lastFutureContexts = gpu && jobs.size() > contexts_.size() ? storage::scaleBytes(jobs.size() - contexts_.size(), contextHostReserve) : 0;
            if (progress_) progress_("native-task-reload", 0, jobs.size()); const auto reloadStarted = Clock::now();
            for (auto& job : jobs) job.task->reload(scene_); stats_.reloadSeconds += seconds(reloadStarted);
            if (progress_) progress_("native-task-layout-ready", jobs.size(), jobs.size());
            std::vector<atom_scheduling::Job> forecasts; for (const auto& job : jobs) forecasts.push_back(job.forecast);
            atom_scheduling::Pool credits(forecasts, {solverHost, deviceUsed}, hardware, uint32_t(jobs.size()));
            if (gpu && contexts_.size() < jobs.size() && progress_) progress_("native-context-init", contexts_.size(), jobs.size());
            const auto contextStarted = Clock::now();
            if (gpu) while (contexts_.size() < jobs.size()) { auto context = std::make_unique<VkContext>(); context->init(contextOptions); contexts_.push_back(std::move(context)); ++stats_.contextCreates; }
            stats_.contextSeconds += seconds(contextStarted); if (gpu && progress_) progress_("native-context-ready", contexts_.size(), contexts_.size());
            std::exception_ptr failure; std::mutex failureMutex;
            std::atomic<uint32_t> finishedWorkers{0}, finishedSolves{0}; std::mutex finishedMutex; std::condition_variable changed;
            auto work = [&](size_t worker) {
                try {
                    for (;;) {
                        const auto lease = credits.acquire(); if (lease.task >= jobs.size()) return;
                        try { auto options = options_; auto& job = jobs[lease.task]; options.threads = int(lease.threads); options.host_budget_bytes = size_t(job.forecast.credit.host + job.task->residentBytes()); options.device_budget_bytes = size_t(job.forecast.credit.device); options.over_budget_throws = true; options.stats = &job.run.stats;
                            const auto started = Clock::now(); job.run = job.task->solve(options, gpu ? contexts_[worker].get() : nullptr); job.workerSeconds = seconds(started); job.solvedReal = job.run.real;
                        } catch (...) { credits.release(lease.task); throw; }
                        credits.release(lease.task); finishedSolves.fetch_add(1); changed.notify_all();
                    }
                } catch (...) { std::lock_guard<std::mutex> lock(failureMutex); if (!failure) failure = std::current_exception(); credits.stop(); }
            };
            if (progress_) progress_("native-task-solve", completed, descriptors_.size());
            slog::diag(slog::Tag::Map, "[native-scene] solve wave begin: %zu tasks, %zu owned contexts, solver credits host %.1f/device %.1f MiB\n", jobs.size(), contexts_.size(), solverHost / 1048576., deviceUsed / 1048576.);
            const auto waveStarted = Clock::now(); std::vector<std::thread> workers;
            try { for (size_t worker = 0; worker < jobs.size(); ++worker) workers.emplace_back([&, worker] { work(worker); finishedWorkers.fetch_add(1); changed.notify_all(); }); }
            catch (...) { credits.stop(); for (auto& worker : workers) worker.join(); throw; }
            try {
                std::unique_lock<std::mutex> lock(finishedMutex);
                while (finishedWorkers.load() < jobs.size()) { changed.wait_for(lock, std::chrono::milliseconds(200)); if (progress_) progress_("native-task-solve", completed + finishedSolves.load(), descriptors_.size()); }
            } catch (...) { credits.stop(); for (auto& worker : workers) worker.join(); throw; }
            for (auto& worker : workers) worker.join(); const double waveElapsed = seconds(waveStarted); stats_.waveSeconds += waveElapsed;
            if (progress_) progress_("native-task-solve", completed + finishedSolves.load(), descriptors_.size());
            if (failure) std::rethrow_exception(failure);
            if (progress_) progress_("native-task-apply", 0, jobs.size()); const auto applyStarted = Clock::now();
            double workerElapsed = 0, initElapsed = 0, computeElapsed = 0;
            for (auto& job : jobs) { job.task->apply(scene_); ++completed; ++stats_.completed; stats_.gpuCompleted += job.solvedReal != RealCfg::CPU; workerElapsed += job.workerSeconds; initElapsed += job.run.t_init; computeElapsed += job.run.t_solve; }
            const double applyElapsed = seconds(applyStarted); stats_.applySeconds += applyElapsed; stats_.workerSeconds += workerElapsed; stats_.solverInitSeconds += initElapsed; stats_.solverComputeSeconds += computeElapsed;
            if (progress_) progress_("native-task-apply", jobs.size(), jobs.size());
            slog::diag(slog::Tag::Map, "[native-scene] wave wall %.3fs; worker aggregate %.3fs (native init %.3fs, solve %.3fs); main apply %.3fs\n", waveElapsed, workerElapsed, initElapsed, computeElapsed, applyElapsed);
            const auto usage = credits.stats(); stats_.peakWorkers = std::max(stats_.peakWorkers, usage.peak_active); stats_.peakHostCredit = std::max(stats_.peakHostCredit, usage.peak_host); stats_.peakDeviceCredit = std::max(stats_.peakDeviceCredit, usage.peak_device);
            stats_.peakContexts = std::max<uint32_t>(stats_.peakContexts, gpu ? uint32_t(jobs.size()) : 0);
            next += jobs.size(); if (progress_) progress_("native-block-solve", completed, descriptors_.size());
        }
    }
    const ImmutableSceneTopology& index_; CanonicalScene& scene_; BundleOptions options_; size_t budget_; ImmutableSceneTopology::Progress progress_;
    std::vector<Descriptor> descriptors_; std::map<uint64_t, Cached> cache_; std::vector<std::unique_ptr<VkContext>> contexts_; uint64_t nextId_ = 1, tick_ = 0, externalReserve_ = 0; Statistics stats_;
};
}
