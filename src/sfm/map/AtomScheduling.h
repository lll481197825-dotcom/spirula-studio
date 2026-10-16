#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "sfm/core/Cancel.h"

namespace sfm::atom_scheduling {

struct Credit {
    uint64_t host = 0, device = 0;
};
struct Job {
    Credit credit;
    uint64_t solver_host = 0, solver_device = 0, model_bytes = 0;
};
struct Limits {
    uint64_t host = 0, device = 0, context_host = 0, context_device = 0;
    uint32_t hardware = 1, requested = 0;
    bool first_device_context_accounted = false;
};
struct Plan {
    uint32_t workers = 1, cpu_ceiling = 1;
    uint64_t contexts_host = 0, contexts_device = 0;
    Credit pool;
};
inline uint64_t add(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX - a) throw std::length_error("atom resource forecast overflow"); return a + b;
}
inline uint64_t multiply(uint64_t a, uint64_t b) {
    if (b && a > UINT64_MAX / b) throw std::length_error("atom resource forecast overflow"); return a * b;
}
inline Plan plan(const std::vector<Job>& jobs, const Limits& limits) {
    Plan out; if (jobs.empty()) { out.workers = 0; return out; }
    out.cpu_ceiling = std::max<uint32_t>(1, limits.hardware);
    if (limits.requested) out.cpu_ceiling = std::min(out.cpu_ceiling, limits.requested);
    out.cpu_ceiling = std::min<uint32_t>(out.cpu_ceiling, uint32_t(std::min<uint64_t>(jobs.size(), UINT32_MAX)));
    uint64_t largestHost = 0, largestDevice = 0, minimumHost = UINT64_MAX, minimumDevice = UINT64_MAX;
    for (const auto& job : jobs) { largestHost = std::max(largestHost, job.credit.host); largestDevice = std::max(largestDevice, job.credit.device); minimumHost = std::min(minimumHost, job.credit.host); minimumDevice = std::min(minimumDevice, job.credit.device); }
    for (uint32_t workers = out.cpu_ceiling; workers; --workers) {
        const auto host = multiply(workers, limits.context_host), device = multiply(workers - (limits.first_device_context_accounted ? 1 : 0), limits.context_device);
        if (host > limits.host || device > limits.device) continue;
        const auto poolHost = limits.host - host, poolDevice = limits.device - device;
        if (largestHost > poolHost || largestDevice > poolDevice) continue;
        if (minimumHost && workers > poolHost / minimumHost) continue;
        if (minimumDevice && workers > poolDevice / minimumDevice) continue;
        out.workers = workers; out.contexts_host = host; out.contexts_device = device; out.pool = {poolHost, poolDevice}; return out;
    }
    throw std::length_error("atom resource capacity insufficient even for one complete task; shared inputs and accepted models retained");
}

class Pool {
public:
    struct Stats { uint64_t peak_host = 0, peak_device = 0, waits = 0, completed = 0; uint32_t peak_active = 0, peak_cpu = 0; };
    struct Lease { size_t task; uint32_t threads; };
    Pool(const std::vector<Job>& jobs, Credit capacity, uint32_t hardware = 1, uint32_t workers = UINT32_MAX) : jobs_(jobs), capacity_(capacity), claimed_(jobs.size(), 0), quotas_(jobs.size(), 0), hardware_(std::max<uint32_t>(1, hardware)), workers_(std::max<uint32_t>(1, workers)) {
        for (const auto& job : jobs) if (job.credit.host > capacity.host || job.credit.device > capacity.device)
            throw std::length_error("one atom task exceeds empty resource pool capacity");
    }
    Lease acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            cancel::check(); if (stopped_ || claimed_count_ == jobs_.size()) return {jobs_.size(), 0};
            for (size_t task = 0; task < jobs_.size(); ++task) {
                if (claimed_[task] || used_cpu_ == hardware_ || active_ == workers_) continue; const auto credit = jobs_[task].credit;
                if (credit.host > capacity_.host - used_.host || credit.device > capacity_.device - used_.device) continue;
                claimed_[task] = 1; ++claimed_count_; ++active_; used_.host += credit.host; used_.device += credit.device;
                Credit available{capacity_.host - used_.host, capacity_.device - used_.device}; uint32_t possible = 0;
                for (size_t next = 0; next < jobs_.size() && possible < std::min(hardware_, workers_ - active_); ++next) {
                    if (claimed_[next]) continue; const auto need = jobs_[next].credit;
                    if (need.host <= available.host && need.device <= available.device) { available.host -= need.host; available.device -= need.device; ++possible; }
                }
                const uint32_t cpu = hardware_ - used_cpu_; const auto quota = std::max<uint32_t>(1, cpu / (possible + 1));
                quotas_[task] = quota; used_cpu_ += quota; stats_.peak_cpu = std::max(stats_.peak_cpu, used_cpu_);
                stats_.peak_host = std::max(stats_.peak_host, used_.host); stats_.peak_device = std::max(stats_.peak_device, used_.device); stats_.peak_active = std::max(stats_.peak_active, active_); return {task, quota};
            }
            if (!active_) throw std::length_error("pending atom tasks cannot fit free resource pool; original inputs retained");
            ++stats_.waits; changed_.wait_for(lock, std::chrono::milliseconds(50));
        }
    }
    void release(size_t task) {
        std::lock_guard<std::mutex> lock(mutex_); const auto credit = jobs_.at(task).credit;
        if (claimed_.at(task) != 1 || credit.host > used_.host || credit.device > used_.device || !active_) throw std::logic_error("invalid atom resource release");
        claimed_[task] = 2; used_cpu_ -= quotas_.at(task); quotas_[task] = 0; used_.host -= credit.host; used_.device -= credit.device; --active_; ++stats_.completed; changed_.notify_all();
    }
    void stop() { std::lock_guard<std::mutex> lock(mutex_); stopped_ = true; changed_.notify_all(); }
    Stats stats() const { std::lock_guard<std::mutex> lock(mutex_); return stats_; }
private:
    const std::vector<Job>& jobs_; Credit capacity_, used_; std::vector<uint8_t> claimed_; std::vector<uint32_t> quotas_; uint32_t hardware_ = 1, used_cpu_ = 0, workers_ = 1; size_t claimed_count_ = 0; uint32_t active_ = 0;
    bool stopped_ = false; mutable std::mutex mutex_; std::condition_variable changed_; Stats stats_;
};

} // namespace sfm::atom_scheduling
