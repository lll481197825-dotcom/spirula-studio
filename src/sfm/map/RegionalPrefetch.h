#pragma once

#include <functional>
#include <future>
#include <deque>
#include <thread>
#include "sfm/map/ModelStore.h"

namespace sfm::regional {
class ModelReadAhead {
public:
    ModelReadAhead(size_t count, size_t budget, std::function<size_t(size_t)> memory,
                   std::function<Reconstruction(size_t)> read)
        : count_(count), limit_(budget / 2), memory_(std::move(memory)), read_(std::move(read)) {}
    ModelReadAhead(const ModelReadAhead&) = delete;
    ModelReadAhead& operator=(const ModelReadAhead&) = delete;
    Reconstruction get(size_t model) {
        if (model != next_ || model >= count_) throw std::logic_error("regional prefetch read order changed");
        Reconstruction rec;
        if (pending_.empty()) rec = read_(model);
        else {
            if (pending_.front().model != model) throw std::logic_error("regional prefetch queue reordered a model");
            rec = pending_.front().ready.get(); pending_.pop_front();
        }
        next_ = model + 1 == count_ ? 0 : model + 1;
        size_t reserved = memory_(model);
        for (const auto& item : pending_) {
            if (reserved > limit_ || item.memory > limit_ - reserved)
                throw std::length_error("regional prefetch memory estimate changed");
            reserved += item.memory;
        }
        size_t following = pending_.empty() ? model + 1 : pending_.back().model + 1;
        const size_t capacity = std::max(1u, std::thread::hardware_concurrency()) - 1;
        while (following < count_ && pending_.size() < capacity && reserved <= limit_) {
            constexpr size_t buffers = 2u << 20;
            const size_t memory = memory_(following);
            if (memory > limit_ - reserved || buffers > limit_ - reserved - memory) break;
            pending_.push_back({following, memory + buffers,
                std::async(std::launch::async, [this, following] { return read_(following); })});
            reserved += memory + buffers; ++following;
        }
        return rec;
    }
private:
    size_t count_, limit_, next_ = 0;
    std::function<size_t(size_t)> memory_;
    std::function<Reconstruction(size_t)> read_;
    struct Pending { size_t model, memory; std::future<Reconstruction> ready; };
    std::deque<Pending> pending_;
};
}
