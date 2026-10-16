#pragma once

#include "sfm/map/ModelStore.h"
#include "sfm/map/RegionalRecovery.h"
#include <memory>

namespace sfm::regional {

class RepairModelCache {
    struct Entry {
        fs::path path;
        std::string identity;
        std::vector<uint32_t> images;
        std::shared_ptr<model_store_detail::ModelReadIndex> index;
        uint64_t used = 0;
    };
public:
    explicit RepairModelCache(size_t capacity) : capacity_(capacity) {}
    static std::string identity(const fs::path& path) {
        return fs::absolute(path).string() + ":" + std::to_string(fs::file_size(path)) + ":" +
            std::to_string(fs::last_write_time(path).time_since_epoch().count());
    }
    void sync(const std::vector<fs::path>& paths) {
        active_.clear(); std::vector<std::string> identities;
        for (const auto& path : paths) {
            const auto stamp = identity(path); auto& entry = entries_[path];
            if (!entry || entry->identity != stamp) {
                if (entry && entry->index) bytes_ -= entry->index->bytes;
                entry = std::make_shared<Entry>(); entry->path = path; entry->identity = stamp;
                auto views = model_store_detail::readModel(path, false); ++header_reads;
                for (const auto& kv : views.images) entry->images.push_back(kv.first);
                if (identity(path) != stamp) throw std::runtime_error("repair model changed during catalog read");
            }
            active_.push_back(entry); identities.push_back(stamp);
        }
        if (identities == catalog_identity_) return;
        catalog_identity_ = std::move(identities); by_image_.clear();
        for (size_t m = 0; m < active_.size(); ++m) for (uint32_t id : active_[m]->images) by_image_[id].push_back(m);
        std::set<fs::path> retained(paths.begin(), paths.end());
        for (auto at = entries_.begin(); at != entries_.end();) {
            if (retained.count(at->first)) { ++at; continue; }
            if (at->second->index) bytes_ -= at->second->index->bytes;
            at = entries_.erase(at);
        }
    }
    std::vector<std::pair<size_t, size_t>> rank(const Region& context, const std::set<uint32_t>& anchors) const {
        std::vector<size_t> votes(active_.size());
        for (uint32_t id : context.images) if (anchors.count(id)) {
            auto at = by_image_.find(id); if (at != by_image_.end()) for (auto m : at->second) ++votes[m];
        }
        std::vector<std::pair<size_t, size_t>> result;
        for (size_t m = 0; m < votes.size(); ++m) if (votes[m] >= 3) result.emplace_back(votes[m], m);
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        return result;
    }
    bool contains(size_t model, uint32_t image) const {
        const auto& ids = active_.at(model)->images; return std::binary_search(ids.begin(), ids.end(), image);
    }
    const fs::path& path(size_t model) const { return active_.at(model)->path; }
    const std::string& stamp(size_t model) const { return active_.at(model)->identity; }
    Reconstruction subset(size_t model, const std::vector<uint32_t>& images) {
        auto& entry = *active_.at(model); entry.used = ++clock_;
        if (!entry.index) {
            auto index = std::make_shared<model_store_detail::ModelReadIndex>();
            model_store_detail::readModel(entry.path, false, index.get(), capacity_); ++index_builds;
            if (identity(entry.path) != entry.identity) throw std::runtime_error("repair model changed during observation indexing");
            evict(index->bytes);
            bytes_ += index->bytes; entry.index = std::move(index);
        } else ++index_hits;
        auto result = model_store_detail::readModelSubset(entry.path, *entry.index, images); ++subset_reads;
        if (identity(entry.path) != entry.identity) throw std::runtime_error("repair model changed during subset read");
        return result;
    }
    size_t bytes() const { return bytes_; }
    uint64_t header_reads = 0, index_builds = 0, index_hits = 0, subset_reads = 0;
private:
    void evict(size_t reservation) {
        while (bytes_ > capacity_ - reservation) {
            std::shared_ptr<Entry> oldest;
            for (const auto& kv : entries_) if (kv.second->index && (!oldest || kv.second->used < oldest->used)) oldest = kv.second;
            if (!oldest) throw std::logic_error("repair observation cache accounting mismatch");
            bytes_ -= oldest->index->bytes; oldest->index.reset();
        }
    }
    size_t capacity_, bytes_ = 0;
    uint64_t clock_ = 0;
    std::map<fs::path, std::shared_ptr<Entry>> entries_;
    std::vector<std::shared_ptr<Entry>> active_;
    std::vector<std::string> catalog_identity_;
    std::map<uint32_t, std::vector<size_t>> by_image_;
};

class RepairDigest {
public:
    template<class T> void scalar(const T& value) { append(&value, sizeof(value)); }
    template<class T> void vector(const std::vector<T>& value) { scalar(uint64_t(value.size())); append(value.data(), value.size() * sizeof(T)); }
    void string(const std::string& value) { scalar(uint64_t(value.size())); append(value.data(), value.size()); }
    std::string result() const { return std::to_string(hash_); }
private:
    void append(const void* data, size_t bytes) {
        const auto* p = static_cast<const uint8_t*>(data);
        for (size_t k = 0; k < bytes; ++k) { hash_ ^= p[k]; hash_ *= 1099511628211ull; }
    }
    uint64_t hash_ = 1469598103934665603ull;
};

inline std::string repairGeometryIdentity(const Region& context, const Reconstruction& poses, const Reconstruction* seed) {
    RepairDigest out; out.vector(context.core); out.vector(context.images);
    for (uint32_t id : context.images) {
        auto at = poses.images.find(id); out.scalar(uint8_t(at != poses.images.end()));
        if (at == poses.images.end()) continue;
        out.scalar(at->second.pose); out.scalar(at->second.registered); out.scalar(at->second.camera_id);
        model_store_detail::writeCamera(out, poses.cameras.at(at->second.camera_id));
    }
    out.scalar(uint8_t(seed != nullptr)); if (seed) model_store_detail::serializeModel(*seed, out);
    return out.result();
}

}
