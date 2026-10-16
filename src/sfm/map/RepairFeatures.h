#pragma once

#include "sfm/feature/Verification.h"
#include "sfm/core/CameraSetup.h"
#include "sfm/map/RegionalPlan.h"
#include <list>

namespace sfm::regional {

class RepairFeatureCache {
    const MatchesIndex& index_;
    std::filesystem::path directory_;
    size_t budget_, bytes_ = 0, peak_ = 0, loads_ = 0;
    uint32_t dimension_ = 0, dtype_ = 0;
    std::list<uint32_t> lru_;
    std::map<uint32_t, std::list<uint32_t>::iterator> resident_;
    void evict(const std::set<uint32_t>& pinned, size_t incoming) {
        for (auto it = lru_.begin(); bytes_ + incoming > budget_;) {
            if (it == lru_.end()) throw RegionInputOverBudget();
            uint32_t id = *it;
            if (pinned.count(id)) { ++it; continue; }
            bytes_ -= size(id); features[id] = FeatureSet{};
            std::vector<Vec3>().swap(bearings.bearings[id]);
            resident_.erase(id); it = lru_.erase(it);
        }
    }
    size_t size(uint32_t id) const {
        const auto& f = features[id];
        return f.descBytes() + f.keypoints.size() * sizeof(Keypoint) + f.colors.size() +
               f.exif_camera.size() + bearings.bearings[id].size() * sizeof(Vec3);
    }
    size_t estimate(uint32_t id) const {
        if (resident_.count(id)) return size(id);
        return size_t(index_.images.at(id).num_features) *
            (dimension_ * dtypeSize(DType(dtype_)) + sizeof(Keypoint) + sizeof(Vec3) + 3) + (64u << 10);
    }
public:
    std::vector<FeatureSet> features;
    std::vector<const FeatureSet*> view;
    std::vector<Camera> cameras;
    BearingCache bearings;
    RepairFeatureCache(const MatchesIndex& index, std::filesystem::path directory,
                       const CameraSetup& setup, size_t budget)
        : index_(index), directory_(std::move(directory)), budget_(budget), features(index.images.size()) {
        size_t overhead = features.size() * (sizeof(FeatureSet) + sizeof(const FeatureSet*) +
                          2 * sizeof(Camera) + sizeof(std::vector<Vec3>));
        if (budget_ <= overhead) throw RegionInputOverBudget();
        budget_ -= overhead; view.reserve(features.size()); cameras.reserve(features.size());
        for (const auto& image : index.images) if (image.num_features) {
            std::ifstream file(directory_ / (image.name + ".bin"), std::ios::binary);
            char magic[4]{}; uint32_t header[6]{}; file.read(magic, 4);
            file.read(reinterpret_cast<char*>(header), sizeof(header));
            if (!file || std::memcmp(magic, "VKFT", 4) || header[3] != image.num_features || !header[4] || header[5] > 1)
                throw std::runtime_error("invalid repair feature metadata");
            dimension_ = header[4]; dtype_ = header[5]; break;
        }
        for (size_t i = 0; i < features.size(); ++i) {
            view.push_back(&features[i]); cameras.push_back(setup.cameras.at(setup.ids.at(i)));
        }
        bearings.cameras = cameras; bearings.bearings.resize(features.size());
    }
    size_t bytes() const { return bytes_; }
    size_t budget() const { return budget_; }
    size_t peakBytes() const { return peak_; }
    size_t loads() const { return loads_; }
    size_t nextEnd(const std::vector<std::pair<uint32_t, uint32_t>>& pairs, size_t begin) const {
        size_t end = begin, estimated = 0; std::set<uint32_t> pinned;
        while (end < pairs.size() && end - begin < 2048) {
            size_t added = 0;
            for (uint32_t id : {pairs[end].first, pairs[end].second}) if (!pinned.count(id))
                added += estimate(id);
            if (added && end > begin && estimated + added > budget_ / 2) break;
            estimated += added; pinned.insert(pairs[end].first); pinned.insert(pairs[end].second); ++end;
        }
        return end;
    }
    void pin(const std::set<uint32_t>& pinned) {
        for (uint32_t id : pinned) {
            auto old = resident_.find(id);
            if (old != resident_.end()) { lru_.splice(lru_.end(), lru_, old->second); continue; }
            cancel::check();
            evict(pinned, estimate(id));
            const auto path = (directory_ / (index_.images.at(id).name + ".bin")).string(); uint32_t count = 0;
            if (!peekFeatures(path, count) || count != index_.images.at(id).num_features)
                throw std::runtime_error("repair feature file changed or truncated: " + path);
            auto f = readFeatures(path);
            if (f.count() != index_.images.at(id).num_features || f.dim != dimension_ || uint32_t(f.dtype) != dtype_)
                throw std::runtime_error("repair feature identity changed");
            std::vector<Vec3> rays;
            if (cameras[id].wideFov()) {
                rays.reserve(f.count());
                for (const auto& k : f.keypoints) rays.push_back(cameras[id].bearing({k.x, k.y}));
            }
            features[id] = std::move(f); bearings.bearings[id] = std::move(rays);
            size_t incoming = size(id); evict(pinned, incoming);
            lru_.push_back(id); resident_[id] = std::prev(lru_.end());
            bytes_ += incoming; peak_ = std::max(peak_, bytes_); ++loads_;
        }
    }
};

} // namespace sfm::regional
