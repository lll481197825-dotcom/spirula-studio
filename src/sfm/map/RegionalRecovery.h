#pragma once

#include "sfm/map/RegionalPlan.h"
#include "sfm/SfmConfig.h"
#include "sfm/core/CameraSetup.h"
#include "sfm/core/PriorSource.h"

namespace sfm::regional {

class RepairFeatureCache;
class FixedMetricPriorSource : public PriorSource {
public:
    explicit FixedMetricPriorSource(PriorSource& source) : source_(source) {}
    bool has(uint32_t id) const override { return source_.has(id); }
    bool relativeRotation(uint32_t a, uint32_t b, Mat3& r, double& sigma) const override {
        return source_.relativeRotation(a, b, r, sigma);
    }
    std::vector<uint32_t> neighbours(uint32_t id) const override { return source_.neighbours(id); }
    bool position(uint32_t id, Vec3& p) const override { return source_.position(id, p); }
    bool declaredUp(uint32_t id, Vec3& up) const override { return source_.declaredUp(id, up); }
    void disableLevel() override { source_.disableLevel(); }
    PosePriors factors(const std::vector<PosedImage>& images) override;
private:
    PriorSource& source_;
};
std::vector<TwoViewMatches> verifyRepairPairs(const SfmConfig& cfg, RepairFeatureCache& cache,
    IFeatureMatcher& matcher, const std::vector<std::pair<uint32_t, uint32_t>>& pairs,
    uint64_t& putative);

inline uint64_t signatureHash(const std::string& s) {
    uint64_t hash = 14695981039346656037ull;
    for (unsigned char c : s) { hash ^= c; hash *= 1099511628211ull; }
    return hash;
}

std::vector<Reconstruction> reconstructRepair(const SfmConfig& cfg, const MatchesIndex& index,
    const Region& region, const std::filesystem::path& matches, const std::filesystem::path& features,
    const std::string& images, const CameraSetup& cameras, PriorSource* gps,
    const Reconstruction* seed, size_t budget, bool hold_seed = false);

void loadRepairMatches(MatchesIndex& index, const std::filesystem::path& directory);
size_t supplementRepairMatches(const SfmConfig& cfg, MatchesIndex& index, const std::set<uint32_t>& required,
    const std::set<uint32_t>& anchors, const std::vector<Vec3>& positions, const CameraSetup& cameras,
    const std::filesystem::path& features, const std::filesystem::path& directory, size_t neighbours, size_t budget);

inline std::set<uint32_t> registeredIds(const Reconstruction& rec) {
    std::set<uint32_t> ids;
    for (const auto& kv : rec.images) if (kv.second.registered) ids.insert(kv.first);
    return ids;
}

inline bool coversImages(const std::vector<Reconstruction>& models, const std::set<uint32_t>& required) {
    std::set<uint32_t> covered;
    for (const auto& model : models) {
        auto ids = registeredIds(model); covered.insert(ids.begin(), ids.end());
    }
    return std::includes(covered.begin(), covered.end(), required.begin(), required.end());
}

class RepairPairCatalog {
public:
    void sync(const MatchesIndex& index) {
        if (indexed_ > index.pairs.size()) throw std::runtime_error("repair matching index is not append-only");
        adjacent_.resize(index.images.size());
        for (; indexed_ < index.pairs.size(); ++indexed_) {
            const auto& p = index.pairs[indexed_];
            adjacent_.at(p.image1).push_back(indexed_); adjacent_.at(p.image2).push_back(indexed_);
        }
    }
    template<class Images> std::vector<size_t> incident(const Images& images) const {
        std::vector<size_t> result;
        for (uint32_t id : images) {
            const auto& pairs = adjacent_.at(id); result.insert(result.end(), pairs.begin(), pairs.end());
        }
        std::sort(result.begin(), result.end()); result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }
private:
    size_t indexed_ = 0;
    std::vector<std::vector<size_t>> adjacent_;
};

inline Region repairContext(const MatchesIndex& index, const std::set<uint32_t>& required,
                            const std::set<uint32_t>& anchors, size_t support_limit, RepairPairCatalog* catalog = nullptr) {
    Region out; out.core.assign(required.begin(), required.end()); out.images = out.core;
    if (catalog) catalog->sync(index);
    auto visit = [&](const std::set<uint32_t>& ids, auto apply) {
        if (catalog) { for (auto k : catalog->incident(ids)) if (index.pairs[k].count) apply(index.pairs[k]); }
        else for (const auto& p : index.pairs) if (p.count) apply(p);
    };
    std::map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> adjacent;
    visit(required, [&](const auto& p) {
        if (required.count(p.image1) && !required.count(p.image2)) adjacent[p.image1].push_back({p.count, p.image2});
        if (required.count(p.image2) && !required.count(p.image1)) adjacent[p.image2].push_back({p.count, p.image1});
    });
    for (auto& kv : adjacent) std::sort(kv.second.rbegin(), kv.second.rend());
    std::set<uint32_t> selected, bridge;
    size_t bridge_limit = support_limit / 4;
    for (size_t depth = 0;; ++depth) {
        bool have = false;
        for (const auto& kv : adjacent) {
            if (depth >= kv.second.size()) continue;
            have = true; uint32_t id = kv.second[depth].second;
            if (selected.size() >= support_limit) break;
            if (!anchors.count(id) && bridge.size() >= bridge_limit) continue;
            if (selected.insert(id).second && !anchors.count(id)) bridge.insert(id);
        }
        if (!have || selected.size() >= support_limit) break;
    }
    std::map<uint32_t, uint64_t> votes;
    visit(bridge, [&](const auto& p) {
        if (bridge.count(p.image1) && anchors.count(p.image2) && !required.count(p.image2)) votes[p.image2] += p.count;
        if (bridge.count(p.image2) && anchors.count(p.image1) && !required.count(p.image1)) votes[p.image1] += p.count;
    });
    std::vector<std::pair<uint64_t, uint32_t>> rank;
    for (const auto& kv : votes) rank.emplace_back(kv.second, kv.first);
    std::sort(rank.rbegin(), rank.rend());
    for (const auto& kv : rank) if (selected.size() < support_limit) selected.insert(kv.second);
    out.images.insert(out.images.end(), selected.begin(), selected.end());
    std::sort(out.images.begin(), out.images.end()); return out;
}

inline Reconstruction compactSeed(const Reconstruction& seed, const RegionInput& input, bool hold_geometry = false) {
    if (hold_geometry) { const HeldRepairSeedContract heldContract(seed);(void)heldContract; }
    Reconstruction out; out.cameras = seed.cameras; out.rigs = seed.rigs; out.next_point3D_id = seed.next_point3D_id;
    std::map<uint32_t, uint32_t> local;
    for (uint32_t i = 0; i < input.global.size(); ++i) local.emplace(input.global[i], i);
    for (const auto& kv : seed.images) if (kv.second.registered) {
        auto at = local.find(kv.first);
        if (at == local.end()) throw std::runtime_error("regional repair omitted a seed image");
        uint32_t id = at->second; Image im = kv.second; im.id = id;
        im.points2D.clear(); im.point3D_ids.assign(input.original.at(id).size(), kInvalidPoint3D);
        for (size_t k = 0; k < input.original.at(id).size(); ++k) {
            uint32_t f = input.original.at(id).at(k);
            if (f >= kv.second.points2D.size()) throw std::runtime_error("regional repair seed feature out of range");
            const auto& point = input.features.at(id).keypoints.at(k);
            if (hold_geometry && (kv.second.points2D.at(f).x != point.x || kv.second.points2D.at(f).y != point.y))
                throw std::runtime_error("held repair compaction changed an original feature measurement");
            im.points2D.push_back({point.x, point.y});
        }
        out.images.emplace(id, std::move(im));
    }
    for (const auto& kv : seed.points3D) {
        Point3D p = kv.second; p.track.clear();
        for (const auto& e : kv.second.track) {
            auto at = local.find(e.image_id);
            if (at == local.end() || !out.images.count(at->second)) {
                if (hold_geometry) throw std::runtime_error("held repair compaction omitted a seed raw image");
                continue;
            }
            const auto& original = input.original.at(at->second);
            auto f = std::lower_bound(original.begin(), original.end(), e.point2D_idx);
            if (f != original.end() && *f == e.point2D_idx) p.track.push_back({at->second, uint32_t(f - original.begin())});
            else if (hold_geometry) throw std::runtime_error("held repair compaction omitted an assigned seed raw feature");
        }
        if (p.track.size() < (hold_geometry ? 1u : 2u)) continue;
        for (const auto& e : p.track) out.images.at(e.image_id).point3D_ids.at(e.point2D_idx) = kv.first;
        out.points3D.emplace(kv.first, std::move(p));
    }
    for (uint32_t id : seed.rig_detached) if (local.count(id)) out.rig_detached.insert(local.at(id));
    return out;
}

} // namespace sfm::regional
