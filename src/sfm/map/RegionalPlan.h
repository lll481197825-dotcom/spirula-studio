#pragma once

#include "sfm/core/FeatureCompaction.h"
#include "sfm/feature/SpatialBlocks.h"
#include "sfm/map/Atoms.h"
#include "sfm/map/ModelStore.h"
#include "sfm/map/HeldRepairSeed.h"
#include <cmath>
#include <numeric>

namespace sfm::regional {
struct Region { std::vector<uint32_t> core, images; };

inline std::vector<Region> readRegionalPlan(const std::filesystem::path& root, size_t images) {
    model_store_detail::Reader in(root / "plan.bin");
    if (in.scalar<uint32_t>() != 1) throw std::runtime_error("unsupported regional plan");
    std::vector<Region> plan(in.count(16)); std::vector<uint8_t> owner(images);
    for (Region& r : plan) {
        r.core = in.vector<uint32_t>(); r.images = in.vector<uint32_t>();
        if (r.core.empty() || !std::is_sorted(r.images.begin(), r.images.end()) ||
            std::adjacent_find(r.images.begin(), r.images.end()) != r.images.end())
            throw std::runtime_error("invalid regional checkpoint plan");
        for (uint32_t i : r.images) if (i >= images) throw std::runtime_error("regional checkpoint image out of range");
        for (uint32_t i : r.core)
            if (i >= images || owner[i]++ || !std::binary_search(r.images.begin(), r.images.end(), i))
                throw std::runtime_error("invalid regional checkpoint ownership");
    }
    if (std::count(owner.begin(), owner.end(), 1) != images) throw std::runtime_error("incomplete regional checkpoint plan");
    in.finish(); return plan;
}

struct RegionInputOverBudget : std::length_error {
    RegionInputOverBudget() : std::length_error("regional matches and graph exceed input budget") {}
};

inline bool expandAlignmentContext(const MatchesIndex& index, Region& region,
                                   const std::set<uint32_t>& unresolved,
                                   const std::set<uint32_t>& anchors, size_t limit) {
    std::set<uint32_t> included(region.images.begin(), region.images.end());
    std::map<uint32_t, uint64_t> votes;
    for (const auto& pair : index.pairs) {
        auto vote = [&](uint32_t a, uint32_t b) {
            if (unresolved.count(a) && anchors.count(b) && !included.count(b)) votes[b] += pair.count;
        };
        vote(pair.image1, pair.image2); vote(pair.image2, pair.image1);
    }
    std::vector<std::pair<uint64_t, uint32_t>> ranked;
    for (const auto& kv : votes) if (kv.second) ranked.emplace_back(kv.second, kv.first);
    std::sort(ranked.begin(), ranked.end(), [](auto a, auto b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    size_t count = std::min(limit, ranked.size());
    for (size_t k = 0; k < count; ++k) region.images.push_back(ranked[k].second);
    std::sort(region.images.begin(), region.images.end());
    return count != 0;
}

struct SpatialPlanOptions {
    uint64_t seed = 0x8e7f6a49b5c132d0ULL;
    uint64_t distance_budget = 1500000000;
};
struct SpatialPlanStats {
    size_t positioned = 0, missing_positions = 0, clusters = 0, rounds = 0, attempts = 0;
    size_t smallest = 0, largest = 0;
    uint64_t distances = 0;
    double compactness = 0;
    bool budget_limited = false;
};

inline SpatialBlockPlan spatialRegionOwners(const std::vector<Vec3>& gps, size_t target,
    SpatialPlanStats& stats, const SpatialPlanOptions& options = {}) {
    if (target < 2 || gps.empty() || gps.size() > UINT32_MAX)
        throw std::invalid_argument("invalid spatial region input");
    stats = {}; SpatialBlockPlan out; out.owner.assign(gps.size(), UINT32_MAX);
    std::vector<uint32_t> valid;
    for (uint32_t i = 0; i < gps.size(); ++i)
        if (std::isfinite(gps[i].x) && std::isfinite(gps[i].y)) valid.push_back(i);
    stats.positioned = valid.size(); stats.missing_positions = gps.size() - valid.size();
    const size_t count = (gps.size() - 1) / target + 1;
    out.blocks = stats.clusters = valid.empty() ? count : std::min(count, valid.size());
    if (valid.empty()) return out;
    const size_t k = out.blocks, n = valid.size();
    if (k > options.distance_budget / (uint64_t(n) * 33))
        throw std::length_error("spatial kmeans distance budget cannot cover three attempts");
    const uint64_t bound = uint64_t(n) * k * 11;
    const size_t rounds = size_t(std::min<uint64_t>(10, options.distance_budget / (3 * bound)));
    uint64_t random = options.seed ? options.seed : 1;
    auto unit = [&]() {
        random ^= random >> 12; random ^= random << 25; random ^= random >> 27;
        return double((random * 2685821657736338717ULL) >> 11) * (1. / 9007199254740992.);
    };
    const Vec3 origin = gps[valid.front()];
    std::vector<Vec2> points; points.reserve(n);
    for (uint32_t id : valid) points.push_back({gps[id].x - origin.x, gps[id].y - origin.y});
    auto distance = [&](const Vec2& a, const Vec2& b) {
        ++stats.distances; const double x = a.x - b.x, y = a.y - b.y;
        const double value = x * x + y * y;
        if (!std::isfinite(value)) throw std::invalid_argument("spatial coordinates exceed squared-distance range");
        return value;
    };
    size_t best_max = SIZE_MAX, best_min = 0;
    double best_cost = std::numeric_limits<double>::infinity();
    std::vector<uint32_t> best, assignment(n, UINT32_MAX);
    std::vector<double> nearest(n);
    for (size_t round = 0; round < rounds; ++round) {
        double round_cost = std::numeric_limits<double>::infinity();
        std::vector<uint32_t> round_assignment;
        std::vector<size_t> round_sizes;
        for (size_t attempt = 0; attempt < 3; ++attempt) {
            cancel::check(); std::vector<Vec2> centres; centres.reserve(k);
            centres.push_back(points[std::min(n - 1, size_t(unit() * n))]);
            std::fill(nearest.begin(), nearest.end(), std::numeric_limits<double>::infinity());
            for (size_t c = 1; c < k; ++c) {
                long double sum = 0;
                for (size_t i = 0; i < n; ++i) {
                    nearest[i] = std::min(nearest[i], distance(points[i], centres.back())); sum += nearest[i];
                }
                long double threshold = unit() * sum; size_t chosen = n - 1;
                for (size_t i = 0; i < n; ++i) if ((threshold -= nearest[i]) <= 0) { chosen = i; break; }
                centres.push_back(points[chosen]);
            }
            std::fill(assignment.begin(), assignment.end(), UINT32_MAX);
            std::vector<size_t> sizes(k);
            for (size_t iteration = 0; iteration < 10; ++iteration) {
                cancel::check(); std::fill(sizes.begin(), sizes.end(), 0);
                std::vector<Vec2> sum(k);
                for (size_t i = 0; i < n; ++i) {
                    uint32_t owner = assignment[i] < k ? assignment[i] : 0;
                    double value = std::numeric_limits<double>::infinity();
                    for (uint32_t c = 0; c < k; ++c) {
                        const double candidate = distance(points[i], centres[c]);
                        if (candidate < value || (candidate == value && (c == assignment[i] ||
                            (assignment[i] == UINT32_MAX && sizes[c] < sizes[owner])))) { value = candidate; owner = c; }
                    }
                    assignment[i] = owner; nearest[i] = value; ++sizes[owner];
                    sum[owner].x += points[i].x; sum[owner].y += points[i].y;
                }
                for (uint32_t c = 0; c < k; ++c) if (!sizes[c]) {
                    size_t chosen = n;
                    for (size_t i = 0; i < n; ++i) if (sizes[assignment[i]] > 1 &&
                        (chosen == n || nearest[i] > nearest[chosen])) chosen = i;
                    if (chosen == n) throw std::logic_error("spatial kmeans cannot fill empty cluster");
                    const uint32_t previous = assignment[chosen]; --sizes[previous]; ++sizes[c];
                    sum[previous].x -= points[chosen].x; sum[previous].y -= points[chosen].y;
                    sum[c] = points[chosen]; assignment[chosen] = c; nearest[chosen] = 0;
                }
                double movement = 0;
                for (size_t c = 0; c < k; ++c) {
                    Vec2 next{sum[c].x / sizes[c], sum[c].y / sizes[c]};
                    const double x = next.x - centres[c].x, y = next.y - centres[c].y;
                    movement = std::max(movement, x * x + y * y); centres[c] = next;
                }
                if (movement <= .01) break;
            }
            double cost = 0;
            for (size_t i = 0; i < n; ++i) cost += distance(points[i], centres[assignment[i]]);
            if (!std::isfinite(cost)) throw std::invalid_argument("spatial compactness overflow");
            ++stats.attempts;
            if (cost < round_cost) { round_cost = cost; round_assignment = assignment; round_sizes = sizes; }
        }
        ++stats.rounds;
        const auto range = std::minmax_element(round_sizes.begin(), round_sizes.end());
        if (*range.second < best_max || (*range.second == best_max && round_cost < best_cost)) {
            best_max = *range.second; best_min = *range.first; best_cost = round_cost; best = round_assignment;
        }
        if (*range.second - *range.first < 1000) break;
    }
    stats.budget_limited = stats.rounds == rounds && rounds < 10 && best_max - best_min >= 1000;
    stats.compactness = best_cost;
    for (size_t i = 0; i < n; ++i) out.owner[valid[i]] = best[i];
    return out;
}

inline std::vector<Region> planRegions(const MatchesIndex& index, const std::vector<Vec3>& gps,
    size_t requested, SpatialPlanStats* statistics = nullptr, const SpatialPlanOptions& options = {}) {
    if (gps.size() != index.images.size()) throw std::invalid_argument("regional position count differs from images");
    SpatialPlanStats stats;
    SpatialBlockPlan partition = spatialRegionOwners(gps, requested, stats, options);
    std::vector<size_t> sizes(partition.blocks);
    for (uint32_t owner : partition.owner) if (owner != UINT32_MAX) ++sizes[owner];
    std::map<uint32_t, std::map<uint32_t, uint64_t>> missing_votes;
    for (const auto& pair : index.pairs) {
        if (pair.image1 >= gps.size() || pair.image2 >= gps.size()) throw std::invalid_argument("regional pair image out of range");
        auto vote = [&](uint32_t a, uint32_t b) {
            if (partition.owner[a] == UINT32_MAX && partition.owner[b] != UINT32_MAX)
                missing_votes[a][partition.owner[b]] += pair.count;
        };
        vote(pair.image1, pair.image2); vote(pair.image2, pair.image1);
    }
    using Slot = std::pair<size_t, uint32_t>;
    std::priority_queue<Slot, std::vector<Slot>, std::greater<Slot>> available;
    for (uint32_t c = 0; c < sizes.size(); ++c) available.emplace(sizes[c], c);
    for (uint32_t id = 0; id < gps.size(); ++id) if (partition.owner[id] == UINT32_MAX) {
        while (available.top().first != sizes[available.top().second]) available.pop();
        uint32_t owner = available.top().second; uint64_t strongest = 0;
        const auto votes = missing_votes.find(id);
        if (votes != missing_votes.end()) for (const auto& vote : votes->second)
            if (vote.second > strongest) { strongest = vote.second; owner = vote.first; }
        partition.owner[id] = owner; available.emplace(++sizes[owner], owner);
    }
    std::vector<Region> out(partition.blocks);
    std::vector<uint32_t> order(out.size()), remap(out.size()); std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return sizes[a] > sizes[b]; });
    for (uint32_t c = 0; c < order.size(); ++c) remap[order[c]] = c;
    for (uint32_t i = 0; i < gps.size(); ++i) { partition.owner[i] = remap[partition.owner[i]]; out[partition.owner[i]].core.push_back(i); }
    std::vector<std::map<uint32_t, uint64_t>> votes(out.size());
    for (const auto& pair : index.pairs) {
        uint32_t a = partition.owner[pair.image1], b = partition.owner[pair.image2];
        if (a == b) continue;
        votes[a][pair.image2] += pair.count; votes[b][pair.image1] += pair.count;
    }
    for (size_t r = 0; r < out.size(); ++r) {
        out[r].images = out[r].core;
        std::vector<std::pair<uint64_t, uint32_t>> halo;
        for (const auto& kv : votes[r]) halo.emplace_back(kv.second, kv.first);
        std::sort(halo.begin(), halo.end(), [](auto a, auto b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
        const size_t count = std::min(halo.size(), std::max<size_t>(16, out[r].core.size() / 4));
        for (size_t k = 0; k < count; ++k) out[r].images.push_back(halo[k].second);
        std::sort(out[r].images.begin(), out[r].images.end());
    }
    stats.smallest = *std::min_element(sizes.begin(), sizes.end()); stats.largest = *std::max_element(sizes.begin(), sizes.end());
    if (statistics) *statistics = stats;
    return out;
}

struct RegionInput {
    MatchesDatabase db;
    std::vector<FeatureSet> features;
    std::vector<uint32_t> global;
    std::vector<std::vector<uint32_t>> original;
    std::vector<std::vector<Vec2>> original_xy;
};

class RegionalMatchReader {
    struct File { std::ifstream stream; uint64_t touched = 0; };
    const MatchesIndex& index_;
    std::filesystem::path baseline_;
    std::map<uint32_t, File> files_;
    uint64_t clock_ = 0;
public:
    // Thousands of repair batches must share the CRT's limited stream table.
    static constexpr size_t capacity = 32;
    RegionalMatchReader(const MatchesIndex& index, std::filesystem::path baseline)
        : index_(index), baseline_(std::move(baseline)) {}
    size_t openFiles() const { return files_.size(); }
    void read(const MatchesIndex::Entry& pair, std::vector<FeatureMatch>& out) {
        const auto path = pair.source ? std::filesystem::path(index_.sources.at(pair.source - 1)) : baseline_;
        auto failed = [&](const char* reason) {
            return std::runtime_error(std::string(reason) + ": " + path.string() +
                ", source=" + std::to_string(pair.source) + ", images=" + std::to_string(pair.image1) +
                "," + std::to_string(pair.image2) + ", offset=" + std::to_string(pair.offset) +
                ", count=" + std::to_string(pair.count));
        };
        auto found = files_.find(pair.source);
        if (found == files_.end()) {
            if (files_.size() == capacity) {
                auto oldest = std::min_element(files_.begin(), files_.end(), [](const auto& a, const auto& b) {
                    return a.second.touched < b.second.touched;
                });
                files_.erase(oldest);
            }
            File file; file.stream.open(path, std::ios::binary);
            if (!file.stream) throw failed("cannot open regional match source");
            found = files_.emplace(pair.source, std::move(file)).first;
        }
        found->second.touched = ++clock_;
        if (!readPairMatches(found->second.stream, pair, out)) throw failed("cannot read regional match pair");
    }
};

inline RegionInput loadRegion(const MatchesIndex& index, const Region& region,
                              const std::filesystem::path& matches, const std::filesystem::path& features,
                              size_t budget, const Reconstruction* held = nullptr) {
    RegionInput out; out.global = region.images;
    std::vector<uint32_t> local(index.images.size(), UINT32_MAX);
    uint64_t input = 0;
    for (uint32_t id : out.global) input += uint64_t(index.images.at(id).num_features) * (31 + (held ? sizeof(Vec2) : 0));
    for (uint32_t i = 0; i < out.global.size(); ++i) local.at(out.global[i]) = i;
    for (const auto& pair : index.pairs)
        if (local.at(pair.image1) != UINT32_MAX && local.at(pair.image2) != UINT32_MAX)
            input += uint64_t(pair.count) * 32;
    if (input > budget / 2) throw RegionInputOverBudget();
    const auto& metadata = index.metadata;
    out.db.cameras = metadata.cameras; out.db.focal_prior = metadata.focal_prior;
    out.db.focal_measured = metadata.focal_measured;
    for (uint32_t g : out.global) {
        out.db.images.push_back(index.images[g]); out.db.camera_ids.push_back(metadata.camera_ids.at(g));
    }
    RegionalMatchReader reader(index, matches);
    for (const auto& pair : index.pairs) {
        uint32_t a = local.at(pair.image1), b = local.at(pair.image2);
        if (a == UINT32_MAX || b == UINT32_MAX) continue;
        cancel::check(); TwoViewMatches p; p.image1 = a; p.image2 = b; p.config = pair.config;
        reader.read(pair, p.matches.mut());
        out.db.pairs.push_back(std::move(p));
    }
    FeatureCompactionPlan compact = buildFeatureCompactionPlan(out.db);
    if (held) {
        const HeldRepairSeedContract heldContract(*held);(void)heldContract;
        for (const auto& image : held->images) if (image.second.registered) {
            if (image.first >= local.size() || local[image.first] == UINT32_MAX ||
                image.second.points2D.size() != index.images.at(image.first).num_features)
                throw std::runtime_error("held regional feature closure omitted an original image or measurement table");
            auto& mapping = compact.old_to_new.at(local[image.first]);
            for (size_t feature = 0; feature < image.second.point3D_ids.size(); ++feature)
                if (image.second.point3D_ids[feature] != kInvalidPoint3D) mapping.at(feature) = 0;
        }
        compact.stats.compact_features = compact.stats.zero_feature_images = 0;
        for (size_t image = 0; image < compact.old_to_new.size(); ++image) {
            uint64_t next = 0;
            for (auto& feature : compact.old_to_new[image]) if (feature != kUnusedFeature)
                feature = feature_compaction_detail::checkedCompactIndex(next++);
            compact.compact_counts[image] = uint32_t(next); compact.stats.compact_features += next;
            compact.stats.zero_feature_images += next == 0;
        }
    }
    out.original.resize(out.global.size()); out.features.reserve(out.global.size());
    if (held) out.original_xy.resize(out.global.size());
    for (uint32_t i = 0; i < out.global.size(); ++i) {
        cancel::check();
        for (uint32_t f = 0; f < compact.old_to_new[i].size(); ++f)
            if (compact.old_to_new[i][f] != kUnusedFeature) out.original[i].push_back(f);
        FeatureSet fs = readFeatures((features / (index.images[out.global[i]].name + ".bin")).string(), false);
        if (held) {
            auto& xy = out.original_xy.at(i); xy.reserve(fs.count());
            for (const auto& point : fs.keypoints) {
                if (!std::isfinite(point.x) || !std::isfinite(point.y)) throw std::runtime_error("held context original feature XY nonfinite");
                xy.push_back({double(point.x),double(point.y)});
            }
            const auto expected = held->images.find(out.global[i]);
            if (expected != held->images.end() && expected->second.registered) {
                const auto& image = expected->second;
                if (fs.count() != image.points2D.size()) throw std::runtime_error("held original feature-file count changed");
                const auto& camera = held->cameras.at(image.camera_id);
                if (fs.width != camera.width || fs.height != camera.height || fs.exif_orientation != image.exif_orientation ||
                    std::abs(fs.pixelScale() - camera.pixel_scale) > 1e-12 * std::max(1.0,camera.pixel_scale))
                    throw std::runtime_error("held original feature-file measurement frame changed");
                for (size_t feature = 0; feature < fs.count(); ++feature) {
                    const auto xy = image.points2D[feature]; const auto& actual = fs.keypoints[feature];
                    if (xy.x != actual.x || xy.y != actual.y || !std::isfinite(xy.x) || !std::isfinite(xy.y))
                        throw std::runtime_error("held original feature-file measurement differs from accepted seed");
                }
            }
        }
        out.features.push_back(compactFeatureSet(std::move(fs), compact.old_to_new[i], compact.compact_counts[i]));
    }
    remapMatches(out.db, compact, out.features);
    return out;
}

inline void restoreFeatureIds(Reconstruction& rec, const RegionInput& input, const MatchesIndex& index,
                              const Reconstruction* held = nullptr) {
    for (auto& kv : rec.points3D)
        for (TrackElement& e : kv.second.track) e.point2D_idx = input.original.at(e.image_id).at(e.point2D_idx);
    for (auto& kv : rec.images) {
        Image& im = kv.second; if (!im.registered) continue;
        uint32_t global = input.global.at(kv.first);
        std::vector<Vec2> xy(index.images.at(global).num_features);
        std::vector<uint64_t> ids(xy.size(), kInvalidPoint3D);
        for (uint32_t f = 0; f < im.points2D.size(); ++f) {
            uint32_t old = input.original.at(kv.first).at(f); xy.at(old) = im.points2D[f]; ids.at(old) = im.point3D_ids.at(f);
        }
        if (!input.original_xy.empty()) {
            if (input.original_xy.size() != input.global.size()) throw std::runtime_error("regional original measurement closure image count changed");
            const auto& source = input.original_xy.at(kv.first);
            if (source.size() != xy.size()) throw std::runtime_error("regional original measurement closure feature count changed");
            for (uint32_t f = 0; f < im.points2D.size(); ++f) {
                const auto old = input.original.at(kv.first).at(f);
                if (xy[old].x != source[old].x || xy[old].y != source[old].y)
                    throw std::runtime_error("loaded original feature XY changed before global measurement closure");
            }
            xy = source;
        }
        if (held) {
            if (input.original_xy.empty()) throw std::runtime_error("held global restoration lacks original FeatureBin measurement evidence");
            const auto expected = held->images.find(global);
            if (expected != held->images.end() && expected->second.registered) {
                if (expected->second.points2D.size() != xy.size()) throw std::runtime_error("held global feature count changed");
                for (size_t f = 0; f < xy.size(); ++f) {
                    const auto before = expected->second.points2D.at(f);
                    if (xy[f].x != before.x || xy[f].y != before.y)
                        throw std::runtime_error("held loaded feature XY changed before global measurement closure");
                }
            }
        }
        im.points2D = std::move(xy); im.point3D_ids = std::move(ids);
    }
    detail::toGlobalIds(rec, input.global);
    if (held) HeldRepairSeedContract(*held).requireUnchanged(rec,"regional.held-feature-global-closure");
}
} // namespace sfm::regional
