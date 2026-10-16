#pragma once

#include "sfm/map/RegionalTopology.h"
#include <iomanip>

namespace sfm::regional {

struct PrivateGeometryObjective {
    uint64_t observations = 0, invalid = 0;
    double finite = 0, total = 0;
    PrivateGeometryObjective& operator+=(const PrivateGeometryObjective& b) {
        observations += b.observations; invalid += b.invalid; finite += b.finite; total += b.total; return *this;
    }
};
struct PrivateGeometryProgress {
    const char* phase = "";
    uint64_t source = 0, sourceTotal = 0, pointsDone = 0, pointsTotal = 0, groupsDone = 0, groupsTotal = 0;
};
struct PrivateGeometryResult {
    uint64_t visited = 0, changed = 0, rejected = 0, unresolved_alias_groups = 0;
    bool stagePareto = false;
    uint64_t invalid_to_valid_finite_tradeoff = 0, cache_raw_observations = 0, cache_reused_groups = 0, peakMemoryBytes = 0;
    PrivateGeometryObjective before, after;
    fs::path ledger;
};
inline bool privateGeometryAccepted(const PrivateGeometryResult& result) {
    const auto& before = result.before; const auto& after = result.after;
    return before.observations == after.observations && std::isfinite(before.finite) && std::isfinite(before.total) &&
        std::isfinite(after.finite) && std::isfinite(after.total) && after.invalid <= before.invalid &&
        after.finite <= before.finite * (1 + 1e-8) + 1e-8 && after.total <= before.total * (1 + 1e-8) + 1e-8;
}
using PrivateGeometryProgressCallback = std::function<void(const PrivateGeometryProgress&)>;

namespace private_geometry_detail {
using Key = RegionalWorkingStore::PointKey;
using Raw = RegionalWorkingStore::RawObservation;
using topology_storage::Chunks;
using topology_storage::Bits;
using topology_storage::Indices;
struct Row { uint64_t root = 0, slot = 0, feature = 0; Vec2 xy; };
struct Group { uint64_t root = 0, first = 0, count = 0, file = 0; };
struct File { std::string path, hash; uint64_t bytes = 0; };
inline bool ordered(const Row& a, const Row& b) {
    return std::tie(a.root, a.slot, a.feature, a.xy.x, a.xy.y) < std::tie(b.root, b.slot, b.feature, b.xy.x, b.xy.y);
}
inline std::string digest(const fs::path& path) {
    std::ifstream in(path, std::ios::binary); if (!in) throw std::runtime_error("cannot hash private support dependency");
    spirula::Sha256 hash; std::array<char, 65536> bytes;
    while (in) { in.read(bytes.data(), bytes.size()); hash.update(reinterpret_cast<const uint8_t*>(bytes.data()), size_t(in.gcount())); }
    if (!in.eof()) throw std::runtime_error("private support checksum read failed"); return hash.hex();
}
inline void verify(const File& file) {
    struct Seen { uint64_t bytes; fs::file_time_type time; std::string hash; };
    static std::mutex mutex; static std::map<std::string, Seen> seen;
    const auto path = fs::u8path(file.path); const auto bytes = fs::file_size(path); const auto time = fs::last_write_time(path);
    if (bytes != file.bytes) throw std::runtime_error("private support dependency size changed");
    std::lock_guard<std::mutex> lock(mutex); auto old = seen.find(file.path);
    if (old != seen.end() && old->second.bytes == bytes && old->second.time == time && old->second.hash == file.hash) return;
    if (digest(path) != file.hash) throw std::runtime_error("private support dependency checksum changed");
    seen[file.path] = {bytes, time, file.hash};
}
inline PrivateGeometryObjective observationCost(const Row& row, Vec3 xyz, const Reconstruction& poses, double gate) {
    const auto& image = poses.images.at(uint32_t(row.feature >> 32)); const auto& camera = poses.cameras.at(image.camera_id);
    if (!image.registered || !(camera.pixel_scale > 0) || !std::isfinite(camera.pixel_scale)) throw std::runtime_error("private support lacks valid fixed visual camera");
    bool valid = true; const auto error = reprojErrorAt(camera, image.pose, row.xy, xyz, &valid) / camera.pixel_scale;
    if (!std::isfinite(error)) throw std::runtime_error("nonfinite private residual");
    const double cost = error <= gate ? error * error : gate * (2 * error - gate);
    if (!std::isfinite(cost)) throw std::runtime_error("nonfinite private objective");
    return {1, valid ? uint64_t(0) : uint64_t(1), valid ? cost : 0., cost};
}
inline bool strict(const PrivateGeometryObjective& next, const PrivateGeometryObjective& previous) {
    return next.invalid <= previous.invalid && next.finite <= previous.finite && next.total <= previous.total;
}

class SupportCache {
    static constexpr uint64_t magic = 0x3156505053525353ull;
public:
    SupportCache(const fs::path& root, uint64_t budget) : root_(fs::absolute(root)), budget_(budget) { fs::create_directories(root_); }
    uint64_t bytes() const {
        uint64_t size = slots.bytes() + members.bytes() + groups.bytes() + aliases.bytes() + offsets.capacity() * 8 +
            files.capacity() * sizeof(File) + sources_.capacity() * sizeof(std::string) + readers_.size() * ((64u << 10) + 512) + identity_.capacity() + owner_.capacity();
        for (const auto& file : files) size += file.path.capacity() + file.hash.capacity(); for (const auto& source : sources_) size += source.capacity(); return size;
    }
    void guard(uint64_t extra = 0) {
        const auto bound = storage::addBytes(bytes(), extra); peak = std::max(peak, bound);
        if (bound > budget_) throw std::length_error("private geometry exceeds remaining host memory allowance");
    }
    uint64_t global(const Key& key) const {
        if (key.source + 1 >= offsets.size() || key.ordinal >= offsets[key.source + 1] - offsets[key.source]) throw std::out_of_range("private source point identity");
        return offsets[key.source] + key.ordinal;
    }
    uint64_t slot(const Key& key) const { return slots.at(size_t(global(key))); }
    void prepare(RegionalWorkingStore& store, const RegionalTopology& topology) {
        if (fs::exists(root_ / "head.bin")) load();
        guard((store.sourceCount() + 1) * 512);
        std::vector<std::string> sources; std::vector<uint64_t> nextOffsets{0};
        for (size_t source = 0; source < store.sourceCount(); ++source) {
            sources.push_back(store.sourceIdentity(source)); nextOffsets.push_back(storage::addBytes(nextOffsets.back(), store.sourcePointCount(source)));
        }
        if (nextOffsets.back() != topology.points()) throw std::runtime_error("private support topology source count mismatch");
        guard(Bits::allocationBytes(topology.points())); aliases = topology.privateAliasRoots();
        if (identity_ == topology.identity() && sources_ == sources && offsets == nextOffsets && owner_ == topology.ownerIdentity()) {
            validate(topology); return;
        }
        const bool append = owner_ == topology.ownerIdentity() && sources_.size() < sources.size() &&
            std::equal(sources_.begin(), sources_.end(), sources.begin()) && offsets.size() <= nextOffsets.size() && std::equal(offsets.begin(), offsets.end(), nextOffsets.begin());
        const size_t oldSources = append ? sources_.size() : 0, oldMembers = append ? members.size() : 0;
        if (!append) { slots.clear(); slots.shrink_to_fit(); members.release(); groups.release(); files.clear(); }
        const auto newSlotBytes = Indices::allocationBytes(topology.points()); guard(newSlotBytes);
        slots.resize(topology.points()); offsets = std::move(nextOffsets);
        guard(Bits::allocationBytes(topology.points())); Bits changed(topology.points());
        uint64_t temporary = changed.bytes() + (2u << 20);
        const auto stage = root_ / ("build-" + std::to_string(store.sourceCount()) + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())); fs::create_directory(stage);
        const auto delta = stage / "delta.bin", reused = stage / "reused.bin"; std::ofstream raw(delta, std::ios::binary), oldGroups(reused, std::ios::binary);
        auto addMember = [&](const Key& key) {
            const auto point = global(key), root = topology.canonicalPoint(point);
            if (!aliases[root] || slots[size_t(point)]) return;
            if (members.size() == members.capacity()) guard(temporary + Chunks<Key>::block * sizeof(Key) + 128);
            members.push_back(key); slots[size_t(point)] = members.size(); changed[size_t(root)] = true;
        };
        if (append) {
            for (size_t g = 0; g < groups.size(); ++g) { const auto root = topology.canonicalPoint(groups[g].root); if (root != groups[g].root && aliases[root]) changed[size_t(root)] = true; }
            for (size_t point = 0; point < offsets[oldSources]; ++point) {
                if (point % 1048576 == 0) cancel::check(); const auto root = topology.canonicalPoint(point);
                if (aliases[root] && !slots[point]) {
                    const size_t source = size_t(std::upper_bound(offsets.begin(), offsets.end(), point) - offsets.begin() - 1);
                    addMember(store.key(source, point - offsets[source]));
                }
            }
        }
        for (size_t source = oldSources; source < store.sourceCount(); ++source) {
            store.visitSourcePoints(source, [&](Key key, Vec3, const std::vector<Raw>& track) {
                cancel::check(); const auto root = topology.canonicalPoint(global(key)); if (!aliases[root]) return; addMember(key);
                const auto id = slot(key) - 1;
                for (const auto& obs : track) { record(raw, Row{root, id, (uint64_t(obs.image) << 32) | obs.feature, obs.xy}); ++newRaw; }
            });
            events::progress(Stage::Merge, source - oldSources + 1, store.sourceCount() - oldSources);
            slog::diag(slog::Tag::Map, "[private-support] immutable new source scan %zu/%zu, %llu full raw rows\n", source - oldSources + 1, store.sourceCount() - oldSources, (unsigned long long)newRaw);
        }
        if (append) {
            for (size_t slot = oldMembers; slot < members.size(); ++slot) {
                const auto key = members[slot]; if (key.source >= oldSources) continue; const auto root = topology.canonicalPoint(global(key));
                store.visitPoint(key, [&](const Raw& obs) { record(raw, Row{root, slot, (uint64_t(obs.image) << 32) | obs.feature, obs.xy}); ++newRaw; });
            }
            for (size_t g = 0; g < groups.size(); ++g) {
                cancel::check(); auto group = groups[g]; const auto root = topology.canonicalPoint(group.root); if (!aliases[root]) continue;
                if (!changed[size_t(root)]) { record(oldGroups, group); ++reusedGroups; continue; }
                visit(group, [&](Row row) { row.root = topology.canonicalPoint(global(members.at(size_t(row.slot)))); record(raw, row); });
            }
        }
        raw.flush(); oldGroups.flush(); if (!raw || !oldGroups) throw std::runtime_error("cannot flush private support delta"); raw.close(); oldGroups.close(); changed.release(); temporary = 2u << 20;
        const auto remaining = budget_ > bytes() + temporary ? budget_ - bytes() - temporary : 0;
        const auto sortBudget = std::min<uint64_t>(std::max<uint64_t>(sizeof(Row), fs::file_size(delta)), std::min<uint64_t>(256u << 20, remaining / 2)); if (sortBudget < sizeof(Row)) throw std::length_error("private support delta sort has no remaining allowance"); guard(temporary + sortBudget);
        const auto sorted = sort(delta, stage / "sort", size_t(sortBudget));
        const auto newGroups = stage / "groups.bin"; std::ofstream directory(newGroups, std::ios::binary); std::ifstream orderedRows(sorted, std::ios::binary);
        const auto fileIndex = files.size(); files.push_back({sorted.u8string(), digest(sorted), fs::file_size(sorted)});
        Row row; bool have = nextRecord(orderedRows, row); uint64_t first = 0;
        while (have) {
            cancel::check(); Group group{row.root, first, 0, fileIndex};
            do { ++group.count; ++first; have = nextRecord(orderedRows, row); } while (have && row.root == group.root);
            record(directory, group);
        }
        directory.flush(); if (!directory) throw std::runtime_error("cannot flush private support directory"); directory.close();
        groups.release(); std::ifstream before(reused, std::ios::binary), next(newGroups, std::ios::binary); Group a, b; bool hasA = nextRecord(before, a), hasB = nextRecord(next, b);
        while (hasA || hasB) {
            if (groups.size() == groups.capacity()) guard(temporary + Chunks<Group>::block * sizeof(Group) + 128);
            if (hasA && hasB && a.root == b.root) throw std::runtime_error("private support reused/delta root collision");
            if (hasA && (!hasB || a.root < b.root)) { groups.push_back(a); hasA = nextRecord(before, a); }
            else { groups.push_back(b); hasB = nextRecord(next, b); }
        }
        sources_ = std::move(sources); owner_ = topology.ownerIdentity(); identity_ = topology.identity(); validate(topology); save(stage);
        slog::diag(slog::Tag::Map, "[private-support] cache committed: %zu groups/%zu members, %llu new raw rows, %llu unchanged groups reused\n", groups.size(), members.size(), (unsigned long long)newRaw, (unsigned long long)reusedGroups);
    }
    template<class Visit> void visit(const Group& group, Visit callback) {
        const auto& file = files.at(size_t(group.file)); if (group.first > file.bytes / sizeof(Row) || group.count > file.bytes / sizeof(Row) - group.first) throw std::runtime_error("private support range invalid");
        auto found = readers_.find(group.file);
        if (found == readers_.end()) {
            if (readers_.size() == 16) {
                const auto oldest = std::min_element(readers_.begin(), readers_.end(), [](const auto& a, const auto& b) { return a.second.tick < b.second.tick; }); readers_.erase(oldest);
            }
            guard((64u << 10) + 512); ++readerOpens;
            found = readers_.emplace(group.file, CachedReader{std::make_shared<model_store_detail::Reader>(fs::u8path(file.path), 64u << 10), ++readerTick_}).first;
        } else { ++readerHits; found->second.tick = ++readerTick_; }
        auto& in = *found->second.reader; in.seek(storage::scaleBytes(group.first, sizeof(Row)));
        for (uint64_t i = 0; i < group.count; ++i) { const auto row = in.scalar<Row>(); if (row.root != group.root || row.slot >= members.size()) throw std::runtime_error("private support member identity corrupted"); callback(row); }
    }
    uint64_t peak = 0, newRaw = 0, reusedGroups = 0, readerOpens = 0, readerHits = 0;
    Indices slots; Chunks<Key> members; Chunks<Group> groups; Bits aliases; std::vector<uint64_t> offsets;
private:
    static fs::path sort(const fs::path& input, const fs::path& dir, size_t memory) {
        fs::create_directory(dir); storage::requireSpace(dir, storage::scaleBytes(fs::file_size(input), 3));
        std::ifstream in(input, std::ios::binary); if (!in) throw std::runtime_error("cannot open private support delta");
        std::vector<fs::path> runs; uint64_t sequence = 0; const auto capacity = std::max<size_t>(1, memory / sizeof(Row));
        for (;;) {
            cancel::check(); std::vector<Row> chunk; chunk.reserve(capacity); Row row;
            while (chunk.size() < capacity && nextRecord(in, row)) chunk.push_back(row); if (chunk.empty()) break;
            std::sort(chunk.begin(), chunk.end(), ordered); auto path = dir / (std::to_string(sequence++) + ".bin"); std::ofstream out(path, std::ios::binary);
            for (const auto& value : chunk) record(out, value); out.flush(); if (!out) throw std::runtime_error("private support sort write failed"); runs.push_back(path);
        }
        if (runs.empty()) { auto path = dir / "empty.bin"; std::ofstream out(path, std::ios::binary); return path; }
        while (runs.size() > 1) {
            std::vector<fs::path> next;
            for (size_t first = 0; first < runs.size(); first += 32) {
                struct Head { Row row; size_t stream; bool operator<(const Head& b) const { return ordered(b.row, row); } };
                const auto count = std::min<size_t>(32, runs.size() - first); std::vector<std::ifstream> inputs(count); std::priority_queue<Head> heap;
                for (size_t i = 0; i < count; ++i) { inputs[i].open(runs[first + i], std::ios::binary); if (!inputs[i]) throw std::runtime_error("cannot read private support run"); Row row; if (nextRecord(inputs[i], row)) heap.push({row, i}); }
                auto path = dir / (std::to_string(sequence++) + ".bin"); std::ofstream out(path, std::ios::binary); uint64_t processed = 0;
                while (!heap.empty()) { if (++processed % 65536 == 0) cancel::check(); auto head = heap.top(); heap.pop(); record(out, head.row); if (nextRecord(inputs[head.stream], head.row)) heap.push(head); }
                out.flush(); if (!out) throw std::runtime_error("private support merge write failed"); inputs.clear(); for (size_t i = 0; i < count; ++i) fs::remove(runs[first + i]); next.push_back(path);
            }
            runs = std::move(next);
        }
        return runs.front();
    }
    void validate(const RegionalTopology& topology) {
        guard(); if (offsets.empty() || offsets.back() != slots.size()) throw std::runtime_error("private support source ranges corrupted");
        for (size_t g = 0; g < groups.size(); ++g) { const auto& group = groups[g]; if ((g && groups[g - 1].root >= group.root) || group.root >= aliases.size() || !aliases[size_t(group.root)] || topology.canonicalPoint(group.root) != group.root || !group.count || group.file >= files.size()) throw std::runtime_error("private support group topology mismatch"); }
        for (size_t slot = 0; slot < members.size(); ++slot) if (slots.at(size_t(global(members[slot]))) != slot + 1) throw std::runtime_error("private support duplicate member identity");
    }
    void save(const fs::path& stage) {
        guard(2u << 20); const auto path = stage / "cache.bin"; model_store_detail::Writer out(path); out.scalar(magic); out.string(identity_); out.string(owner_);
        out.scalar<uint64_t>(sources_.size()); for (const auto& source : sources_) out.string(source); out.vector(offsets);
        out.scalar<uint64_t>(members.size()); for (size_t i = 0; i < members.size(); ++i) out.scalar(members[i]);
        out.scalar<uint64_t>(groups.size()); for (size_t i = 0; i < groups.size(); ++i) out.scalar(groups[i]);
        out.scalar<uint64_t>(files.size()); for (const auto& file : files) { out.string(file.path); out.string(file.hash); out.scalar(file.bytes); } out.finish();
        model_store_detail::Writer head(root_ / "head.part"); head.scalar(magic); head.string(path.u8string()); head.string(digest(path)); head.finish(); replaceFile(root_ / "head.part", root_ / "head.bin");
    }
    std::string string(model_store_detail::Reader& in) { const auto count = in.count(1); guard(count); std::string value(count, '\0'); in.read(value.data(), count); return value; }
    void load() {
        guard(2u << 20); model_store_detail::Reader head(root_ / "head.bin", 64u << 10); if (head.scalar<uint64_t>() != magic) throw std::runtime_error("private support head format mismatch");
        const auto path = string(head), hash = string(head); head.finish(); verify({path, hash, fs::file_size(fs::u8path(path))});
        model_store_detail::Reader in(fs::u8path(path), 64u << 10); if (in.scalar<uint64_t>() != magic) throw std::runtime_error("private support format mismatch"); identity_ = string(in); owner_ = string(in);
        auto count = in.count(8); guard(count * 128); for (size_t i = 0; i < count; ++i) sources_.push_back(string(in));
        count = in.count(8); guard(count * 8); offsets.resize(count); in.read(offsets.data(), count); if (offsets.empty() || offsets.front() || !std::is_sorted(offsets.begin(), offsets.end()) || offsets.size() != sources_.size() + 1) throw std::runtime_error("private support source ranges invalid");
        guard(Indices::allocationBytes(offsets.back())); slots.resize(size_t(offsets.back()));
        count = in.count(sizeof(Key)); guard((count + Chunks<Key>::block) * sizeof(Key)); members.reserve(count);
        for (size_t i = 0; i < count; ++i) { const auto key = in.scalar<Key>(); if (slots.at(size_t(global(key)))) throw std::runtime_error("private support repeated member"); members.push_back(key); slots[size_t(global(key))] = i + 1; }
        count = in.count(sizeof(Group)); guard((count + Chunks<Group>::block) * sizeof(Group)); groups.reserve(count); for (size_t i = 0; i < count; ++i) groups.push_back(in.scalar<Group>());
        count = in.count(24); guard(count * 256); for (size_t i = 0; i < count; ++i) { File file; file.path = string(in); file.hash = string(in); file.bytes = in.scalar<uint64_t>(); if (file.bytes % sizeof(Row)) throw std::runtime_error("private support track length invalid"); verify(file); files.push_back(std::move(file)); } in.finish();
    }
    fs::path root_; uint64_t budget_; std::string identity_, owner_; std::vector<std::string> sources_; struct CachedReader { std::shared_ptr<model_store_detail::Reader> reader; uint64_t tick; };
    std::vector<File> files; std::map<uint64_t, CachedReader> readers_; uint64_t readerTick_ = 0;
};

} // namespace private_geometry_detail
} // namespace sfm::regional

namespace sfm::regional {

// Refines complete affected private raw support; caller retains/accepts the pending
// XYZ generation only after this stage and the full original-observation audit.
inline PrivateGeometryResult refinePrivateGeometry(RegionalWorkingStore& store, const RegionalTopology& topology,
        const Reconstruction& fixedPoses, double gate, int threads, uint64_t memoryBytes, const fs::path& cacheRoot,
        const std::set<uint32_t>* affectedImages = nullptr, PrivateGeometryProgressCallback progress = {}) {
    using namespace private_geometry_detail;
    if (!(gate > 0) || !std::isfinite(gate) || memoryBytes < (8u << 20)) throw std::invalid_argument("private geometry needs a positive gate and 8 MiB remaining allowance");
    PrivateGeometryResult result;
    if (affectedImages) {
        bool allRegistered = true; for (const auto& item : fixedPoses.images) if (item.second.registered && !affectedImages->count(item.first)) { allRegistered = false; break; }
        if (allRegistered) affectedImages = nullptr;
    }
    const auto phase = fs::absolute(cacheRoot) / ("phase-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())); fs::create_directories(phase);
    result.ledger = phase / "private-geometry.csv";
    std::ofstream ledger(result.ledger); ledger << std::setprecision(17) << "kind,root_or_source,ordinal,observations,before_invalid,after_invalid,before_finite,after_finite,before_total,after_total,result,candidate_x,candidate_y,candidate_z\n";
    if (!ledger) throw std::runtime_error("cannot create private geometry ledger");
    events::stage_begin(Stage::Merge);
    struct EndStage { ~EndStage() { events::stage_end(Stage::Merge); } } stage;
    if (affectedImages && affectedImages->empty()) {
        result.stagePareto = true; ledger << "stage,0,0,0,0,0,0,0,0,0,no_projection_changes,0,0,0\n";
        ledger.flush(); if (!ledger) throw std::runtime_error("private empty-phase ledger flush failed"); return result;
    }
    SupportCache cache(cacheRoot, memoryBytes); cache.prepare(store, topology);
    result.cache_raw_observations = cache.newRaw; result.cache_reused_groups = cache.reusedGroups;
    uint64_t live = 2u << 20;
    Bits affected, affectedRoots; std::vector<uint8_t> affectedSources(store.sourceCount(), affectedImages ? 0 : 1); live += affectedSources.capacity();
    if (affectedImages) {
        cache.guard(live + 2 * Bits::allocationBytes(topology.points())); affected.resize(topology.points()); affectedRoots.resize(topology.points());
        live += affected.bytes() + affectedRoots.bytes();
        for (uint32_t image : *affectedImages) {
            cancel::check(); store.visitImageObservations(image, [&](const Raw& obs) {
                const auto point = cache.global(obs.point); affected[size_t(point)] = true; if (!topology.shared(obs.point)) affectedSources.at(size_t(obs.point.source)) = 1;
                const auto root = topology.canonicalPoint(point); if (cache.aliases[size_t(root)]) affectedRoots[size_t(root)] = true;
            });
        }
    }
    cache.guard(live + (cache.members.size() + Chunks<Vec3>::block) * sizeof(Vec3));
    Chunks<Vec3> values; values.reserve(cache.members.size()); live += values.bytes();
    for (size_t slot = 0; slot < cache.members.size(); ++slot) {
        if (slot % 65536 == 0) cancel::check(); const auto& key = cache.members[slot]; const auto root = topology.canonicalPoint(cache.global(key));
        const bool active = cache.aliases[size_t(root)] && (!affectedImages || affectedRoots[size_t(root)]);
        values.push_back(active ? store.pointXYZ(key) : Vec3{}); if (active) affectedSources.at(size_t(key.source)) = 1;
    }
    struct Job {
        Key key; uint64_t root = 0; Vec3 original, candidate;
        std::vector<Row> rows; std::vector<Observation> track;
        PrivateGeometryObjective before, after; bool accepted = false, tradeoff = false, noop = false;
    };
    std::vector<Job> jobs; uint64_t jobBytes = 0;
    const auto remaining = memoryBytes > cache.bytes() + live ? memoryBytes - cache.bytes() - live : 0;
    const auto batchLimit = std::min<uint64_t>(64u << 20, remaining / 3);
    if (batchLimit < sizeof(Job) + 2 * sizeof(Row) + 2 * sizeof(Observation) + 2 * sizeof(PointView)) throw std::length_error("private geometry has no remaining triangulation workspace");
    uint64_t done = 0, total = 0, groupsDone = 0, groupsTotal = 0; std::atomic<uint64_t> computedGroups{0};
    for (size_t source = 0; source < store.sourceCount(); ++source) total += store.sourcePointCount(source);
    for (size_t g = 0; g < cache.groups.size(); ++g) if (!affectedImages || affectedRoots[size_t(cache.groups[g].root)]) ++groupsTotal;
    std::mutex progressMutex; auto lastProgress = std::chrono::steady_clock::now();
    auto report = [&](const char* label, size_t source, bool force) {
        std::lock_guard<std::mutex> lock(progressMutex); const auto now = std::chrono::steady_clock::now(); if (!force && std::chrono::duration<double>(now - lastProgress).count() < 2) return;
        lastProgress = now; const auto completed = std::max(groupsDone, computedGroups.load()); PrivateGeometryProgress value{label, source, store.sourceCount(), done, total, completed, groupsTotal};
        events::progress(Stage::Merge, done + completed, total + groupsTotal); if (progress) progress(value);
        slog::diag(slog::Tag::Map, "[private-geometry] %s source %zu/%zu: %llu/%llu point ordinals, %llu/%llu full-support alias groups\n", label, source, store.sourceCount(), (unsigned long long)done, (unsigned long long)total, (unsigned long long)completed, (unsigned long long)groupsTotal);
    };
    auto addRows = [&](Job& job, const std::vector<Raw>& track) {
        const auto extra = track.size() * uint64_t(sizeof(Row) + sizeof(Observation) + sizeof(PointView)); cache.guard(live + jobBytes + extra + (jobs.capacity() * 2 + 1) * sizeof(Job));
        job.rows.reserve(track.size()); job.track.reserve(track.size());
        for (const auto& obs : track) { const auto feature = (uint64_t(obs.image) << 32) | obs.feature; job.rows.push_back({job.root, UINT64_MAX, feature, obs.xy}); job.track.push_back({job.root, feature, obs.xy}); }
        jobBytes += extra + sizeof(Job);
    };
    auto solve = [&](bool aliases) {
        cache.guard(live + jobBytes + jobs.capacity() * sizeof(Job));
        auto& pool = bacpu::Pool::get(); const int workers = threads > 0 ? std::min(threads, pool.size()) : pool.size();
        pool.runChecked(int(jobs.size()), workers, [&](int index, int) {
            cancel::check(); auto& job = jobs[size_t(index)]; if (job.noop) { job.candidate = job.original; job.accepted = true; return; }
            for (const auto& row : job.rows) job.before += observationCost(row, aliases ? values.at(size_t(row.slot)) : job.original, fixedPoses, gate);
            job.candidate = refineUnifiedPoint(job.original, job.track, fixedPoses, gate);
            for (const auto& row : job.rows) job.after += observationCost(row, job.candidate, fixedPoses, gate);
            job.tradeoff = job.after.invalid < job.before.invalid && job.after.finite > job.before.finite && job.after.total <= job.before.total;
            job.accepted = std::isfinite(job.candidate.norm()) && (strict(job.after, job.before) || job.tradeoff);
            if (aliases) { ++computedGroups; report("alias triangulation", 0, false); }
        });
        for (auto& job : jobs) {
            if (job.noop) continue;
            result.before += job.before; result.after += job.accepted ? job.after : job.before; result.invalid_to_valid_finite_tradeoff += job.tradeoff;
            if (aliases) {
                ++groupsDone; ++result.visited;
                if (job.accepted) {
                    uint64_t previous = UINT64_MAX;
                    for (const auto& row : job.rows) if (row.slot != previous) { result.changed += (values.at(size_t(row.slot)) - job.candidate).norm() > 0; values.at(size_t(row.slot)) = job.candidate; previous = row.slot; }
                } else { ++result.rejected; ++result.unresolved_alias_groups; }
            } else { ++result.visited; result.changed += job.accepted && (job.original - job.candidate).norm() > 0; result.rejected += !job.accepted; }
            if (!job.accepted || job.tradeoff) ledger << (aliases ? "alias" : "single") << ',' << (aliases ? job.root : job.key.source) << ',' << job.key.ordinal << ',' << job.before.observations << ',' << job.before.invalid << ',' << job.after.invalid << ',' << job.before.finite << ',' << job.after.finite << ',' << job.before.total << ',' << job.after.total << ',' << (job.tradeoff ? "invalid_to_valid_finite_tradeoff_candidate" : "unresolved_kept_original") << ',' << job.candidate.x << ',' << job.candidate.y << ',' << job.candidate.z << '\n';
        }
    };
    // Alias support is immutable; original XYZ remains per member until a complete group passes.
    for (size_t g = 0; g < cache.groups.size(); ++g) {
        cancel::check(); const auto group = cache.groups[g]; if (affectedImages && !affectedRoots[size_t(group.root)]) continue;
        const auto needed = storage::scaleBytes(group.count, sizeof(Row) + sizeof(Observation) + sizeof(PointView)) + sizeof(Job);
        if (!jobs.empty() && jobBytes + needed > batchLimit) { solve(true); jobs.clear(); jobBytes = 0; report("alias triangulation", 0, false); }
        cache.guard(live + jobBytes + needed + (jobs.capacity() * 2 + 1) * sizeof(Job)); Job job; job.root = group.root; job.rows.reserve(size_t(group.count)); job.track.reserve(size_t(group.count));
        Vec3 sum{};
        cache.visit(group, [&](Row row) { job.rows.push_back(row); job.track.push_back({row.root, row.feature, row.xy}); sum = sum + values.at(size_t(row.slot)); });
        job.original = sum * (1. / double(group.count)); jobBytes += needed; jobs.push_back(std::move(job));
        if (jobs.size() >= 8192) { solve(true); jobs.clear(); jobBytes = 0; report("alias triangulation", 0, false); }
    }
    if (!jobs.empty()) { solve(true); jobs.clear(); jobBytes = 0; } report("alias triangulation", 0, true);
    for (size_t source = 0; source < store.sourceCount(); ++source) {
        cancel::check(); uint64_t sourceDone = 0;
        if (!affectedSources.at(source)) { done += store.sourcePointCount(source); report("unchanged source parameters reused", source + 1, false); continue; }
        store.rewriteSourceXYZ(source, [&](auto emit) {
            auto flush = [&] {
                if (jobs.empty()) return; solve(false);
                for (const auto& job : jobs) emit(job.accepted ? job.candidate : job.original);
                done += jobs.size(); sourceDone += jobs.size(); jobs.clear(); jobBytes = 0; report("single-alias triangulation", source + 1, false);
            };
            auto visit = [&](Key key, Vec3 xyz, const std::vector<Raw>* rawTrack) {
                cancel::check(); const auto point = cache.global(key), root = topology.canonicalPoint(point);
                if (topology.shared(key) || cache.aliases[size_t(root)] || (affectedImages && !affected[size_t(point)])) {
                    if (!jobs.empty() && jobBytes + sizeof(Job) > batchLimit) flush();
                    const auto slot = cache.slot(key); Job job; job.key = key; job.noop = true;
                    job.original = cache.aliases[size_t(root)] && slot && (!affectedImages || affectedRoots[size_t(root)]) ? values.at(size_t(slot - 1)) : xyz;
                    cache.guard(live + jobBytes + (jobs.capacity() * 2 + 1) * sizeof(Job)); jobBytes += sizeof(Job); jobs.push_back(std::move(job));
                    if (jobs.size() >= 8192) flush(); return;
                }
                std::vector<Raw> track;
                if (!rawTrack) { store.visitPoint(key, [&](const Raw& row) { cache.guard(live + jobBytes + (track.capacity() * 2 + 1) * sizeof(Raw)); track.push_back(row); }); rawTrack = &track; }
                const auto need = rawTrack->size() * uint64_t(sizeof(Row) + sizeof(Observation) + sizeof(PointView)) + sizeof(Job);
                if (!jobs.empty() && jobBytes + need > batchLimit) flush(); Job job; job.key = key; job.root = root; job.original = xyz; addRows(job, *rawTrack); jobs.push_back(std::move(job));
                if (jobs.size() >= 8192) flush();
            };
            if (!affectedImages) store.visitSourcePoints(source, [&](Key key, Vec3 xyz, const std::vector<Raw>& raw) { visit(key, xyz, &raw); });
            else store.visitSourceParameters(source, [&](Key key, Vec3 xyz) { visit(key, xyz, nullptr); });
            flush();
        });
        if (sourceDone != store.sourcePointCount(source)) throw std::runtime_error("private XYZ producer point coverage mismatch"); report("source XYZ complete", source + 1, true);
    }
    result.stagePareto = privateGeometryAccepted(result);
    ledger << "stage,0,0," << result.before.observations << ',' << result.before.invalid << ',' << result.after.invalid << ',' << result.before.finite << ',' << result.after.finite << ',' << result.before.total << ',' << result.after.total << ',' << (result.stagePareto ? "stage_pareto_candidate" : "stage_pareto_rejected_candidate_retained") << ",0,0,0\n";
    ledger.flush(); if (!ledger) throw std::runtime_error("private geometry ledger flush failed"); result.peakMemoryBytes = cache.peak;
    slog::diag(slog::Tag::Map, "[private-geometry] affected private complete raw support (final all-source workingQuality still required): %llu observations, invalid %llu -> %llu, finite %.17g -> %.17g, total %.17g -> %.17g; original sources/cameras/poses unchanged\n", (unsigned long long)result.before.observations, (unsigned long long)result.before.invalid, (unsigned long long)result.after.invalid, result.before.finite, result.after.finite, result.before.total, result.after.total);
    return result;
}

} // namespace sfm::regional
