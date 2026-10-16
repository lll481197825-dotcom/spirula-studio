#pragma once

#include "sfm/map/CanonicalScene.h"
#include <queue>
#include <chrono>
#include <cwctype>

namespace sfm::regional {
namespace scene_index_detail {
using Path = std::filesystem::path;
using Reader = model_store_detail::Reader;
using Writer = model_store_detail::Writer;
using Progress = std::function<void(const char*, uint64_t, uint64_t)>;
inline Progress throttledProgress(Progress callback) {
    if (!callback) return {};
    struct State { std::string phase; std::chrono::steady_clock::time_point last{}; };
    auto state = std::make_shared<State>();
    return [callback = std::move(callback), state](const char* phase, uint64_t done, uint64_t total) {
        const auto now = std::chrono::steady_clock::now();
        if (state->phase != phase || done == total || now - state->last >= std::chrono::seconds(2)) { callback(phase, done, total); state->phase = phase; state->last = now; }
    };
}

template<class Row, class Less> Path sortFile(const Path& input, const Path& directory, size_t allowance, Less less) {
    if (allowance < (8u << 20)) throw std::length_error("scene index sort requires 8 MiB");
    const uint64_t bytes = std::filesystem::file_size(input);
    if (bytes % sizeof(Row)) throw std::runtime_error("scene index partial record");
    std::filesystem::create_directories(directory);
    const size_t capacity = std::max<size_t>(1, (allowance - (3u << 20)) / sizeof(Row));
    std::vector<Row> buffer; buffer.reserve(capacity);
    std::vector<Path> runs;
    {
        Reader in(input); uint64_t remaining = bytes / sizeof(Row);
        do {
            cancel::check(); const size_t take = size_t(std::min<uint64_t>(remaining, capacity));
            buffer.resize(take); if (take) in.read(buffer.data(), take);
            std::sort(buffer.begin(), buffer.end(), less);
            const Path run = directory / ("run-" + std::to_string(runs.size()) + ".bin");
            Writer out(run); for (const auto& row : buffer) out.scalar(row); out.finish(); runs.push_back(run);
            remaining -= take;
        } while (remaining);
        in.finish();
    }
    buffer.clear(); buffer.shrink_to_fit();
    uint64_t serial = runs.size();
    while (runs.size() > 1) {
        std::vector<Path> next;
        for (size_t i = 0; i < runs.size(); i += 2) {
            cancel::check(); if (i + 1 == runs.size()) { next.push_back(runs[i]); continue; }
            const Path merged = directory / ("run-" + std::to_string(serial++) + ".bin");
            {
                Reader a(runs[i]), b(runs[i + 1]); Writer out(merged);
                uint64_t na = std::filesystem::file_size(runs[i]) / sizeof(Row), nb = std::filesystem::file_size(runs[i + 1]) / sizeof(Row);
                Row ra{}, rb{}; if (na) ra = a.scalar<Row>(); if (nb) rb = b.scalar<Row>();
                while (na || nb) {
                    if (na && (!nb || !less(rb, ra))) { out.scalar(ra); if (--na) ra = a.scalar<Row>(); }
                    else { out.scalar(rb); if (--nb) rb = b.scalar<Row>(); }
                }
                a.finish(); b.finish(); out.finish();
            }
            std::filesystem::remove(runs[i]); std::filesystem::remove(runs[i + 1]); next.push_back(merged);
        }
        runs = std::move(next);
    }
    return runs.front();
}
}

class ImmutableSceneTopology {
public:
    using PointKey = RegionalWorkingStore::PointKey;
    using Path = std::filesystem::path;
    using Reader = model_store_detail::Reader;
    using Writer = model_store_detail::Writer;
    struct Row {
        uint64_t canonical = 0;
        PointKey point;
        uint64_t rawOrdinal = 0;
        uint32_t image = 0, feature = 0;
        Vec2 xy;
    };
    struct Range { uint64_t first = 0, count = 0; };
    struct PhotoPosting { uint64_t row = 0, root = 0; };
    struct Seed { uint64_t root = 0, weight = 0; Vec3 xyz; };
    using Canonical = std::function<uint64_t(PointKey)>;
    using SeedResolver = std::function<Vec3(PointKey, Vec3)>;
    using Progress = std::function<void(const char*, uint64_t, uint64_t)>;
    ImmutableSceneTopology(const Path& root, size_t budget) : root_(std::filesystem::absolute(root)), budget_(budget) {
        if (budget < (16u << 20)) throw std::length_error("scene index requires 16 MiB");
        std::filesystem::create_directories(root_);
    }
    static std::string identityFor(const RegionalWorkingStore& catalog, const std::vector<uint32_t>& owners,
                                  const std::string& topologyIdentity, uint64_t slots, const std::string& seedIdentity = {}) {
        spirula::Sha256 hash;
        auto append = [&](const std::string& value) { uint64_t n = value.size(); hash.update(reinterpret_cast<const uint8_t*>(&n), sizeof(n)); hash.update(reinterpret_cast<const uint8_t*>(value.data()), value.size()); };
        append("immutable-scene-csr-v2"); append(topologyIdentity); append(seedIdentity);
        hash.update(reinterpret_cast<const uint8_t*>(&slots), sizeof(slots));
        hash.update(reinterpret_cast<const uint8_t*>(owners.data()), owners.size() * sizeof(uint32_t));
        uint64_t count = 0, sourcePoints = 0;
        for (size_t source = 0; source < catalog.sourceCount(); ++source) { append(catalog.sourceLabel(source)); append(catalog.sourceRawIdentity(source)); count = storage::addBytes(count, catalog.sourceObservationCount(source)); sourcePoints = storage::addBytes(sourcePoints, catalog.sourcePointCount(source)); }
        if (sourcePoints != slots) throw std::runtime_error("scene source-global point layout differs");
        return hash.hex();
    }
    void prepare(const RegionalWorkingStore& catalog, const std::vector<uint32_t>& owners,
                 const std::string& topologyIdentity, uint64_t slots, const Canonical& canonical,
                 const SeedResolver& seedResolver = {}, const std::string& seedIdentity = {}, Progress progress = {}) {
        progress = scene_index_detail::throttledProgress(std::move(progress));
        const std::string expected = identityFor(catalog, owners, topologyIdentity, slots, seedIdentity);
        uint64_t count = 0;
        for (size_t source = 0; source < catalog.sourceCount(); ++source) count = storage::addBytes(count, catalog.sourceObservationCount(source));
        if (identity_ == expected) return;
        readers_.clear(); const Path target = root_ / expected;
        if (std::filesystem::exists(target / "manifest.bin")) { open(target, expected); return; }
        const std::string partial = expected + ".part-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        Path stage = root_ / partial; for (uint64_t suffix = 1; std::filesystem::exists(stage); ++suffix) stage = root_ / (partial + "-" + std::to_string(suffix));
        const uint64_t disk = storage::addBytes(storage::scaleBytes(count, sizeof(Row) * 3 + 40), storage::scaleBytes(slots, 100));
        storage::requireSpace(root_, disk); std::filesystem::create_directories(stage);
        struct Posting { uint32_t image = 0, reserved = 0; uint64_t row = 0, root = 0; };
        uint64_t scanned = 0;
        {
            Writer raw(stage / "input.bin"), seeds(stage / "seed-input.bin"), aliases(stage / "aliases.bin");
            for (size_t source = 0; source < catalog.sourceCount(); ++source) {
                catalog.visitImmutableSourcePointsDetailed(source, [&](PointKey key, const auto&, Vec3 imported, const auto& track) {
                    cancel::check(); const uint64_t root = canonical(key); if (root >= slots) throw std::runtime_error("canonical root outside scene layout");
                    aliases.scalar(key); aliases.scalar(root);
                    const Vec3 seed = seedResolver ? seedResolver(key, imported) : imported;
                    if (!std::isfinite(seed.norm())) throw std::runtime_error("nonfinite scene seed");
                    seeds.scalar(Seed{root, std::max<uint64_t>(1, track.size()), seed});
                    for (const auto& observation : track) {
                        if (observation.image >= owners.size() || owners[observation.image] == UINT32_MAX || !(observation.point == key)) throw std::runtime_error("scene raw row has no unique photo owner");
                        raw.scalar(Row{root, key, observation.ordinal, observation.image, observation.feature, observation.xy}); ++scanned;
                    }
                });
                if (progress) progress("source-scan", source + 1, catalog.sourceCount());
            }
            raw.finish(); seeds.finish(); aliases.finish();
        }
        if (scanned != count) throw std::runtime_error("scene original observation coverage differs");
        auto pointSort = [](const Row& a, const Row& b) { return std::tie(a.canonical, a.image, a.feature, a.point.source, a.point.ordinal, a.rawOrdinal) < std::tie(b.canonical, b.image, b.feature, b.point.source, b.point.ordinal, b.rawOrdinal); };
        if (progress) progress("point-sort", 0, count);
        const Path ordered = scene_index_detail::sortFile<Row>(stage / "input.bin", stage / "point-sort", budget_ / 2, pointSort);
        std::filesystem::rename(ordered, stage / "rows.bin");
        {
            Reader in(stage / "rows.bin"); Writer ranges(stage / "point-ranges.bin"), pointOwners(stage / "point-owners.bin"), postings(stage / "posting-input.bin");
            uint64_t cursor = 0; Row row{}; if (count) row = in.scalar<Row>();
            for (uint64_t root = 0; root < slots; ++root) {
                const uint64_t first = cursor; uint32_t owner = UINT32_MAX; bool seen = false, mixed = false;
                while (cursor < count && row.canonical == root) {
                    uint32_t next = owners.at(row.image); if (seen && owner != next) mixed = true; else if (!seen) owner = next; seen = true;
                    postings.scalar(Posting{row.image, 0, cursor, row.canonical}); if (++cursor < count) row = in.scalar<Row>();
                }
                ranges.scalar(Range{first, cursor - first}); pointOwners.scalar(mixed ? UINT32_MAX : owner);
            }
            if (cursor != count) throw std::runtime_error("scene canonical row range incomplete");
            in.finish(); ranges.finish(); pointOwners.finish(); postings.finish();
        }
        if (progress) progress("photo-sort", 0, count);
        const Path photos = scene_index_detail::sortFile<Posting>(stage / "posting-input.bin", stage / "photo-sort", budget_ / 2,
            [](const Posting& a, const Posting& b) { return std::tie(a.image, a.row) < std::tie(b.image, b.row); });
        {
            Reader in(photos); Writer ranges(stage / "photo-ranges.bin"), postings(stage / "photo-rows.bin");
            uint64_t cursor = 0; Posting row{}; if (count) row = in.scalar<Posting>();
            for (uint64_t image = 0; image < owners.size(); ++image) {
                const uint64_t first = cursor;
                while (cursor < count && row.image == image) { postings.scalar(PhotoPosting{row.row, row.root}); if (++cursor < count) row = in.scalar<Posting>(); }
                ranges.scalar(Range{first, cursor - first});
            }
            if (cursor != count) throw std::runtime_error("scene photo CSR incomplete");
            in.finish(); ranges.finish(); postings.finish();
        }
        const Path seedRows = scene_index_detail::sortFile<Seed>(stage / "seed-input.bin", stage / "seed-sort", budget_ / 2,
            [](const Seed& a, const Seed& b) { return a.root < b.root; });
        {
            Reader in(seedRows); Writer out(stage / "seeds.bin"); const uint64_t n = std::filesystem::file_size(seedRows) / sizeof(Seed);
            uint64_t position = 0; Seed seed{}; if (n) seed = in.scalar<Seed>();
            while (position < n) {
                const uint64_t root = seed.root; uint64_t weight = 0; Vec3 sum{};
                while (position < n && seed.root == root) { weight = storage::addBytes(weight, seed.weight); sum = sum + seed.xyz * double(seed.weight); if (++position < n) seed = in.scalar<Seed>(); }
                const Vec3 mean = sum * (1.0 / double(weight)); if (!std::isfinite(mean.norm())) throw std::runtime_error("nonfinite canonical seed mean");
                out.scalar(Seed{root, weight, mean});
            }
            in.finish(); out.finish();
        }
        for (const auto& file : {stage / "input.bin", stage / "posting-input.bin", stage / "seed-input.bin", photos, seedRows}) std::filesystem::remove(file);
        {
            Writer manifest(stage / "manifest.bin"); manifest.scalar<uint64_t>(1); manifest.string(expected); manifest.scalar(slots); manifest.scalar(count); manifest.scalar<uint64_t>(owners.size());
            for (const char* name : files()) { manifest.string(name); manifest.scalar<uint64_t>(std::filesystem::file_size(stage / name)); manifest.string(spirula::sha256_file((stage / name).u8string())); }
            manifest.finish();
        }
        std::filesystem::rename(stage, target); open(target, expected); if (progress) progress("index-ready", count, count);
    }
    const std::string& identity() const { return identity_; }
    void restore(const std::string& identity) { if (identity_ == identity) return; readers_.clear(); open(root_ / identity, identity); }
    Path forkImmutable(const Path& destination) const {
        if (identity_.empty() || directory_.empty()) throw std::logic_error("scene index fork requires a verified immutable index");
        const auto source = std::filesystem::canonical(root_), target = std::filesystem::weakly_canonical(std::filesystem::absolute(destination));
        auto contains = [](const Path& parent, const Path& child) {
            auto a = parent.begin(), b = child.begin();
            for (; a != parent.end(); ++a, ++b) {
                if (b == child.end()) return false;
#ifdef _WIN32
                auto x = a->wstring(), y = b->wstring();
                std::transform(x.begin(), x.end(), x.begin(), [](wchar_t c) { return std::towlower(c); });
                std::transform(y.begin(), y.end(), y.begin(), [](wchar_t c) { return std::towlower(c); });
                if (x != y) return false;
#else
                if (*a != *b) return false;
#endif
            }
            return true;
        };
        if (contains(source, target) || contains(target, source) || std::filesystem::exists(target))
            throw std::runtime_error("immutable index fork requires a new destination outside the source tree");
        ImmutableSceneTopology verified(root_, budget_); verified.restore(identity_);
        const auto manifestHash = spirula::sha256_file((directory_ / "manifest.bin").u8string());
        const Path stage = target.parent_path() / (target.filename().u8string() + ".part-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        if (std::filesystem::exists(stage)) throw std::runtime_error("immutable index fork partial destination already exists");
        std::filesystem::create_directories(stage / identity_);
        for (const char* name : files()) {
            if (std::filesystem::is_symlink(directory_ / name)) throw std::runtime_error("immutable index fork source file is a symbolic link");
            storage::retainFile(directory_ / name, stage / identity_ / name);
        }
        if (std::filesystem::is_symlink(directory_ / "manifest.bin")) throw std::runtime_error("immutable index fork manifest is a symbolic link");
        storage::retainFile(directory_ / "manifest.bin", stage / identity_ / "manifest.bin");
        const auto seedContext = directory_ / "layout-seed-context.txt";
        if (std::filesystem::exists(seedContext)) {
            if (!std::filesystem::is_regular_file(seedContext) || std::filesystem::is_symlink(seedContext) || std::filesystem::file_size(seedContext) > 1024)
                throw std::runtime_error("immutable index fork layout context framing changed");
            const auto before = spirula::sha256_file(seedContext.u8string());
            const auto copy = stage / identity_ / seedContext.filename();
            storage::retainFile(seedContext, copy);
            if (spirula::sha256_file(copy.u8string()) != before || spirula::sha256_file(seedContext.u8string()) != before)
                throw std::runtime_error("immutable index fork layout context bytes changed");
        }
        ImmutableSceneTopology restored(stage, budget_); restored.restore(identity_);
        if (spirula::sha256_file((stage / identity_ / "manifest.bin").u8string()) != manifestHash)
            throw std::runtime_error("immutable index fork manifest bytes changed");
        std::filesystem::rename(stage, target);
        return target / identity_;
    }
    uint64_t pointSlots() const { return slots_; }
    uint64_t observationCount() const { return count_; }
    uint64_t imageSlots() const { return images_; }
    uint64_t residentBytes() const { return (uint64_t(readers_.size()) * (1u << 20)) + 4096; }
    Range pointRange(uint64_t root) const { if (root >= slots_) throw std::out_of_range("scene root"); return scalar<Range>("point-ranges.bin", root); }
    uint32_t pointOwner(uint64_t root) const { if (root >= slots_) throw std::out_of_range("scene root"); return scalar<uint32_t>("point-owners.bin", root); }
    template<class Visit> void visitPoint(uint64_t root, Visit visit) const {
        const auto range = pointRange(root); auto& in = reader("rows.bin"); in.seek(storage::scaleBytes(range.first, sizeof(Row)));
        for (uint64_t i = 0; i < range.count; ++i) { auto row = in.scalar<Row>(); if (row.canonical != root) throw std::runtime_error("scene point CSR identity differs"); visit(row); }
    }
    template<class Visit> void visitPhoto(uint32_t image, Visit visit) const {
        if (image >= images_) throw std::out_of_range("scene image"); const auto range = scalar<Range>("photo-ranges.bin", image);
        auto& postings = reader("photo-rows.bin"); postings.seek(storage::scaleBytes(range.first, sizeof(PhotoPosting)));
        auto& rows = reader("rows.bin");
        for (uint64_t i = 0; i < range.count; ++i) { const auto posting = postings.scalar<PhotoPosting>(); if (posting.row >= count_) throw std::runtime_error("scene photo posting outside raw ledger"); rows.seek(storage::scaleBytes(posting.row, sizeof(Row))); auto row = rows.scalar<Row>(); if (row.image != image || row.canonical != posting.root) throw std::runtime_error("scene photo posting differs"); visit(row); }
    }
    template<class Visit> void visitPhotoRoots(uint32_t image, Visit visit) const {
        if (image >= images_) throw std::out_of_range("scene image"); const auto range = scalar<Range>("photo-ranges.bin", image);
        auto& postings = reader("photo-rows.bin"); postings.seek(storage::scaleBytes(range.first, sizeof(PhotoPosting)));
        for (uint64_t i = 0; i < range.count; ++i) { const auto posting = postings.scalar<PhotoPosting>(); if (posting.row >= count_ || posting.root >= slots_) throw std::runtime_error("scene photo root posting outside immutable layout"); visit(posting.root); }
    }
    template<class Visit> void visitRaw(Visit visit) const { Reader in(directory_ / "rows.bin"); for (uint64_t i = 0; i < count_; ++i) { if (!(i & 65535)) cancel::check(); visit(in.scalar<Row>()); } in.finish(); }
    template<class Visit> void visitSeeds(Visit visit) const { const auto path = directory_ / "seeds.bin"; Reader in(path); for (uint64_t i = 0, n = std::filesystem::file_size(path) / sizeof(Seed); i < n; ++i) visit(in.scalar<Seed>()); in.finish(); }
    void emitAliases(const CanonicalScene::AliasEmit& emit) const { Reader in(directory_ / "aliases.bin"); for (uint64_t i = 0; i < slots_; ++i) { auto key = in.scalar<PointKey>(); emit(key, in.scalar<uint64_t>()); } in.finish(); }
private:
    static std::vector<const char*> files() { return {"rows.bin", "point-ranges.bin", "point-owners.bin", "photo-ranges.bin", "photo-rows.bin", "seeds.bin", "aliases.bin"}; }
    template<class T> T scalar(const char* name, uint64_t ordinal) const { auto& in = reader(name); in.seek(storage::scaleBytes(ordinal, sizeof(T))); return in.scalar<T>(); }
    Reader& reader(const char* name) const { auto& value = readers_[name]; if (!value) value = std::make_unique<Reader>(directory_ / name); return *value; }
    void open(const Path& directory, const std::string& expected) {
        Reader manifest(directory / "manifest.bin"); if (manifest.scalar<uint64_t>() != 1 || manifest.string() != expected) throw std::runtime_error("scene index manifest identity differs");
        const uint64_t slots = manifest.scalar<uint64_t>(), count = manifest.scalar<uint64_t>(), images = manifest.scalar<uint64_t>();
        for (const char* name : files()) if (manifest.string() != name || manifest.scalar<uint64_t>() != std::filesystem::file_size(directory / name) || manifest.string() != spirula::sha256_file((directory / name).u8string())) throw std::runtime_error("scene index file identity differs");
        manifest.finish();
        if (std::filesystem::file_size(directory / "rows.bin") != storage::scaleBytes(count, sizeof(Row)) ||
            std::filesystem::file_size(directory / "point-ranges.bin") != storage::scaleBytes(slots, sizeof(Range)) ||
            std::filesystem::file_size(directory / "point-owners.bin") != storage::scaleBytes(slots, sizeof(uint32_t)) ||
            std::filesystem::file_size(directory / "photo-ranges.bin") != storage::scaleBytes(images, sizeof(Range)) ||
            std::filesystem::file_size(directory / "photo-rows.bin") != storage::scaleBytes(count, sizeof(PhotoPosting)) ||
            std::filesystem::file_size(directory / "aliases.bin") != storage::scaleBytes(slots, sizeof(PointKey) + sizeof(uint64_t)) ||
            std::filesystem::file_size(directory / "seeds.bin") % sizeof(Seed)) throw std::runtime_error("scene index structural coverage differs");
        directory_ = directory; slots_ = slots; count_ = count; images_ = images; identity_ = expected;
    }
    Path root_, directory_; size_t budget_; uint64_t slots_ = 0, count_ = 0, images_ = 0; std::string identity_;
    mutable std::map<std::string, std::unique_ptr<Reader>> readers_;
};
static_assert(sizeof(ImmutableSceneTopology::Row) == 64, "scene raw row layout changed");
}
