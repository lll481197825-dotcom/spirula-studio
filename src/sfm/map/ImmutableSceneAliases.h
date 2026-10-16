#pragma once
#include "sfm/map/SceneTopology.h"
#include "core/HostMemory.h"
#include <numeric>

namespace sfm::regional {
class ImmutableSceneAliases {
public:
    using PointKey = RegionalWorkingStore::PointKey;
    using Path = std::filesystem::path;
    using Reader = model_store_detail::Reader;
    using Writer = model_store_detail::Writer;
    using Progress = ImmutableSceneTopology::Progress;
    ImmutableSceneAliases(const Path& root, size_t budget) : root_(std::filesystem::absolute(root)), budget_(budget) {
        if (!budget_) {
            const auto physical = spirula::physicalRamBytes(), available = spirula::availableRamBytes();
            if (!physical || !available) throw std::runtime_error("alias auto budget needs actual physical memory");
            budget_ = std::min<size_t>(physical / 5 * 4, available - std::min<size_t>(size_t(4) << 30, available / 4));
        }
        if (budget_ < (16u << 20)) throw std::length_error("immutable alias index requires 16 MiB");
        std::filesystem::create_directories(root_);
    }
    static std::string dependencyIdentity(const RegionalWorkingStore& catalog, const std::vector<uint32_t>& owners) {
        spirula::Sha256 hash;
        append(hash, "immutable-scene-aliases-v1"); const uint64_t images = owners.size();
        hash.update(reinterpret_cast<const uint8_t*>(&images), sizeof(images)); hash.update(reinterpret_cast<const uint8_t*>(owners.data()), owners.size() * sizeof(uint32_t));
        for (size_t source = 0; source < catalog.sourceCount(); ++source) {
            append(hash, catalog.sourceLabel(source)); append(hash, catalog.sourceIdentity(source)); append(hash, catalog.sourceRawIdentity(source));
            const uint64_t points = catalog.sourcePointCount(source), raw = catalog.sourceObservationCount(source);
            hash.update(reinterpret_cast<const uint8_t*>(&points), sizeof(points)); hash.update(reinterpret_cast<const uint8_t*>(&raw), sizeof(raw));
        }
        return hash.hex();
    }
    void prepare(const RegionalWorkingStore& catalog, const std::vector<uint32_t>& owners, Progress progress = {}) {
        catalog.verifyImmutable(); catalog_ = &catalog;
        std::vector<uint64_t> offsets{0}; uint64_t observations = 0;
        for (size_t source = 0; source < catalog.sourceCount(); ++source) {
            offsets.push_back(storage::addBytes(offsets.back(), catalog.sourcePointCount(source)));
            observations = storage::addBytes(observations, catalog.sourceObservationCount(source));
        }
        const auto expected = dependencyIdentity(catalog, owners); const uint64_t count = offsets.back();
        if (count > SIZE_MAX) throw std::length_error("immutable aliases exceed host address space");
        const uint64_t mapBytes = storage::scaleBytes(count, sizeof(uint64_t));
        const uint64_t fixed = storage::addBytes(catalog.residentBytes(), mapBytes + offsets.capacity() * sizeof(uint64_t) + (4u << 20));
        if (fixed + (8u << 20) > budget_) throw std::length_error("immutable alias map plus sort minimum exceeds budget");
        requirePhysical(mapBytes + (4u << 20));
        if (identity_ == expected) return;
        roots_.clear(); roots_.shrink_to_fit(); offsets_ = std::move(offsets); observations_ = observations;
        const Path target = root_ / expected;
        if (std::filesystem::exists(target / "manifest.bin")) { open(target, expected, count, observations); return; }
        const Path stage = root_ / (expected + ".part-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        storage::requireSpace(root_, storage::addBytes(storage::scaleBytes(observations, sizeof(Link) * 3), storage::scaleBytes(count, sizeof(uint64_t))));
        std::filesystem::create_directory(stage); roots_.resize(size_t(count)); std::iota(roots_.begin(), roots_.end(), uint64_t(0));
        {
            Writer out(stage / "links-input.bin"); uint64_t done = 0; PointKey previous{}; bool havePrevious = false;
            catalog.visitObservations([&](const RegionalWorkingStore::RawObservation& row) {
                if (!(done & 65535)) { cancel::check(); if (progress) progress("alias-raw-scan", done, observations); }
                if (row.image >= owners.size() || owners[row.image] == UINT32_MAX || row.point.source >= catalog.sourceCount() || row.point.ordinal >= catalog.sourcePointCount(size_t(row.point.source)))
                    throw std::runtime_error("immutable alias raw point/photo owner invalid");
                if (!havePrevious || !(row.point == previous)) {
                    if (!(catalog.key(size_t(row.point.source), row.point.ordinal) == row.point)) throw std::runtime_error("immutable alias original local point ID differs");
                    previous = row.point; havePrevious = true;
                }
                if (!std::isfinite(row.xy.x) || !std::isfinite(row.xy.y)) throw std::runtime_error("immutable alias raw XY nonfinite");
                out.scalar(Link{(uint64_t(row.image) << 32) | row.feature, offsets_.at(size_t(row.point.source)) + row.point.ordinal, row.xy}); ++done;
            });
            if (done != observations) throw std::runtime_error("immutable alias raw coverage changed"); out.finish();
        }
        const size_t allowance = size_t(std::min<uint64_t>(budget_ - fixed, std::max<uint64_t>(8u << 20, spirula::availableRamBytes() / 2)));
        if (progress) progress("alias-feature-sort", 0, observations);
        const Path sorted = scene_index_detail::sortFile<Link>(stage / "links-input.bin", stage / "sort", allowance,
            [](const Link& a, const Link& b) { return std::tie(a.feature, a.point) < std::tie(b.feature, b.point); });
        {
            Reader in(sorted); uint64_t done = 0, feature = 0, firstPoint = 0; Vec2 xy{}; bool haveFeature = false;
            while (done < observations) {
                if (!(done & 65535)) { cancel::check(); if (progress) progress("alias-union", done, observations); }
                const auto row = in.scalar<Link>(); if (row.point >= count) throw std::runtime_error("immutable alias link point outside layout");
                if (!haveFeature || row.feature != feature) { haveFeature = true; feature = row.feature; xy = row.xy; firstPoint = row.point; }
                else {
                    if (row.xy.x != xy.x || row.xy.y != xy.y) throw std::runtime_error("same original image/feature has conflicting actual XY");
                    const auto a = find(firstPoint), b = find(row.point); if (a != b) roots_[size_t(std::max(a, b))] = std::min(a, b);
                }
                ++done;
            }
            in.finish();
        }
        rootCount_ = 0;
        {
            Writer out(stage / "roots.bin");
            for (uint64_t point = 0; point < count; ++point) {
                if (!(point & 65535)) { cancel::check(); if (progress) progress("alias-root-compress", point, count); }
                roots_[size_t(point)] = find(point); rootCount_ += roots_[size_t(point)] == point; out.scalar(roots_[size_t(point)]);
            }
            out.finish();
        }
        std::filesystem::remove(stage / "links-input.bin"); std::filesystem::remove(sorted);
        const auto rootsHash = fileHash(stage / "roots.bin");
        {
            Writer out(stage / "manifest.bin"); out.scalar<uint64_t>(1); out.string(expected); out.scalar(count); out.scalar(observations); out.scalar(rootCount_);
            out.vector(offsets_); out.string(rootsHash); out.finish();
        }
        std::filesystem::rename(stage, target); identity_ = expected; directory_ = target; rootHash_ = rootsHash;
        if (progress) progress("alias-ready", count, count);
    }
    const std::string& identity() const { return identity_; }
    const std::string& rootsHash() const { return rootHash_; }
    uint64_t points() const { return roots_.size(); }
    uint64_t canonicalPoints() const { return rootCount_; }
    uint64_t observations() const { return observations_; }
    uint64_t residentBytes() const { return roots_.capacity() * sizeof(uint64_t) + offsets_.capacity() * sizeof(uint64_t) + identity_.capacity() + rootHash_.capacity() + 4096; }
    uint64_t canonicalGlobal(uint64_t point) const { return roots_.at(size_t(point)); }
    uint64_t canonical(PointKey key) const {
        if (!catalog_ || key.source >= catalog_->sourceCount() || key.ordinal >= catalog_->sourcePointCount(size_t(key.source))) throw std::out_of_range("immutable alias source key");
        if (!(catalog_->key(size_t(key.source), key.ordinal) == key)) throw std::runtime_error("immutable alias original local ID changed");
        return roots_.at(size_t(offsets_.at(size_t(key.source)) + key.ordinal));
    }
    void emitAliases(const CanonicalScene::AliasEmit& emit) const {
        if (!catalog_) throw std::logic_error("immutable alias catalogue missing");
        for (size_t source = 0; source < catalog_->sourceCount(); ++source) for (uint64_t ordinal = 0; ordinal < catalog_->sourcePointCount(source); ++ordinal) {
            if (!(ordinal & 65535)) cancel::check(); const auto key = catalog_->key(source, ordinal); emit(key, roots_.at(size_t(offsets_[source] + ordinal)));
        }
    }
private:
    struct Link { uint64_t feature = 0, point = 0; Vec2 xy; };
    static void append(spirula::Sha256& hash, const std::string& value) {
        const uint64_t n = value.size(); hash.update(reinterpret_cast<const uint8_t*>(&n), sizeof(n)); hash.update(reinterpret_cast<const uint8_t*>(value.data()), value.size());
    }
    static std::string fileHash(const Path& path) { const auto hash = spirula::sha256_file(path.string()); if (hash.empty()) throw std::runtime_error("immutable alias file hash unavailable"); return hash; }
    static void requirePhysical(uint64_t allocation) {
        const auto available = spirula::availableRamBytes(); if (available && allocation > available - std::min<size_t>(size_t(4) << 30, available / 4)) throw std::length_error("immutable alias map exceeds physical RAM headroom");
    }
    uint64_t find(uint64_t point) {
        auto root = point; while (roots_.at(size_t(root)) != root) root = roots_.at(size_t(root));
        while (point != root) { const auto next = roots_.at(size_t(point)); roots_[size_t(point)] = root; point = next; } return root;
    }
    void open(const Path& directory, const std::string& expected, uint64_t points, uint64_t observations) {
        if (std::filesystem::file_size(directory / "manifest.bin") > budget_ / 16) throw std::length_error("immutable alias manifest exceeds allowance");
        Reader manifest(directory / "manifest.bin");
        if (manifest.scalar<uint64_t>() != 1 || manifest.string() != expected || manifest.scalar<uint64_t>() != points || manifest.scalar<uint64_t>() != observations)
            throw std::runtime_error("immutable alias checkpoint identity/count differs");
        const uint64_t rootCount = manifest.scalar<uint64_t>(); const auto offsets = manifest.vector<uint64_t>(); const auto hash = manifest.string(); manifest.finish();
        if (offsets != offsets_ || rootCount > points || std::filesystem::file_size(directory / "roots.bin") != storage::scaleBytes(points, sizeof(uint64_t)) || fileHash(directory / "roots.bin") != hash)
            throw std::runtime_error("immutable alias checkpoint mapping differs");
        roots_.resize(size_t(points)); Reader in(directory / "roots.bin"); in.read(roots_.data(), roots_.size()); in.finish(); uint64_t actualRoots = 0;
        for (uint64_t point = 0; point < points; ++point) { const auto root = roots_[size_t(point)]; if (root > point || roots_[size_t(root)] != root) throw std::runtime_error("immutable alias checkpoint has noncanonical root"); actualRoots += root == point; }
        if (actualRoots != rootCount) throw std::runtime_error("immutable alias checkpoint root count differs");
        identity_ = expected; rootHash_ = hash; rootCount_ = rootCount; directory_ = directory;
    }
    Path root_, directory_; size_t budget_ = 0;
    const RegionalWorkingStore* catalog_ = nullptr;
    std::vector<uint64_t> roots_, offsets_;
    uint64_t rootCount_ = 0, observations_ = 0; std::string identity_, rootHash_;
};
}
