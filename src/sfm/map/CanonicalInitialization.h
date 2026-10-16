#pragma once
#include "sfm/map/CanonicalScene.h"
#include <optional>

namespace sfm::regional {
struct CanonicalInitialSource {
    std::string label;
    std::filesystem::path original, aligned;
    uint64_t memory = 0;
    Sim3 transform;
    uint64_t originalBytes = 0;
    std::string originalSHA;
};
struct CanonicalInitialStage {
    Reconstruction poses;
    struct Deferred { std::string label; std::filesystem::path original; uint64_t memory = 0, region = 0; uint64_t originalBytes = 0; std::string originalSHA; };
    std::vector<CanonicalInitialSource> sources;
    std::vector<Deferred> deferred;
    std::vector<uint32_t> targets, fixedImages;
};
namespace canonical_initial_detail {
using Path = std::filesystem::path;
using Reader = model_store_detail::Reader;
using Writer = model_store_detail::Writer;
struct Digest {
    spirula::Sha256 hash;
    template<class T> void scalar(const T& value) { hash.update(reinterpret_cast<const uint8_t*>(&value), sizeof(value)); }
    template<class T> void vector(const std::vector<T>& values) { scalar<uint64_t>(values.size()); hash.update(reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(T)); }
    void string(const std::string& value) { scalar<uint64_t>(value.size()); hash.update(reinterpret_cast<const uint8_t*>(value.data()), value.size()); }
};
inline std::string hash(const Path& path) { const auto result = spirula::sha256_file(path.u8string()); if (result.empty()) throw std::runtime_error("canonical initialization SHA unavailable"); return result; }
inline std::string nativeDigest(const Reconstruction& model) { Digest out; model_store_detail::serializeModel(model, out); return out.hash.hex(); }
inline Path aligned(const Path& directory, const Reconstruction& model) {
    std::filesystem::create_directories(directory); const auto expected = nativeDigest(model); const auto path = directory / (expected + ".bin");
    if (std::filesystem::exists(path)) {
        if (std::filesystem::file_size(path) == model_store_detail::serializedModelBytes(model) && hash(path) == expected) return path;
        const auto retained = directory / ("rejected-" + expected + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
        std::filesystem::rename(path, retained);
    }
    const auto temporary = directory / (expected + ".part-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    model_store_detail::writeModel(model, temporary);
    if (hash(temporary) != expected) throw std::runtime_error("canonical aligned native serialization identity differs");
    storage::replaceFile(temporary, path); return path;
}
}
inline void saveCanonicalInitialStage(const std::filesystem::path& directory, const std::string& signature,
                                     const CanonicalInitialStage& stage) {
    using namespace canonical_initial_detail;
    std::filesystem::create_directories(directory); const auto poseHash = nativeDigest(stage.poses);
    const auto posePath = aligned(directory, stage.poses);
    Writer out(directory / "manifest.part"); out.scalar<uint64_t>(2); out.string(signature); out.string(posePath.filename().u8string()); out.string(poseHash);
    out.scalar<uint64_t>(stage.sources.size()); std::set<std::string> labels;
    for (const auto& source : stage.sources) {
        if (source.label.empty() || !labels.insert(source.label).second) throw std::runtime_error("canonical initialization source labels conflict");
        out.string(source.label); out.string(std::filesystem::canonical(source.original).u8string()); out.scalar<uint64_t>(std::filesystem::file_size(source.original)); out.string(hash(source.original));
        out.string(std::filesystem::canonical(source.aligned).u8string()); out.scalar<uint64_t>(std::filesystem::file_size(source.aligned)); out.string(hash(source.aligned)); out.scalar(source.memory); out.scalar(source.transform);
    }
    out.scalar<uint64_t>(stage.deferred.size());
    for (const auto& source : stage.deferred) {
        if (source.label.empty() || !labels.insert(source.label).second) throw std::runtime_error("canonical initial deferred labels conflict");
        out.string(source.label); out.string(std::filesystem::canonical(source.original).u8string());
        out.scalar<uint64_t>(std::filesystem::file_size(source.original)); out.string(hash(source.original)); out.scalar(source.memory); out.scalar(source.region);
    }
    out.vector(stage.targets); out.vector(stage.fixedImages);
    out.finish(); storage::replaceFile(directory / "manifest.part", directory / "manifest.bin");
}
inline std::optional<CanonicalInitialStage> loadCanonicalInitialStage(const std::filesystem::path& directory,
    const std::string& signature, const std::vector<std::pair<std::string, std::filesystem::path>>& originals, size_t budget,
    bool allowAdditionalSources = false) {
    using namespace canonical_initial_detail; const auto path = directory / "manifest.bin";
    if (!std::filesystem::exists(path)) return std::nullopt;
    if (std::filesystem::file_size(path) > budget / 16) throw std::length_error("canonical initialization metadata exceeds allowance");
    Reader in(path); const auto version = in.scalar<uint64_t>(); if (version != 2 || in.string() != signature) throw std::runtime_error("canonical initialization run signature differs");
    const auto name = in.string(), poseHash = in.string();
    if (std::filesystem::path(name).filename() != std::filesystem::path(name) || hash(directory / name) != poseHash) throw std::runtime_error("canonical initial pose snapshot changed");
        std::map<std::string, std::filesystem::path> expected;
    for (const auto& source : originals) if (!expected.emplace(source).second) throw std::runtime_error("canonical initial current source labels conflict");
    std::set<std::string> loaded;
    CanonicalInitialStage out; const size_t count = in.count(64);
    if (!allowAdditionalSources && count > originals.size()) throw std::runtime_error("canonical initial source catalogue count differs");
    for (size_t source = 0; source < count; ++source) {
        CanonicalInitialSource row; row.label = in.string(); row.original = std::filesystem::u8path(in.string());
        const auto originalBytes = in.scalar<uint64_t>(); const auto originalHash = in.string();
        row.aligned = std::filesystem::u8path(in.string()); const auto alignedBytes = in.scalar<uint64_t>(); const auto alignedHash = in.string();
        row.memory = in.scalar<uint64_t>(); row.transform = in.scalar<Sim3>(); row.originalBytes = originalBytes; row.originalSHA = originalHash;
        if ((!expected.empty() && (!expected.count(row.label) || std::filesystem::canonical(row.original) != std::filesystem::canonical(expected.at(row.label)))) ||
            !loaded.insert(row.label).second ||
            std::filesystem::file_size(row.original) != originalBytes || hash(row.original) != originalHash ||
            std::filesystem::file_size(row.aligned) != alignedBytes || hash(row.aligned) != alignedHash)
            throw std::runtime_error("canonical original/aligned source identity changed");
        if (!(row.transform.scale > 0) || !std::isfinite(row.transform.scale) || !step::finite(row.transform.t)) throw std::runtime_error("canonical initial transform nonfinite");
        Pose rotation{row.transform.R, {}}; step::validatePose(rotation); out.sources.push_back(std::move(row));
    }
    const size_t deferred = in.count(40);
    for (size_t source = 0; source < deferred; ++source) {
        CanonicalInitialStage::Deferred row; row.label = in.string(); row.original = std::filesystem::u8path(in.string());
        const auto bytes = in.scalar<uint64_t>(); const auto checksum = in.string(); row.memory = in.scalar<uint64_t>(); row.region = in.scalar<uint64_t>(); row.originalBytes = bytes; row.originalSHA = checksum;
        if ((!expected.empty() && (!expected.count(row.label) || std::filesystem::canonical(row.original) != std::filesystem::canonical(expected.at(row.label)))) ||
            !loaded.insert(row.label).second ||
            std::filesystem::file_size(row.original) != bytes || hash(row.original) != checksum) throw std::runtime_error("canonical deferred original identity changed");
        out.deferred.push_back(std::move(row));
    }
    if (!allowAdditionalSources && loaded.size() != expected.size()) throw std::runtime_error("canonical initial selected/deferred coverage incomplete");
    out.targets = in.vector<uint32_t>(); out.fixedImages = in.vector<uint32_t>();
    in.finish(); const auto poseBytes = std::filesystem::file_size(directory / name);
    if (storage::scaleBytes(poseBytes, 10) > budget / 2) throw std::length_error("canonical initial P/K exceeds allowance");
    out.poses = model_store_detail::readModel(directory / name); if (modelResidentBytes(out.poses) > budget / 2) throw std::length_error("canonical initial native P/K exceeds allowance"); return out;
}
}
