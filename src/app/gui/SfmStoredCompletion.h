#pragma once
#include "core/Sha256.h"
#include <array>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace gui {
struct SfmStoredCompletion {
    bool known = false;
    bool partial = false;
};
namespace sfm_stored_completion_detail {
inline std::map<std::string, std::string> fields(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("missing stored SfM proof");
    std::map<std::string, std::string> result;
    std::string line;
    size_t bytes = 0;
    while (std::getline(input, line)) {
        bytes += line.size();
        if (bytes > (4u << 20)) throw std::runtime_error("oversized stored SfM proof");
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto at = line.find('=');
        if (at == std::string::npos) continue;
        const auto key = line.substr(0, at);
        if (key == "source") continue;
        if (!result.emplace(key, line.substr(at + 1)).second)
            throw std::runtime_error("duplicate stored SfM proof field");
    }
    if (!input.eof()) throw std::runtime_error("unreadable stored SfM proof");
    return result;
}
inline bool hash(const std::string& value) {
    if (value.size() != 64) return false;
    for (char c : value) if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}
inline uint64_t count(const std::string& value) {
    uint64_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        throw std::runtime_error("invalid stored SfM proof count");
    return result;
}
}
// A status snapshot has no model identity; only the export receipt binds a reused verdict.
inline SfmStoredCompletion read_sfm_stored_completion(const std::filesystem::path& model) {
    using namespace sfm_stored_completion_detail;
    try {
        auto proof = fields(model / "verified-export.txt");
        auto binding = fields(model / "raw-alias-binding.txt");
        const bool complete = proof.at("full_capture_verified") == "true";
        if (proof.at("schema") != "canonical-scene-export-v3" || proof.at("reader_verified") != "1" ||
            (proof.at("full_capture_verified") != "false" && !complete) ||
            !hash(proof.at("scene_binding")) || binding.at("schema") != "immutable-raw-alias-v1" ||
            proof.at("scene_binding") != binding.at("scene")) return {};
        const auto raw = count(proof.at("raw_observations"));
        if (!raw || count(proof.at("lineage_rows")) != raw || proof.at("raw_type_coverage_verified") != "1" ||
            count(proof.at("unresolved_raw_observations")) != 0) return {};
        const auto failures = count(proof.at("failed_checks"));
        if (complete && (failures != 0 || count(proof.at("raw_invalid")) != 0 ||
                         count(proof.at("photo_components")) != 1)) return {};
        if (!complete && failures == 0) return {};
        auto hashes = fields(model / "formal-file-sha256.txt");
        constexpr std::array<const char*, 6> names = {"cameras.bin", "images.bin", "points3D.bin",
            "raw-alias-map.bin", "directional-points.bin", "retained-observations.bin"};
        if (hashes.size() != names.size()) return {};
        if (!hash(binding.at("alias_sha256")) || binding.at("alias_sha256") != hashes.at("raw-alias-map.bin")) return {};
        for (const auto* name : names) {
            const auto& expected = hashes.at(name);
            if (!hash(expected) || spirula::sha256_file(model / name) != expected) return {};
        }
        return {true, !complete};
    } catch (const std::exception&) { return {}; }
}
template <class ResolveColmap>
inline SfmStoredCompletion read_sfm_workspace_completion(const std::filesystem::path& workspace,
                                                        ResolveColmap resolve) {
    try {
        if (std::filesystem::exists(workspace / "transforms.json")) return {};
        const auto model = resolve(workspace.u8string());
        return model.empty() ? SfmStoredCompletion{} : read_sfm_stored_completion(std::filesystem::u8path(model));
    } catch (const std::exception&) { return {}; }
}
}
