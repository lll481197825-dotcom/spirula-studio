#pragma once
#include "sfm/map/CanonicalInitialization.h"
#include <sstream>
#include <cstring>

namespace sfm::regional {
struct CanonicalOriginalRequirement {
    std::string label;
    std::filesystem::path path;
    uint64_t bytes = 0, memory = 0;
    std::string sha;
};
struct CanonicalOriginalCoverageRow {
    std::string label, originalSHA, acceptedDigest, reason;
    uint64_t observations = 0, matched = 0;
    size_t acceptedSource = SIZE_MAX;
    bool preserved = false;
};
struct CanonicalOriginalCoverage {
    std::string acceptedIdentity;
    std::vector<CanonicalOriginalCoverageRow> rows;
    bool complete = false;
    std::string csv() const {
        auto quoted = [](std::string text) { size_t at = 0; while ((at = text.find('"', at)) != std::string::npos) { text.insert(at, 1, '"'); at += 2; } return '"' + text + '"'; };
        std::ostringstream out;
        out << "original_source,raw_observations,preserved,scope,accepted_source,matched_original_raw,original_sha256,accepted_source_digest,reason\n";
        for (const auto& row : rows) {
            out << quoted(row.label) << ',' << row.observations << ',' << (row.preserved ? "true" : "false")
                << ",immutable_source_raw_multiset_geometry_requires_successful_typed_formal_Reader,";
            if (row.acceptedSource != SIZE_MAX) out << row.acceptedSource;
            out << ',' << row.matched << ',' << row.originalSHA << ',' << quoted(row.acceptedDigest) << ',' << quoted(row.reason) << '\n';
        }
        return out.str();
    }
};
inline std::vector<CanonicalOriginalRequirement> canonicalOriginalRequirements(const CanonicalInitialStage& initial,
    const std::filesystem::path& evidence, const std::string& manifest) {
    std::map<std::string, CanonicalOriginalRequirement> identities;
    auto add = [&](const auto& row) {
        if (row.label.empty() || row.originalSHA.size() != 64 || !row.originalBytes ||
            std::filesystem::u8path(row.label).filename() != std::filesystem::u8path(row.label) ||
            !identities.emplace(row.label, CanonicalOriginalRequirement{row.label, evidence / std::filesystem::u8path(row.label), row.originalBytes, row.memory, row.originalSHA}).second)
            throw std::runtime_error("original source stage identity is incomplete or duplicated");
    };
    for (const auto& row : initial.sources) add(row);
    for (const auto& row : initial.deferred) add(row);
    std::vector<CanonicalOriginalRequirement> result; std::set<std::string> seen; std::istringstream input(manifest); std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto split = line.find(':'); if (split == std::string::npos || split == 0) throw std::runtime_error("invalid original source evidence manifest");
        const auto label = line.substr(0, split); const auto found = identities.find(label);
        if (found == identities.end() || !seen.insert(label).second) throw std::runtime_error("original evidence labels differ from the verified initial stage");
        size_t used = 0; const auto bytes = std::stoull(line.substr(split + 1), &used);
        if (used != line.size() - split - 1 || bytes != found->second.bytes) throw std::runtime_error("original evidence source size differs from the verified initial stage");
        result.push_back(found->second);
    }
    if (result.empty() || seen.size() != identities.size()) throw std::runtime_error("original evidence manifest omitted required initial sources");
    return result;
}
template<class Normalize> CanonicalOriginalCoverage verifyCanonicalOriginalCoverage(const RegionalWorkingStore& catalog,
    const CanonicalScene& scene, const std::vector<CanonicalOriginalRequirement>& required, Normalize normalize,
    size_t allowance) {
    if (!scene.initialized() || scene.editing() || required.empty() || allowance < (16u << 20))
        throw std::invalid_argument("original coverage requires an idle accepted scene and explicit original identities");
    scene.requireBinding(CanonicalScene::bindingFor(catalog, scene.binding().topology, true, scene.sourceCount()));
    CanonicalOriginalCoverage result; result.acceptedIdentity = scene.identity(); std::set<std::string> labels;
    for (const auto& requirement : required) {
        if (!labels.insert(requirement.label).second) throw std::runtime_error("duplicate original source requirement");
        CanonicalOriginalCoverageRow row; row.label = requirement.label; row.originalSHA = requirement.sha;
        for (size_t source = 0; source < scene.sourceCount(); ++source) if (scene.binding().sources[source].label == row.label) row.acceptedSource = source;
        try {
            if (requirement.sha.size() != 64 || std::filesystem::is_symlink(requirement.path) ||
                std::filesystem::file_size(requirement.path) != requirement.bytes ||
                spirula::sha256_file(requirement.path.u8string()) != requirement.sha)
                throw std::runtime_error("original source differs from initial SHA/size identity");
            if (requirement.memory > allowance) throw std::length_error("original source model exceeds coverage allowance");
            auto original = model_store_detail::readModel(requirement.path); normalize(original);
            for (const auto& point : original.points3D) row.observations = storage::addBytes(row.observations, point.second.track.size());
            if (row.acceptedSource == SIZE_MAX) throw std::runtime_error("original source is not admitted in the accepted scene binding");
            row.acceptedDigest = scene.binding().sources[row.acceptedSource].digest;
            auto views = catalog.sourceViews(row.acceptedSource); normalize(views);
            struct Range { uint64_t first = 0; const Image* image = nullptr; };
            uint64_t slots = 0, features = 0;
            for (const auto& item : original.images) if (item.second.registered) { slots = std::max<uint64_t>(slots, uint64_t(item.first) + 1); features = storage::addBytes(features, item.second.points2D.size()); }
            const auto scratch = storage::addBytes(storage::scaleBytes(slots, sizeof(Range)), storage::scaleBytes(features, sizeof(uint64_t)));
            const auto live = storage::addBytes(modelResidentBytes(original), modelResidentBytes(views));
            if (storage::addBytes(live, scratch) > allowance) throw std::length_error("original raw multiset counters exceed allowance");
            const auto available = spirula::availableRamBytes();
            if (available && scratch > available - std::min<size_t>(available / 4, size_t(4) << 30)) throw std::length_error("original coverage counters exceed physical RAM headroom");
            std::vector<Range> ranges(size_t(slots), Range{}); std::vector<uint64_t> remaining(size_t(features), 0); uint64_t cursor = 0;
            for (const auto& item : original.images) if (item.second.registered) {
                const auto found = views.images.find(item.first);
                if (found == views.images.end() || item.second.id != item.first || found->second.id != item.first || !item.second.registered || !found->second.registered ||
                    item.second.name != found->second.name || item.second.camera_id != found->second.camera_id ||
                    item.second.points2D.size() != catalog.sourceImageFeatureCount(row.acceptedSource, item.first))
                    throw std::runtime_error("accepted source lost original image or feature identity");
                ranges[item.first] = {cursor, &item.second}; cursor = storage::addBytes(cursor, item.second.points2D.size());
            }
            for (const auto& point : original.points3D) for (const auto& observation : point.second.track) {
                const auto& range = ranges.at(observation.image_id);
                if (!range.image) throw std::runtime_error("original raw references a missing image");
                const auto& image = *range.image;
                if (observation.point2D_idx >= image.points2D.size() || observation.point2D_idx >= image.point3D_ids.size() || image.point3D_ids[observation.point2D_idx] != point.first)
                    throw std::runtime_error("original immutable raw association is invalid");
                const auto xy = image.points2D[observation.point2D_idx];
                if (!std::isfinite(xy.x) || !std::isfinite(xy.y)) throw std::runtime_error("original raw coordinate is nonfinite");
                auto& count = remaining.at(size_t(range.first + observation.point2D_idx)); count = storage::addBytes(count, 1);
            }
            catalog.visitImmutableSourcePointsDetailed(row.acceptedSource, [&](auto, const auto&, Vec3, const auto& track) {
                for (const auto& observation : track) {
                    if (observation.image >= ranges.size()) continue; const auto& range = ranges[observation.image];
                    if (!range.image || observation.feature >= range.image->points2D.size()) continue;
                    auto& count = remaining[size_t(range.first + observation.feature)]; const auto expected = range.image->points2D[observation.feature];
                    if (count && std::memcmp(&expected, &observation.xy, sizeof(Vec2)) == 0) { --count; ++row.matched; }
                }
            });
            if (row.matched != row.observations) throw std::runtime_error("accepted source omitted original raw row or duplicate multiplicity");
            row.preserved = true;
        } catch (const std::exception& error) { row.reason = error.what(); }
        cancel::check(); result.rows.push_back(std::move(row));
    }
    result.complete = result.rows.size() == required.size() && std::all_of(result.rows.begin(), result.rows.end(), [](const auto& row) { return row.preserved; });
    if (scene.editing() || scene.identity() != result.acceptedIdentity) throw std::runtime_error("accepted scene changed during original raw coverage verification");
    return result;
}
}
