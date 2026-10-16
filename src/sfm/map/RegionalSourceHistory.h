#pragma once

#include "sfm/map/RegionalPriors.h"
#include "core/Sha256.h"
#include "data/Json.h"
#include <cctype>
#include <sstream>

namespace sfm::regional::history {

struct PinnedSource {
    std::string name, path, sha256;
    uint64_t photos = 0;
    bool positioned = false, certified = false;
};
struct Manifest {
    std::string feature_digest;
    uint64_t input_count = 0;
    std::vector<PinnedSource> sources;
};

inline bool validHash(const std::string& value) {
    return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == std::string::npos;
}
// Use the shared SHA implementation with filesystem paths, including UTF-8 names on Windows.
inline std::string fullHash(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("historical source cannot be read: " + path.u8string());
    spirula::Sha256 hash; std::vector<char> bytes(1u << 20);
    while (input) {
        input.read(bytes.data(), std::streamsize(bytes.size()));
        if (input.gcount()) hash.update(reinterpret_cast<const uint8_t*>(bytes.data()), size_t(input.gcount()));
    }
    if (!input.eof()) throw std::runtime_error("historical source checksum read failed");
    return hash.hex();
}
inline void validate(const Manifest& manifest) {
    if (manifest.feature_digest.empty() || !manifest.input_count || manifest.sources.empty())
        throw std::runtime_error("historical source identity is incomplete");
    std::set<std::string> names, paths;
    for (const auto& source : manifest.sources) {
        if (source.name.empty() || !validHash(source.sha256) || !source.photos || source.photos > manifest.input_count ||
            !std::filesystem::u8path(source.path).is_absolute() || (!source.positioned && !source.certified) ||
            !names.insert(source.name).second || !paths.insert(source.path).second)
            throw std::runtime_error("invalid or duplicate historical source certificate");
    }
}
inline void writeManifest(const Manifest& manifest, const std::filesystem::path& path) {
    validate(manifest);
    model_store_detail::Writer out(path);
    out.string("SPIRULA-HISTORICAL-SOURCE-CERTIFICATES"); out.scalar<uint32_t>(1);
    out.string(manifest.feature_digest); out.scalar<uint64_t>(manifest.input_count);
    out.scalar<uint64_t>(manifest.sources.size());
    for (const auto& source : manifest.sources) {
        out.string(source.name); out.string(source.path); out.string(source.sha256);
        out.scalar<uint64_t>(source.photos); out.scalar<uint8_t>(source.positioned); out.scalar<uint8_t>(source.certified);
    }
    out.finish();
}
inline Manifest readManifest(const std::filesystem::path& path) {
    model_store_detail::Reader in(path);
    if (in.string() != "SPIRULA-HISTORICAL-SOURCE-CERTIFICATES" || in.scalar<uint32_t>() != 1)
        throw std::runtime_error("unsupported historical source manifest");
    Manifest manifest; manifest.feature_digest = in.string(); manifest.input_count = in.scalar<uint64_t>();
    const auto n = in.count(34);
    for (size_t i = 0; i < n; ++i) {
        PinnedSource source; source.name = in.string(); source.path = in.string(); source.sha256 = in.string();
        source.photos = in.scalar<uint64_t>();
        const auto position = in.scalar<uint8_t>(), certificate = in.scalar<uint8_t>();
        if (position > 1 || certificate > 1) throw std::runtime_error("invalid historical source flag");
        source.positioned = position != 0; source.certified = certificate != 0;
        manifest.sources.push_back(std::move(source));
    }
    in.finish(); validate(manifest); return manifest;
}
inline Manifest pinJson(const std::filesystem::path& input, const std::filesystem::path& output,
                        const std::string& digest, uint64_t count) {
    std::ifstream file(input, std::ios::binary);
    if (!file) throw std::runtime_error("historical source input map missing");
    const std::string bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    const auto json = json_parse(bytes);
    const auto* list = json.is_array() ? &json : json.find("sources");
    if (!list || !list->is_array()) throw std::runtime_error("historical source input map needs source array");
    Manifest manifest; manifest.feature_digest = digest; manifest.input_count = count;
    for (const auto& row : list->arr) {
        auto text = [&](const char* key) {
            const auto* value = row.find(key);
            if (!value || value->type != JsonValue::Type::String || value->str.empty())
                throw std::runtime_error(std::string("historical source missing ") + key);
            return value->str;
        };
        auto integer = [&](const char* key, uint64_t maximum) {
            const auto* value = row.find(key);
            if (!value || value->type != JsonValue::Type::Number || !std::isfinite(value->num) ||
                value->num < 0 || value->num > double(maximum) || value->num != std::floor(value->num))
                throw std::runtime_error(std::string("historical source invalid ") + key);
            return uint64_t(value->num);
        };
        PinnedSource source; source.name = text("source"); source.path = text("frame_candidate");
        source.sha256 = text("candidate_sha"); source.photos = integer("photos", count);
        source.positioned = integer("position_required", 1) != 0;
        source.certified = integer("gauge_required", 1) != 0;
        manifest.sources.push_back(std::move(source));
    }
    validate(manifest);
    for (const auto& source : manifest.sources) {
        const auto path = std::filesystem::u8path(source.path);
        if (fullHash(path) != source.sha256) throw std::runtime_error("historical source SHA mismatch: " + source.name);
        const auto views = model_store_detail::readModel(path, false);
        if (views.numRegistered() != source.photos) throw std::runtime_error("historical source photo count mismatch: " + source.name);
        for (const auto& image : views.images) if (image.second.registered && image.first >= count)
            throw std::runtime_error("historical source image ID outside original input");
    }
    writeManifest(manifest, output); return manifest;
}
inline std::vector<RegionalSourceFrame> loadPinned(const std::filesystem::path& path,
    const std::string& digest, uint64_t count, PriorSource& gps, const AttitudeRef& attitude,
    size_t expected_sources = 20, size_t expected_certified = 18) {
    const auto manifest = readManifest(path);
    if (manifest.feature_digest != digest || manifest.input_count != count || manifest.sources.size() != expected_sources)
        throw std::runtime_error("historical source manifest input/cohort identity mismatch");
    size_t certified = 0, positioned = 0; std::vector<RegionalSourceFrame> frames;
    for (const auto& source : manifest.sources) {
        certified += source.certified; positioned += source.positioned;
        const auto model = std::filesystem::u8path(source.path);
        if (fullHash(model) != source.sha256) throw std::runtime_error("historical source SHA mismatch: " + source.name);
        RegionalSourceFrame frame; frame.name = source.name;
        frame.positioned = source.positioned; frame.certified = source.certified;
        frame.views = model_store_detail::readModel(model, false);
        if (frame.views.numRegistered() != source.photos) throw std::runtime_error("historical source count mismatch: " + source.name);
        for (const auto& image : frame.views.images) if (image.second.registered && image.first >= count)
            throw std::runtime_error("historical source photo outside original input");
        if (!regionalSourceAccepted(frame, frame.views, gps, attitude))
            throw std::runtime_error("original historical source certificate no longer verifies: " + source.name);
        frames.push_back(std::move(frame));
    }
    if (certified != expected_certified || positioned != expected_sources)
        throw std::runtime_error("historical position/certified source count changed");
    return frames;
}
inline std::vector<RegionalSourceFrame> loadRequired(const std::filesystem::path& root,
    const std::string& digest, uint64_t count, PriorSource& gps, const AttitudeRef& attitude, bool required) {
    const auto path = root / "historical-source-certificates.bin";
    const auto marker = root / "historical-source-certificates.required";
    if (!required && !std::filesystem::exists(path) && !std::filesystem::exists(marker)) return {};
    std::ifstream input(marker); std::string expected, extra;
    if (!(input >> expected) || (input >> extra) || !validHash(expected) || !std::filesystem::is_regular_file(path) ||
        fullHash(path) != expected)
        throw std::runtime_error("required historical source proof missing or modified");
    return loadPinned(path, digest, count, gps, attitude);
}
inline std::string csvCell(std::string text) {
    std::string out = "\"";
    for (char c : text) { if (c == '"') out += '"'; out += c; }
    return out + "\"";
}
// Audit only: never copy old poses into the candidate or weaken original certificate flags.
inline bool verifyFinal(const std::vector<RegionalSourceFrame>& frames, const Reconstruction& poses,
    PriorSource& gps, const AttitudeRef& attitude, const std::filesystem::path& csv) {
    std::ofstream output(csv); if (!output) throw std::runtime_error("cannot write historical source final proof");
    output << "source,photos,position_required,gauge_required,gps_inliers,position_passed,gauge_passed,accepted,reason\n";
    bool all = true;
    for (const auto& frame : frames) {
        bool accepted = false, position = false, gauge = false; size_t inliers = 0; std::string reason;
        try {
            const auto views = regionalSourceViews(frame, poses);
            const auto quality = regionalGpsQuality(views, gps); inliers = quality.inliers; position = quality.accepted();
            gauge = regionalGpsGauge(views, gps, attitude);
            accepted = (!frame.positioned || position) && (!frame.certified || gauge);
            if (!accepted) reason = "historical position or certified gauge failed";
        } catch (const std::exception& error) { reason = error.what(); }
        all &= accepted;
        output << csvCell(frame.name) << ',' << frame.views.numRegistered() << ',' << frame.positioned << ','
            << frame.certified << ',' << inliers << ',' << position << ',' << gauge << ',' << accepted << ',' << csvCell(reason) << '\n';
    }
    output.flush(); if (!output) throw std::runtime_error("historical final proof write failed");
    return all;
}

} // namespace sfm::regional::history
