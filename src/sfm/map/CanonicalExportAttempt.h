#pragma once
#include "sfm/map/CanonicalScene.h"
#include "sfm/core/Resume.h"
#include <chrono>

namespace sfm::regional {
struct CanonicalExportAttempt {
    std::filesystem::path scratch, pending;
};
inline CanonicalExportAttempt reserveCanonicalExportAttempt(const std::filesystem::path& coordination,
    const std::filesystem::path& sparse, const CanonicalScene& scene) {
    if (!scene.initialized() || scene.editing()) throw std::logic_error("canonical export requires an idle accepted scene");
    std::filesystem::create_directories(coordination);
    std::filesystem::create_directories(sparse);
    const auto token = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    for (uint64_t attempt = 0; ; ++attempt) {
        const auto name = std::to_string(scene.generation()) + "-" + token + "-" + std::to_string(attempt);
        const auto claim = coordination / ("export-attempt-" + name);
        const auto pending = sparse / ("canonical-isolated-" + name);
        if (std::filesystem::exists(pending) || !std::filesystem::create_directory(claim)) continue;
        resume::store(claim / "request.txt", "accepted_identity=" + scene.identity() + "\ngeneration=" +
            std::to_string(scene.generation()) + "\nraw=" + std::to_string(scene.observations()) +
            "\npending=" + pending.u8string() + "\nfull_capture_verified=false\n");
        return {claim / "scratch", pending};
    }
}
}
