#pragma once
#include "sfm/map/CanonicalScene.h"
#include "sfm/map/SceneTopology.h"
#include "sfm/core/Resume.h"
#include <chrono>
#include <map>
#include <sstream>

namespace sfm::regional {
inline std::map<std::string, std::string> canonicalResumeFields(const std::filesystem::path& path) {
    std::map<std::string, std::string> fields; std::istringstream input(resume::recorded(path)); std::string line;
    while (std::getline(input, line)) { const auto at = line.find('='); if (at != std::string::npos) fields.emplace(line.substr(0, at), line.substr(at + 1)); }
    return fields;
}
inline uint64_t canonicalCompletedRounds(const std::filesystem::path& coordination, const CanonicalScene& scene) {
    const std::string phase = scene.acceptedSnapshot().phase, marker = ".cohort-round=";
    const auto ordinal = phase.find(marker);
    if (ordinal != std::string::npos) {
        const auto base = phase.substr(0, ordinal);
        if (base != "canonical-scene-round-fullraw-v1" && base != "canonical-scene-round-fullraw-step-v1")
            throw std::runtime_error("canonical atomic round marker has an invalid producer phase");
        return std::stoull(phase.substr(ordinal + marker.size()));
    }
    uint64_t rounds = 0, generation = scene.generation();
    while (generation) {
        const auto row = canonicalResumeFields(coordination / ("round-" + std::to_string(generation) + "-acceptance.txt"));
        if (row.empty() || !row.count("raw") || !row.count("accepted") || !row.count("generation") ||
            row.at("accepted") != "1" || std::stoull(row.at("generation")) != generation || std::stoull(row.at("raw")) != scene.observations()) break;
        std::filesystem::path snapshot; const auto prefix = "accepted-" + std::to_string(generation) + "-";
        for (const auto& entry : std::filesystem::directory_iterator(scene.acceptedSnapshot().directory.parent_path())) {
            if (!entry.is_directory() || entry.path().filename().u8string().rfind(prefix, 0) != 0) continue;
            if (!snapshot.empty()) throw std::runtime_error("canonical legacy generation has multiple accepted snapshots"); snapshot = entry.path();
        }
        if (snapshot.empty()) break;
        const auto retained = scene.inspectRetainedSnapshot(snapshot);
        if (!(retained.binding == scene.binding()) || retained.generation != generation || retained.observations != scene.observations()) break;
        if (retained.phase != "canonical-scene-round-fullraw-v1" && retained.phase != "canonical-scene-round-fullraw-step-v1") break;
        ++rounds; --generation;
    }
    return rounds;
}
inline bool canonicalRoundsFinished(const std::filesystem::path& coordination, const CanonicalScene& scene) {
    const auto record = canonicalResumeFields(coordination / ("cohort-rounds-" + scene.binding().topology + ".txt"));
    for (const char* field : {"accepted_identity", "complete", "generation", "raw", "rounds"}) if (!record.count(field)) return false;
    return !record.empty() && record.at("accepted_identity") == scene.identity() && record.count("complete") && record.at("complete") == "1" &&
        std::stoull(record.at("generation")) == scene.generation() && std::stoull(record.at("raw")) == scene.observations() &&
        std::stoull(record.at("rounds")) == canonicalCompletedRounds(coordination, scene);
}
inline void storeCanonicalRounds(const std::filesystem::path& coordination, const CanonicalScene& scene, uint64_t rounds,
    bool complete = false, const std::string& termination = {}, uint64_t attemptedRounds = UINT64_MAX) {
    if (attemptedRounds == UINT64_MAX) attemptedRounds = rounds;
    if (attemptedRounds < rounds || (!termination.empty() && termination != "running" && termination != "rejected" &&
        termination != "round_limit" && termination != "small_improvement"))
        throw std::invalid_argument("canonical round checkpoint has invalid attempt or termination state");
    if ((!complete && !termination.empty() && termination != "running") || (complete && termination == "running"))
        throw std::invalid_argument("canonical round checkpoint termination disagrees with stopped state");
    resume::store(coordination / ("cohort-rounds-" + scene.binding().topology + ".txt"),
        "accepted_identity=" + scene.identity() + "\ngeneration=" + std::to_string(scene.generation()) +
        "\nraw=" + std::to_string(scene.observations()) + "\nrounds=" + std::to_string(rounds) +
        "\nattempted_rounds=" + std::to_string(attemptedRounds) + "\ncomplete=" + std::to_string(complete) +
        "\ntermination=" + termination + "\nfull_capture_verified=false\n");
}
inline std::filesystem::path storeCanonicalRoundAssessment(const std::filesystem::path& coordination,
    const CanonicalScene& scene, uint64_t attemptedRound, const std::string& assessment) {
    if (!scene.initialized() || scene.editing() || !attemptedRound || assessment.empty())
        throw std::invalid_argument("canonical round evidence requires an idle accepted snapshot and complete assessment");
    const auto token = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto directory = coordination / "round-evidence"; std::filesystem::create_directories(directory);
    for (uint64_t suffix = 0; ; ++suffix) {
        const auto claim = directory / (scene.binding().topology + "-" + std::to_string(scene.generation()) + "-" +
            std::to_string(attemptedRound) + "-" + token + "-" + std::to_string(suffix));
        if (!std::filesystem::create_directory(claim)) continue;
        const auto path = claim / "assessment.txt";
        resume::store(path, "accepted_identity=" + scene.identity() + "\ntopology=" + scene.binding().topology +
            "\nattempted_round=" + std::to_string(attemptedRound) + "\n" + assessment);
        return path;
    }
}
inline void retainCanonicalFlatEvidence(const std::filesystem::path& from, const std::filesystem::path& to) {
    if (!std::filesystem::is_directory(from) || std::filesystem::exists(to)) throw std::runtime_error("canonical resume evidence source or destination invalid");
    std::filesystem::create_directory(to);
    for (const auto& entry : std::filesystem::directory_iterator(from)) {
        if (!entry.is_regular_file() || entry.is_symlink()) throw std::runtime_error("canonical resume evidence must contain regular files");
        storage::retainFile(entry.path(), to / entry.path().filename());
        const auto before = spirula::sha256_file(entry.path().string()), after = spirula::sha256_file((to / entry.path().filename()).string());
        if (before.empty() || after.empty() || before != after)
            throw std::runtime_error("canonical resume evidence copy digest differs");
    }
}
inline bool forkCanonicalResume(const std::filesystem::path& previous, const std::filesystem::path& destination, size_t budget) {
    if (!std::filesystem::exists(previous / "scene" / "head.bin")) return false;
    if (!std::filesystem::is_regular_file(previous / "immutable-raw" / "manifest.bin")) throw std::runtime_error("canonical previous immutable source catalogue is missing");
    const auto source = std::filesystem::canonical(previous), target = std::filesystem::weakly_canonical(std::filesystem::absolute(destination));
    for (auto ancestor = target; !ancestor.empty();) {
        if (std::filesystem::exists(ancestor) && std::filesystem::equivalent(ancestor, source)) throw std::invalid_argument("canonical resume destination is physically inside source");
        const auto parent = ancestor.parent_path(); if (parent == ancestor) break; ancestor = parent;
    }
    const auto relative = target.lexically_relative(source), inverse = source.lexically_relative(target);
    if (source == target || (!relative.empty() && *relative.begin() != std::filesystem::path("..")) ||
        (!inverse.empty() && *inverse.begin() != std::filesystem::path(".."))) throw std::invalid_argument("canonical resume source and destination must be disjoint");
    if (std::filesystem::exists(destination)) throw std::runtime_error("canonical resume destination already exists without a restored head");
    const auto stage = std::filesystem::path(destination.string() + ".resume-part-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(stage);
    {
        CanonicalScene accepted(previous / "scene", budget, CanonicalScene::OpenMode::ReadOnly);
        if (!accepted.sourceCount()) throw std::runtime_error("canonical accepted prefix has no source catalogue");
        RegionalWorkingStore raw(previous / "immutable-raw", budget, RegionalWorkingStore::OpenMode::ReadOnly);
        raw.forkImmutablePrefix(stage / "immutable-raw", accepted.sourceCount()); accepted.forkAccepted(stage / "scene");
        if (!std::filesystem::is_regular_file(previous / "indices" / accepted.binding().topology / "manifest.bin"))
            throw std::runtime_error("canonical previous immutable topology is missing");
        ImmutableSceneTopology topology(previous / "indices", budget); topology.restore(accepted.binding().topology);
        topology.forkImmutable(stage / "indices");
        retainCanonicalFlatEvidence(previous / "initialization", stage / "initialization");
        RegionalWorkingStore restoredRaw(stage / "immutable-raw", budget); CanonicalScene restoredScene(stage / "scene", budget);
        restoredScene.requireBinding(CanonicalScene::bindingFor(restoredRaw, topology.identity()));
        if (restoredScene.identity() != accepted.identity() || restoredScene.observations() != accepted.observations())
            throw std::runtime_error("canonical accepted prefix restore changed identity or observation multiset");
        const auto rounds = canonicalCompletedRounds(previous, accepted);
        const bool legacy = accepted.acceptedSnapshot().phase.find(".cohort-round=") == std::string::npos;
        for (uint64_t offset = 0; legacy && offset < rounds; ++offset) {
            const uint64_t generation = accepted.generation() - offset; const auto prefix = "accepted-" + std::to_string(generation) + "-";
            for (const auto& entry : std::filesystem::directory_iterator(accepted.acceptedSnapshot().directory.parent_path())) {
                if (!entry.is_directory() || entry.path().filename().u8string().rfind(prefix, 0) != 0) continue;
                const auto verified = accepted.inspectRetainedSnapshot(entry.path());
                if (!(verified.binding == accepted.binding())) throw std::runtime_error("canonical legacy resume ancestry changed binding");
                const auto target = stage / "scene" / entry.path().filename();
                if (!std::filesystem::exists(target)) retainCanonicalFlatEvidence(entry.path(), target);
            }
            const auto ledger = previous / ("round-" + std::to_string(generation) + "-acceptance.txt");
            storage::retainFile(ledger, stage / ledger.filename());
        }
        const auto oldCheckpoint = canonicalResumeFields(previous / ("cohort-rounds-" + accepted.binding().topology + ".txt"));
        const bool boundCheckpoint = oldCheckpoint.count("accepted_identity") && oldCheckpoint.count("generation") &&
            oldCheckpoint.count("raw") && oldCheckpoint.count("rounds") && oldCheckpoint.at("accepted_identity") == accepted.identity() &&
            std::stoull(oldCheckpoint.at("generation")) == accepted.generation() &&
            std::stoull(oldCheckpoint.at("raw")) == accepted.observations() && std::stoull(oldCheckpoint.at("rounds")) == rounds;
        const auto termination = boundCheckpoint && oldCheckpoint.count("termination") ? oldCheckpoint.at("termination") : std::string{};
        const auto attempted = boundCheckpoint && oldCheckpoint.count("attempted_rounds") ? std::stoull(oldCheckpoint.at("attempted_rounds")) : rounds;
        storeCanonicalRounds(stage, restoredScene, rounds, canonicalRoundsFinished(previous, accepted), termination, attempted);
        std::ostringstream proof; proof << "accepted_identity=" << accepted.identity() << "\ngeneration=" << accepted.generation()
            << "\naccepted_sources=" << accepted.sourceCount() << "\nraw=" << accepted.observations() << "\ncompleted_rounds=" << rounds
            << "\nprevious=" << previous.u8string() << "\nfull_capture_verified=false\n";
        resume::store(stage / "accepted-prefix-fork.txt", proof.str());
    }
    std::filesystem::rename(stage, destination); return true;
}
}
