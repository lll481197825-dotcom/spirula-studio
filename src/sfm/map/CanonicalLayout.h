#pragma once
#include "sfm/map/ImmutableSceneAliases.h"
#include "sfm/map/SceneTopology.h"
#include "sfm/core/Resume.h"

namespace sfm::regional {
enum class CanonicalLayoutPhase { Cold, ResumeAccepted, AppendSources };
struct PreparedCanonicalLayout {
    CanonicalLayoutPhase phase = CanonicalLayoutPhase::Cold;
    std::unique_ptr<ImmutableSceneTopology> index;
    CanonicalScene::Binding binding;
    std::string seedIdentity;
    size_t acceptedSources = 0;
};
inline std::string canonicalLayoutSeedRecord(const std::filesystem::path& coordination,
    const std::string& aliases, const std::string& parent = {}) {
    if (parent.empty()) return (coordination / ("layout-seed-" + aliases + ".txt")).u8string();
    spirula::Sha256 hash;
    for (const auto& value : {aliases, parent}) { const uint64_t count = value.size(); hash.update(reinterpret_cast<const uint8_t*>(&count), sizeof(count)); hash.update(reinterpret_cast<const uint8_t*>(value.data()), value.size()); }
    return (coordination / ("layout-seed-" + hash.hex() + ".txt")).u8string();
}
inline PreparedCanonicalLayout prepareCanonicalLayout(const std::filesystem::path& coordination,
    const RegionalWorkingStore& catalog, const CanonicalScene& scene, const std::vector<uint32_t>& owners,
    size_t indexBudget, ImmutableSceneTopology::Progress progress = {}) {
    if (scene.editing()) throw std::logic_error("canonical layout selection requires an idle accepted scene");
    if (!catalog.sourceCount() || owners.empty()) throw std::invalid_argument("canonical layout requires immutable sources and photo ownership");
    PreparedCanonicalLayout result;
    result.acceptedSources = scene.initialized() ? scene.sourceCount() : 0;
    if (catalog.sourceCount() < result.acceptedSources) throw std::runtime_error("canonical immutable source catalogue shortened");
    if (scene.initialized()) {
        const auto prefix = CanonicalScene::bindingFor(catalog, scene.acceptedSnapshot().binding.topology, true, result.acceptedSources);
        if (!(prefix == scene.acceptedSnapshot().binding)) throw std::runtime_error("canonical accepted immutable raw prefix changed");
        if (catalog.sourceCount() == result.acceptedSources) {
            result.phase = CanonicalLayoutPhase::ResumeAccepted;
            const auto& accepted = scene.acceptedSnapshot();
            if (!std::filesystem::is_regular_file(coordination / "indices" / accepted.binding.topology / "manifest.bin"))
                throw std::runtime_error("canonical accepted immutable index is missing; retained head cannot be rebuilt from current parameters");
            result.index = std::make_unique<ImmutableSceneTopology>(coordination / "indices", indexBudget);
            result.index->restore(accepted.binding.topology);
            if (result.index->pointSlots() != accepted.pointSlots || result.index->observationCount() != accepted.observations ||
                result.index->imageSlots() != owners.size()) throw std::runtime_error("canonical accepted immutable index coverage changed");
            const auto aliases = ImmutableSceneAliases::dependencyIdentity(catalog, owners);
            const auto context = coordination / "indices" / accepted.binding.topology / "layout-seed-context.txt";
            if (std::filesystem::exists(context) && (std::filesystem::is_symlink(context) || std::filesystem::file_size(context) > 1024))
                throw std::runtime_error("canonical accepted layout seed context framing changed");
            const std::vector<std::string> seeds = std::filesystem::exists(context) ? std::vector<std::string>{resume::recorded(context)} :
                std::vector<std::string>{resume::recorded(std::filesystem::u8path(canonicalLayoutSeedRecord(coordination, aliases))), "immutable-aligned-initial-seed", ""};
            bool ownersMatch = false;
            for (const auto& seed : seeds) if (ImmutableSceneTopology::identityFor(catalog, owners, aliases, accepted.pointSlots, seed) == accepted.binding.topology) {
                ownersMatch = true; result.seedIdentity = seed; break;
            }
            if (!ownersMatch) throw std::runtime_error("canonical accepted photo ownership or immutable layout provenance changed");
            result.binding = prefix; scene.requireBinding(result.binding);
            if (progress) progress("layout-resume-accepted", accepted.observations, accepted.observations);
            return result;
        }
        result.phase = CanonicalLayoutPhase::AppendSources;
    }
    ImmutableSceneAliases aliases(coordination / "aliases", indexBudget);
    aliases.prepare(catalog, owners, progress);
    result.seedIdentity = scene.initialized() ? scene.identity() : "immutable-aligned-initial-seed";
    const auto seedRecord = std::filesystem::u8path(canonicalLayoutSeedRecord(coordination, aliases.identity(), scene.initialized() ? scene.identity() : ""));
    const auto savedSeed = resume::recorded(seedRecord);
    if (!std::filesystem::exists(seedRecord)) resume::store(seedRecord, result.seedIdentity);
    else if (savedSeed != result.seedIdentity) throw std::runtime_error("canonical candidate layout seed has a different accepted parent; retained");
    result.index = std::make_unique<ImmutableSceneTopology>(coordination / "indices", indexBudget);
    result.index->prepare(catalog, owners, aliases.identity(), aliases.points(),
        [&](auto key) { return aliases.canonical(key); },
        [&](auto key, Vec3 imported) {
            return key.source < result.acceptedSources ? scene.acceptedLandmark(scene.acceptedCanonical(key)).value : imported;
        }, result.seedIdentity, progress);
    result.binding = CanonicalScene::bindingFor(catalog, result.index->identity());
    const auto context = coordination / "indices" / result.index->identity() / "layout-seed-context.txt";
    if (!std::filesystem::exists(context)) resume::store(context, result.seedIdentity);
    else if (resume::recorded(context) != result.seedIdentity) throw std::runtime_error("canonical candidate immutable index seed context changed");
    return result;
}
}
