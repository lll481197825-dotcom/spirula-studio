#include "sfm/map/CanonicalLayout.h"
#include "sfm/map/CanonicalResume.h"
#include "sfm/map/CanonicalInitialization.h"
#include "sfm/map/CanonicalAdmission.h"
#include "sfm/map/SceneCoordinator.h"
#include <iostream>

using namespace sfm;
using namespace sfm::regional;
using Path = std::filesystem::path;
static constexpr size_t budget = 64u << 20;
static void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> static void rejects(F action, const char* message) {
    bool failed = false; try { action(); } catch (const std::exception&) { failed = true; } check(failed, message);
}
static std::string sha(const Path& path) { auto value = spirula::sha256_file(path.string()); check(!value.empty(), "fixture SHA missing"); return value; }
static std::map<std::string, std::string> tree(const Path& path) {
    std::map<std::string, std::string> result;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(path))
        result.emplace(entry.path().lexically_relative(path).generic_u8string(), entry.is_directory() ? "directory" : sha(entry.path()));
    return result;
}
static Reconstruction input(int source) {
    Reconstruction rec; Camera camera; camera.id = 0; camera.width = camera.height = 100;
    camera.fx = camera.fy = 80; camera.cx = camera.cy = 50; rec.cameras.emplace(0, camera);
    for (uint32_t id = 0; id < (source == 2 ? 3u : 2u); ++id) {
        Image image; image.id = id; image.camera_id = 0; image.name = "photo-" + std::to_string(id);
        image.registered = true; image.pose.t.x = -double(id); image.points2D.resize(3); image.point3D_ids.assign(3, kInvalidPoint3D);
        rec.images.emplace(id, image);
    }
    auto add = [&](uint64_t id, Vec3 xyz, uint32_t feature, std::vector<TrackElement> track) {
        Point3D point; point.xyz = xyz; point.track = std::move(track);
        for (const auto& row : point.track) {
            auto& image = rec.images.at(row.image_id); image.points2D[row.point2D_idx] = camera.project(xyz + image.pose.t);
            image.point3D_ids[row.point2D_idx] = id;
        }
        rec.points3D.emplace(id, std::move(point));
    };
    add(source == 0 ? 10 : source == 1 ? 20 : 30, {0, 0, 8}, 0,
        source == 0 ? std::vector<TrackElement>{{0, 0}, {0, 0}, {1, 0}} : source == 1 ?
        std::vector<TrackElement>{{0, 0}, {1, 0}} : std::vector<TrackElement>{{0, 0}, {2, 0}});
    if (source == 1) add(22, {1, 0, 9}, 1, {{0, 1}, {1, 1}});
    if (source == 2) add(32, {2, 1, 10}, 2, {{0, 2}, {1, 2}, {2, 2}});
    rec.next_point3D_id = 40; return rec;
}
static Reconstruction poses(const RegionalWorkingStore& raw) {
    auto result = raw.poses(); result.points3D.clear();
    for (auto& image : result.images) { image.second.points2D.clear(); image.second.point3D_ids.clear(); }
    return result;
}
static size_t seedRecords(const Path& coordination) {
    size_t count = 0; for (const auto& entry : std::filesystem::directory_iterator(coordination))
        count += entry.is_regular_file() && entry.path().filename().u8string().find("layout-seed-") == 0;
    return count;
}
int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("fresh owned fixture directory required");
    const Path root = std::filesystem::absolute(argv[1]); check(!std::filesystem::exists(root), "fixture output exists"); std::filesystem::create_directories(root);
    const Path previous = root / "previous", resumed = root / "resumed";
    RegionalWorkingStore raw(previous / "immutable-raw", budget); std::vector<Path> originals;
    for (int source = 0; source < 3; ++source) { originals.push_back(root / ("input-" + std::to_string(source) + ".bin")); model_store_detail::writeModel(input(source), originals.back()); }
    raw.importModel(originals[0], input(0), "source0"); raw.importModel(originals[1], input(1), "source1");
    CanonicalScene scene(previous / "scene", budget); const std::vector<uint32_t> owners{0, 1, 1}; std::vector<std::string> stages;
    const auto progress = [&](const char* stage, uint64_t, uint64_t) { stages.emplace_back(stage); };
    auto cold = prepareCanonicalLayout(previous, raw, scene, owners, budget, progress);
    check(cold.phase == CanonicalLayoutPhase::Cold && cold.seedIdentity == "immutable-aligned-initial-seed", "cold lifecycle selection incorrect");
    check(ImmutableSceneAliases::dependencyIdentity(raw, owners) != ImmutableSceneAliases::dependencyIdentity(raw, {1, 0, 1}), "alias dependency hash ignored ownership");
    scene.initializeValidated(cold.binding, poses(raw), [&](const auto& emit) { cold.index->emitAliases(emit); },
        completeSceneSeed(*cold.index, poses(raw), 3, budget, 1), [&](const auto& candidate) {
            const auto q = sceneQuality(*cold.index, candidate, 3, budget, 1); check(q.observations == 7 && !q.invalid_projections, "cold full raw validation incomplete");
        }, "canonical-initial-fullsupport-baseline-v1");
    scene.beginCandidate(); scene.applyPointUpdates({{0, {0.001, 0, 8}}}); scene.commit("canonical-scene-round-fullraw-v1.cohort-round=1");
    scene.beginCandidate(); auto camera = scene.poses().cameras.at(0); camera.fx = camera.fy = 80.01; scene.setCamera(camera);
    scene.commit("canonical-scene-round-fullraw-v1.cohort-round=2"); storeCanonicalRounds(previous, scene, 2, true);
    CanonicalInitialStage initialization; initialization.poses = scene.poses(); initialization.fixedImages = {0};
    initialization.sources = {{"source0", originals[0], originals[0], modelResidentBytes(input(0)), {}},
        {"source1", originals[1], originals[1], modelResidentBytes(input(1)), {}}};
    saveCanonicalInitialStage(previous / "initialization", "layout-fixture", initialization);
    raw.importModel(originals[2], input(2), "dirty-unaccepted-source"); const auto originalState = tree(previous);
    check(forkCanonicalResume(previous, resumed, budget), "accepted cohort fork failed");
    check(tree(previous) == originalState, "fork mutated original accepted or dirty evidence");
    CanonicalScene restored(resumed / "scene", budget); RegionalWorkingStore prefix(resumed / "immutable-raw", budget);
    const auto resumedBefore = tree(resumed); const auto oldHead = sha(resumed / "scene/head.bin"), oldTopology = restored.binding().topology;
    prefix.resetCounters(); stages.clear(); auto warm = prepareCanonicalLayout(resumed, prefix, restored, owners, budget, progress);
    check(warm.phase == CanonicalLayoutPhase::ResumeAccepted && warm.index->identity() == oldTopology && warm.binding == restored.binding(), "resumed production selector replaced accepted topology");
    check(stages == std::vector<std::string>{"layout-resume-accepted"} && prefix.counters().source_full_reads == 0 &&
        prefix.counters().raw_bytes == 0 && prefix.counters().index_bytes == 0 && prefix.counters().parameter_bytes == 0,
        "resume performed a cold source/alias/index scan");
    check(tree(resumed) == resumedBefore && canonicalCompletedRounds(resumed, restored) == 2, "resume wrote seed/index or changed accepted round ordinal");
    const auto legacy = root / "legacy-fork-missing-context";
    std::filesystem::copy(resumed, legacy, std::filesystem::copy_options::recursive);
    std::filesystem::remove(legacy / "indices" / oldTopology / "layout-seed-context.txt");
    const auto legacyAlias = ImmutableSceneAliases::dependencyIdentity(prefix, owners);
    resume::store(std::filesystem::u8path(canonicalLayoutSeedRecord(legacy, legacyAlias)), restored.identity());
    const auto legacyBefore = tree(legacy);
    {
        CanonicalScene legacyScene(legacy / "scene", budget); RegionalWorkingStore legacyRaw(legacy / "immutable-raw", budget);
        auto selected = prepareCanonicalLayout(legacy, legacyRaw, legacyScene, owners, budget);
        check(selected.phase == CanonicalLayoutPhase::ResumeAccepted && selected.index->identity() == oldTopology && tree(legacy) == legacyBefore,
            "legacy stale-current-head seed marker rebuilt topology or changed accepted evidence");
        rejects([&] { prepareCanonicalLayout(legacy, legacyRaw, legacyScene, {1, 0, 1}, budget); }, "legacy proof accepted same-size owner drift");
    }
    const auto changedContext = root / "changed-seed-context";
    std::filesystem::copy(resumed, changedContext, std::filesystem::copy_options::recursive);
    resume::store(changedContext / "indices" / oldTopology / "layout-seed-context.txt", "wrong-seed-token");
    const auto changedBefore = tree(changedContext);
    {
        CanonicalScene changedScene(changedContext / "scene", budget); RegionalWorkingStore changedRaw(changedContext / "immutable-raw", budget);
        rejects([&] { prepareCanonicalLayout(changedContext, changedRaw, changedScene, owners, budget); }, "changed retained seed context was ignored");
        check(tree(changedContext) == changedBefore, "context mismatch rewrote immutable evidence");
    }
    const auto missingIndex = root / "missing-bound-index";
    std::filesystem::copy(resumed, missingIndex, std::filesystem::copy_options::recursive);
    std::filesystem::rename(missingIndex / "indices" / oldTopology / "manifest.bin", missingIndex / "indices" / oldTopology / "manifest-retained-evidence.bin");
    const auto missingBefore = tree(missingIndex);
    {
        CanonicalScene missingScene(missingIndex / "scene", budget); RegionalWorkingStore missingRaw(missingIndex / "immutable-raw", budget);
        rejects([&] { prepareCanonicalLayout(missingIndex, missingRaw, missingScene, owners, budget); }, "missing head-bound index was rebuilt from current parameters");
        check(tree(missingIndex) == missingBefore, "missing accepted index refusal rewrote retained artifacts");
    }
    const auto ownerState = tree(resumed);
    rejects([&] { prepareCanonicalLayout(resumed, prefix, restored, {1, 0, 1}, budget); }, "same-size owner partition drift accepted");
    check(tree(resumed) == ownerState, "owner mismatch created cold artifacts");
    const Path wrongRaw = root / "wrong-raw"; RegionalWorkingStore foreign(wrongRaw, budget);
    foreign.importModel(originals[0], input(0), "foreign-source0"); foreign.importModel(originals[1], input(1), "source1");
    rejects([&] { prepareCanonicalLayout(resumed, foreign, restored, owners, budget); }, "accepted immutable raw prefix drift accepted");
    check(sha(resumed / "scene/head.bin") == oldHead, "prefix rejection altered accepted head");
    prefix.importModel(originals[2], input(2), "new-source2"); stages.clear();
    auto append = prepareCanonicalLayout(resumed, prefix, restored, owners, budget, progress);
    check(append.phase == CanonicalLayoutPhase::AppendSources && append.acceptedSources == 2 &&
        append.index->observationCount() == 12 && sha(resumed / "scene/head.bin") == oldHead, "append preparation changed accepted authority or raw count");
    const auto initialRecords = seedRecords(resumed); const auto firstAppendIdentity = append.index->identity();
    auto nextPoses = poses(prefix); for (const auto& item : restored.poses().images) nextPoses.images.at(item.first) = item.second;
    nextPoses.cameras = restored.poses().cameras;
    auto admittedSeed = prepareCanonicalAdmissionSeed(*warm.index, *append.index, restored, nextPoses, 3, budget, 1);
    restored.beginLayoutCandidate(append.binding, nextPoses, [&](const auto& emit) { append.index->emitAliases(emit); }, admittedSeed.producer);
    const auto candidate = sceneQuality(*append.index, restored, 3, budget, 1);
    check(candidate.observations == 12 && !candidate.invalid_projections, "append candidate lost complete raw multiplicity");
    restored.sealCandidate("controlled-append-candidate"); const auto rejected = restored.abort("fixture-admission-gate-rejection");
    check(std::filesystem::is_regular_file(rejected / "points.bin") && restored.sourceCount() == 2 && sha(resumed / "scene/head.bin") == oldHead,
        "append rejection did not preserve full candidate and unchanged accepted head");
    auto oldXYZ = restored.pointXYZ(0); restored.beginCandidate(); restored.applyPointUpdates({{0, oldXYZ + Vec3{0.0001, 0, 0}}});
    restored.commit("canonical-scene-round-fullraw-v1.cohort-round=3");
    auto appendDifferentParent = prepareCanonicalLayout(resumed, prefix, restored, owners, budget);
    check(appendDifferentParent.phase == CanonicalLayoutPhase::AppendSources && appendDifferentParent.index->identity() != firstAppendIdentity &&
        seedRecords(resumed) == initialRecords + 1, "legal same-source append with different accepted parent collided with old marker");
    const auto parentHead = sha(resumed / "scene/head.bin");
    auto nextSeed = prepareCanonicalAdmissionSeed(*warm.index, *appendDifferentParent.index, restored, nextPoses, 3, budget, 1);
    restored.beginLayoutCandidate(appendDifferentParent.binding, nextPoses, [&](const auto& emit) { appendDifferentParent.index->emitAliases(emit); }, nextSeed.producer);
    const auto beforePrefix = scenePrefixQuality(*warm.index, restored, 3, budget, 1, true), afterPrefix = scenePrefixQuality(*warm.index, restored, 3, budget, 1, false);
    check(canonicalPrefixAccepted(beforePrefix, afterPrefix), "controlled admission regressed old full raw objective");
    check(sceneQuality(*appendDifferentParent.index, restored, 3, budget, 1).observations == 12, "new full raw objective incomplete");
    restored.commit("canonical-new-cohort-fullraw-prefix-v1");
    check(sha(resumed / "scene/head.bin") != parentHead && restored.sourceCount() == 3, "verified candidate layout did not publish atomically");
    {
        CanonicalScene reopened(resumed / "scene", budget); const auto state = tree(resumed);
        auto selected = prepareCanonicalLayout(resumed, prefix, reopened, owners, budget);
        check(selected.phase == CanonicalLayoutPhase::ResumeAccepted && selected.index->identity() == reopened.binding().topology && tree(resumed) == state,
            "accepted appended cohort cannot reopen without rebuilding its layout");
    }
    const auto appendedFork = root / "appended-cohort-fork";
    check(forkCanonicalResume(resumed, appendedFork, budget), "accepted appended cohort fork failed");
    {
        CanonicalScene forkScene(appendedFork / "scene", budget); RegionalWorkingStore forkRaw(appendedFork / "immutable-raw", budget);
        const auto state = tree(appendedFork); auto selected = prepareCanonicalLayout(appendedFork, forkRaw, forkScene, owners, budget);
        check(selected.phase == CanonicalLayoutPhase::ResumeAccepted && selected.index->identity() == restored.binding().topology && tree(appendedFork) == state,
            "append-parent seed context lost through fork/reopen");
    }
    check(tree(previous) == originalState, "whole lifecycle changed original accepted/source evidence");
    std::cout << "CanonicalLayout cold/commits/fork/production resume/owners/raw prefix/append reject/new parent/commit/reopen passed\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
