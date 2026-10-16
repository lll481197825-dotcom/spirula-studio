#include "sfm/map/CanonicalResume.h"
#include "sfm/map/CanonicalInitialization.h"
#include "sfm/map/ImmutableSceneAliases.h"
#include <iostream>
#include <cwctype>
using namespace sfm;
using namespace sfm::regional;
using Path = std::filesystem::path;
static void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> static void rejects(F action, const char* message) {
    bool failed = false; try { action(); } catch (const std::exception&) { failed = true; } check(failed, message);
}
static std::string sha(const Path& path) { auto value = spirula::sha256_file(path.string()); check(!value.empty(), "test SHA unavailable"); return value; }
static Reconstruction model(bool second) {
    Reconstruction rec; Camera camera; camera.id = 0; camera.width = 100; camera.height = 100; camera.fx = camera.fy = 80; camera.cx = camera.cy = 50;
    rec.cameras.emplace(0, camera);
    for (uint32_t i = 0; i < 2; ++i) { Image image; image.id = i; image.camera_id = 0; image.name = "photo-" + std::to_string(i);
        image.registered = true; image.pose.t.x = double(i); image.points2D = {{50, 50}, {55, 50}}; image.point3D_ids.assign(2, kInvalidPoint3D); rec.images.emplace(i, image); }
    Point3D point; point.xyz = {0, 0, 8}; point.track = {{0, 0}, {1, 0}}; if (!second) point.track.insert(point.track.begin(), {0, 0});
    const uint64_t id = second ? 20 : 10; rec.points3D.emplace(id, point); for (auto& image : rec.images) image.second.point3D_ids[0] = id;
    if (second) { point.xyz = {1, 0, 9}; point.track = {{0, 1}, {1, 1}}; rec.points3D.emplace(22, point); for (auto& image : rec.images) image.second.point3D_ids[1] = 22; }
    rec.next_point3D_id = 30; return rec;
}
static Reconstruction poseTable(const RegionalWorkingStore& raw) {
    auto rec = raw.poses(); rec.points3D.clear(); for (auto& image : rec.images) { image.second.points2D.clear(); image.second.point3D_ids.clear(); } return rec;
}
int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("fresh owned fixture directory required"); const Path root = std::filesystem::absolute(argv[1]);
    check(!std::filesystem::exists(root), "fixture output already exists"); std::filesystem::create_directories(root);
    const Path previous = root / "previous", next = root / "next";
    RegionalWorkingStore raw(previous / "immutable-raw", 64u << 20);
    std::vector<Path> original; for (int i = 0; i < 3; ++i) { original.push_back(root / ("source-" + std::to_string(i) + ".bin")); model_store_detail::writeModel(model(i == 1), original.back()); }
    raw.importModel(original[0], model(false), "source0"); raw.importModel(original[1], model(true), "source1");
    ImmutableSceneAliases aliases(previous / "aliases", 64u << 20); aliases.prepare(raw, {0, 1});
    ImmutableSceneTopology index(previous / "indices", 64u << 20); index.prepare(raw, {0, 1}, aliases.identity(), aliases.points(), [&](auto key) { return aliases.canonical(key); });
    CanonicalScene scene(previous / "scene", 64u << 20); const auto poses = poseTable(raw);
    scene.initialize(CanonicalScene::bindingFor(raw, index.identity()), poses, [&](const auto& emit) { aliases.emitAliases(emit); },
        [&](const auto& emit) { index.visitSeeds([&](const auto& seed) { emit(seed.root, seed.xyz); }); });
    scene.beginCandidate(); scene.commit("canonical-scene-round-fullraw-v1");
    resume::store(previous / "round-1-acceptance.txt", "generation=1\nraw=7\naccepted=1\n");
    check(canonicalCompletedRounds(previous, scene) == 1, "valid legacy complete round lost");
    const auto alternate = root / "sameN-other-cohort"; CanonicalScene other(alternate / "scene", 64u << 20);
    const auto aliasProducer = [&](const auto& emit) { aliases.emitAliases(emit); };
    const auto seedProducer = [&](const auto& emit) { index.visitSeeds([&](const auto& seed) { emit(seed.root, seed.xyz); }); };
    other.initialize(scene.binding(), poses, aliasProducer, seedProducer);
    other.beginCandidate(); other.commit("canonical-scene-round-fullraw-v1");
    auto otherBinding = other.binding(); otherBinding.topology = "different-layout-same-raw-count";
    other.beginLayoutCandidate(otherBinding, poses, aliasProducer, seedProducer); other.commit("canonical-new-cohort-fullraw-prefix-v1");
    resume::store(alternate / "round-2-acceptance.txt", "generation=2\nraw=7\naccepted=1\n");
    resume::store(alternate / "round-1-acceptance.txt", "generation=1\nraw=7\naccepted=1\n");
    check(canonicalCompletedRounds(alternate, other) == 0, "same-N layout promotion or other topology record counted as a BA round");
    scene.beginCandidate(); scene.commit("canonical-scene-round-fullraw-v1.cohort-round=1");
    storeCanonicalRounds(previous, scene, 99, true);
    check(canonicalCompletedRounds(previous, scene) == 1, "external record overrode atomic head round ordinal");
    check(!canonicalRoundsFinished(previous, scene), "external complete record bypassed head ordinal");
    const auto record = previous / ("cohort-rounds-" + scene.binding().topology + ".txt");
    resume::store(record, "accepted_identity=" + scene.identity() + "\ngeneration=0\nraw=1\nrounds=1\ncomplete=1\n");
    check(canonicalCompletedRounds(previous, scene) == 1 && !canonicalRoundsFinished(previous, scene), "stale generation/raw record trusted");
    resume::store(record, "accepted_identity=" + scene.identity() + "\ncomplete=1\n");
    check(canonicalCompletedRounds(previous, scene) == 1 && !canonicalRoundsFinished(previous, scene), "partial nonauthoritative complete record blocked or bypassed atomic head");
    storeCanonicalRounds(previous, scene, 1, false);
    CanonicalInitialStage stage; stage.poses = poses; stage.fixedImages = {0}; stage.targets = {1};
    stage.sources = {{"source0", original[0], original[0], modelResidentBytes(model(false)), {}}, {"source1", original[1], original[1], modelResidentBytes(model(true)), {}}};
    stage.deferred.push_back({"weak-source", original[2], modelResidentBytes(model(false)), 1});
    saveCanonicalInitialStage(previous / "initialization", "fixture-signature", stage);
    raw.importModel(original[2], model(false), "dirty-unaccepted-source");
    const auto originalHead = sha(previous / "scene" / "head.bin"), originalManifest = sha(previous / "immutable-raw" / "manifest.bin");
    rejects([&] { forkCanonicalResume(previous, previous / "nested-v2", 64u << 20); }, "resume fork wrote inside its previous coordination root");
#ifdef _WIN32
    auto casePrevious = previous.wstring(); std::transform(casePrevious.begin(), casePrevious.end(), casePrevious.begin(),
        [](wchar_t c) { return wchar_t(std::towupper(c)); });
    rejects([&] { forkCanonicalResume(previous, Path(casePrevious) / "nested-case-v2", 64u << 20); }, "Windows case variant bypassed whole coordination fork boundary");
#endif
    check(!std::filesystem::exists(previous / "nested-v2"), "invalid nested fork exposed a destination");
    check(forkCanonicalResume(previous, next, 64u << 20), "native accepted-prefix helper failed");
    CanonicalScene restored(next / "scene", 64u << 20); RegionalWorkingStore prefix(next / "immutable-raw", 64u << 20);
    check(prefix.sourceCount() == 2 && raw.sourceCount() == 3 && restored.identity() == scene.identity() && restored.observations() == 7 &&
        restored.binding() == CanonicalScene::bindingFor(prefix, restored.binding().topology), "helper included dirty suffix or changed accepted authority");
    ImmutableSceneTopology restoredIndex(next / "indices", 64u << 20); restoredIndex.restore(restored.binding().topology);
    uint64_t rows = 0; restoredIndex.visitRaw([&](auto row) { check(row.point.source < 2, "dirty raw entered restored index"); ++rows; }); check(rows == 7, "full duplicate raw coverage changed");
    const auto initialization = loadCanonicalInitialStage(next / "initialization", "fixture-signature", {}, 64u << 20, true);
    check(initialization && initialization->deferred.size() == 1 && initialization->deferred[0].label == "weak-source" &&
        initialization->fixedImages == stage.fixedImages && initialization->targets == stage.targets, "fork dropped deferred queue or original held references");
    const auto oldPoints = sha(scene.acceptedSnapshot().directory / "points.bin");
    restored.beginCandidate(); restored.applyPointUpdates({{0, {3, 4, 8}}}); restored.commit("canonical-scene-round-fullraw-v1.cohort-round=2");
    check(sha(scene.acceptedSnapshot().directory / "points.bin") == oldPoints && sha(previous / "scene" / "head.bin") == originalHead &&
        sha(previous / "immutable-raw" / "manifest.bin") == originalManifest, "helper COW altered original accepted state");
    rejects([&] { scene.inspectRetainedSnapshot(next / "scene" / scene.acceptedSnapshot().directory.filename()); }, "foreign retained snapshot inspected");
    check(scene.inspectRetainedSnapshot(scene.acceptedSnapshot().directory).identity == scene.identity(), "owned retained snapshot failed native validation");
    rejects([&] { raw.forkImmutablePrefix(root / "zero-prefix", 0); }, "zero-source immutable prefix fork accepted");
    const auto missingRaw = root / "missing-original-raw";
    scene.forkAccepted(missingRaw / "scene");
    rejects([&] { forkCanonicalResume(missingRaw, root / "missing-raw-target", 64u << 20); }, "missing accepted raw catalog silently created");
    check(!std::filesystem::exists(missingRaw / "immutable-raw"), "failed helper mutated original root by creating a missing raw catalog");
    rejects([&] { forkCanonicalResume(previous, next, 64u << 20); }, "existing fork target accepted");
    check(!forkCanonicalResume(root / "missing", root / "unused", 64u << 20), "missing previous head invented accepted state");
    std::cout << "CanonicalResume: dirty-prefix exclusion, full duplicate raw, held/deferred scope, head-bound round ordinal, SHA and COW passed\n"; return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
