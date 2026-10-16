#include "sfm/map/CanonicalOriginalCoverage.h"
#include "sfm/map/CanonicalImageIdentity.h"
#include "sfm/map/RegionalLandmarks.h"
#include <iostream>

using namespace sfm;
using namespace sfm::regional;
using Path = std::filesystem::path;
static constexpr size_t budget = 64u << 20;
static void check(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
static std::string sha(const Path& path) { const auto hash = spirula::sha256_file(path.u8string()); check(hash.size() == 64, "SHA unavailable"); return hash; }
static Reconstruction input(bool missing = false) {
    Reconstruction rec; Camera camera; camera.id = 0; camera.width = camera.height = 100;
    camera.fx = camera.fy = 80; camera.cx = camera.cy = 50; rec.cameras.emplace(0, camera);
    Point3D point; point.xyz = {0, 0, 8}; point.track = {{0, 0}, {1, 0}};
    if (!missing) point.track.push_back({0, 0});
    for (uint32_t id = 0; id < 2; ++id) {
        Image image; image.id = id; image.camera_id = 0; image.name = "photo-" + std::to_string(id);
        image.registered = true; image.pose.t.x = -double(id);
        image.points2D = {camera.project(point.xyz + image.pose.t)}; image.point3D_ids = {10}; rec.images.emplace(id, image);
    }
    rec.points3D.emplace(10, point); rec.next_point3D_id = 11; return rec;
}
static void initialize(CanonicalScene& scene, const RegionalWorkingStore& raw) {
    auto poses = raw.poses(); for (auto& image : poses.images) { image.second.points2D.clear(); image.second.point3D_ids.clear(); }
    poses.points3D.clear();
    scene.initialize(CanonicalScene::bindingFor(raw, "coverage-fixture-topology"), poses,
        [](const auto& emit) { emit({0, 0, 10}, 0); }, [](const auto& emit) { emit(0, {0, 0, 8}); });
}
static void real(const Path& originalRoot, const Path& coordination, const Path& output) {
    check(!std::filesystem::exists(output), "audit output must be new");
    const auto available = spirula::availableRamBytes();
    const size_t allowance = available ? size_t(available - std::min<uint64_t>(available / 4, uint64_t(4) << 30)) : size_t(2) << 30;
    check(allowance >= (128u << 20), "insufficient RAM headroom for read-only audit");
    const auto initial = loadCanonicalInitialStage(coordination / "initialization",
        regionalBaseSignature(resume::recorded(originalRoot / "signature.txt")), {}, allowance / 4, true);
    check(bool(initial), "verified initial source stage absent");
    const auto required = canonicalOriginalRequirements(*initial, originalRoot / "source-evidence-v1",
        resume::recorded(originalRoot / "source-evidence-v1" / "inputs.txt"));
    RegionalWorkingStore raw(coordination / "immutable-raw", allowance / 4, RegionalWorkingStore::OpenMode::ReadOnly);
    CanonicalScene scene(coordination / "scene", allowance / 4, CanonicalScene::OpenMode::ReadOnly);
    struct Names { std::string name, export_name; }; std::map<uint32_t, Names> names;
    for (const auto& item : scene.poses().images) {
        const auto found = raw.poses().images.find(item.first); check(found != raw.poses().images.end(), "accepted catalogue image missing");
        names.emplace(item.first, Names{item.second.name, found->second.name});
    }
    const auto headSHA = sha(coordination / "scene" / "head.bin"), manifestSHA = sha(coordination / "immutable-raw" / "manifest.bin");
    const auto result = verifyCanonicalOriginalCoverage(raw, scene, required,
        [&](Reconstruction& model) { canonicalizeImageIdentity(model, names); }, allowance / 2);
    check(sha(coordination / "scene" / "head.bin") == headSHA && sha(coordination / "immutable-raw" / "manifest.bin") == manifestSHA,
        "read-only audit changed accepted evidence");
    std::filesystem::create_directories(output); resume::store(output / "original-observation-coverage.csv", result.csv());
    std::ostringstream proof; proof << "scope=original_raw_multiset_only_requires_formal_Reader_and_capture_gates\naccepted=" << result.acceptedIdentity
        << "\noriginal_sources_complete=" << result.complete << "\nrequired_sources=" << result.rows.size()
        << "\naccepted_head_sha256=" << headSHA << "\nimmutable_manifest_sha256=" << manifestSHA << "\nfull_capture_verified=false\n";
    resume::store(output / "source-completeness.txt", proof.str()); std::cout << proof.str();
}
int main(int argc, char** argv) try {
    if (argc == 5 && std::string(argv[1]) == "--real") { real(std::filesystem::u8path(argv[2]), std::filesystem::u8path(argv[3]), std::filesystem::u8path(argv[4])); return 0; }
    check(argc == 2, "usage: <fresh-output> or --real <original-root> <coordination> <new-output>");
    const Path root = std::filesystem::absolute(std::filesystem::u8path(argv[1])); check(!std::filesystem::exists(root), "fixture output exists");
    const Path evidence = root / "source-evidence-v1"; std::filesystem::create_directories(evidence);
    const Path original = evidence / "weak-original.bin"; model_store_detail::writeModel(input(), original);
    CanonicalInitialStage stage; stage.poses = input(); stage.poses.points3D.clear();
    for (auto& image : stage.poses.images) { image.second.points2D.clear(); image.second.point3D_ids.clear(); }
    stage.deferred.push_back({"weak-original.bin", original, modelResidentBytes(input()), 0});
    saveCanonicalInitialStage(root / "initialization", "coverage-fixture", stage);
    const auto initialSHA = sha(root / "initialization/manifest.bin");
    const auto loaded = loadCanonicalInitialStage(root / "initialization", "coverage-fixture", {}, budget, true);
    check(loaded && loaded->deferred.size() == 1 && loaded->deferred[0].originalBytes == std::filesystem::file_size(original) &&
        loaded->deferred[0].originalSHA == sha(original), "existing v2 parser lost verified original identity");
    const auto manifest = "weak-original.bin:" + std::to_string(std::filesystem::file_size(original)) + "\n";
    const auto required = canonicalOriginalRequirements(*loaded, evidence, manifest);
    auto run = [&](const char* name, bool missing, const std::string& label, bool suffix) {
        const auto directory = root / name; RegionalWorkingStore raw(directory / "immutable-raw", budget);
        const auto native = directory / "accepted-input.bin"; model_store_detail::writeModel(input(missing), native);
        raw.importModel(native, input(missing), label); CanonicalScene scene(directory / "scene", budget); initialize(scene, raw);
        if (suffix) raw.importModel(original, input(), "weak-original.bin");
        const auto oldHead = sha(directory / "scene/head.bin"), oldRaw = sha(directory / "immutable-raw/manifest.bin");
        const auto result = verifyCanonicalOriginalCoverage(raw, scene, required, [](Reconstruction&) {}, budget);
        check(sha(directory / "scene/head.bin") == oldHead && sha(directory / "immutable-raw/manifest.bin") == oldRaw,
            "coverage verification mutated accepted head or source evidence");
        check(result.rows.size() == 1 && result.rows[0].observations == 3, "original multiset count changed"); return result;
    };
    const size_t staleDeferredQueue = 1;
    const auto complete = run("stale-queue", false, "weak-original.bin", false);
    check(staleDeferredQueue == 1 && complete.complete && complete.rows[0].matched == 3,
        "stale queue overruled complete accepted raw multiset");
    const auto missing = run("missing-duplicate", true, "weak-original.bin", false);
    check(!missing.complete && missing.rows[0].matched == 2, "one missing duplicate raw row was hidden");
    check(!run("wrong-label", false, "wrong-label", false).complete, "wrong source label passed");
    check(!run("unadmitted-suffix", false, "first-source", true).complete, "unadmitted catalogue source counted as accepted");
    auto altered = required; altered[0].sha[0] = altered[0].sha[0] == '0' ? '1' : '0';
    { RegionalWorkingStore raw(root / "stale-queue/immutable-raw", budget); CanonicalScene scene(root / "stale-queue/scene", budget);
      check(!verifyCanonicalOriginalCoverage(raw, scene, altered, [](Reconstruction&) {}, budget).complete, "original SHA mismatch passed"); }
    check(sha(root / "initialization/manifest.bin") == initialSHA && sha(original) == required[0].sha,
        "legacy initialization or original source was modified");
    std::cout << "Original source SHA/accepted prefix/full raw duplicates/stale queue/missing row/wrong label/unadmitted suffix passed\n"; return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
