#include "sfm/map/SceneCoordinator.h"
#include "sfm/map/NativeSceneTasks.h"
#include "sfm/map/RegionalPriors.h"
#include "sfm/map/CanonicalCapacity.h"
#include <iostream>

using namespace sfm;
using namespace sfm::regional;
using Path = std::filesystem::path;
static void check(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
static bool exact(Vec3 a, Vec3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
static bool close(double a, double b) { return std::abs(a - b) <= 1e-9 * std::max(1., std::abs(a)); }
template<class Action> static void rejects(Action action, const char* reason) { bool failed = false; try { action(); } catch (const std::exception&) { failed = true; } check(failed, reason); }
static Vec3 truth(uint32_t point) { return {0.2 * (point % 8) - .6, .18 * (point / 8) - .4, 5 + .07 * (point % 7)}; }
static void capacityTest(const Path& root) {
    std::filesystem::create_directories(root);
    constexpr uint64_t physicalSlots = 16;
    { model_store_detail::Writer points(root / "points.bin"), kinds(root / "kinds.bin");
      for (uint64_t i = 0; i < physicalSlots; ++i) { points.scalar(Vec3{}); kinds.scalar<uint8_t>(0); } points.finish(); kinds.finish(); }
    const auto pointBytes = std::filesystem::file_size(root / "points.bin"), kindBytes = std::filesystem::file_size(root / "kinds.bin");
    check(pointBytes == physicalSlots * 3 * sizeof(double) && kindBytes == physicalSlots,
        "physical typed snapshot payload differs from XYZ plus one kind byte per slot");
    const auto physicalPayloadPerSlot = (pointBytes + kindBytes) / physicalSlots;
    CanonicalDiskInputs input; input.rawRows = 34369746; input.pointSlots = 6638229; input.imageSlots = 10650;
    input.posesBytes = 2205730; input.snapshotMetadataBytes = 4096; input.stepReferenceBytes = 8192;
    input.exportPeakBytes = 5214772584; input.compactRejectedSteps = true; input.aliasHardlinks = true;
    const auto compact = canonicalCohortDiskForecast(input);
    input.compactRejectedSteps = false; const auto full = canonicalCohortDiskForecast(input);
    const auto snapshotPayload = input.pointSlots * physicalPayloadPerSlot;
    const auto completeTrialCount = input.rounds * input.backtrackSteps;
    // Compact evidence retains all proposals and one full archive fallback.
    check(full.roundArchives - compact.roundArchives == completeTrialCount * (snapshotPayload - input.stepReferenceBytes) - snapshotPayload, "compact capacity dropped proposal/base or fallback full candidate");
    rejects([&] { storage::checkAvailable("fixture", full.peakAdditional, compact.peakAdditional); }, "full archives incorrectly admitted compact-only available capacity");
    input.compactRejectedSteps = true; input.aliasHardlinks = false;
    const auto fallback = canonicalCohortDiskForecast(input);
    check(fallback.roundArchives > compact.roundArchives && fallback.trialTemporary > compact.trialTemporary, "unknown hardlink capacity silently assumed sharing");
    auto unsupported = input; unsupported.rawRows = 0; unsupported.pointSlots = 1000000;
    const auto unsupportedForecast = canonicalCohortDiskForecast(unsupported);
    check(unsupportedForecast.indexCold >= 172000000, "cold seed sorting under-reserved points without raw observations");
    input.aliasesPrepared = input.indexPrepared = input.layoutPrepared = true;
    const auto warm = canonicalCohortDiskForecast(input);
    check(!warm.aliasCold && !warm.aliasRetained && !warm.indexCold && !warm.indexRetained && !warm.layoutSeed && warm.roundArchives == fallback.roundArchives, "warm forecast charged existing layout or lost new archives");
    input.pointSlots = UINT64_MAX;
    rejects([&] { canonicalCohortDiskForecast(input); }, "overflowing canonical disk footprint admitted");
}
struct FixtureGps : PriorSource {
    bool has(uint32_t image) const override { return image < 4; }
    bool relativeRotation(uint32_t, uint32_t, Mat3&, double&) const override { return false; }
    std::vector<uint32_t> neighbours(uint32_t) const override { return {}; }
    PosePriors factors(const std::vector<PosedImage>&) override { return {}; }
    bool position(uint32_t image, Vec3& value) const override { if (image >= 4) return false; value = {.6 * image, 0, 0}; return true; }
};
static Reconstruction model(uint32_t source) {
    Reconstruction result; result.cameras[0] = Camera::defaultFor(0, 800, 600, 500, CamModel::Pinhole); result.cameras[0].pixel_scale = 2;
    result.cameras[1] = Camera::defaultFor(1, 800, 600, 500, CamModel::FullOpenCV); result.cameras[1].pixel_scale = 3; result.cameras[1].k1 = .01; result.cameras[1].p1 = .001; result.cameras[1].k6 = .00001;
    for (uint32_t i = 0; i < 4; ++i) { Image image; image.id = i; image.camera_id = i % 2; image.registered = true; image.name = "photo-" + std::to_string(i); image.pose = {mat3Identity(), {-.6 * i, 0, 0}}; image.points2D.resize(40); image.point3D_ids.assign(40, kInvalidPoint3D); result.images.emplace(i, image); }
    for (uint32_t i = 0; i < 40; ++i) {
        Point3D point; point.xyz = truth(i) + Vec3{source ? -.03 : .02, .015, .035};
        const uint64_t id = uint64_t(source + 1) * 1000 + i;
        for (auto& image : result.images) { image.second.points2D[i] = result.cameras.at(image.second.camera_id).project(mul(image.second.pose.R, truth(i)) + image.second.pose.t); image.second.point3D_ids[i] = id; point.track.push_back({image.first, i}); }
        if (!source && !i) point.track.push_back({0, 0}); result.points3D.emplace(id, point);
    }
    result.images.at(1).pose.t.x += .08; result.next_point3D_id = uint64_t(source + 1) * 1000 + 40; return result;
}
static void siblingOwnershipTest(const Path& root) {
    std::filesystem::create_directories(root); auto original = model(0);
    for (auto& point : original.points3D) {
        auto& track = point.second.track;
        track.erase(std::remove_if(track.begin(), track.end(), [](const auto& row) { return row.image_id > 1; }), track.end());
    }
    for (uint32_t child = 2; child < 4; ++child) original.images.at(child).point3D_ids.assign(40, kInvalidPoint3D);
    for (uint32_t child = 2; child < 4; ++child) {
        const uint32_t feature = 40 + child - 2; const uint64_t id = 1000 + feature;
        Point3D point; point.xyz = truth(feature) + Vec3{.02, .015, .035};
        for (auto& image : original.images) { image.second.points2D.resize(42); image.second.point3D_ids.resize(42, kInvalidPoint3D); }
        for (uint32_t image : {uint32_t(child - 2), child}) {
            auto& view = original.images.at(image); view.points2D[feature] = original.cameras.at(view.camera_id).project(mul(view.pose.R, truth(feature)) + view.pose.t);
            view.point3D_ids[feature] = id; point.track.push_back({image, feature});
        }
        original.points3D.emplace(id, point);
    }
    original.next_point3D_id = 1042;
    const auto input = root / "source.bin"; model_store_detail::writeModel(original, input);
    const auto originalHash = spirula::sha256_file(input.string()); RegionalWorkingStore catalog(root / "catalog", 128u << 20);
    catalog.importModel(input, original, "held-anchor-siblings"); ImmutableSceneTopology index(root / "index", 32u << 20);
    const std::vector<uint32_t> owners(4, 0); index.prepare(catalog, owners, "same-owner-held-anchors", 42, [](auto key) { return key.ordinal; });
    CanonicalScene scene(root / "scene", 128u << 20);
    scene.initialize(CanonicalScene::bindingFor(catalog, index.identity()), catalog.poses(), [&](const auto& emit) { index.emitAliases(emit); },
        [&](const auto& emit) { index.visitSeeds([&](const auto& seed) { emit(seed.root, seed.xyz); }); });
    BundleOptions options; options.real = RealCfg::CPU; options.refine_intrinsics = false; options.threads = 2; options.loss = "huber"; options.loss_param = 3;
    std::vector<CanonicalBundle> children(2); const std::set<uint32_t> held{0, 1};
    children[0].prepare(index, scene, {2}, 0, held, options, 64u << 20); children[1].prepare(index, scene, {3}, 0, held, options, 64u << 20);
    SceneCoordinator::requireIndependentTasks(children, 64u << 20);
    for (const auto& child : children) {
        uint64_t anchorOnly = 0, movable = 0;
        for (size_t point = 0; point < child.points.size(); ++point) {
            if (child.points[point] < 40) { ++anchorOnly; check(child.problem.fixed_points[point] != 0, "same-owner child writes a held-anchor-only canonical XYZ"); }
            else movable += !child.problem.fixed_points[point];
        }
        check(anchorOnly == 40 && movable == 1, "sibling fixture lost complete anchor tracks or its uniquely owned private point");
    }
    const auto before = sceneQuality(index, scene, 3, 32u << 20, 2);
    scene.beginCandidate();
    for (auto& child : children) { child.reload(scene); child.solve(options); child.apply(scene); }
    triangulateScene(index, scene, 3, 32u << 20, 2); const auto after = sceneQuality(index, scene, 3, 32u << 20, 2);
    uint64_t oracleRows = 0, oracleInvalid = 0; double oracleFinite = 0, oracleTotal = 0;
    for (const auto& point : original.points3D) for (const auto& row : point.second.track) {
        const auto& image = scene.poses().images.at(row.image_id); const auto& camera = scene.poses().cameras.at(image.camera_id);
        const auto ordinal = point.first - 1000; bool valid = true;
        const auto error = reprojErrorAt(camera, image.pose, original.images.at(row.image_id).points2D.at(row.point2D_idx), scene.pointXYZ(ordinal), &valid) / camera.pixel_scale;
        const auto cost = error <= 3 ? error * error : 3 * (2 * error - 3); ++oracleRows; oracleInvalid += !valid; oracleTotal += cost; if (valid) oracleFinite += cost;
    }
    check(before.observations == 85 && after.observations == oracleRows && after.invalid_projections == oracleInvalid && close(after.finite_cost, oracleFinite) && close(after.cost, oracleTotal) && after.cost < before.cost,
        "two sibling native solves/fullsupport triangle changed original raw multiset or failed its independent projection oracle");
    scene.abort("sibling-native-geometry-proof"); check(spirula::sha256_file(input.string()) == originalHash, "sibling task rewrote original raw source");
}
static void promotionTest(const Path& root) {
    std::filesystem::create_directories(root); RegionalWorkingStore catalog(root / "catalog", 128u << 20);
    auto fragment = [&](uint32_t source) {
        auto rec = model(0); rec.points3D.clear();
        for (auto& image : rec.images) { image.second.points2D.resize(2); image.second.point3D_ids.assign(2, kInvalidPoint3D); const auto xy = rec.cameras.at(image.second.camera_id).project(mul(image.second.pose.R, truth(0)) + image.second.pose.t); image.second.points2D.assign(2, xy); }
        rec.images.at(1).pose.t.x += .08;
        Point3D point; point.xyz = truth(0) + Vec3{source == 0 ? .2 : -.1, .02, .03}; const uint64_t local = 100 + source;
        point.track = source == 0 ? std::vector<TrackElement>{{0, 0}, {1, 0}, {0, 0}} : source == 1 ? std::vector<TrackElement>{{2, 1}, {3, 1}} : std::vector<TrackElement>{{0, 0}, {2, 1}};
        for (const auto& row : point.track) rec.images.at(row.image_id).point3D_ids.at(row.point2D_idx) = local;
        rec.points3D.emplace(local, point); rec.next_point3D_id = local + 1; return rec;
    };
    for (uint32_t source = 0; source < 2; ++source) { auto rec = fragment(source); const auto path = root / ("original-" + std::to_string(source) + ".bin"); model_store_detail::writeModel(rec, path); catalog.importModel(path, rec, "original-" + std::to_string(source)); }
    const std::vector<uint32_t> owners{0, 0, 1, 1}; ImmutableSceneTopology oldIndex(root / "old-index", 32u << 20);
    oldIndex.prepare(catalog, owners, "two-private-canonical-roots", 2, [](auto key) { return key.source; });
    check(oldIndex.pointOwner(0) == 0 && oldIndex.pointOwner(1) == 1, "promotion fixture did not start with owner-private points");
    CanonicalScene scene(root / "scene", 128u << 20); scene.initialize(CanonicalScene::bindingFor(catalog, oldIndex.identity()), catalog.poses(), [&](const auto& emit) { oldIndex.emitAliases(emit); }, [&](const auto& emit) { oldIndex.visitSeeds([&](const auto& seed) { emit(seed.root, seed.xyz); }); });
    const auto key0 = catalog.key(0, 0), key1 = catalog.key(1, 0); check((scene.xyz(key0) - scene.xyz(key1)).norm() > .25, "promotion did not expose different accepted private XYZ");
    const auto acceptedIdentity = scene.identity(); const auto acceptedHead = spirula::sha256_file((root / "scene" / "head.bin").string());
    const auto acceptedPointSHA = spirula::sha256_file((scene.acceptedSnapshot().directory / "points.bin").string());
    const auto oldPoint0 = scene.pointXYZ(0), oldPoint1 = scene.pointXYZ(1);
    auto bridge = fragment(2); const auto rejectedPath = root / "uncertified-bridge.bin"; model_store_detail::writeModel(bridge, rejectedPath);
    const auto rejectedSHA = spirula::sha256_file(rejectedPath.string());
    const auto rejectedCatalogPath = catalog.forkImmutablePrefix(root / "uncertified-catalog", 2);
    {
        RegionalWorkingStore rejectedCatalog(rejectedCatalogPath, 128u << 20); rejectedCatalog.importModel(rejectedPath, bridge, "uncertified-bridge");
        ImmutableSceneTopology rejectedIndex(root / "uncertified-index", 32u << 20);
        rejectedIndex.prepare(rejectedCatalog, owners, "uncertified-old-root-fusion", 3, [](auto) { return uint64_t(0); },
            [&](auto key, Vec3 imported) { return key.source < 2 ? scene.acceptedPointXYZ(scene.acceptedCanonical(key)) : imported; }, acceptedIdentity);
        const auto rejectedPoses = scene.poses();
        bool identityRejected = false;
        try { scene.beginLayoutCandidate(CanonicalScene::bindingFor(rejectedCatalog, rejectedIndex.identity()), rejectedPoses,
            [&](const auto& emit) { rejectedIndex.emitAliases(emit); }, completeSceneSeed(rejectedIndex, rejectedPoses, 3, 32u << 20, 4)); }
        catch (const std::exception& error) { identityRejected = std::string(error.what()).find("accepted alias/type identity") != std::string::npos; }
        check(identityRejected, "uncertified merger was not rejected for its actual old canonical identity change");
        if (scene.editing()) scene.abort("uncertified-old-root-fusion-rejected");
        uint64_t rejectedRows = 0; rejectedCatalog.visitObservations([&](const auto&) { ++rejectedRows; });
        check(rejectedRows == 7 && scene.identity() == acceptedIdentity && scene.observations() == 5 &&
            spirula::sha256_file((root / "scene" / "head.bin").string()) == acceptedHead &&
            exact(scene.pointXYZ(0), oldPoint0) && exact(scene.pointXYZ(1), oldPoint1) &&
            spirula::sha256_file((scene.acceptedSnapshot().directory / "points.bin").string()) == acceptedPointSHA &&
            spirula::sha256_file(rejectedPath.string()) == rejectedSHA,
            "rejected old-root fusion changed accepted parameters, head or retained incoming multiset");
    }
    // A separate valid source adds new support to each root without fusing old roots.
    auto valid = fragment(2); valid.points3D.clear(); for (auto& image : valid.images) image.second.point3D_ids.assign(2, kInvalidPoint3D);
    Point3D shared; shared.xyz = truth(0) + Vec3{.015, .01, .025}; shared.track = {{0, 0}, {2, 0}, {2, 0}};
    Point3D privatePoint = shared; privatePoint.track = {{2, 1}};
    valid.points3D.emplace(200, shared); valid.points3D.emplace(201, privatePoint); valid.next_point3D_id = 202;
    for (const auto& point : valid.points3D) for (const auto& row : point.second.track) valid.images.at(row.image_id).point3D_ids.at(row.point2D_idx) = point.first;
    const auto path = root / "valid-support-append.bin"; model_store_detail::writeModel(valid, path); const auto validSHA = spirula::sha256_file(path.string());
    catalog.importModel(path, valid, "valid-support-append"); ImmutableSceneTopology joined(root / "joined-index", 32u << 20);
    joined.prepare(catalog, owners, "private-to-shared-preserving-old-identities", 4,
        [](auto key) { return key.source < 2 ? key.source : key.ordinal; },
        [&](auto key, Vec3 imported) { return key.source < 2 ? scene.acceptedPointXYZ(scene.acceptedCanonical(key)) : imported; }, acceptedIdentity);
    check(joined.pointOwner(0) == UINT32_MAX && joined.pointRange(0).count == 6 && joined.pointRange(1).count == 3,
        "legal private-to-shared append lost full duplicate support");
    const auto appendPoses = scene.poses(); scene.beginLayoutCandidate(CanonicalScene::bindingFor(catalog, joined.identity()), appendPoses,
        [&](const auto& emit) { joined.emitAliases(emit); }, completeSceneSeed(joined, appendPoses, 3, 32u << 20, 4)); scene.commit("fixture-legal-support-append");
    check(scene.canonical(key0) == 0 && scene.canonical(key1) == 1 && scene.canonical(catalog.key(2, 0)) == 0 && scene.canonical(catalog.key(2, 1)) == 1 &&
        scene.sourceCount() == 3 && scene.observations() == 9, "legal append rewrote old roots or lost new source multiplicity");
    BundleOptions bundle; bundle.real = RealCfg::CPU; bundle.refine_intrinsics = false; bundle.threads = 2; SceneCoordinator::Options options; options.bundle = bundle; options.allowance = 64u << 20;
    uint32_t gates = 0; SceneCoordinator coordinator(scene, joined, options, [&](const auto& before, const auto& after) { ++gates; return std::abs(before.images.at(1).pose.t.x - after.images.at(1).pose.t.x) > .06 ? std::vector<std::string>{"protected-source-frame"} : std::vector<std::string>{}; });
    auto pose = scene.poses().images.at(1).pose; pose.t.x -= .08; std::vector<CanonicalBundle> empty;
    const auto result = coordinator.round(empty, [&](auto&, auto& candidate, const auto&) { candidate.setPose(1, pose); });
    check(result.accepted && result.alpha == .5 && gates == 2, "promotion/coupled halfstep did not pass full original gates");
    double materializedCost = 0; uint64_t count = 0;
    for (size_t source = 0; source < catalog.sourceCount(); ++source) {
        const auto output = root / ("native-" + std::to_string(source) + ".bin"); model_store_detail::writeModel(scene.materialize(catalog, source), output); const auto native = model_store_detail::readModel(output);
        for (const auto& point : native.points3D) for (const auto& row : point.second.track) { const auto& image = scene.poses().images.at(row.image_id); const auto& camera = scene.poses().cameras.at(image.camera_id); bool valid = true; const double error = reprojErrorAt(camera, image.pose, native.images.at(row.image_id).points2D.at(row.point2D_idx), point.second.xyz, &valid) / camera.pixel_scale; check(valid, "canonical native materialization produced an invalid projection"); materializedCost += error <= 3 ? error * error : 3 * (2 * error - 3); ++count; }
    }
    check(count == 9 && close(materializedCost, result.after.cost), "promotion accepted fullraw and untouched native Reader disagree without parameter override");
    CanonicalScene restored(root / "scene", 128u << 20); check(restored.canonical(key0) == 0 && restored.canonical(key1) == 1 && restored.canonical(catalog.key(2, 0)) == 0 && restored.observations() == 9 &&
        close(sceneQuality(joined, restored, 3, 32u << 20, 4).cost, materializedCost) && spirula::sha256_file(path.string()) == validSHA &&
        spirula::sha256_file(rejectedPath.string()) == rejectedSHA, "promotion reopen changed canonical accepted meaning or either original candidate source");
}
static void baselineCacheTest(const Path& root) {
    std::filesystem::create_directories(root); const auto original = model(0); const auto source = root / "source.bin";
    model_store_detail::writeModel(original, source); const auto originalSHA = spirula::sha256_file(source.string());
    RegionalWorkingStore catalog(root / "catalog", 128u << 20); catalog.importModel(source, original, "baseline-cache-source");
    ImmutableSceneTopology index(root / "index", 32u << 20); index.prepare(catalog, {0, 0, 1, 1}, "baseline-cache-native", 40, [](auto key) { return key.ordinal; });
    CanonicalScene scene(root / "scene", 128u << 20);
    scene.initialize(CanonicalScene::bindingFor(catalog, index.identity()), catalog.poses(), [&](const auto& emit) { index.emitAliases(emit); },
        [&](const auto& emit) { index.visitSeeds([&](const auto& seed) { emit(seed.root, seed.xyz); }); });
    SceneCoordinator::Options options; options.allowance = 64u << 20; options.bundle.real = RealCfg::CPU; options.bundle.threads = 2; options.bundle.refine_intrinsics = false;
    SceneCoordinator controller(scene, index, options, [&](const auto&, const auto&) { return std::vector<std::string>{}; }); std::vector<CanonicalBundle> empty;
    auto improve = [&](double correction) { auto pose = scene.poses().images.at(1).pose; pose.t.x -= correction; return controller.round(empty, [&](auto&, auto& candidate, const auto&) { candidate.setPose(1, pose); }); };
    const auto first = improve(.04);
    check(first.accepted && !first.statistics.baselineCacheHits && first.statistics.completeEvaluations == 2 && first.before.observations == 161 && first.after.observations == 161,
        "first cache fixture round skipped its independent complete baseline or actual accepted candidate");
    const auto warm = improve(.02);
    check(warm.accepted && warm.statistics.baselineCacheHits == 1 && warm.statistics.completeEvaluations == 1 && warm.statistics.qualityRows == 161 && warm.after.observations == 161,
        "same-controller accepted snapshot did not reuse baseline or omitted accepted fullraw evaluation");
    scene.beginCandidate(); scene.commit("external-identical-parameters-new-accepted-identity");
    const auto invalidated = improve(.01);
    check(invalidated.accepted && !invalidated.statistics.baselineCacheHits && invalidated.statistics.completeEvaluations == 2 && invalidated.statistics.qualityRows == 322,
        "external accepted identity change reused stale baseline quality");
    const auto identity = scene.identity(), head = spirula::sha256_file((root / "scene" / "head.bin").string());
    const auto points = scene.acceptedSnapshot().directory / "points.bin"; const auto pointSHA = spirula::sha256_file(points.string());
    char before = 0; { std::fstream file(points, std::ios::binary | std::ios::in | std::ios::out); file.read(&before, 1); const char damaged = char(before ^ 1); file.seekp(0); file.write(&damaged, 1); file.flush(); check(bool(file), "cache tamper fixture could not change its owned point file"); }
    bool rejected = false, executed = false;
    try { controller.round(empty, [&](auto&, auto&, const auto&) { executed = true; }); } catch (const std::exception&) { rejected = true; }
    { std::fstream file(points, std::ios::binary | std::ios::in | std::ios::out); file.write(&before, 1); file.flush(); check(bool(file), "cache tamper fixture failed to restore original point byte"); }
    check(rejected && !executed && !scene.editing() && scene.identity() == identity && spirula::sha256_file((root / "scene" / "head.bin").string()) == head && spirula::sha256_file(points.string()) == pointSHA,
        "warm cached quality bypassed accepted point SHA verification or changed accepted state");
    check(spirula::sha256_file(source.string()) == originalSHA, "cached baseline path changed immutable original raw source");
}
int main(int argc, char** argv) try {
    const Path root = argc > 1 ? Path(argv[1]) : Path("scene-coordinator-fixture");
    if (std::filesystem::exists(root)) throw std::runtime_error("test output must be new directory"); std::filesystem::create_directories(root);
    capacityTest(root / "physical-capacity-proof");
    RegionalWorkingStore catalog(root / "catalog", 128u << 20);
    std::vector<std::string> sourceHash;
    for (uint32_t source = 0; source < 2; ++source) { auto rec = model(source); auto path = root / ("source-" + std::to_string(source) + ".bin"); model_store_detail::writeModel(rec, path); sourceHash.push_back(spirula::sha256_file(path.string())); catalog.importModel(path, rec, "source-" + std::to_string(source)); }
    ImmutableSceneTopology index(root / "index", 32u << 20); const std::vector<uint32_t> owners{0, 0, 1, 1};
    catalog.resetCounters();
    index.prepare(catalog, owners, "fixture-full-feature-union", 80, [](auto key) { return key.ordinal; });
    check(catalog.counters().parameter_bytes == 0, "immutable scene CSR/seed read retired sourceXYZ parameter files");
    check(index.pointSlots() == 80 && index.observationCount() == 321, "immutable fullraw coverage lost duplicate rows");
    uint64_t photos = 0; for (uint32_t image = 0; image < 4; ++image) index.visitPhoto(image, [&](const auto& row) { check(row.image == image, "photo CSR crossed image"); ++photos; }); check(photos == 321, "photo postings lost raw multiplicity");
    check(index.pointOwner(0) == UINT32_MAX && index.pointRange(0).count == 9, "canonical shared/fullraw support differs");
    Vec3 seed0{}; index.visitSeeds([&](const auto& seed) { if (!seed.root) seed0 = seed.xyz; });
    check(close(seed0.x, truth(0).x + (.02 * 5 - .03 * 4) / 9), "canonical seed selected first alias instead of fullraw weighting");
    const auto poses = catalog.poses(); CanonicalScene scene(root / "scene", 128u << 20);
    scene.initialize(CanonicalScene::bindingFor(catalog, index.identity()), poses, [&](const auto& emit) { index.emitAliases(emit); }, [&](const auto& emit) { index.visitSeeds([&](const auto& seed) { emit(seed.root, seed.xyz); }); });
    const auto quality = sceneQuality(index, scene, 3, 32u << 20, 4); check(quality.observations == 321 && !quality.invalid_projections, "global scene quality omitted original rows");
    std::cerr << "fullraw quality complete\n";
    double reference = 0;
    catalog.visitObservations([&](const auto& row) { const auto& image = scene.poses().images.at(row.image); const auto& camera = scene.poses().cameras.at(image.camera_id); const double error = reprojErrorAt(camera, image.pose, row.xy, scene.xyz(row.point)) / camera.pixel_scale; reference += error <= 3 ? error * error : 3 * (2 * error - 3); });
    check(close(reference, quality.cost), "CSR quality differs from immutable original raw oracle");
    BundleOptions options; options.real = RealCfg::CPU; options.max_iters = 2; options.refine_intrinsics = false; options.host_budget_bytes = 64u << 20; options.threads = 2; options.loss_param = 3;
    std::vector<CanonicalBundle> tasks(2); tasks[0].prepare(index, scene, {0, 1}, 0, {0}, options, 64u << 20); tasks[1].prepare(index, scene, {2, 3}, 1, {2}, options, 64u << 20);
    std::cerr << "dense templates complete\n";
    check(tasks[0].problem.num_obs == 161 && tasks[1].problem.num_obs == 160, "dense bundle deduplicated original raw rows");
    double nativeCost = 0;
    for (const auto& task : tasks) for (uint32_t o = 0; o < task.problem.num_obs; ++o) {
        const auto& p = task.problem; const auto& group = p.groups.at(p.image_group.at(p.obs_image[o])); double r[2], jc[36], jp[6];
        bacpu::withModel(group.model, [&](auto m) { using M = decltype(m); bacpu::jacobian<M>(&p.poses[6 * p.obs_image[o]], &p.intr[group.intr_offset], &p.points[3 * p.obs_point[o]], &p.obs_xy[2 * o], r, jc, jp); });
        const double e = std::hypot(r[0], r[1]); nativeCost += e <= 3 ? e * e : 3 * (2 * e - 3);
    }
    check(close(nativeCost, quality.cost), "native mixed pixel-scale 2/3 BA objective differs from fullraw ledger");
    check(std::all_of(tasks[0].problem.fixed_points.begin(), tasks[0].problem.fixed_points.end(), [](uint32_t fixed) { return fixed != 0; }), "common points became block-private");
    SceneCoordinator::requireIndependentTasks(tasks, 64u << 20);
    auto overlap = std::vector<CanonicalBundle>(2); overlap[0].prepare(index, scene, {0, 1}, 0, {0}, options, 64u << 20); overlap[1].prepare(index, scene, {1, 2}, 1, {2}, options, 64u << 20);
    rejects([&] { SceneCoordinator::requireIndependentTasks(overlap, 64u << 20); }, "overlapping mutable P tasks admitted concurrently");
    std::cerr << "native solve begin\n";
    scene.beginCandidate(); tasks[0].reload(scene); tasks[0].solve(options); tasks[0].apply(scene); const auto solved = sceneQuality(index, scene, 3, 32u << 20, 2); check(solved.cost < quality.cost, "dense native solve did not improve actual global pose residual"); scene.abort("native-fixture-only");
    std::cerr << "native solve complete\n";
    uint32_t gates = 0; SceneCoordinator::Options coordinatorOptions; coordinatorOptions.bundle = options; coordinatorOptions.allowance = 64u << 20;
    SceneCoordinator coordinator(scene, index, coordinatorOptions, [&](const auto& before, const auto& after) { ++gates; const double movement = std::abs(after.images.at(1).pose.t.x - before.images.at(1).pose.t.x); return movement > .06 ? std::vector<std::string>{"source-protection-fixture"} : std::vector<std::string>{}; });
    auto correctedPose = scene.poses().images.at(1).pose; correctedPose.t.x -= .08;
    const auto accepted = coordinator.round(tasks, [&](auto&, auto& candidate, const auto&) { candidate.setPose(1, correctedPose); });
    check(accepted.accepted && accepted.alpha == .5 && gates == 2 && accepted.after.cost < accepted.before.cost, "coupled halfstep/fullraw/gate composition failed");
    check(accepted.statistics.completeEvaluations == 2 && accepted.statistics.gateRejections == 1 && !accepted.statistics.boundedRejections &&
        accepted.statistics.qualityRows == 642 && accepted.before.observations == 321 && accepted.after.observations == 321,
        "accepted half-step skipped complete raw acceptance or re-evaluated source-rejected full candidate");
    check(scene.xyz(catalog.key(0, 0)).x == scene.xyz(catalog.key(1, 0)).x, "accepted aliases have two XYZ authorities");
    const auto head = spirula::sha256_file((root / "scene" / "head.bin").string()); const auto identity = scene.identity();
    uint32_t refusedCalls = 0; SceneCoordinator refused(scene, index, coordinatorOptions, [&](const auto&, const auto&) { ++refusedCalls; return std::vector<std::string>{"always-refuse-original-gate"}; });
    const auto refusedResult = refused.round(tasks, [&](auto&, auto& candidate, const auto&) { candidate.setPose(1, correctedPose); });
    check(!refusedResult.accepted && refusedCalls == 13 && refusedResult.preserved.size() == 13 && scene.identity() == identity && spirula::sha256_file((root / "scene" / "head.bin").string()) == head, "full gate rejection changed accepted head or dropped candidates");
    check(refusedResult.statistics.completeEvaluations == 1 && refusedResult.statistics.gateRejections == 13 &&
        !refusedResult.statistics.boundedRejections && refusedResult.statistics.qualityRows == 321 && refusedResult.statistics.candidateQualitySeconds == 0 &&
        refusedResult.before.observations == 321 && refusedResult.after.observations == 321,
        "source-rejected trials scanned candidate raw or counted partial/unevaluated quality as complete");
    check(std::filesystem::exists(refusedResult.preserved.front() / "points.bin"), "complete coupled proposal was compacted away");
    for (size_t trial = 1; trial < refusedResult.preserved.size(); ++trial) {
        const auto& directory = refusedResult.preserved[trial];
        check(!std::filesystem::exists(directory / "points.bin") && std::filesystem::exists(directory / "step-reference.bin"), "rejected interpolated trial retained a full dense XYZ or lost its recipe");
        const auto snapshot = scene.verifyRejectedStep(directory); uint64_t roots = 0;
        scene.visitRejectedStepPoints(directory, [&](uint64_t rootId, Vec3 xyz) { check(rootId < snapshot.pointSlots && std::isfinite(xyz.norm()), "archived native candidate returned invalid XYZ"); ++roots; });
        check(roots == snapshot.canonicalPoints && snapshot.binding == scene.binding(), "archived candidate lost canonical roots or original raw binding");
    }
    CanonicalScene archiveReopen(root / "scene", 128u << 20);
    check(archiveReopen.identity() == identity && archiveReopen.verifyRejectedStep(refusedResult.preserved.back()).observations == 321, "reopening archived evidence altered accepted head or raw multiset");
    {
        const auto reference = refusedResult.preserved.back() / "step-reference.bin";
        std::fstream file(reference, std::ios::in | std::ios::out | std::ios::binary); char byte = 0;
        file.seekg(0); file.read(&byte, 1); const char damaged = char(byte ^ 1); file.seekp(0); file.write(&damaged, 1); file.flush();
        rejects([&] { scene.verifyRejectedStep(refusedResult.preserved.back()); }, "tampered rejected reference accepted as exact preserved coordinates");
        file.seekp(0); file.write(&byte, 1); file.flush(); file.close();
        check(scene.verifyRejectedStep(refusedResult.preserved.back()).observations == 321, "restored recipe failed native verification");
    }
    uint32_t archiveRefusalCalls = 0;
    SceneCoordinator archiveRefusal(scene, index, coordinatorOptions, [&](const auto&, const auto&) {
        if (++archiveRefusalCalls == 2) {
            auto altered = scene.poses().images.at(2).pose; altered.t.x += .001; scene.setPose(2, altered);
        }
        return std::vector<std::string>{"original-gate-refuses-changed-proposal"};
    });
    const auto archiveFailure = archiveRefusal.round(tasks, [&](auto&, auto& candidate, const auto&) { candidate.setPose(1, correctedPose); });
    check(!archiveFailure.accepted && archiveRefusalCalls == 2 && archiveFailure.preserved.size() == 2 &&
        std::filesystem::exists(archiveFailure.preserved.back() / "points.bin") &&
        std::find(archiveFailure.rejected.begin(), archiveFailure.rejected.end(), "native_step_archive_refused_full_candidate_retained") != archiveFailure.rejected.end(),
        "actual post-interpolation pose mutation bypassed archive closure or wrote repeated full fallback arrays");
    check(scene.identity() == identity && spirula::sha256_file((root / "scene" / "head.bin").string()) == head && !scene.editing(), "archive fallback changed accepted authority");
    const auto fallbackPose = model_store_detail::readModel(archiveFailure.preserved.back() / "poses.bin");
    check(close(fallbackPose.images.at(2).pose.t.x, scene.poses().images.at(2).pose.t.x + .001), "full archive fallback omitted the actual changed pose");
    scene.beginCandidate(); auto camera = scene.poses().cameras.at(0); camera.fx *= 1.04; camera.fy *= 1.04; scene.setCamera(camera);
    const auto kBefore = sceneQuality(index, scene, 3, 32u << 20, 2); options.refine_intrinsics = true;
    const auto intrinsic = refineSceneIntrinsics(index, scene, options, 3, 32u << 20); const auto kAfter = sceneQuality(index, scene, 3, 32u << 20, 2);
    check(intrinsic.observations == 321 && intrinsic.camerasUpdated >= 1 && kAfter.cost < kBefore.cost, "global K normal update did not optimize complete duplicated raw objective"); scene.abort("K-fixture-only");
    std::vector<Region> plan{{{0, 1}, {0, 1, 2}}, {{2, 3}, {0, 2, 3}}};
    std::map<uint32_t, Pose> held{{0, scene.poses().images.at(0).pose}, {2, scene.poses().images.at(2).pose}};
    FixtureGps gps; const auto absolutePriors = regionalGpsPriors(scene.poses(), gps);
    check(absolutePriors.centres.size() == 4 && absolutePriors.gps.ok, "native scheduling fixture omitted absolute ENU GPS factors");
    auto runTasks = [&](int workers, RealCfg real) {
        BundleOptions run = options; run.threads = workers; run.real = real; run.refine_intrinsics = false; run.max_iters = 3; run.priors = &absolutePriors;
        const auto ownerThread = std::this_thread::get_id(); std::set<std::string> reported;
        NativeSceneTasks planner(index, scene, plan, held, run, 2ull << 30, [&](const char* phase, uint64_t, uint64_t) { check(std::this_thread::get_id() == ownerThread, "GPU worker mutated main-thread progress callback"); reported.insert(phase); }); scene.beginCandidate(); std::vector<CanonicalBundle> unused;
        planner.execute()(unused, scene, run); triangulateScene(index, scene, 3, 32u << 20, 4); const auto q = sceneQuality(index, scene, 3, 32u << 20, 2); const auto parameters = scene.poses(); const auto stats = planner.statistics(); scene.abort("scheduler-equivalence-fixture");
        scene.beginCandidate(); planner.execute()(unused, scene, run); triangulateScene(index, scene, 3, 32u << 20, 4); const auto warmQuality = sceneQuality(index, scene, 3, 32u << 20, 2); scene.abort("scheduler-warm-fixture"); const auto warm = planner.statistics();
        check(warm.prepares == 2 && warm.cacheHits >= 2, "second native round rebuilt immutable task arrays");
        check(warm.contextCreates == stats.contextCreates, "second native round recreated device/pipeline context");
        check(reported.count("native-task-prepare") && reported.count("native-task-solve") && reported.count("native-task-apply") && warm.prepareSeconds > 0 && warm.reloadSeconds > 0 && warm.applySeconds > 0 && warm.workerSeconds > 0 && warm.solverComputeSeconds > 0 && warm.waveSeconds > 0,
              "native phases did not expose actual prepare/solve/apply timing");
        check(close(q.cost, warmQuality.cost), "warm same native queue changed complete raw geometry objective");
        return std::make_tuple(q, parameters, warm);
    };
    const auto serial = runTasks(1, RealCfg::CPU), parallel = runTasks(2, RealCfg::CPU);
    check(close(std::get<0>(serial).cost, std::get<0>(parallel).cost) && std::get<2>(parallel).completed == 4 && std::get<2>(parallel).peakWorkers == 1,
          "native CPU phase did not preserve serial global-pool scheduling");
    for (const auto& image : std::get<1>(serial).images) check((image.second.pose.t - std::get<1>(parallel).images.at(image.first).pose.t).norm() < 1e-8, "parallel ownership changed global pose geometry");
    if (argc > 2 && std::string(argv[2]) == "--gpu") {
        const auto serialGpu = runTasks(1, RealCfg::F32), parallelGpu = runTasks(2, RealCfg::F32);
        check(std::abs(std::get<0>(serialGpu).cost - std::get<0>(parallelGpu).cost) < 1e-3 && std::get<2>(parallelGpu).peakWorkers == 2 && std::get<2>(parallelGpu).peakContexts == 2 && std::get<2>(parallelGpu).gpuCompleted == 4 && std::get<2>(parallelGpu).peakDeviceCredit > 0,
              "independent two-context native GPU tasks did not match serial queue");
        std::cout << "native GPU serial/parallel cost " << std::get<0>(serialGpu).cost << "/" << std::get<0>(parallelGpu).cost << ", device credits " << std::get<2>(parallelGpu).peakDeviceCredit << '\n';
    }
    CanonicalScene restored(root / "scene", 128u << 20); check(restored.identity() == identity && close(sceneQuality(index, restored, 3, 32u << 20, 2).cost, accepted.after.cost), "accepted scene restore changed fullraw quality");
    ImmutableSceneTopology reopened(root / "index", 32u << 20); reopened.prepare(catalog, owners, "fixture-full-feature-union", 80, [](auto key) { return key.ordinal; }); check(reopened.identity() == index.identity(), "immutable warm index rebuilt identity");
    const auto interrupted = root / "partial-index"; std::filesystem::create_directories(interrupted / (index.identity() + ".part"));
    { model_store_detail::Writer out(interrupted / (index.identity() + ".part") / "old-partial.bin"); out.scalar<uint64_t>(123); out.finish(); }
    const auto partialHash = spirula::sha256_file((interrupted / (index.identity() + ".part") / "old-partial.bin").string());
    ImmutableSceneTopology recoveredIndex(interrupted, 32u << 20); recoveredIndex.prepare(catalog, owners, "fixture-full-feature-union", 80, [](auto key) { return key.ordinal; });
    check(recoveredIndex.identity() == index.identity() && spirula::sha256_file((interrupted / (index.identity() + ".part") / "old-partial.bin").string()) == partialHash, "interrupted index blocked recovery or discarded old evidence");
    for (size_t source = 0; source < 2; ++source) check(sourceHash[source] == spirula::sha256_file((root / ("source-" + std::to_string(source) + ".bin")).string()), "canonical pipeline rewrote frozen original source");
    baselineCacheTest(root / "baseline-cache");
    promotionTest(root / "private-promotion");
    siblingOwnershipTest(root / "same-owner-siblings");
    std::cout << "single-scene CSR/dense/fullraw/K/halfstep/rollback/restore tests passed\n"; return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
