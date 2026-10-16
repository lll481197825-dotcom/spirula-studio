#include "sfm/map/SceneCoordinator.h"
#include "sfm/map/SceneVisualEvidence.h"
#include "sfm/map/CanonicalAdmission.h"
#include "sfm/map/CanonicalLayout.h"
#include "sfm/map/CanonicalRecovery.h"
#include "sfm/map/RegionalBootstrap.h"
#include <iostream>
using namespace sfm;
using namespace sfm::regional;
using Path = std::filesystem::path;
static void require(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
template<class Action> static void rejects(Action action, const char* why) { bool caught = false; try { action(); } catch (const std::exception&) { caught = true; } require(caught, why); }
static std::string sha(const Path& path) { auto value = spirula::sha256_file(path.string()); require(!value.empty(), "fixture SHA failed"); return value; }
static bool zero(Vec3 value) { return value.x == 0 && value.y == 0 && value.z == 0; }
static Reconstruction fixture(unsigned metric, unsigned retained) {
    Reconstruction r; Camera c; c.id = 0; c.model = CamModel::OpenCV; c.width = 1000; c.height = 800; c.fx = c.fy = 700; c.cx = 500; c.cy = 400; c.pixel_scale = 2; r.cameras.emplace(0, c);
    for (uint32_t id = 0; id < 2; ++id) { Image im; im.id = id; im.camera_id = 0; im.name = "image-" + std::to_string(id); im.registered = true; im.pose = {mat3Identity(), {-double(id), 0, 0}}; im.points2D.resize(metric + retained); im.point3D_ids.assign(metric + retained, kInvalidPoint3D); r.images.emplace(id, im); }
    for (unsigned i = 0; i < metric + retained; ++i) {
        Point3D p; p.xyz = {.1 * i, .03 * i, 10 + .05 * i};
        if (i < metric) p.track = {{0, i}, {1, i}};
        else { p.track = {{0, i}, {0, i}}; p.xyz = {12345, -9876, -54}; }
        for (const auto& row : p.track) { auto& image = r.images.at(row.image_id); image.points2D[row.point2D_idx] = i < metric ? c.project(p.xyz + image.pose.t) : Vec2{20. + i, 30. + i}; image.point3D_ids[row.point2D_idx] = i; }
        r.points3D.emplace(i, std::move(p));
    }
    r.next_point3D_id = metric + retained; return r;
}
static Reconstruction poses(const RegionalWorkingStore& catalog) { auto result = catalog.poses(); for (auto& image : result.images) { image.second.points2D.clear(); image.second.point3D_ids.clear(); } return result; }
static void cold(const Path& root, unsigned metric, unsigned retained) {
    const auto input = fixture(metric, retained); model_store_detail::writeModel(input, root / "source.bin"); const auto immutable = sha(root / "source.bin");
    RegionalWorkingStore catalog(root / "catalog", 64u << 20); catalog.importModel(root / "source.bin", input, "input");
    ImmutableSceneAliases aliases(root / "aliases", 64u << 20); aliases.prepare(catalog, {0, 1});
    ImmutableSceneTopology index(root / "indices", 64u << 20); index.prepare(catalog, {0, 1}, aliases.identity(), aliases.points(), [&](auto key) { return aliases.canonical(key); });
    CanonicalScene scene(root / "scene", 64u << 20); const auto table = poses(catalog);
    auto classified = std::make_shared<std::vector<uint64_t>>();
    const auto producer = completeSceneSeed(index, table, 3, 16u << 20, 2, {}, {}, 0, classified);
    scene.beginLayoutCandidate(CanonicalScene::bindingFor(catalog, index.identity()), table, [&](const auto& emit) { index.emitAliases(emit); }, producer);
    if (metric) {
        scene.applyRetainedObservations({0}); rejects([&] { sceneQuality(index, scene, 3, 16u << 20, 2); }, "two-view metric root silently classified as unconstrained raw");
        scene.abort("invalid-retained-classification"); require(!scene.initialized(), "invalid initial type classification published a head");
        scene.beginLayoutCandidate(CanonicalScene::bindingFor(catalog, index.identity()), table, [&](const auto& emit) { index.emitAliases(emit); }, producer);
    }
    require(classified->size() == retained, "cold seed did not classify complete one-photo support"); scene.applyRetainedObservations(*classified);
    std::vector<PhotoQuality> photos; const auto q = sceneQuality(index, scene, 3, 16u << 20, 2, &photos);
    require(q.observations == 2 * metric + 2 * retained && q.retained_observations == 2 * retained && !q.directional_observations && !q.invalid_projections && q.cost < 1e-12 && q.finite_cost < 1e-12, "retained raw changed projection cost or allraw multiplicity");
    require(photos[0].observations == metric && photos[0].retained_observations == 2 * retained && photos[1].observations == metric, "raw retained observations inflated per-photo constraint denominator");
    const auto visual = sceneVisualEvidence(index, scene, 3, 16u << 20, 2);
    require(visual.landmarks[0] == metric && visual.landmarks[1] == metric && visual.retained_observations == retained && visual.uniqueObservations == 2 * metric + retained, "retained duplicates inflated metric landmarks or unique evidence");
    if (!metric) require(visual.components == 2 && !sceneFiniteImprovementSmall(q, q, .001), "all-retained observations fabricated connectivity or convergence");
    if (metric == 11) require(visual.landmarks[0] < 12 && visual.landmarks[1] < 12, "retained raw bypassed original twelve-landmark gate");
    for (auto id : *classified) { require(scene.landmark(id).kind == LandmarkKind::RetainedObservation && zero(scene.landmark(id).value), "retained root contains invented XYZ"); rejects([&] { scene.pointXYZ(id); }, "retained root escaped finite XYZ API"); }
    const auto beforeKinds = scene.landmarkKindsHash(); require(triangulateScene(index, scene, 3, 16u << 20, 2) == scene.canonicalPoints() && beforeKinds == scene.landmarkKindsHash(), "fullsupport stage changed retained type");
    BundleOptions options; options.refine_intrinsics = true; options.threads = 2;
    if (!metric) { const auto intr = refineSceneIntrinsics(index, scene, options, 3, 16u << 20); require(intr.observations == q.observations && intr.retained_observations == q.retained_observations && !intr.valid && !intr.camerasUpdated, "retained rows participated in intrinsic solve"); }
    writeProjectionQuality(root / "quality.bin", q); const auto read = readProjectionQuality(root / "quality.bin"); require(read.observations == q.observations && read.retained_observations == q.retained_observations, "projection ledger lost retained raw scope");
    scene.commit("retained-cold-diagnostic"); const auto accepted = scene.identity(), points = sha(scene.acceptedSnapshot().directory / "points.bin");
    require(scene.acceptedSnapshot().schema == 3, "retained snapshots are not explicit schema3");
    if (retained) {
        auto recovery = input; recovery.points3D.clear(); for (auto& item : recovery.images) item.second.point3D_ids.assign(item.second.points2D.size(), kInvalidPoint3D);
        const uint64_t id = classified->front(); Point3D borrowed; borrowed.xyz = {}; borrowed.track = {{0, uint32_t(id)}}; recovery.points3D.emplace(id, borrowed); recovery.images.at(0).point3D_ids.at(size_t(id)) = id; recovery.next_point3D_id = scene.pointSlots();
        bool typed = false; try { restoreAcceptedFeatureBindings(index, scene, recovery, 16u << 20); } catch (const CanonicalRepairBindingRejected&) { typed = true; }
        require(typed && scene.identity() == accepted, "finite recovery failed to close nonmetric borrowed-root promotion");
    }
    scene.beginCandidate(); rejects([&] { scene.applyRetainedObservations(*classified); }, "ordinary round reclassified accepted roots");
    if (!classified->empty()) { const auto id = classified->front(); rejects([&] { scene.applyLandmarkUpdates({{id, {{0, 0, 1}, LandmarkKind::RetainedObservation}}}); }, "nonzero retained payload accepted"); }
    auto proposalPose = scene.poses().images.at(0).pose; proposalPose.t.x += .01; scene.setPose(0, proposalPose); auto full = scene.sealCandidate(); full.directory = scene.abort("full-proposal");
    scene.resetCounters(); scene.beginInterpolatedCandidate(full, .5); require(!scene.counters().pointCopiedBytes && scene.counters().pointWrittenBytes == scene.pointSlots() * sizeof(Vec3), "retained interpolation did not stream a single native array"); const auto rejected = scene.preserveInterpolatedRejection(full, .5, "source-gate-rejected");
    require(!std::filesystem::exists(rejected / "points.bin"), "retained coupled step did not retain native compact evidence");
    uint64_t replayed = 0; scene.visitRejectedStepLandmarks(rejected, [&](uint64_t, CanonicalLandmark value) { if (value.kind == LandmarkKind::RetainedObservation) { require(zero(value.value), "rejected replay manufactured retained XYZ"); ++replayed; } });
    require(replayed == retained && scene.identity() == accepted && sha(scene.acceptedSnapshot().directory / "points.bin") == points, "rejected retained step changed accepted head");
    const auto fork = scene.forkAccepted(root / "fork"); { CanonicalScene reopened(fork, 64u << 20, CanonicalScene::OpenMode::ReadOnly); require(reopened.identity() == accepted && reopened.landmarkKindsHash() == beforeKinds, "typed retained fork/reopen drift"); for (auto id : *classified) require(zero(reopened.landmark(id).value), "reopened retained payload not zero"); }
    require(sha(root / "source.bin") == immutable && catalog.sourceObservationCount(0) == q.observations, "retained classification changed immutable source/raw");
    if (metric && retained) {
        auto incoming = input; incoming.points3D.clear(); for (auto& item : incoming.images) item.second.point3D_ids.assign(item.second.points2D.size(), kInvalidPoint3D);
        const uint32_t feature = metric; Point3D point; point.xyz = {0, 0, 8}; point.track = {{0, feature}, {1, feature}};
        incoming.images.at(0).point3D_ids[feature] = incoming.images.at(1).point3D_ids[feature] = 42; incoming.points3D.emplace(42, point); incoming.next_point3D_id = 43;
        model_store_detail::writeModel(incoming, root / "incoming-two-view.bin"); catalog.importModel(root / "incoming-two-view.bin", incoming, "incoming-two-view");
        ImmutableSceneAliases nextAliases(root / "aliases", 64u << 20); nextAliases.prepare(catalog, {0, 1}); ImmutableSceneTopology next(root / "indices", 64u << 20);
        next.prepare(catalog, {0, 1}, nextAliases.identity(), nextAliases.points(), [&](auto key) { return nextAliases.canonical(key); }, [&](auto key, Vec3 xyz) { return key.source == 0 ? scene.acceptedLandmark(scene.acceptedCanonical(key)).value : xyz; }, scene.identity());
        const auto nextPoses = poses(catalog); const auto seed = prepareCanonicalAdmissionSeed(index, next, scene, nextPoses, 3, 16u << 20, 2);
        scene.beginLayoutCandidate(CanonicalScene::bindingFor(catalog, next.identity()), nextPoses, [&](const auto& emit) { next.emitAliases(emit); }, seed.producer);
        scene.applyRetainedObservations(seed.evidence->newRetainedObservations); scene.applyLandmarkTypes(seed.evidence->newDirections);
        require(scene.landmark(metric).kind == LandmarkKind::RetainedObservation && zero(scene.landmark(metric).value), "append changed accepted retained kind");
        rejects([&] { sceneQuality(next, scene, 3, 16u << 20, 2); }, "new second view silently promoted or discarded an accepted retained observation");
        scene.abort("explicit-depth-promotion-required"); require(scene.identity() == accepted && scene.landmarkKindsHash() == beforeKinds && sha(root / "source.bin") == immutable, "closed promotion changed original source or accepted state");
    }
}
static void restoredResidual(const Path& root) {
    const auto original = fixture(1, 0); auto repaired = original; repaired.points3D.at(0).track = {{0, 0}}; repaired.images.at(1).point3D_ids[0] = kInvalidPoint3D;
    require(preserveRepairObservations(original, repaired, 3) == 2 && repaired.points3D.size() == 2, "preserved original raw did not expose residual grouping");
    require(raw_lineage::verify(original, repaired) == 2, "residual restoration lost immutable raw");
    model_store_detail::writeModel(original, root / "original.bin"); const auto originalSHA = sha(root / "original.bin"); model_store_detail::writeModel(repaired, root / "repaired.bin");
    RegionalWorkingStore catalog(root / "catalog", 64u << 20); catalog.importModel(root / "repaired.bin", repaired, "repaired-residual");
    ImmutableSceneAliases aliases(root / "aliases", 64u << 20); aliases.prepare(catalog, {0, 1}); ImmutableSceneTopology index(root / "indices", 64u << 20); index.prepare(catalog, {0, 1}, aliases.identity(), aliases.points(), [&](auto key) { return aliases.canonical(key); });
    CanonicalScene scene(root / "scene", 64u << 20); const auto table = poses(catalog); auto retained = std::make_shared<std::vector<uint64_t>>();
    scene.beginLayoutCandidate(CanonicalScene::bindingFor(catalog, index.identity()), table, [&](const auto& emit) { index.emitAliases(emit); }, completeSceneSeed(index, table, 3, 16u << 20, 2, {}, {}, 0, retained)); scene.applyRetainedObservations(*retained);
    const auto q = sceneQuality(index, scene, 3, 16u << 20, 2); const auto visual = sceneVisualEvidence(index, scene, 3, 16u << 20, 2);
    require(q.observations == 2 && q.retained_observations == 2 && !q.cost && !q.invalid_projections && retained->size() == 2 && !visual.landmarks[0] && !visual.landmarks[1] && visual.components == 2, "post-repair orphan residuals counted as metric or dropped raw");
    scene.commit("post-repair-singleton-diagnostic"); require(sha(root / "original.bin") == originalSHA, "original multi-view grouping evidence changed");
}
int main(int argc, char** argv) try {
    const Path root = argc > 1 ? std::filesystem::u8path(argv[1]) : std::filesystem::temp_directory_path() / ("retained-scene-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (std::filesystem::exists(root)) throw std::runtime_error("fixture destination must be new"); std::filesystem::create_directories(root);
    for (const char* name : {"eleven", "empty-metric", "restored"}) std::filesystem::create_directory(root / name);
    cold(root / "eleven", 11, 12); cold(root / "empty-metric", 0, 12); restoredResidual(root / "restored");
    std::cout << "retained raw: native cold seed, original multi-view residual restoration, metric12 exclusion, fullraw/unique coverage, intrinsic exclusion, schema3 archive/fork/reopen passed\n"; return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
