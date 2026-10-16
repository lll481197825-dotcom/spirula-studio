#include "sfm/map/CanonicalAdmission.h"
#include "sfm/map/ImmutableSceneAliases.h"
#include <iostream>
using namespace sfm;
using namespace sfm::regional;
using Path = std::filesystem::path;
static void require(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
template<class Action> static void rejects(Action action, const char* why) { bool caught = false; try { action(); } catch (const std::exception&) { caught = true; } require(caught, why); }
static std::string sha(const Path& path) { auto value = spirula::sha256_file(path.string()); require(!value.empty(), "fixture SHA failed"); return value; }
static bool exact(Vec3 a, Vec3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
static Reconstruction fixture(bool incoming) {
    Reconstruction r; Camera c; c.id = 0; c.model = CamModel::OpenCV; c.width = 1000; c.height = 800; c.fx = c.fy = 700; c.cx = 500; c.cy = 400; c.pixel_scale = 2; r.cameras.emplace(0, c);
    for (uint32_t id = 0; id < 2; ++id) { Image im; im.id = id; im.camera_id = 0; im.name = "image-" + std::to_string(id); im.registered = true;
        im.pose = {mat3Identity(), {-double(id), 0, 0}}; im.points2D.resize(5); im.point3D_ids.assign(5, kInvalidPoint3D); r.images.emplace(id, im); }
    auto add = [&](uint64_t id, Vec3 xyz, uint32_t feature, std::vector<TrackElement> track, const std::function<Vec2(uint32_t)>& xy) {
        Point3D p; p.xyz = xyz; p.track = track; r.points3D.emplace(id, p);
        for (const auto& row : track) { auto& im = r.images.at(row.image_id); im.points2D[feature] = xy(row.image_id); im.point3D_ids[feature] = id; }
    };
    if (!incoming) add(10, {0, 0, 10}, 0, {{0, 0}, {0, 0}, {1, 0}}, [&](uint32_t id) { return c.project(mul(r.images.at(id).pose.R, Vec3{0, 0, 10}) + r.images.at(id).pose.t); });
    else {
        add(20, {0, 0, -3}, 1, {{0, 1}, {0, 1}, {1, 1}}, [&](uint32_t) { return c.project(Vec3{0, 0, 1}); });
        add(21, {1, 0, 8}, 2, {{0, 2}, {1, 2}}, [&](uint32_t id) { return c.project(Vec3{1, 0, 8} + r.images.at(id).pose.t); });
        add(22, {0, 0, -3}, 3, {{0, 3}, {1, 3}}, [&](uint32_t id) { return Vec2{500 + 100 * double(id), 400}; });
        add(23, {0, 0, -3}, 4, {{0, 4}, {0, 4}}, [&](uint32_t) { return Vec2{500, 400}; });
    }
    r.next_point3D_id = incoming ? 24 : 11; return r;
}
static Reconstruction poseTable(const RegionalWorkingStore& catalog) { auto r = catalog.poses(); r.points3D.clear(); for (auto& item : r.images) { item.second.points2D.clear(); item.second.point3D_ids.clear(); } return r; }
static void geometry() {
    const auto r = fixture(true); const Vec3 d = Vec3{.1, 0, 1}.normalized();
    CanonicalLandmark value{d, LandmarkKind::UnitDirection}; bool valid = false;
    const auto pixel = r.cameras.at(0).project(d); auto pose = r.images.at(0).pose;
    const auto before = reprojLandmarkError(r.cameras.at(0), pose, pixel, value, &valid); require(valid && before < 1e-9, "native direction residual mismatch");
    pose.t = {100000, -90000, 80000}; require(reprojLandmarkError(r.cameras.at(0), pose, pixel, value, &valid) == before && valid, "direction incorrectly depended on translation");
    require(reprojLandmarkError(r.cameras.at(0), pose, pixel, {d * -1, LandmarkKind::UnitDirection}, &valid) == 1e30 && !valid, "antipodal pinhole ray accepted");
    Pose rotated{so3Exp(Vec3{0, M_PI, 0}), {}};
    const CanonicalLandmark negativeWorldZ{{0, 0, -1}, LandmarkKind::UnitDirection};
    require(reprojLandmarkError(r.cameras.at(0), rotated, {500, 400}, negativeWorldZ, &valid) < 1e-8 && valid, "world Z used as direction cheirality");
    std::vector<Observation> support{{0, 0, pixel}, {0, 0, pixel}, {0, uint64_t(1) << 32, pixel}};
    const auto fit = fitUnitDirection(support, r, 3, 1);
    require(fit.accepted && fit.observations == 3 && fit.inliers == 3 && fit.distinctPhotos == 2 && fit.maxWorldRayAngleDegrees < 1e-4, "direction fit lost duplicate native support");
    auto single = support; single.back().point = 0; require(!fitUnitDirection(single, r, 3, 1).accepted, "duplicate rows fabricated two supporting photos");
    auto parallax = support; parallax.back().xy.x += 100; require(!fitUnitDirection(parallax, r, 3, 1).accepted, "observable finite parallax converted to direction");
    Camera spherical = Camera::defaultFor(0, 1000, 500, 1., CamModel::Equirect); spherical.pixel_scale = 2;
    const Vec3 seamDirection = spherical.bearing({.5, 250});
    const double seam = reprojLandmarkError(spherical, Pose{mat3Identity(), {}}, {999.5, 250}, {seamDirection, LandmarkKind::UnitDirection}, &valid);
    require(valid && std::abs(seam / spherical.pixel_scale - .5) < 1e-8, "typed spherical seam or extraction scale changed");
    const auto mid = interpolateLandmark({{1, 0, 0}, LandmarkKind::UnitDirection}, {{0, 1, 0}, LandmarkKind::UnitDirection}, .5);
    require(std::abs(mid.value.norm() - 1) < 1e-12 && std::abs(mid.value.x - std::sqrt(.5)) < 1e-12, "direction interpolation not on S2");
    rejects([&] { interpolateLandmark(value, {d * -1, value.kind}, .5); }, "antipodal S2 path silently invented");
    rejects([&] { interpolateLandmark(value, {d, LandmarkKind::FiniteXYZ}, .5); }, "ordinary interpolation changed kind");
    const CanonicalLandmark retained{{}, LandmarkKind::RetainedObservation}; validateLandmark(retained);
    require(exact(interpolateLandmark(retained, retained, .5).value, {}), "retained interpolation invented geometry");
    rejects([&] { validateLandmark({{0, 0, 1}, LandmarkKind::RetainedObservation}); }, "retained payload accepted fake XYZ");
    rejects([&] { reprojLandmarkError(r.cameras.at(0), pose, pixel, retained); }, "retained observation exposed a visual residual");
    require(isRetainedObservationTrack(single) && !isRetainedObservationTrack(support), "distinct-photo classification confused duplicate raw with support");
    require(refineLandmark(retained, single, r, 3).kind == LandmarkKind::RetainedObservation, "retained fullsupport refinement created XYZ");
    const auto refined = refineLandmark({Vec3{.101, .002, 1}.normalized(), value.kind}, support, r, 3);
    require(refined.kind == value.kind && landmark_geometry_detail::objective(refined, support, r, 3).finite < 1e-9, "full-support S2 refinement did not improve true objective");
}
static void legacyFinite(const Path& root, CanonicalScene& source) {
    const auto saved = source.acceptedSnapshot(); const auto directory = root / "legacy" / saved.directory.filename(); std::filesystem::create_directories(directory);
    for (const char* file : {"aliases.bin", "points.bin", "poses.bin"}) std::filesystem::copy_file(saved.directory / file, directory / file);
    model_store_detail::Writer out(directory / "snapshot.bin"); out.scalar<uint64_t>(0x314e4353454e4143ull); out.scalar<uint32_t>(1);
    out.scalar(saved.generation); out.scalar(saved.pointSlots); out.scalar(saved.canonicalPoints); out.scalar(saved.observations); out.string(saved.phase);
    out.string(saved.binding.rawCatalog); out.string(saved.binding.topology); out.scalar<uint64_t>(saved.binding.sources.size());
    for (const auto& s : saved.binding.sources) { out.string(s.label); out.string(s.digest); out.scalar(s.points); out.scalar(s.observations); }
    for (const char* file : {"aliases.bin", "points.bin", "poses.bin"}) { out.string(sha(directory / file)); out.scalar<uint64_t>(std::filesystem::file_size(directory / file)); } out.finish();
    model_store_detail::Writer head(root / "legacy" / "head.bin"); head.scalar<uint64_t>(0x314448534e414343ull); head.string(directory.filename().u8string()); head.string(sha(directory / "snapshot.bin")); head.finish();
    const auto v2 = root / "legacy-v2" / saved.directory.filename(); std::filesystem::create_directories(v2);
    for (const char* file : {"aliases.bin", "points.bin", "poses.bin", "kinds.bin", "snapshot.bin"}) std::filesystem::copy_file(saved.directory / file, v2 / file);
    { std::fstream file(v2 / "snapshot.bin", std::ios::binary | std::ios::in | std::ios::out); const uint32_t schema = 2; file.seekp(8); file.write(reinterpret_cast<const char*>(&schema), sizeof(schema)); }
    { model_store_detail::Writer head2(root / "legacy-v2" / "head.bin"); head2.scalar<uint64_t>(0x314448534e414343ull); head2.string(v2.filename().u8string()); head2.string(sha(v2 / "snapshot.bin")); head2.finish(); }
    const auto v2head = sha(root / "legacy-v2" / "head.bin");
    { CanonicalScene read(root / "legacy-v2", 64u << 20, CanonicalScene::OpenMode::ReadOnly); require(read.acceptedSnapshot().schema == 2 && exact(read.pointXYZ(0), source.pointXYZ(0)), "legacy schema2 finite restore changed geometry"); }
    require(sha(root / "legacy-v2" / "head.bin") == v2head, "legacy schema2 read mutated accepted head");
    const auto original = sha(root / "legacy" / "head.bin"), points = sha(directory / "points.bin");
    { CanonicalScene read(root / "legacy", 64u << 20, CanonicalScene::OpenMode::ReadOnly); require(read.acceptedSnapshot().schema == 1 && exact(read.pointXYZ(0), source.pointXYZ(0)), "legacy finite restore mismatch"); require(!read.landmarkKindsHash().empty(), "legacy finite types unbound"); }
    require(sha(root / "legacy" / "head.bin") == original && !std::filesystem::exists(directory / "kinds.bin"), "legacy read mutated source");
    { CanonicalScene migrate(root / "legacy", 64u << 20); migrate.beginCandidate(); migrate.commit("schema3-finite-only"); require(migrate.acceptedSnapshot().schema == 3 && sha(directory / "points.bin") == points && sha(migrate.acceptedSnapshot().directory / "points.bin") == points, "legacy candidate migration changed geometry"); }
}
int main(int argc, char** argv) try {
    geometry(); const Path root = argc > 1 ? std::filesystem::u8path(argv[1]) : std::filesystem::temp_directory_path() / ("canonical-landmark-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (std::filesystem::exists(root)) throw std::runtime_error("fixture destination must be new"); std::filesystem::create_directories(root);
    RegionalWorkingStore catalog(root / "catalog", 64u << 20); const auto original = fixture(false); model_store_detail::writeModel(original, root / "original.bin"); catalog.importModel(root / "original.bin", original, "original");
    const std::vector<uint32_t> owners{0, 1}; ImmutableSceneAliases oldAliases(root / "aliases", 64u << 20); oldAliases.prepare(catalog, owners);
    ImmutableSceneTopology oldIndex(root / "indices", 64u << 20); oldIndex.prepare(catalog, owners, oldAliases.identity(), oldAliases.points(), [&](auto key) { return oldAliases.canonical(key); });
    CanonicalScene scene(root / "scene", 64u << 20); const auto poses = poseTable(catalog);
    scene.initialize(CanonicalScene::bindingFor(catalog, oldIndex.identity()), poses, [&](const auto& emit) { oldIndex.emitAliases(emit); }, [&](const auto& emit) { oldIndex.visitSeeds([&](const auto& seed) { emit(seed.root, seed.xyz); }); });
    legacyFinite(root, scene); const auto oldXYZ = scene.pointXYZ(0); const auto oldID = scene.identity();
    const auto incoming = fixture(true); model_store_detail::writeModel(incoming, root / "incoming.bin"); catalog.importModel(root / "incoming.bin", incoming, "incoming"); const auto immutableSHA = sha(root / "incoming.bin");
    ImmutableSceneAliases aliases(root / "aliases", 64u << 20); aliases.prepare(catalog, owners); ImmutableSceneTopology index(root / "indices", 64u << 20);
    index.prepare(catalog, owners, aliases.identity(), aliases.points(), [&](auto key) { return aliases.canonical(key); }, [&](auto key, Vec3 xyz) { return key.source == 0 ? scene.acceptedLandmark(scene.acceptedCanonical(key)).value : xyz; }, scene.identity());
    const auto nextPoses = poseTable(catalog); auto seed = prepareCanonicalAdmissionSeed(oldIndex, index, scene, nextPoses, 3, 32u << 20, 2, {}, {}, 1);
    scene.beginLayoutCandidate(CanonicalScene::bindingFor(catalog, index.identity()), nextPoses, [&](const auto& emit) { index.emitAliases(emit); }, seed.producer);
    require(seed.evidence->newDirections.size() == 1 && seed.evidence->newDirections[0].first == 1, "admission did not classify only the invalid angular new root");
    require(seed.evidence->oldPreservedRoots == 1 && seed.evidence->oldBefore.observations == 3 && seed.evidence->newBefore.observations == 9 && exact(scene.pointXYZ(0), oldXYZ), "admission changed old root or raw multiplicity");
    require(seed.evidence->newRetainedObservations == std::vector<uint64_t>{4}, "single-photo duplicate raw was not retained separately");
    scene.applyLandmarkTypes(seed.evidence->newDirections); scene.applyRetainedObservations(seed.evidence->newRetainedObservations); require(scene.landmark(1).kind == LandmarkKind::UnitDirection && scene.landmark(2).kind == LandmarkKind::FiniteXYZ && scene.landmark(3).kind == LandmarkKind::FiniteXYZ && scene.landmark(4).kind == LandmarkKind::RetainedObservation && exact(scene.landmark(4).value, {}), "admission type contract changed finite controls");
    rejects([&] { scene.applyLandmarkTypes({{0, {0, 0, 1}}}); }, "admission changed accepted finite kind");
    rejects([&] { scene.pointXYZ(4); }, "retained raw escaped finite-only getter");
    rejects([&] { scene.applyRetainedObservations({0}); }, "layout changed an accepted finite root to retained");
    rejects([&] { scene.pointXYZ(1); }, "direction escaped finite-only getter"); scene.commit("typed-admission-diagnostic");
    const auto quality = sceneQuality(index, scene, 3, 16u << 20, 2); require(quality.observations == 12 && quality.retained_observations == 2 && quality.directional_observations == 3, "typed fullraw accounting did not preserve duplicate orphan observations");
    const auto typedID = scene.identity(), typedHash = scene.landmarkKindsHash(); const auto typedValue = scene.landmark(1).value;
    uint64_t sourceAliases = 0; CanonicalScene::PointKey lastAlias{};
    scene.visitSourceAliases([&](auto key, uint64_t rootID) {
        const auto expected = catalog.key(size_t(key.source), key.ordinal); require(key == expected && rootID == sourceAliases++, "bulk source aliases changed stored local IDs, order or roots"); lastAlias = key;
    });
    require(sourceAliases == 5 && lastAlias.source == 1 && lastAlias.ordinal == 3 && lastAlias.localPoint == 23, "bulk alias stream omitted the final source/ordinal/local row");
    CanonicalScene invalidAliases(root / "invalid-bulk-aliases", 64u << 20);
    invalidAliases.beginLayoutCandidate(scene.binding(), scene.poses(), [&](const auto& emit) {
        uint64_t global = 0; for (size_t source = 0; source < catalog.sourceCount(); ++source) for (uint64_t ordinal = 0; ordinal < catalog.sourcePointCount(source); ++ordinal, ++global)
            emit(catalog.key(source, ordinal), global < 3 ? global : global == 3 ? 1 : 3);
    }, [&](const auto& emit) { scene.visitCanonicalLandmarks([&](uint64_t id, CanonicalLandmark value) { if (id < 3) emit(id, value.value); }); });
    rejects([&] { invalidAliases.visitSourceAliases([&](auto, uint64_t) {}); }, "bulk alias stream accepted a final-row alias to another alias");
    invalidAliases.abort("invalid-bulk-alias-retained"); require(!invalidAliases.initialized() && scene.identity() == typedID, "bad alias stream changed accepted authority");
    scene.beginCandidate(); rejects([&] { scene.applyLandmarkTypes({{1, {0, 0, 1}}}); }, "round changed landmark type");
    const auto beforeRejectedBatch = scene.pointXYZ(0);
    rejects([&] { scene.applyLandmarkUpdates({{0, {{7, 8, 9}, LandmarkKind::FiniteXYZ}}, {1, {{0, 0, 2}, LandmarkKind::UnitDirection}}}); }, "nonunit typed batch partially applied");
    require(exact(scene.pointXYZ(0), beforeRejectedBatch) && exact(scene.landmark(1).value, typedValue), "failed typed validation changed earlier root");
    rejects([&] { scene.applyPointUpdates({{1, {9, 0, 9}}}); }, "finite updater changed a direction");
    rejects([&] { scene.applyLandmarkUpdates({{1, {{0, 0, 1}, LandmarkKind::FiniteXYZ}}}); }, "typed updater changed immutable kind");
    const auto proposed = Vec3{.102, .001, 1}.normalized(); scene.applyLandmarkUpdates({{1, {proposed, LandmarkKind::UnitDirection}}}); auto full = scene.sealCandidate(); full.directory = scene.abort("full-proposal");
    scene.resetCounters(); scene.beginCandidate(); scene.interpolateCandidate(full, .5); const auto copied = scene.sealCandidate("copied-step-oracle");
    require(scene.counters().pointCopiedBytes == scene.pointSlots() * sizeof(Vec3), "copied interpolation control did not copy one complete native array"); scene.abort("copied-step-oracle-preserved");
    scene.resetCounters(); scene.beginInterpolatedCandidate(full, .5); const auto midpoint = scene.landmark(1).value;
    require(scene.counters().pointCopiedBytes == 0 && scene.counters().pointWrittenBytes == scene.pointSlots() * sizeof(Vec3), "direct interpolation copied baseline or wrote more than one native point array");
    require(scene.sealCandidate("direct-step-oracle").pointHash == copied.pointHash, "copy-free interpolation changed the complete coupled point-array SHA");
    const auto exactMid = interpolateLandmark({typedValue, LandmarkKind::UnitDirection}, {proposed, LandmarkKind::UnitDirection}, .5).value;
    require(exact(midpoint, exactMid) && scene.landmarkKindsHash() == typedHash, "coupled half step changed types or S2 geometry");
    const auto rejected = scene.preserveInterpolatedRejection(full, .5, "source-gate-rejected"); require(!std::filesystem::exists(rejected / "points.bin"), "typed rejection failed exact compact proof");
    uint64_t replayed = 0; scene.visitRejectedStepLandmarks(rejected, [&](uint64_t id, CanonicalLandmark value) { if (id == 4) require(value.kind == LandmarkKind::RetainedObservation && exact(value.value, {}), "archive reconstructed fake retained geometry"); if (id == 1) require(value.kind == LandmarkKind::UnitDirection && exact(value.value, midpoint), "archive replay changed typed midpoint"); ++replayed; }); require(replayed == 5 && scene.identity() == typedID, "rejection advanced accepted snapshot");
    auto wrongProposal = full; wrongProposal.identity.front() = wrongProposal.identity.front() == 'a' ? 'b' : 'a';
    const auto protectedHead = sha(root / "scene" / "head.bin"), protectedPoints = sha(scene.acceptedSnapshot().directory / "points.bin");
    rejects([&] { scene.beginInterpolatedCandidate(wrongProposal, .5); }, "copy-free interpolation skipped sealed-proposal SHA validation");
    require(!scene.editing() && scene.identity() == typedID && sha(root / "scene" / "head.bin") == protectedHead && sha(scene.acceptedSnapshot().directory / "points.bin") == protectedPoints,
        "failed copy-free interpolation left a live transaction or changed accepted parameters");
    // Break the immutable retained link before deliberately corrupting only a candidate.
    scene.beginCandidate(); const auto candidateKind = scene.sealCandidate().directory / "kinds.bin";
    const auto privateKind = candidateKind.parent_path() / "private-kinds-copy.bin";
    std::filesystem::copy_file(candidateKind, privateKind); std::filesystem::remove(candidateKind); std::filesystem::rename(privateKind, candidateKind);
    { std::fstream file(candidateKind, std::ios::binary | std::ios::in | std::ios::out); const char zero = 0; file.seekp(1); file.write(&zero, 1); }
    rejects([&] { scene.sealCandidate(); }, "ordinary round silently rebound modified landmark types"); scene.abort("typed-kind-integrity-rejected");
    require(scene.identity() == typedID && sha(scene.acceptedSnapshot().directory / "kinds.bin") == typedHash, "kind tamper changed accepted immutable head");
    const auto fork = scene.forkAccepted(root / "fork"); { CanonicalScene reopened(fork, 64u << 20, CanonicalScene::OpenMode::ReadOnly); require(reopened.identity() == typedID && reopened.landmarkKindsHash() == typedHash && exact(reopened.landmark(1).value, typedValue), "typed fork/reopen changed authority"); }
    const auto tamper = root / "tamper"; std::filesystem::copy(root / "scene", tamper, std::filesystem::copy_options::recursive);
    { std::fstream file(tamper / scene.acceptedSnapshot().directory.filename() / "kinds.bin", std::ios::binary | std::ios::in | std::ios::out); const char zero = 0; file.seekp(1); file.write(&zero, 1); }
    rejects([&] { CanonicalScene changed(tamper, 64u << 20, CanonicalScene::OpenMode::ReadOnly); }, "kind SHA tamper accepted");
    const auto changedAccepted = root / "changed-accepted"; std::filesystem::copy(root / "scene", changedAccepted, std::filesystem::copy_options::recursive);
    { CanonicalScene changed(changedAccepted, 64u << 20); const auto changedPath = changed.acceptedSnapshot().directory / "kinds.bin";
      { std::fstream file(changedPath, std::ios::binary | std::ios::in | std::ios::out); const char zero = 0; file.seekp(1); file.write(&zero, 1); }
      const auto validBinding = changed.binding(); const auto validPoses = changed.poses(); bool correctFailure = false;
      try { changed.beginLayoutCandidate(validBinding, validPoses, [&](const auto& emit) {
          uint64_t global = 0; for (size_t source = 0; source < catalog.sourceCount(); ++source) for (uint64_t ordinal = 0; ordinal < catalog.sourcePointCount(source); ++ordinal) { emit(catalog.key(source, ordinal), global); ++global; }
      }, [&](const auto& emit) { scene.visitCanonicalLandmarks([&](uint64_t id, CanonicalLandmark value) { emit(id, value.value); }); }); }
      catch (const std::exception& error) { correctFailure = std::string(error.what()).find("accepted canonical landmark kinds changed") != std::string::npos; }
      require(correctFailure, "changed accepted kind SHA copied into a new layout or failed for unrelated fixture error");
      if (changed.editing()) changed.abort("accepted-type-source-corrupt"); }
    require(sha(root / "incoming.bin") == immutableSHA && scene.observations() == 12 && scene.identity() != oldID && exact(scene.pointXYZ(0), oldXYZ), "typed path changed immutable source/raw/accepted finite root");
    std::cout << "typed landmarks: fullraw angular admission, finite controls, S2, legacy migration, kind SHA, coupled rollback/archive/fork passed; diagnostic fixture is not full capture acceptance\n"; return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
