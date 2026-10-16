#include "sfm/map/CanonicalScene.h"
#include "sfm/map/ImmutableSceneAliases.h"
#include "sfm/map/CanonicalRecovery.h"
#include "sfm/map/CanonicalInitialization.h"
#include "sfm/map/CanonicalAdmission.h"
#include "sfm/map/CanonicalImageIdentity.h"
#include <iostream>
#include <cwctype>

using namespace sfm;
using Scene = regional::CanonicalScene;
using Catalog = regional::RegionalWorkingStore;
using Path = std::filesystem::path;
static void check(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
template<class Action> static void rejects(Action action, const char* why) {
    bool failed = false; try { action(); } catch (const std::exception&) { failed = true; } check(failed, why);
}
static std::string sha(const Path& path) { const auto hash = spirula::sha256_file(path.string()); check(!hash.empty(), "hash failed"); return hash; }
static bool exact(Vec3 a, Vec3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
static Reconstruction model(bool second) {
    Reconstruction rec; Camera camera; camera.id = 4; camera.width = 1000; camera.height = 800;
    camera.fx = camera.fy = 700; camera.cx = 500; camera.cy = 400; rec.cameras.emplace(camera.id, camera);
    for (uint32_t id = 0; id < 2; ++id) {
        Image image; image.id = id; image.camera_id = 4; image.name = "photo-" + std::to_string(id); image.registered = true;
        image.pose = {mat3Identity(), {double(id), 0, 0}}; image.points2D = {{10, 20}, {30, 40}}; image.point3D_ids.assign(2, kInvalidPoint3D);
        rec.images.emplace(id, image);
    }
    Point3D point; point.xyz = {second ? 101. : 1., 2, 8}; point.rgb[0] = 13; point.rgb[1] = 29; point.rgb[2] = 47; point.error = 0.75;
    point.track = {{0, 0}, {1, 0}}; if (!second) point.track.insert(point.track.begin(), {0, 0});
    const uint64_t local = second ? 20 : 10; rec.points3D.emplace(local, point);
    for (auto& image : rec.images) image.second.point3D_ids[0] = local;
    if (second) { point.xyz = {5, 6, 12}; point.track = {{0, 1}, {1, 1}}; rec.points3D.emplace(22, point); for (auto& image : rec.images) image.second.point3D_ids[1] = 22; }
    rec.next_point3D_id = second ? 23 : 11; return rec;
}
static Reconstruction poses(const Catalog& catalog) { auto result = catalog.poses(); for (auto& image : result.images) { image.second.points2D.clear(); image.second.point3D_ids.clear(); } result.points3D.clear(); return result; }
static void truncate(const Path& source, const Path& target) {
    std::filesystem::copy_file(source, target, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::resize_file(target, std::filesystem::file_size(target) - 1);
}
static void readOnlyAcceptedTest(const Path& root, const Scene& accepted) {
    const auto copy = root / "read-only-source";
    std::filesystem::copy(accepted.acceptedSnapshot().directory.parent_path(), copy, std::filesystem::copy_options::recursive);
    const auto partial = copy / "candidate-uncommitted-readonly";
    std::filesystem::create_directory(partial); { std::ofstream out(partial / "partial.bin"); out << "retained uncommitted evidence"; }
    auto tree = [](const Path& path, bool locked = false) {
        std::map<std::string, std::string> result;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
            const auto key = entry.path().lexically_relative(path).generic_u8string();
            result.emplace(key, entry.is_directory() ? "directory" : locked && entry.path().filename() == "transaction.lock" ?
                "exclusive-lock-size=" + std::to_string(entry.file_size()) : sha(entry.path()));
        }
        return result;
    };
    const auto before = tree(copy);
    {
        Scene read(copy, 64u << 20, Scene::OpenMode::ReadOnly);
        check(read.identity() == accepted.identity() && exact(read.pointXYZ(0), accepted.pointXYZ(0)), "read-only accepted restore changed parameters");
        check(read.inspectRetainedSnapshot(read.acceptedSnapshot().directory).identity == accepted.identity(), "read-only strong snapshot inspection failed");
        read.forkAccepted(root / "read-only-fork");
        rejects([&] { read.beginCandidate(); }, "read-only candidate transaction started");
        rejects([&] { read.beginLayoutCandidate({}, {}, {}, {}); }, "read-only layout transaction started");
        rejects([&] { read.setPose(0, read.poses().images.at(0).pose); }, "read-only pose writable");
        rejects([&] { read.setCamera(read.poses().cameras.begin()->second); }, "read-only calibration writable");
        rejects([&] { read.applyPointUpdates({{0, {1, 2, 3}}}); }, "read-only XYZ writable");
        rejects([&] { read.rewriteCanonicalPoints({}); }, "read-only XYZ producer writable");
        rejects([&] { read.sealCandidate(); }, "read-only proposal sealed");
        rejects([&] { read.commit("invalid"); }, "read-only head committed");
        rejects([&] { read.abort(); }, "read-only candidate retained by mutation");
        rejects([&] { read.interpolateCandidate({}, 0.5); }, "read-only interpolation started");
        rejects([&] { read.preserveInterpolatedRejection({}, 0.5, "invalid"); }, "read-only trial archive started");
        rejects([&] { Scene writer(copy, 64u << 20); }, "read-only lease allowed a concurrent writer");
    }
    check(tree(copy) == before, "read-only open/fork/mutation refusal changed original paths or SHA");
    {
        Scene writer(copy, 64u << 20); writer.beginCandidate();
        const auto locked = tree(copy, true);
        rejects([&] { Scene read(copy, 64u << 20, Scene::OpenMode::ReadOnly); }, "read-only open ignored an active writer lock");
        check(tree(copy, true) == locked, "read-only refusal changed writer lock or pending evidence");
        writer.abort("fixture-writer-rejection");
    }
    const auto noLock = root / "read-only-no-lock";
    std::filesystem::copy(copy, noLock, std::filesystem::copy_options::recursive);
    std::filesystem::remove(noLock / "transaction.lock"); const auto noLockBefore = tree(noLock);
    { Scene read(noLock, 64u << 20, Scene::OpenMode::ReadOnly); }
    check(tree(noLock) == noLockBefore && !std::filesystem::exists(noLock / "transaction.lock"), "read-only open created a missing lock");
    const auto absent = root / "read-only-absent";
    rejects([&] { Scene read(absent, 64u << 20, Scene::OpenMode::ReadOnly); }, "read-only open accepted missing root");
    check(!std::filesystem::exists(absent), "read-only open created a missing source root");
}
static void admissionRecoveryTest(const Path& root, Catalog& catalog) {
    struct InputName { std::string name, export_name; };
    const std::map<uint32_t, InputName> explicitNames{{0, {"nested/photo-0", "nested/photo-0.JPG"}}, {1, {"photo-1", "photo-1.jpeg"}}};
    auto named = poses(catalog); named.images.at(0).name = "nested/photo-0.JPG";
    const auto originalNames = named; regional::canonicalizeImageIdentity(named, explicitNames);
    check(named.images.at(0).name == "nested/photo-0" && originalNames.images.at(0).name == "nested/photo-0.JPG", "canonical name mapping mutated immutable source or guessed extension");
    auto foreign = named; foreign.images.at(1).name = "photo-1.JPG";
    rejects([&] { regional::canonicalizeImageIdentity(foreign, explicitNames); }, "foreign extension silently mapped to another input");
    foreign = named; foreign.images.at(1).id = 0;
    rejects([&] { regional::canonicalizeImageIdentity(foreign, explicitNames); }, "wrong native source image ID mapped by filename");
    regional::ImmutableSceneAliases aliases(root / "admission-aliases", 64u << 20); aliases.prepare(catalog, {0, 1, 2});
    regional::ImmutableSceneTopology oldIndex(root / "admission-index", 64u << 20); catalog.resetCounters();
    oldIndex.prepare(catalog, {0, 1, 2}, aliases.identity(), aliases.points(), [&](auto key) { return aliases.canonical(key); });
    check(catalog.counters().parameter_bytes == 0, "immutable initialization read retired source XYZ");
    const auto binding = Scene::bindingFor(catalog, oldIndex.identity());
    auto poseTable = poses(catalog); poseTable.cameras.at(4).pixel_scale = 2;
    for (auto& image : poseTable.images) image.second.name += ".JPG";
    const Scene::AliasProducer producer = [&](const auto& emit) { aliases.emitAliases(emit); };
    const Scene::PointProducer seed = [&](const auto& emit) { oldIndex.visitSeeds([&](const auto& point) { emit(point.root, point.xyz); }); };
    Scene rejectedInitial(root / "rejected-initial", 64u << 20);
    rejects([&] { rejectedInitial.initializeValidated(binding, poseTable, producer, seed, [&](const Scene& proposal) {
        check(!proposal.initialized() && proposal.editing(), "initial validator saw an accepted head");
        check(regional::sceneQuality(oldIndex, proposal, 3, 16u << 20, 2).observations == 7, "initial validator missed duplicate raw");
        throw std::runtime_error("fixture initial source gate rejection");
    }); }, "initial gate rejection became accepted");
    check(!rejectedInitial.initialized() && !rejectedInitial.editing() && !std::filesystem::exists(root / "rejected-initial" / "head.bin"), "initial failure left a head or active transaction");
    bool retained = false; for (const auto& entry : std::filesystem::directory_iterator(root / "rejected-initial")) retained |= entry.path().filename().string().find("rejected-") == 0;
    check(retained, "failed initial candidate discarded");
    Scene scene(root / "admission-scene", 64u << 20);
    scene.initializeValidated(binding, poseTable, producer, seed, [&](const Scene& candidate) {
        check(regional::sceneQuality(oldIndex, candidate, 3, 16u << 20, 2).observations == 7, "initial full multiset gate incomplete");
    });
    const auto featureDir = root / "recovery-features"; std::filesystem::create_directories(featureDir);
    for (const auto& image : poseTable.images) {
        FeatureSet features; features.width = 1000; features.height = 800; features.extract_width = 500; features.extract_height = 400;
        features.dim = 1; features.keypoints = {{10, 20}, {30, 40}}; features.descriptors = {1, 2};
        writeFeatures((featureDir / ("photo-" + std::to_string(image.first) + ".bin")).string(), features);
    }
    const auto featurePath = [&](uint32_t id) { return featureDir / ("photo-" + std::to_string(id) + ".bin"); };
    const auto recovery = regional::gatherCanonicalSeed(oldIndex, scene, {0, 1}, featurePath, 32u << 20);
    check(recovery.points3D.size() == 2 && recovery.points3D.at(0).track.size() == 2 && recovery.points3D.at(2).track.size() == 2 &&
        recovery.images.at(0).points2D.at(0).x == 10 && recovery.cameras.at(4).pixel_scale == 2 && exact(recovery.points3D.at(0).xyz, scene.pointXYZ(0)),
        "canonical PnP seed lost original units, distinct feature support or unique authority");
    rejects([&] { regional::gatherCanonicalSeed(oldIndex, scene, {0}, featurePath, 1u << 20); }, "unbounded canonical recovery budget accepted");
    auto changed = readFeatures((featureDir / "photo-0.bin").string(), false); changed.keypoints[0].x += 1;
    writeFeatures((featureDir / "photo-0.bin").string(), changed);
    rejects([&] { regional::gatherCanonicalSeed(oldIndex, scene, {0}, featurePath, 32u << 20); }, "conflicting original feature XY accepted");
    const auto initialDir = root / "initial-stage"; regional::CanonicalInitialStage stage; stage.poses = poseTable;
    stage.sources.push_back({"source0", root / "source-0.bin", regional::canonical_initial_detail::aligned(root / "staged-aligned", model(false)), modelResidentBytes(model(false)), Sim3{}});
    stage.deferred.push_back({"source1", root / "source-1.bin", modelResidentBytes(model(true)), 1}); stage.fixedImages = {0}; stage.targets = {1};
    regional::saveCanonicalInitialStage(initialDir, "run-original-signature", stage);
    const auto restored = regional::loadCanonicalInitialStage(initialDir, "run-original-signature", {{"source1", root / "source-1.bin"}, {"source0", root / "source-0.bin"}}, 64u << 20);
    check(restored && restored->sources.size() == 1 && restored->deferred.size() == 1 && restored->fixedImages == stage.fixedImages && restored->targets == stage.targets,
        "selected/deferred initial stage lost scope or held/recovery identities");
    const auto heldOnly = regional::loadCanonicalInitialStage(initialDir, "run-original-signature", {}, 64u << 20, true);
    check(heldOnly && exact(heldOnly->poses.images.at(0).pose.t, stage.poses.images.at(0).pose.t), "held source restored from current rather than original stage pose");
    rejects([&] { regional::loadCanonicalInitialStage(initialDir, "wrong-signature", {}, 64u << 20, true); }, "initial stage wrong signature accepted");
    check(regional::canonical_initial_detail::aligned(root / "staged-aligned", model(false)) == stage.sources[0].aligned, "complete aligned input not reused");
    const auto badStage = root / "truncated-initial-stage"; std::filesystem::copy(initialDir, badStage, std::filesystem::copy_options::recursive);
    const auto poseFile = badStage / (regional::canonical_initial_detail::nativeDigest(stage.poses) + ".bin"); std::filesystem::resize_file(poseFile, std::filesystem::file_size(poseFile) - 1);
    rejects([&] { regional::loadCanonicalInitialStage(badStage, "run-original-signature", {}, 64u << 20, true); }, "changed original held P/K accepted");
    const auto before = regional::scenePrefixQuality(oldIndex, scene, 3, 16u << 20, 2, true);
    const auto fullBefore = regional::sceneQuality(oldIndex, scene, 3, 16u << 20, 2);
    check(before.observations == 7 && before.cost == fullBefore.cost && before.finite_cost == fullBefore.finite_cost, "prefix validator objective differs on identical raw cohort");
    auto third = model(false); auto point = third.points3D.at(10); point.xyz = {81, 2, 8}; point.track = {{0, 0}, {2, 0}, {2, 0}};
    third.points3D.clear(); third.points3D.emplace(80, point); third.images.erase(1);
    auto image2 = third.images.at(0); image2.id = 2; image2.name = "photo-2"; image2.pose.t.x = 2;
    third.images.emplace(2, image2); for (auto& image : third.images) image.second.point3D_ids[0] = 80; third.next_point3D_id = 81;
    const auto thirdPath = root / "source-2.bin"; model_store_detail::writeModel(third, thirdPath); catalog.importModel(thirdPath, third, "source2");
    scene.requireBinding(Scene::bindingFor(catalog, oldIndex.identity(), true, 2));
    catalog.resetCounters();
    check(scene.materialize(catalog, 0).points3D.at(10).track.size() == 3, "pending immutable suffix tore accepted source prefix");
    check(catalog.counters().parameter_bytes == 0, "canonical materialize read retired XYZ authority");
    regional::CanonicalInitialStage admission; admission.poses = poses(catalog); admission.targets = {2};
    admission.sources.push_back({"source2", thirdPath, regional::canonical_initial_detail::aligned(root / "admission-aligned", third), modelResidentBytes(third), Sim3{}});
    regional::saveCanonicalInitialStage(root / "pending-admission", scene.identity(), admission);
    const auto resumedAdmission = regional::loadCanonicalInitialStage(root / "pending-admission", scene.identity(), {{"source2", thirdPath}}, 64u << 20);
    check(resumedAdmission && resumedAdmission->poses.images.at(2).registered && resumedAdmission->targets == admission.targets &&
        catalog.sourceLabel(2) == resumedAdmission->sources.at(0).label, "already imported raw source lost persistent admission P/K or target identity");
    regional::ImmutableSceneAliases nextAliases(root / "admission-aliases", 64u << 20); nextAliases.prepare(catalog, {0, 1, 2});
    regional::ImmutableSceneTopology nextIndex(root / "admission-index", 64u << 20);
    nextIndex.prepare(catalog, {0, 1, 2}, nextAliases.identity(), nextAliases.points(), [&](auto key) { return nextAliases.canonical(key); },
        [&](auto key, Vec3 xyz) { return key.source < 2 ? scene.acceptedPointXYZ(scene.acceptedCanonical(key)) : xyz; }, scene.identity());
    const auto nextBinding = Scene::bindingFor(catalog, nextIndex.identity()); auto nextPoses = poses(catalog); nextPoses.cameras.at(4).pixel_scale = 2;
    nextPoses.images.at(0).name = poseTable.images.at(0).name; nextPoses.images.at(1).name = poseTable.images.at(1).name;
    const auto oldIdentity = scene.identity(); const auto headSHA = sha(root / "admission-scene" / "head.bin");
    scene.beginLayoutCandidate(nextBinding, nextPoses, [&](const auto& emit) { nextAliases.emitAliases(emit); }, [&](const auto& emit) { emit(0, Vec3{100000, 2, 8}); emit(2, scene.acceptedPointXYZ(2)); });
    const auto regressed = regional::scenePrefixQuality(oldIndex, scene, 3, 16u << 20, 2);
    check(regressed.observations == 7 && !regional::canonicalPrefixAccepted(before, regressed) && regional::sceneQuality(nextIndex, scene, 3, 16u << 20, 2).observations == 10,
        "new cohort finite count hid previous raw objective regression");
    scene.abort("prefix-rejected");
    check(scene.identity() == oldIdentity && sha(root / "admission-scene" / "head.bin") == headSHA, "prefix rejection promoted layout");
    scene.beginLayoutCandidate(nextBinding, nextPoses, [&](const auto& emit) { nextAliases.emitAliases(emit); }, [&](const auto& emit) { emit(0, scene.acceptedPointXYZ(0)); emit(2, scene.acceptedPointXYZ(2)); });
    const auto unchanged = regional::scenePrefixQuality(oldIndex, scene, 3, 16u << 20, 2);
    check(unchanged.cost == before.cost && regional::canonicalPrefixAccepted(before, unchanged), "same-oldraw gate compared old/new cohort counts");
    scene.commit("prefix-fullraw-passed");
    Scene warm(root / "admission-scene", 64u << 20); warm.requireBinding(nextBinding);
    check(warm.observations() == 10 && warm.sourceCount() == 3 && warm.pointXYZ(0).x == 41, "admitted canonical prefix restore changed shared authority");
    regional::ImmutableSceneTopology competingIndex(root / "competing-index", 64u << 20);
    const std::vector<uint64_t> offsets{0, 1, 3};
    competingIndex.prepare(catalog, {0, 1, 2}, "different-canonical-for-one-feature", 4, [&](auto key) { return offsets.at(size_t(key.source)) + key.ordinal; });
    Scene competing(root / "competing-scene", 64u << 20);
    competing.initialize(Scene::bindingFor(catalog, competingIndex.identity()), poses(catalog), [&](const auto& emit) {
        for (size_t source = 0; source < catalog.sourceCount(); ++source) for (uint64_t ordinal = 0; ordinal < catalog.sourcePointCount(source); ++ordinal)
            emit(catalog.key(source, ordinal), offsets.at(source) + ordinal);
    }, [&](const auto& emit) { competingIndex.visitSeeds([&](const auto& point) { emit(point.root, point.xyz); }); });
    for (uint32_t id = 0; id < 3; ++id) {
        FeatureSet features; features.width = 1000; features.height = 800; features.dim = 1;
        features.keypoints = {{10, 20}, {30, 40}}; features.descriptors = {1, 2}; writeFeatures((featureDir / ("photo-" + std::to_string(id) + ".bin")).string(), features);
    }
    rejects([&] { regional::gatherCanonicalSeed(competingIndex, competing, {0}, [&](uint32_t id) { return featureDir / ("photo-" + std::to_string(id) + ".bin"); }, 32u << 20); }, "one original feature assigned to competing canonical points");
}
static void rejectedStepArchiveTest(const Path& root, Scene& scene, const Scene::Snapshot& proposal) {
    const auto head = sha(root / "scene" / "head.bin"), identity = scene.identity();
    std::vector<Path> archives;
    for (int level = 1; level <= 12; ++level) {
        const double alpha = std::ldexp(1., -level); scene.beginCandidate(); scene.interpolateCandidate(proposal, alpha);
        std::map<uint64_t, Vec3> expected; scene.visitCanonicalPoints([&](uint64_t id, Vec3 xyz) { expected.emplace(id, xyz); });
        const auto trial = scene.sealCandidate("fixture-rejected-raw-step");
        const auto archived = scene.preserveInterpolatedRejection(proposal, alpha, "fixture-fullraw-rejection");
        check(!std::filesystem::exists(archived / "points.bin") && std::filesystem::exists(archived / "step-reference.bin"), "eligible exact step retained redundant full XYZ instead of reference");
        const auto verified = scene.verifyRejectedStep(archived); check(verified.identity == trial.identity && verified.pointHash == trial.pointHash, "archived native reconstruction changed original trial SHA");
        uint64_t visited = 0; scene.visitRejectedStepPoints(archived, [&](uint64_t id, Vec3 xyz) { check(exact(expected.at(id), xyz), "native archived XYZ differs from actual rejected trial"); ++visited; });
        check(visited == 2 && scene.identity() == identity && sha(root / "scene" / "head.bin") == head, "archived rejection altered accepted head or root coverage");
        archives.push_back(archived);
    }
    check(std::filesystem::exists(proposal.directory / "points.bin"), "full coupled proposal was compacted");
    Scene reopened(root / "scene", 64u << 20); check(reopened.identity() == identity, "accepted reopen depended on archived trial representation");
    check(reopened.verifyRejectedStep(archives.front()).pointHash == scene.verifyRejectedStep(archives.front()).pointHash, "archived trial cannot be verified after restart");
    scene.beginCandidate(); scene.interpolateCandidate(proposal, .5); scene.setPose(0, scene.poses().images.at(0).pose);
    const auto mutated = scene.preserveInterpolatedRejection(proposal, .5, "fixture-post-mutator");
    check(std::filesystem::exists(mutated / "points.bin"), "post-interpolate mutator silently compressed a full candidate");
    scene.beginCandidate(); scene.interpolateCandidate(proposal, .5);
    const auto wrongAlpha = scene.preserveInterpolatedRejection(proposal, .25, "fixture-alpha-mismatch");
    check(std::filesystem::exists(wrongAlpha / "points.bin"), "mismatching archive alpha discarded full candidate");
    auto corrupt = [&](const char* label) {
        const auto target = root / "scene" / (std::string("rejected-archive-corrupt-") + label);
        std::filesystem::copy(archives.front(), target, std::filesystem::copy_options::recursive); return target;
    };
    const auto alphaBad = corrupt("alpha");
    {
        model_store_detail::Reader in(alphaBad / "step-reference.bin"); in.scalar<uint64_t>(); in.scalar<uint32_t>();
        for (int i = 0; i < 4; ++i) in.string(); const auto offset = in.position();
        std::fstream out(alphaBad / "step-reference.bin", std::ios::binary | std::ios::in | std::ios::out); const double changed = .25;
        out.seekp(std::streamoff(offset)); out.write(reinterpret_cast<const char*>(&changed), sizeof(changed)); out.flush();
    }
    rejects([&] { scene.verifyRejectedStep(alphaBad); }, "archived recipe alpha tamper accepted without trial point SHA equality");
    const auto hashBad = corrupt("basehash");
    {
        model_store_detail::Reader in(hashBad / "step-reference.bin"); in.scalar<uint64_t>(); in.scalar<uint32_t>(); in.string(); const auto offset = in.position() + sizeof(uint64_t);
        std::fstream out(hashBad / "step-reference.bin", std::ios::binary | std::ios::in | std::ios::out); const char changed = 'z'; out.seekp(std::streamoff(offset)); out.write(&changed, 1); out.flush();
    }
    rejects([&] { scene.verifyRejectedStep(hashBad); }, "archived recipe changed immutable base hash accepted");
    const auto missingBase = corrupt("missingbase");
    {
        model_store_detail::Reader in(missingBase / "step-reference.bin"); in.scalar<uint64_t>(); in.scalar<uint32_t>(); const auto offset = in.position() + sizeof(uint64_t);
        const auto name = in.string();
        std::fstream out(missingBase / "step-reference.bin", std::ios::binary | std::ios::in | std::ios::out); const char changed = 'X';
        out.seekp(std::streamoff(offset + name.size() - 1)); out.write(&changed, 1); out.flush();
    }
    rejects([&] { scene.verifyRejectedStep(missingBase); }, "archived recipe missing base accepted");
    const auto truncated = corrupt("truncated"); std::filesystem::resize_file(truncated / "step-reference.bin", 17);
    rejects([&] { scene.verifyRejectedStep(truncated); }, "truncated archived recipe accepted");
    const auto posesBad = corrupt("poses"); const auto value = model_store_detail::readModel(posesBad / "poses.bin"); auto changed = value; changed.images.at(0).pose.t.x += 1;
    model_store_detail::writeModel(changed, posesBad / "poses.bin");
    rejects([&] { scene.verifyRejectedStep(posesBad); }, "archived actual trial pose mutation accepted");
    const auto receiptBad = corrupt("receipt"); std::filesystem::resize_file(receiptBad / "step-reference-ready.bin", 4);
    rejects([&] { scene.verifyRejectedStep(receiptBad); }, "truncated producer receipt accepted");
    scene.beginCandidate(); auto noChange = scene.sealCandidate("unchanged-fullproposal"); noChange.directory = scene.abort();
    scene.beginCandidate(); scene.interpolateCandidate(noChange, .5); const auto noChangeTrial = scene.preserveInterpolatedRejection(noChange, .5, "unchanged-halfstep");
    check(!std::filesystem::exists(noChangeTrial / "points.bin"), "unchanged exact rejected trial cannot be archived");
    {
        model_store_detail::Reader in(noChangeTrial / "step-reference.bin"); in.scalar<uint64_t>(); in.scalar<uint32_t>();
        for (int i = 0; i < 4; ++i) in.string(); const auto offset = in.position();
        std::fstream out(noChangeTrial / "step-reference.bin", std::ios::binary | std::ios::in | std::ios::out); const double changedAlpha = .25;
        out.seekp(std::streamoff(offset)); out.write(reinterpret_cast<const char*>(&changedAlpha), sizeof(changedAlpha)); out.flush();
    }
    rejects([&] { scene.verifyRejectedStep(noChangeTrial); }, "alpha tamper accepted when proposal/base XYZ and poses identical");
    check(scene.identity() == identity && sha(root / "scene" / "head.bin") == head, "archive corruption verification changed accepted head");
}
int main(int argc, char** argv) try {
    const Path root = argc > 1 ? Path(argv[1]) : Path("canonical-scene-fixture");
    if (std::filesystem::exists(root)) throw std::runtime_error("fixture output must be new owned directory"); std::filesystem::create_directories(root);
    const auto source0 = root / "source-0.bin", source1 = root / "source-1.bin";
    const auto first = model(false), second = model(true); model_store_detail::writeModel(first, source0); model_store_detail::writeModel(second, source1);
    const auto sourceSHA0 = sha(source0), sourceSHA1 = sha(source1);
    Catalog catalog(root / "catalog", 64u << 20); catalog.importModel(source0, first, "source0"); catalog.importModel(source1, second, "source1");
    regional::ImmutableSceneAliases nativeAliases(root / "native-aliases", 64u << 20); catalog.resetCounters(); nativeAliases.prepare(catalog, {0, 1});
    check(nativeAliases.points() == 3 && nativeAliases.canonicalPoints() == 2 && nativeAliases.observations() == 7 && nativeAliases.canonical(catalog.key(1, 0)) == 0 && nativeAliases.canonical(catalog.key(1, 1)) == 2,
        "native immutable feature union/root/multiset differs");
    check(catalog.counters().parameter_bytes == 0, "immutable alias builder read retired source XYZ");
    const auto unionIdentity = nativeAliases.identity(), rootsHash = nativeAliases.rootsHash();
    regional::ImmutableSceneAliases warmAliases(root / "native-aliases", 64u << 20); warmAliases.prepare(catalog, {0, 1});
    check(warmAliases.identity() == unionIdentity && warmAliases.rootsHash() == rootsHash && warmAliases.canonicalGlobal(1) == 0, "cold/warm aliases differ");
    rejects([&] { warmAliases.canonical({1, 0, 999}); }, "immutable aliases accepted wrong local point ID");
    const auto badAliasCopy = root / "bad-alias-copy"; std::filesystem::copy(root / "native-aliases", badAliasCopy, std::filesystem::copy_options::recursive);
    std::filesystem::resize_file(badAliasCopy / unionIdentity / "roots.bin", 1);
    rejects([&] { regional::ImmutableSceneAliases broken(badAliasCopy, 64u << 20); broken.prepare(catalog, {0, 1}); }, "truncated immutable roots restored");
    auto badModel = second; badModel.images.at(0).points2D[0].x += 1;
    const auto badSource = root / "conflicting-observed-XY.bin"; model_store_detail::writeModel(badModel, badSource);
    Catalog conflicting(root / "conflicting-catalog", 64u << 20); conflicting.importModel(source0, first, "source0"); conflicting.importModel(badSource, badModel, "source1");
    rejects([&] { regional::ImmutableSceneAliases invalid(root / "conflicting-aliases", 64u << 20); invalid.prepare(conflicting, {0, 1}); }, "conflicting actual shared feature XY was unioned");
    const auto binding = Scene::bindingFor(catalog, "immutable-topology-0-0-2");
    const Scene::AliasProducer aliases = [&](const auto& emit) { emit(catalog.key(0, 0), 0); emit(catalog.key(1, 0), 0); emit(catalog.key(1, 1), 2); };
    uint64_t rawRows = 0; Vec3 weighted{}; uint64_t weight = 0;
    for (size_t source = 0; source < 2; ++source) catalog.visitSourcePoints(source, [&](auto key, auto xyz, const auto& track) {
        rawRows += track.size(); if (key.localPoint == 10 || key.localPoint == 20) { weighted = weighted + xyz * double(track.size()); weight += track.size(); }
    });
    weighted = weighted * (1. / double(weight)); check(rawRows == 7 && weight == 5 && exact(weighted, {41, 2, 8}), "seed did not use complete duplicated original tracks");
    const Scene::PointProducer seed = [&](const auto& emit) { emit(0, weighted); emit(2, Vec3{5, 6, 12}); };
    const auto work = root / "scene"; Scene scene(work, 64u << 20); scene.initialize(binding, poses(catalog), aliases, seed);
    check(scene.generation() == 0 && scene.pointSlots() == 3 && scene.canonicalPoints() == 2 && scene.observations() == 7, "canonical import counts disagree");
    check(scene.canonical(catalog.key(1, 0)) == 0 && exact(scene.xyz(catalog.key(1, 0)), weighted), "shared alias has separate coordinate authority");
    rejects([&] { scene.pointXYZ(1); }, "noncanonical alias coordinate readable");
    rejects([&] { scene.canonical({1, 0, 999}); }, "local lineage ID mismatch accepted");
    auto otherBinding = binding; otherBinding.rawCatalog = "different-raw"; rejects([&] { scene.requireBinding(otherBinding); }, "wrong raw binding accepted");
    const auto appearance = catalog.pointAppearance(catalog.key(0, 0)); check(appearance.rgb[1] == 29 && appearance.error == 0.75, "read-only point metadata lost");
    for (size_t source = 0; source < 2; ++source) {
        auto out = scene.materialize(catalog, source); check(exact(out.points3D.at(source ? 20 : 10).xyz, weighted), "materialization ignored canonical authority");
        check(out.points3D.at(source ? 20 : 10).track.size() == (source ? 2 : 3), "original duplicate observations discarded");
        model_store_detail::writeModel(out, root / ("materialized-" + std::to_string(source) + ".bin"));
        const auto roundtrip = model_store_detail::readModel(root / ("materialized-" + std::to_string(source) + ".bin"));
        check(roundtrip.points3D.at(source ? 20 : 10).track.size() == (source ? 2 : 3), "native Reader changed multiplicity");
    }
    catalog.beginGeneration(); catalog.setPointXYZ(catalog.key(1, 0), {9000, 9000, 9000}); catalog.commitGeneration("retired-source-only");
    check(exact(scene.xyz(catalog.key(1, 0)), weighted) && exact(scene.materialize(catalog, 1).points3D.at(20).xyz, weighted), "retired source XYZ remained runtime authority");
    const auto accepted0 = scene.acceptedSnapshot(); const auto head0 = sha(work / "head.bin");
    scene.beginCandidate(); rejects([&] { scene.applyPointUpdates({{1, {1, 1, 1}}}); }, "alias coordinate writable");
    rejects([&] { scene.applyPointUpdates({{0, {1, 1, 1}}, {0, {2, 2, 2}}}); }, "duplicate point updates accepted");
    scene.applyPointUpdates({{0, {81, 4, 16}}, {2, {7, 8, 14}}}); auto pose = scene.poses().images.at(0).pose; pose.t = {8, 0, 0}; scene.setPose(0, pose);
    auto camera = scene.poses().cameras.at(4); camera.fx = camera.fy = 900; scene.setCamera(camera);
    auto full = scene.sealCandidate(); const auto rejected = scene.abort("full-raw-gate-rejection"); full.directory = rejected;
    check(sha(work / "head.bin") == head0 && exact(scene.pointXYZ(0), weighted) && scene.poses().cameras.at(4).fx == 700, "reject did not atomically restore all parameters");
    scene.beginCandidate(); scene.interpolateCandidate(full, 0.5);
    check(exact(scene.pointXYZ(0), {61, 3, 12}) && exact(scene.xyz(catalog.key(1, 0)), {61, 3, 12}) && scene.poses().images.at(0).pose.t.x == 4 && scene.poses().cameras.at(4).fx == 800,
        "unified half step used different XYZ/P/K baseline");
    const auto accepted1 = scene.commit("complete-raw-gates-passed"); check(accepted1.generation == 1 && !accepted1.identity.empty(), "accepted snapshot missing generation/SHA");
    {
        Scene restored(work, 64u << 20); restored.requireBinding(binding);
        check(restored.identity() == accepted1.identity && exact(restored.pointXYZ(0), {61, 3, 12}) && restored.poses().cameras.at(4).fx == 800, "complete accepted snapshot restore failed");
    }
    const auto head1 = sha(work / "head.bin");
    {
        Scene crashed(work, 64u << 20); crashed.beginCandidate(); crashed.applyPointUpdates({{0, {777, 777, 777}}});
        crashed.setPose(0, {mat3Identity(), {99, 0, 0}}); crashed.sealCandidate("uncommitted");
    }
    {
        Scene restored(work, 64u << 20); check(sha(work / "head.bin") == head1 && exact(restored.pointXYZ(0), {61, 3, 12}), "interrupted candidate contaminated accepted head");
        bool retained = false; for (const auto& entry : std::filesystem::directory_iterator(work)) retained |= entry.path().filename().u8string().find("interrupted-") == 0;
        check(retained, "interrupted candidate was discarded");
    }
    const auto corrupt = root / "corrupt-copy"; std::filesystem::copy(work, corrupt, std::filesystem::copy_options::recursive);
    truncate(accepted1.directory / "points.bin", corrupt / accepted1.directory.filename() / "points.bin");
    rejects([&] { Scene broken(corrupt, 64u << 20); }, "truncated accepted canonical XYZ restored");
    auto newBinding = binding; newBinding.topology = "append-topology"; newBinding.rawCatalog = "append-raw-catalog"; newBinding.sources.push_back({"source2", "original-native-SHA", 1, 3});
    scene.beginLayoutCandidate(newBinding, scene.poses(), [&](const auto& emit) {
        aliases(emit); emit({2, 0, 80}, 0);
    }, [&](const auto& emit) { check(scene.acceptedCanonical(catalog.key(1, 0)) == 0, "append lost accepted alias identity"); emit(0, scene.acceptedPointXYZ(0) * (5. / 8.) + Vec3{81, 3, 12} * (3. / 8.)); emit(2, scene.acceptedPointXYZ(2)); });
    check(scene.xyz({2, 0, 80}).x == 68.5 && scene.observations() == 10 && scene.canonicalPoints() == 2, "append recreated alias authority");
    scene.abort("append-rejected"); check(scene.sourceCount() == 2 && scene.identity() == accepted1.identity && sha(work / "head.bin") == head1, "rejected layout promotion changed accepted catalogue");
    scene.beginCandidate(); rejects([&] { scene.interpolateCandidate(full, 0); }, "alpha zero accepted"); scene.abort();
    scene.beginCandidate(); auto originalXYZ = scene.pointXYZ(0);
    rejects([&] { scene.applyPointUpdates({{0, {std::numeric_limits<double>::quiet_NaN(), 1, 2}}}); }, "nonfinite canonical XYZ update accepted");
    check(exact(scene.pointXYZ(0), originalXYZ), "invalid XYZ batch changed candidate");
    rejects([&] { scene.rewriteCanonicalPoints([&](const auto& emit) { emit(0, Vec3{1, 2, 3}); }); }, "incomplete root producer accepted");
    check(exact(scene.pointXYZ(0), originalXYZ), "failed producer replaced complete canonical XYZ");
    scene.abort();
    scene.beginCandidate(); scene.applyPointUpdates({{0, {91, 5, 20}}, {2, {9, 10, 18}}});
    auto nextPose = scene.poses().images.at(1).pose; nextPose.R = so3Exp({0.02, -0.04, 0.06}); nextPose.t = {3, 2, 1}; scene.setPose(1, nextPose);
    auto nextK = scene.poses().cameras.at(4); nextK.fx = nextK.fy = 850; scene.setCamera(nextK);
    auto endpoint = scene.sealCandidate(); endpoint.directory = scene.abort();
    scene.beginCandidate(); scene.interpolateCandidate(endpoint, 1);
    check(exact(scene.pointXYZ(0), {91, 5, 20}) && scene.poses().images.at(1).pose.R == nextPose.R && exact(scene.poses().images.at(1).pose.t, nextPose.t) && scene.poses().cameras.at(4).fx == 850,
        "alpha-one endpoint changed proposal XYZ/P/K"); scene.abort();
    auto external = endpoint; external.directory = root / "external-rejected";
    std::filesystem::copy(endpoint.directory, external.directory, std::filesystem::copy_options::recursive);
    scene.beginCandidate(); rejects([&] { scene.interpolateCandidate(external, 0.5); }, "external proposal path accepted");
    auto unsealed = endpoint; unsealed.identity.clear(); rejects([&] { scene.interpolateCandidate(unsealed, 0.5); }, "unsealed proposal identity accepted"); scene.abort();
    auto changedNames = scene.poses(); changedNames.images.at(0).name = "wrong-same-length";
    scene.beginLayoutCandidate(binding, changedNames, aliases, seed); auto nameProposal = scene.sealCandidate(); nameProposal.directory = scene.abort();
    scene.beginCandidate(); rejects([&] { scene.interpolateCandidate(nameProposal, 0.5); }, "photo identity changed within proposal");
    check(exact(scene.pointXYZ(0), {61, 3, 12}) && scene.poses().images.at(0).name == "photo-0", "wrong identity rejection mutated parameters"); scene.abort();
    auto changedMetadata = scene.poses(); changedMetadata.images.at(0).exif_orientation = 3;
    scene.beginLayoutCandidate(binding, changedMetadata, aliases, seed); auto metadataProposal = scene.sealCandidate(); metadataProposal.directory = scene.abort();
    scene.beginCandidate(); rejects([&] { scene.interpolateCandidate(metadataProposal, 0.5); }, "same-size measurement metadata change accepted"); scene.abort();
    const auto invalidWork = root / "invalid-root"; Scene invalidScene(invalidWork, 64u << 20);
    rejects([&] { invalidScene.initialize(binding, poses(catalog), [&](const auto& emit) { emit(catalog.key(0, 0), 0); emit(catalog.key(1, 0), 0); emit(catalog.key(1, 1), 1); },
        [&](const auto& emit) { emit(0, weighted); }); }, "alias mapped to a nonroot accepted");
    check(!std::filesystem::exists(invalidWork / "head.bin"), "invalid topology became accepted"); if (invalidScene.editing()) invalidScene.abort();
    const auto autoWork = root / "auto-budget"; Scene adaptive(autoWork, 0); check(adaptive.residentBytes() < (16u << 20), "adaptive scene eagerly allocated hardware budget");
    check(sha(source0) == sourceSHA0 && sha(source1) == sourceSHA1, "immutable source model changed"); catalog.verifyImmutable();
    check(sha(accepted0.directory / "points.bin") != sha(accepted1.directory / "points.bin"), "fixture failed to exercise coordinate mutation");
    check(scene.residentBytes() < (64u << 20), "canonical memory budget not bounded");
    rejectedStepArchiveTest(root, scene, endpoint);
    admissionRecoveryTest(root, catalog);
    const auto originalHead = sha(work / "head.bin"), originalPoints = sha(scene.acceptedSnapshot().directory / "points.bin");
    const auto rawForkPath = catalog.forkImmutablePrefix(root / "fork-immutable-prefix", scene.sourceCount());
    Catalog rawFork(rawForkPath, 64u << 20);
    check(rawFork.sourceCount() == 2 && catalog.sourceCount() == 3 &&
        Scene::bindingFor(rawFork, scene.binding().topology) == scene.binding(), "immutable fork did not discard only the unaccepted suffix");
    uint64_t oldRows = 0, forkRows = 0;
    for (size_t source = 0; source < rawFork.sourceCount(); ++source) {
        std::vector<Catalog::RawObservation> retained;
        catalog.visitObservations([&](auto row) { if (row.point.source == source) { retained.push_back(row); ++oldRows; } });
        size_t at = 0; rawFork.visitObservations([&](auto row) {
            if (row.point.source != source) return;
            const auto& old = retained.at(at++); check(row.point == old.point && row.ordinal == old.ordinal && row.image == old.image &&
                row.feature == old.feature && row.xy.x == old.xy.x && row.xy.y == old.xy.y, "fork changed raw order, duplicate multiplicity or XY"); ++forkRows;
        }); check(at == retained.size(), "fork raw prefix incomplete");
    }
    check(oldRows == forkRows && forkRows == 7, "fork accepted raw multiset changed");
    const auto sceneForkPath = scene.forkAccepted(root / "fork-scene"); Scene forkScene(sceneForkPath, 64u << 20);
    check(forkScene.identity() == scene.identity() && forkScene.generation() == scene.generation() &&
        forkScene.observations() == 7 && forkScene.binding() == Scene::bindingFor(rawFork, scene.binding().topology), "fork accepted generation/SHA/raw binding changed");
    forkScene.beginCandidate(); forkScene.applyPointUpdates({{0, {177, 188, 199}}}); forkScene.commit("fork-native-COW-proof");
    check(sha(work / "head.bin") == originalHead && sha(scene.acceptedSnapshot().directory / "points.bin") == originalPoints &&
        exact(scene.pointXYZ(0), {61, 3, 12}) && exact(forkScene.pointXYZ(0), {177, 188, 199}), "fork COW changed original accepted hardlinks");
    rejects([&] { scene.forkAccepted(work); }, "fork into original root accepted");
    rejects([&] { scene.forkAccepted(root / "fork-scene"); }, "fork into existing destination accepted");
    rejects([&] { catalog.forkImmutablePrefix(root / "fork-immutable-prefix", 2); }, "raw fork into existing destination accepted");
    rejects([&] { catalog.forkImmutablePrefix(root / "invalid-prefix-fork", 4); }, "out-of-range raw prefix accepted");
#ifdef _WIN32
    auto caseScene = work.wstring(), caseRaw = (root / "catalog").wstring();
    std::transform(caseScene.begin(), caseScene.end(), caseScene.begin(), [](wchar_t c) { return wchar_t(std::towupper(c)); });
    std::transform(caseRaw.begin(), caseRaw.end(), caseRaw.begin(), [](wchar_t c) { return wchar_t(std::towupper(c)); });
    rejects([&] { scene.forkAccepted(Path(caseScene) / "nested-case-fork"); }, "Windows case variant bypassed source scene boundary");
    rejects([&] { catalog.forkImmutablePrefix(Path(caseRaw) / "nested-case-fork", 2); }, "Windows case variant bypassed source raw boundary");
#endif
    const auto tamperedForkRoot = root / "tampered-prefix-source";
    std::filesystem::copy(rawForkPath, tamperedForkRoot, std::filesystem::copy_options::recursive);
    const auto protectedRaw = tamperedForkRoot / "source-0" / "observations.bin";
    const auto protectedHash = sha(protectedRaw); const auto protectedBytes = std::filesystem::file_size(protectedRaw);
    {
        Catalog leasedRaw(tamperedForkRoot, 64u << 20, Catalog::OpenMode::ReadOnly);
#ifdef _WIN32
        rejects([&] { std::filesystem::resize_file(protectedRaw, 1); }, "live immutable source lease allowed a destructive write");
        check(sha(protectedRaw) == protectedHash && std::filesystem::file_size(protectedRaw) == protectedBytes,
            "blocked live-lease mutation changed immutable source bytes");
#endif
        leasedRaw.verifyImmutable();
    }
    std::filesystem::resize_file(protectedRaw, 1);
    rejects([&] { Catalog coldTamperedRaw(tamperedForkRoot, 64u << 20); coldTamperedRaw.forkImmutablePrefix(root / "tampered-prefix-destination", 2); },
        "cold native verification admitted a truncated immutable raw source");
    check(!std::filesystem::exists(root / "tampered-prefix-destination"), "invalid fork exposed an accepted destination");
    readOnlyAcceptedTest(root, scene);
    std::cout << "CanonicalScene: unique XYZ/P/K, 7/10-row multiset prefix, initial/admission crash stages, held SHA, immutable-only materialize, original-unit recovery, reject/blend/append/restore passed\n"; return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
