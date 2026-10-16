#include "sfm/map/CanonicalLayout.h"
#include "sfm/map/CanonicalResume.h"
#include "sfm/map/CanonicalInitialization.h"
#include "sfm/map/CanonicalAdmission.h"
#include "sfm/map/CanonicalRecovery.h"
#include "sfm/map/CanonicalImageIdentity.h"
#include "sfm/map/CanonicalAcceptance.h"
#include "sfm/map/CanonicalExportAttempt.h"
#include "sfm/map/SceneCoordinator.h"
#include "sfm/map/SceneVisualEvidence.h"
#include "sfm/map/HeldRepairSeed.h"
#include <iostream>

using namespace sfm;
using namespace sfm::regional;
namespace fs = std::filesystem;
static void check(bool condition, const char* reason) { if (!condition) throw std::runtime_error(reason); }
template<class Action> static void rejects(Action action, const char* reason) {
    bool rejected = false; try { action(); } catch (const std::exception&) { rejected = true; }
    check(rejected, reason);
}
static std::string sha(const fs::path& path) { const auto value = spirula::sha256_file(path.u8string()); check(!value.empty(), "fixture SHA unavailable"); return value; }
static Vec3 centre(uint32_t image) { return {double(image % 2) * 20, double(image / 2) * 25, double(image % 3) + 30}; }
static Vec3 truth(uint32_t point) { return {6 + .8 * (point % 6), 8 + .7 * (point / 6), 80 + .4 * (point % 3)}; }
struct FlowGps : PriorSource {
    bool has(uint32_t image) const override { return image < 4; }
    bool position(uint32_t image, Vec3& value) const override { if (!has(image)) return false; value = centre(image); return true; }
    bool relativeRotation(uint32_t, uint32_t, Mat3&, double&) const override { return false; }
    std::vector<uint32_t> neighbours(uint32_t) const override { return {}; }
    PosePriors factors(const std::vector<PosedImage>&) override { return {}; }
};
static Reconstruction original() {
    Reconstruction rec; rec.cameras[0] = Camera::defaultFor(0, 800, 600, 500, CamModel::Pinhole); rec.cameras[0].pixel_scale = 2;
    for (uint32_t image = 0; image < 4; ++image) {
        Image view; view.id = image; view.camera_id = 0; view.name = "photo-" + std::to_string(image) + ".JPG";
        view.registered = image < 3; view.pose = {mat3Identity(), centre(image) * -1};
        view.points2D.resize(25); view.point3D_ids.assign(25, kInvalidPoint3D);
        for (uint32_t point = 0; point < 25; ++point) {
            const auto xy = rec.cameras.at(0).project(truth(point) + view.pose.t);
            view.points2D[point] = {double(float(xy.x)), double(float(xy.y))};
        }
        rec.images.emplace(image, std::move(view));
    }
    for (uint32_t point = 0; point < 24; ++point) {
        Point3D value; value.xyz = truth(point) + Vec3{.01, -.01, .02};
        for (uint32_t image = 0; image < 3; ++image) { value.track.push_back({image, point}); rec.images.at(image).point3D_ids[point] = point; }
        value.track.push_back({0, point}); rec.points3D.emplace(point, std::move(value));
    }
    rec.next_point3D_id = 24; return rec;
}
static void requireAbsolute(const ExportFinalAcceptance& proof, uint32_t registered) {
    check(proof.historical_evaluated && proof.historical_verified, "explicit source history scope lost");
    for (const auto& photo : proof.photos) if (photo.image < registered)
        check(photo.gps_verified && photo.scale_verified && photo.attitude_verified && photo.reasons.empty(), "actual GPS/scale/attitude/source gate failed");
}
static void assessmentTest(const fs::path& coordination, CanonicalScene& scene) {
    const auto legacy = coordination / "round-2-acceptance.txt";
    if (!fs::exists(legacy)) resume::store(legacy, "generation=" + std::to_string(scene.generation()) +
        "\nraw=" + std::to_string(scene.observations()) + "\naccepted=1\nlegacy_evidence=true\n");
    const auto legacyHash = sha(legacy), headHash = sha(coordination / "scene" / "head.bin");
    const auto pointHash = sha(scene.acceptedSnapshot().directory / "points.bin"), identity = scene.identity();
    const auto first = storeCanonicalRoundAssessment(coordination, scene, 2, "accepted=0\nrejected=source_certificate_unknown\n");
    const auto firstHash = sha(first);
    const auto second = storeCanonicalRoundAssessment(coordination, scene, 2, "accepted=0\nrejected=fullraw_no_descent\n");
    check(first != second && first.parent_path() != second.parent_path() && sha(first) == firstHash && sha(legacy) == legacyHash,
        "same-head same-ordinal round evidence overwrote an earlier assessment or legacy round ledger");
    for (const auto& path : {first, second}) {
        const auto fields = canonicalResumeFields(path);
        check(fields.at("accepted_identity") == identity && fields.at("topology") == scene.binding().topology &&
            fields.at("attempted_round") == "2" && fields.at("accepted") == "0" && fields.count("rejected"),
            "round assessment lost its actual accepted snapshot binding or candidate rejection");
    }
    scene.beginCandidate();
    rejects([&] { storeCanonicalRoundAssessment(coordination, scene, 2, "accepted=0\n"); }, "editing scene wrote accepted round evidence");
    scene.abort("round-evidence-guard-fixture");
    CanonicalScene missing(coordination / "assessment-uninitialized", 64u << 20);
    rejects([&] { storeCanonicalRoundAssessment(coordination, missing, 2, "accepted=0\n"); }, "uninitialized scene wrote accepted round evidence");
    check(scene.identity() == identity && sha(coordination / "scene" / "head.bin") == headHash &&
        sha(scene.acceptedSnapshot().directory / "points.bin") == pointHash && sha(legacy) == legacyHash,
        "log-only evidence writer changed accepted authority or the original ledger");
}
int main(int argc, char** argv) try {
    const fs::path root = argc > 1 ? fs::u8path(argv[1]) : fs::path("canonical-full-flow-fixture");
    check(!fs::exists(root), "flow fixture requires a new output directory"); fs::create_directories(root / "features");
    const auto frozen = original(); const auto source = root / "original-source.bin"; model_store_detail::writeModel(frozen, source);
    if (argc > 2 && std::string(argv[2]) == "--assessment-only") {
        RegionalWorkingStore raw(root / "immutable-raw", 64u << 20); raw.importModel(source, frozen, "evidence-only-source");
        CanonicalScene scene(root / "scene", 64u << 20);
        scene.initialize(CanonicalScene::bindingFor(raw, "evidence-only-immutable-topology"), raw.poses(),
            [&](const auto& emit) { for (uint64_t point = 0; point < 24; ++point) emit(raw.key(0, point), point); },
            [&](const auto& emit) { for (const auto& point : frozen.points3D) emit(point.first, point.second.xyz); });
        assessmentTest(root, scene);
        std::cout << "round evidence-only passed: unique same-head attempts, legacy ledger and accepted authority unchanged; no geometry run\n";
        return 0;
    }
    const auto sourceHash = sha(source); std::map<uint32_t, ExportInputPhoto> inputs; std::vector<std::string> featureHashes;
    auto features = [&](uint32_t id) { return root / "features" / ("photo-" + std::to_string(id) + ".bin"); };
    fs::create_directories(root / "inputs");
    for (const auto& item : frozen.images) {
        FeatureSet feature; feature.width = 800; feature.height = 600; feature.extract_width = 400; feature.extract_height = 300;
        feature.keypoints.resize(25); feature.descriptors.resize(25 * feature.dim);
        for (size_t point = 0; point < 25; ++point) { feature.keypoints[point].x = float(item.second.points2D[point].x); feature.keypoints[point].y = float(item.second.points2D[point].y); }
        writeFeatures(features(item.first).u8string(), feature); featureHashes.push_back(sha(features(item.first)));
        const auto path = root / "inputs" / item.second.name; std::ofstream(path) << "native fixture input identity";
        inputs[item.first] = {path.u8string(), "photo-" + std::to_string(item.first), item.second.name};
    }
    FlowGps gps; AttitudeRef attitude; Region region; std::vector<uint32_t> owners(4, 0);
    for (uint32_t image = 0; image < 4; ++image) { attitude.image_ids.push_back(image); attitude.world_from_cam.push_back(mat3Identity()); region.core.push_back(image); region.images.push_back(image); }
    attitude.registered = 4; const std::vector<Region> plan{region};
    RegionalSourceFrame frame; frame.name = "immutable-original"; frame.views = frozen; frame.positioned = frame.certified = true;
    const std::vector<RegionalSourceFrame> sourceFrames{frame};
    const fs::path previous = root / "previous", next = root / "resumed";
    RegionalWorkingStore catalog(previous / "immutable-raw", 128u << 20); catalog.importModel(source, frozen, frame.name);
    CanonicalScene scene(previous / "scene", 128u << 20); auto poses = catalog.poses(); canonicalizeImageIdentity(poses, inputs); poses.images.at(1).pose.t.x += .3;
    auto cold = prepareCanonicalLayout(previous, catalog, scene, owners, 32u << 20);
    check(cold.phase == CanonicalLayoutPhase::Cold, "cold production selector did not initialize");
    scene.initializeValidated(cold.binding, poses, [&](const auto& emit) { cold.index->emitAliases(emit); },
        completeSceneSeed(*cold.index, poses, 3, 32u << 20, 2), [&](const auto& candidate) {
            const auto q = sceneQuality(*cold.index, candidate, 3, 32u << 20, 2);
            check(q.observations == 96 && q.invalid_projections == 0, "cold initialization lost original duplicate raw");
        }, "fixture-initial-fullraw");
    CanonicalInitialStage initial; initial.poses = scene.poses(); initial.fixedImages = {0, 2}; initial.targets = {3};
    initial.sources.push_back({frame.name, source, source, modelResidentBytes(frozen), {}});
    saveCanonicalInitialStage(previous / "initialization", "native-full-flow", initial);
    auto absolute = [&](const Reconstruction& candidate, const fs::path& evidence) {
        fs::create_directories(evidence);
        return canonicalCaptureAcceptance(candidate, inputs, plan, gps, attitude, sourceFrames, {}, false, evidence);
    };
    BundleOptions bundle; bundle.real = RealCfg::CPU; bundle.threads = 2; bundle.refine_intrinsics = false; bundle.max_iters = 1;
    SceneCoordinator::Options options; options.bundle = bundle; options.allowance = 64u << 20; options.backtrackSteps = 4;
    std::vector<CanonicalBundle> tasks(1); tasks[0].prepare(*cold.index, scene, {0, 1, 2}, 0, {0, 2}, bundle, 64u << 20);
    uint32_t gateCalls = 0;
    SceneCoordinator coordinator(scene, *cold.index, options, [&](const auto&, const auto& candidate) {
        ++gateCalls; const auto proof = absolute(candidate, root / "round-gates"); requireAbsolute(proof, 3); return std::vector<std::string>{};
    });
    for (uint64_t round = 1; round <= 2; ++round) {
        coordinator.setRoundOrdinal(round); const auto result = coordinator.round(tasks);
        check(result.accepted && result.after.observations == 96 && result.after.cost < result.before.cost && result.after.finite_cost < result.before.finite_cost,
            "real native BA round did not improve the unchanged full original multiset");
        storeCanonicalRounds(previous, scene, round);
        std::cout << "NATIVE_ROUND " << round << " before=" << result.before.cost << " after=" << result.after.cost << " alpha=" << result.alpha << '\n';
    }
    check(gateCalls >= 2 && canonicalCompletedRounds(previous, scene) == 2, "native accepted round ordinal or gates lost");
    assessmentTest(previous, scene);
    const auto previousHead = sha(previous / "scene" / "head.bin"), previousXYZ = sha(scene.acceptedSnapshot().directory / "points.bin");
    check(forkCanonicalResume(previous, next, 128u << 20), "production accepted-prefix fork failed");
    RegionalWorkingStore resumedRaw(next / "immutable-raw", 128u << 20); CanonicalScene resumed(next / "scene", 128u << 20);
    uint32_t scans = 0; resumedRaw.resetCounters();
    auto resumedLayout = prepareCanonicalLayout(next, resumedRaw, resumed, owners, 32u << 20,
        [&](const char* phase, uint64_t, uint64_t) { if (std::string(phase) == "source-scan") ++scans; });
    check(resumedLayout.phase == CanonicalLayoutPhase::ResumeAccepted && resumedLayout.index->identity() == cold.index->identity() && !scans &&
        resumed.identity() == scene.identity() && canonicalCompletedRounds(next, resumed) == 2, "fork resume rebuilt topology from current accepted parameters");
    auto shiftedOwners = owners; shiftedOwners[1] = 1;
    rejects([&] { prepareCanonicalLayout(next, resumedRaw, resumed, shiftedOwners, 32u << 20); }, "same-size owner mutation bypassed accepted layout identity");
    const auto heldSeed = gatherCanonicalSeed(*resumedLayout.index, resumed, {0, 1, 2}, features, 64u << 20);
    auto recovered = heldSeed; recovered.images.emplace(3, frozen.images.at(3)); recovered.images.at(3).registered = true; recovered.images.at(3).name = inputs.at(3).name;
    for (auto& point : recovered.points3D) { recovered.images.at(3).point3D_ids[point.first] = point.first; point.second.track.push_back({3, uint32_t(point.first)}); }
    recovered.points3D.at(0).track.push_back({3, 0});
    auto& changedTrack = recovered.points3D.at(0).track;
    changedTrack.erase(std::remove_if(changedTrack.begin(), changedTrack.end(), [](const auto& row) { return row.image_id == 1 && row.point2D_idx == 0; }), changedTrack.end());
    recovered.points3D.at(1).track.push_back({1, 0}); recovered.images.at(1).point3D_ids[0] = 1;
    Point3D privatePoint; privatePoint.xyz = truth(24) + Vec3{.02, .01, .03}; privatePoint.track = {{2, 24}, {3, 24}};
    recovered.points3D.emplace(24, privatePoint); recovered.images.at(2).point3D_ids[24] = recovered.images.at(3).point3D_ids[24] = 24; recovered.next_point3D_id = 25;
    const auto rawBeforeRestore = countObservations(recovered);
    const auto restoredBindings = restoreAcceptedFeatureBindings(*resumedLayout.index, resumed, recovered, 64u << 20);
    check(restoredBindings.reassociated_features == 1 && restoredBindings.moved_raw_rows == 1 && countObservations(recovered) == rawBeforeRestore,
        "recovery did not repair old target binding or dropped original incoming raw");
    HeldRepairSeedContract(heldSeed).requireUnchanged(recovered);
    const auto recoveredOriginal = root / "recovered-native.bin"; model_store_detail::writeModel(recovered, recoveredOriginal);
    const auto recoveredHash = sha(recoveredOriginal);
    auto incoming = recovered; canonicalizeSourceImageIdentity(incoming, inputs, resumedRaw.poses());
    const auto incomingPath = root / "incoming-catalogue.bin"; model_store_detail::writeModel(incoming, incomingPath); const auto incomingHash = sha(incomingPath);
    resumedRaw.importModel(incomingPath, incoming, "held-native-recovery");
    auto appended = prepareCanonicalLayout(next, resumedRaw, resumed, owners, 32u << 20);
    check(appended.phase == CanonicalLayoutPhase::AppendSources, "production selector omitted appended source");
    auto starting = resumed.poses(); starting.images.at(3).registered = true; starting.images.at(3).pose = recovered.images.at(3).pose;
    const auto prefixBefore = scenePrefixQuality(*resumedLayout.index, resumed, 3, 32u << 20, 2, true);
    auto seed = prepareCanonicalAdmissionSeed(*resumedLayout.index, *appended.index, resumed, starting, 3, 64u << 20, 2);
    resumed.beginLayoutCandidate(appended.binding, starting, [&](const auto& emit) { appended.index->emitAliases(emit); }, seed.producer);
    const auto prefixAfter = scenePrefixQuality(*resumedLayout.index, resumed, 3, 32u << 20, 2);
    check(canonicalPrefixAccepted(prefixBefore, prefixAfter) && prefixAfter.observations == 96 && seed.evidence->oldPreservedRoots == 24 && seed.evidence->oldRefinedRoots == 0,
        "appended recovery reseeded unchanged accepted roots or violated old raw objective");
    requireAbsolute(absolute(resumed.poses(), root / "admission-gates"), 4);
    const auto joinedQuality = sceneQuality(*appended.index, resumed, 3, 32u << 20, 2);
    const auto joinedVisual = sceneVisualEvidence(*appended.index, resumed, 3, 32u << 20, 2);
    check(joinedQuality.observations == 96 + rawBeforeRestore && joinedQuality.invalid_projections == 0 && joinedVisual.components == 1,
        "new source complete multiset or connected visual evidence changed");
    for (uint32_t image = 0; image < 4; ++image) check(joinedVisual.landmarks.at(image) >= 12 && photoQualityAccepted(joinedVisual.photos.at(image), 3), "recovery lacks independently supported visual landmarks");
    resumed.commit("canonical-new-cohort-fullraw-prefix-v1");
    const auto beforeHalf = resumed.poses(); const auto correctedPose = frozen.images.at(1).pose;
    const auto correction = (beforeHalf.images.at(1).pose.t - correctedPose.t).norm();
    check(correction > 1e-8, "real native BA fixture has no remaining pose correction for coupled step");
    uint32_t halfGates = 0;
    SceneCoordinator protectedStep(resumed, *appended.index, options, [&](const auto& baseline, const auto& candidate) {
        ++halfGates; requireAbsolute(absolute(candidate, root / "half-step-gates"), 4);
        return (candidate.images.at(1).pose.t - baseline.images.at(1).pose.t).norm() > correction * .6 ?
            std::vector<std::string>{"protected_source_pose_step"} : std::vector<std::string>{};
    });
    protectedStep.setRoundOrdinal(1); std::vector<CanonicalBundle> noTasks;
    const auto half = protectedStep.round(noTasks, [&](auto&, auto& candidate, const auto&) { candidate.setPose(1, correctedPose); });
    check(half.accepted && half.alpha == .5 && halfGates == 2 && half.after.observations == joinedQuality.observations &&
        half.after.invalid_projections == 0 && half.after.finite_cost < half.before.finite_cost && half.after.cost < half.before.cost,
        "append coupled half-step did not preserve the full raw objective and call every original gate");
    storeCanonicalRounds(next, resumed, 1);
    std::cout << "COUPLED_HALF before=" << half.before.cost << " after=" << half.after.cost << " raw=" << half.after.observations << '\n';
    const auto appendedIdentity = resumed.identity(), appendedXYZ = sha(resumed.acceptedSnapshot().directory / "points.bin");
    const auto appendedHead = sha(next / "scene" / "head.bin"); uint32_t rejectionGates = 0;
    SceneCoordinator denied(resumed, *appended.index, options, [&](const auto&, const auto&) { ++rejectionGates; return std::vector<std::string>{"required_source_certificate_absent"}; });
    std::vector<CanonicalBundle> joinedTasks(1); joinedTasks[0].prepare(*appended.index, resumed, {0, 1, 2, 3}, 0, {0, 2}, bundle, 64u << 20);
    denied.setRoundOrdinal(2); const auto rejected = denied.round(joinedTasks);
    check(!rejected.accepted && rejectionGates == 5 && rejected.preserved.size() == 5 && !resumed.editing() && resumed.identity() == appendedIdentity &&
        sha(next / "scene" / "head.bin") == appendedHead && sha(resumed.acceptedSnapshot().directory / "points.bin") == appendedXYZ,
        "all-gate rejection escaped a trial, changed accepted parameters, or lost rejected evidence");
    check(fs::is_regular_file(rejected.preserved.front() / "points.bin"), "full rejected proposal not preserved");
    for (size_t i = 1; i < rejected.preserved.size(); ++i) check(resumed.verifyRejectedStep(rejected.preserved[i]).observations == joinedQuality.observations, "compact rejected trial cannot reconstruct native original scope");
    const auto acceptedRounds = canonicalCompletedRounds(next, resumed);
    storeCanonicalRounds(next, resumed, acceptedRounds, true, "rejected", acceptedRounds + 1);
    CanonicalScene reopened(next / "scene", 128u << 20); auto reopenedLayout = prepareCanonicalLayout(next, resumedRaw, reopened, owners, 32u << 20);
    const auto stopped = canonicalResumeFields(next / ("cohort-rounds-" + reopened.binding().topology + ".txt"));
    check(canonicalRoundsFinished(next, reopened) && canonicalCompletedRounds(next, reopened) == acceptedRounds && stopped.at("termination") == "rejected" &&
        stopped.at("full_capture_verified") == "false" && reopened.identity() == appendedIdentity, "reopen retried rejected round or promoted rejection to convergence/success");
    check(fs::is_regular_file(next / "indices" / appended.index->identity() / "layout-seed-context.txt"), "append layout lacks preserved parent provenance");
    const auto secondFork = root / "append-fork"; check(forkCanonicalResume(next, secondFork, 128u << 20), "appended accepted fork failed");
    RegionalWorkingStore secondRaw(secondFork / "immutable-raw", 128u << 20); CanonicalScene secondScene(secondFork / "scene", 128u << 20);
    auto secondLayout = prepareCanonicalLayout(secondFork, secondRaw, secondScene, owners, 32u << 20);
    check(secondLayout.index->identity() == appended.index->identity() && secondScene.identity() == appendedIdentity, "appended fork lost seed context or changed topology authority");
    const auto forkStopped = canonicalResumeFields(secondFork / ("cohort-rounds-" + secondScene.binding().topology + ".txt"));
    check(canonicalRoundsFinished(secondFork, secondScene) && canonicalCompletedRounds(secondFork, secondScene) == acceptedRounds &&
        forkStopped.at("termination") == "rejected" && forkStopped.at("attempted_rounds") == std::to_string(acceptedRounds + 1) &&
        forkStopped.at("full_capture_verified") == "false" && sha(next / "scene" / "head.bin") == appendedHead,
        "fork dropped rejected stop reason/attempted ordinal or changed its accepted source head");
    CanonicalScene uninitialized(root / "uninitialized", 64u << 20);
    rejects([&] { reserveCanonicalExportAttempt(next, root / "sparse", uninitialized); }, "uninitialized export admitted");
    reopened.beginCandidate(); rejects([&] { reserveCanonicalExportAttempt(next, root / "sparse", reopened); }, "editing export admitted"); reopened.abort("editing-export-probe");
    VerifiedExportOptions exportOptions; exportOptions.cache_bytes = 8u << 20; exportOptions.reader_memory_bytes = 64u << 20;
    exportOptions.inputs = inputs; exportOptions.owners = &owners; exportOptions.input_path = [&](uint32_t image) { return inputs.at(image).path; };
    exportOptions.accepted_raw = sceneQuality(*reopenedLayout.index, reopened, 3, 32u << 20, 2);
    const auto firstAttempt = reserveCanonicalExportAttempt(next, root / "sparse", reopened);
    auto doExport = [&](const CanonicalScene& accepted, const CanonicalExportAttempt& attempt) {
        return exportVerifiedScene(resumedRaw, accepted, features, attempt.scratch, attempt.pending, exportOptions,
            [&](const Reconstruction& readback, const Quality&, const std::map<uint32_t, PhotoQuality>&, const ExportStats&) {
                return absolute(readback, attempt.pending / "constraints");
            });
    };
    const auto first = doExport(reopened, firstAttempt);
    for (const auto& failure : first.failures) std::cerr << "FLOW_EXPORT_FAILURE " << failure.reason << '\n';
    check(first.verified && first.reader_verified && first.raw.observations == 96 + rawBeforeRestore && first.lineage_rows == first.raw.observations &&
        first.stats.points == 25 && first.unique_quality.size() == 4, "actual formal export Reader/full raw/unique point contract failed");
    const auto firstXYZ = sha(firstAttempt.pending / "points3D.bin"), firstPhotos = sha(firstAttempt.pending / "export-per-photo.csv");
    std::ifstream photoLedger(firstAttempt.pending / "export-per-photo.csv"); std::string line; std::getline(photoLedger, line);
    check(line.find("unique_visual_landmarks") != std::string::npos && line.find("verified,reason") != std::string::npos, "formal per-photo ledger omitted visual or measured status");
    auto csvFields = [](const std::string& row) {
        std::vector<std::string> fields(1); bool quoted = false;
        for (size_t i = 0; i < row.size(); ++i) {
            if (row[i] == '"') {
                if (quoted && i + 1 < row.size() && row[i + 1] == '"') { fields.back() += '"'; ++i; }
                else quoted = !quoted;
            } else if (row[i] == ',' && !quoted) fields.emplace_back();
            else if (row[i] != '\r') fields.back() += row[i];
        }
        check(!quoted, "formal photo CSV has an unterminated quoted field"); return fields;
    };
    const auto columns = csvFields(line);
    auto column = [&](const char* name) { const auto at = std::find(columns.begin(), columns.end(), name); check(at != columns.end(), "formal photo CSV column missing"); return size_t(at - columns.begin()); };
    const auto verifiedColumn = column("verified"), reasonColumn = column("reason");
    std::set<uint32_t> reported;
    while (std::getline(photoLedger, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto image = uint32_t(std::stoul(line.substr(0, line.find(','))));
        const auto values = csvFields(line);
        check(inputs.count(image) && reported.insert(image).second && values.size() == columns.size() && values.at(verifiedColumn) == "1" && values.at(reasonColumn) == "verified",
            "serialized photo ledger duplicated, omitted, or failed an input photo");
    }
    check(reported.size() == inputs.size(), "serialized per-photo status did not cover every input");
    const auto actual = Reconstruction::readBinary(firstAttempt.pending.u8string());
    check(actual.numRegistered() == 4 && actual.points3D.size() == 25 && countObservations(actual) == 98, "formal native Reader did not preserve actual unique photo/point support");
    for (uint32_t image = 0; image < 4; ++image) check(first.constraints.at(image).gps_verified && first.constraints.at(image).scale_verified && first.constraints.at(image).attitude_verified && first.unique_landmarks.at(image) >= 12,
        "a photo lacks full measured formal status");
    CanonicalScene exportReopened(next / "scene", 128u << 20); const auto secondAttempt = reserveCanonicalExportAttempt(next, root / "sparse", exportReopened);
    check(firstAttempt.pending != secondAttempt.pending && firstAttempt.scratch != secondAttempt.scratch, "same-head reopen reused a previous export attempt");
    const auto second = doExport(exportReopened, secondAttempt);
    check(second.verified && second.reader_verified && sha(firstAttempt.pending / "points3D.bin") == firstXYZ && sha(firstAttempt.pending / "export-per-photo.csv") == firstPhotos,
        "same-head second export changed first proof or failed actual native readback");
    check(sha(source) == sourceHash && sha(recoveredOriginal) == recoveredHash && sha(incomingPath) == incomingHash &&
        sha(previous / "scene" / "head.bin") == previousHead && sha(scene.acceptedSnapshot().directory / "points.bin") == previousXYZ, "complete lifecycle mutated a frozen source or pre-fork accepted snapshot");
    for (uint32_t image = 0; image < 4; ++image) check(sha(features(image)) == featureHashes.at(image), "native lifecycle changed input features");
    resume::store(root / "proof.txt", "native_multistage_fixture=true\nreal_capture_verified=false\nrecovery_scope=accepted_seed_and_binding_handoff_not_mapper\nregistered=4\nraw=" + std::to_string(first.raw.observations) +
        "\naccepted_rounds_before_append=2\nrejected_trials=5\nunique_formal_points=25\nformal_reader_twice=true\nper_photo_status_count=4\n");
    std::cout << "canonical full flow passed: native BA, fork/resume, restored bindings, append/fullraw, rejected/reopen, twice formal Reader and all-photo status\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
